import os
import re
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from dotenv import dotenv_values
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from data.processing_data_raw import read_audios_dir, load_data_raw
from models.factory import get_model
from train.training_utils import train_loop, evaluate_metrics, validate_real_data
from utils.plotting import (
    plot_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
)

# --------------------------- Configuración global ----------------------------
config = dotenv_values(".env.og")

audio_dir_name = "wavs_e_22050Hz_normalized_057"
validation_type = "kfold"  # solo se usa kfold en este script

experiment_ids = [40, 41, 42, 43,]
batch_sizes    = [8, 16, 32, 64]
lrs            = [1e-5, 2e-5, 5e-5, 8e-5] 
weight_decays  = [1e-4, 1e-4, 1e-4, 1e-4]

TARGET_LEN = int(0.57 * 22_050)  # 570 ms a 22,05 kHz
all_model_names = ["resnet", "inception", "conv", "lstm_fcn", "times_net"]
BASE_RESULTS_DIR = "OGresults5Goodfinal"



def base_id(fname: str) -> str:
    """Extrae el identificador base sin sufijos de augmentación/"generated"."""
    stem = Path(fname).stem  # quita .wav
    return re.split(r"_aug\d+_generated", stem)[0]


def k_fold_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 5):
    gkf = GroupKFold(n_splits=k)
    for tr, va in gkf.split(X, y, groups):
        yield tr, va


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDispositivo detectado: {device}\n")

for exp_id, bs, lr, wd in zip(experiment_ids, batch_sizes, lrs, weight_decays):
    print(f"=== Experimento {exp_id} | BS={bs}, LR={lr}, WD={wd} ===")

    exp_dir = os.path.join(
        BASE_RESULTS_DIR,
        f"normalized_22050_{exp_id}_bs{bs}_lr{lr}_wd{wd}",
    )
    os.makedirs(exp_dir, exist_ok=True)


    
    # 1. Cargar listas de archivos  -----------------------------------------
    audio_dir = f"./data/{audio_dir_name}"
    real_files, generated_files, label_dict = read_audios_dir(audio_dir)

    # Elige 30 controles + 30 patológicos (reales) para test a ciegas
    real_files_clean = [f for f in real_files if "_generated" not in f]
    real_control = [f for f in real_files_clean if f.endswith("_c")]
    real_patologico = [f for f in real_files_clean if f.endswith("_p")]

    assert len(real_control) >= 30 and len(real_patologico) >= 30, "Dataset insuficiente"
    random.seed(42)
    real_val_names = random.sample(real_control, 30) + random.sample(real_patologico, 30)
    real_train_names = [f for f in real_files_clean if f not in real_val_names]

    # Filtra sintéticos cuya base aparezca en validación real
    real_val_bases = {f.replace("_generated", "") for f in real_val_names}
    filtered_generated_files = [
        f for f in generated_files if f.replace("_generated", "") not in real_val_bases
    ]

    train_names = real_train_names + filtered_generated_files
    
    # 2. Vectores de grupos (etiquetas se obtendrán en el paso 3) -------------
    groups = np.array([base_id(f) for f in train_names])        # (N,)
    
    

    # 3. Carga de características y etiquetas (alineadas)
    X_train, y_train_labels = load_data_raw(  # y_train_labels ya viene correcta (0/1)
        train_names, label_dict, audio_dir, TARGET_LEN
    )
    X_test,  y_test_labels  = load_data_raw(
        real_val_names, label_dict, audio_dir, TARGET_LEN
    )
    
    # One-hot global (manteniendo el nombre 'enc' y 'y_onehot')
    enc       = OneHotEncoder(sparse_output=False)
    y_onehot  = enc.fit_transform(y_train_labels.reshape(-1, 1))   # (N,2)
    
    # Tensores que el resto del script espera
    X_tensor       = torch.tensor(X_train, dtype=torch.float32)
    y_tensor       = torch.tensor(y_onehot, dtype=torch.float32)
    
    X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor  = torch.tensor(
        enc.transform(y_test_labels.reshape(-1, 1)), dtype=torch.float32
    )
    
    # (comprobación opcional)
    for i in range(5):
        print(train_names[i], y_train_labels[i])

    # 4. CV por modelo------------------------------------------------------------
    for model_name in all_model_names:
        print(f"\n--- Modelo: {model_name} ---")
        model_dir = os.path.join(exp_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        output_name = os.path.join(model_dir, f"{audio_dir_name}_{exp_id}_{model_name}")

        fold_metrics = {"acc": [], "prec": [], "rec": [], "f1": []}
        best_val_acc = 0.0
        best_model_path = ""

        for fold, (tr_idx, val_idx) in enumerate(k_fold_split(X_tensor, y_tensor, groups, k=5)):
            print(f"Fold {fold+1}: {len(tr_idx)} train – {len(val_idx)} val")

            X_tr, X_val = X_tensor[tr_idx], X_tensor[val_idx]
            y_tr, y_val = y_tensor[tr_idx], y_tensor[val_idx]

            train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=bs)

            model = get_model(model_name, input_size=X_tr.shape[2], output_size=2, device=device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=15)

            save_path = os.path.join(model_dir, f"{model_name}_fold{fold+1}_best.pth")
            model, history = train_loop(
                model,
                train_loader,
                val_loader,
                nn.CrossEntropyLoss(),
                optimizer,
                scheduler,
                device,
                model_name,
                epochs=500,
                patience=15,
                save_path=save_path,
                plot_path=f"{output_name}_fold{fold+1}",
            )

            # Métricas del fold
            acc, prec, rec, f1 = evaluate_metrics(model, X_val, y_val, device, model_name)
            for k, v in zip(fold_metrics.keys(), [acc, prec, rec, f1]):
                fold_metrics[k].append(v)

            if history["val_acc"][-1] > best_val_acc:
                best_val_acc = history["val_acc"][-1]
                best_model_path = save_path

        # ----------- Resumen CV -----------
        def stat(name: str, vals: List[float]) -> str:
            mu, sig = np.mean(vals), np.std(vals)
            cv = sig / mu * 100 if mu else 0
            return f"{name}: media={mu:.4f} | std={sig:.4f} | cv={cv:.2f}%"

        print("\nResultados K‑Fold")
        for metric, vals in fold_metrics.items():
            print(stat(metric, vals))

        with open(f"{output_name}_kfold_metrics.txt", "w", encoding="utf-8") as fh:
            fh.write("Resultados K‑Fold\n")
            for m, v in fold_metrics.items():
                fh.write(stat(m, v) + "\n")

        # ----------- Evaluación final (150 reales) -----------
       
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=16)

        print("\nEvaluación en 150 audios reales no vistos…")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device).eval()
        probs, preds, labels = validate_real_data(model, test_loader, device, model_name)

        plot_classification_report(labels, preds, ["Control", "Patológico"], output_name)
        plot_confusion_matrix(labels, preds, ["Control", "Patológico"], output_name)
        plot_roc_curve(labels, probs, output_name)

print("\nEntrenamiento completado noLeak")
