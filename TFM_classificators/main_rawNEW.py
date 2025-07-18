import os
import re
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from data.processing_data_raw import read_audios_dir, load_data_raw
from models.factory import get_model
from train.training_utils import train_loop, validate_real_data, evaluate_metrics
from utils.plotting import (
    plot_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
)
from models.inceptiontime_model import InceptionTimeEnsemble

# --------------------------- .env -------------------------------------------
load_dotenv(".env")
AUDIO_DIR = os.getenv("AUDIO_DIR_NAME", "wavs_e_22050Hz_normalized_057")

# --------------------------- Experimentos ------------------------------------

experiment_ids = [40, 41, 42, 43]
batch_sizes   = [8, 16, 32, 64]
lrs           = [1e-5, 3e-5, 6e-5, 8e-5]   # subo el de batch16 y ajusto linealmente
weight_decays = [2e-4, 1e-4, 1e-4, 1e-4]   # WD un pelín mayor para lotes pequeños

TARGET_LEN = int(0.57 * 22_050)
all_model_names = ["resnet", "inception", "conv", "lstm_fcn", "times_net"]
BASE_RESULTS_DIR = "results5Goodfinal_noLeak"


def base_id(fname: str) -> str:
    """Devuelve identificador base (original + sintéticos)."""
    stem = Path(fname).stem
    return re.split(r"_aug\d+_generated", stem)[0]


def k_fold_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 5):
    gkf = GroupKFold(n_splits=k)
    for tr, va in gkf.split(X, y, groups):
        yield tr, va



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDispositivo: {device}\n")

for exp_id, bs, lr, wd in zip(experiment_ids, batch_sizes, lrs, weight_decays):
    print(f"=== Experimento {exp_id} | BS={bs}, LR={lr}, WD={wd} ===")
    exp_dir = os.path.join(BASE_RESULTS_DIR, f"V5_{exp_id}_bs{bs}_lr{lr}_wd{wd}")
    os.makedirs(exp_dir, exist_ok=True)

    # 1) Listas de audio ------------------------------------------------------
    audio_dir = f"./data/{AUDIO_DIR}"
    real_files, gen_files, label_dict = read_audios_dir(audio_dir)

    real_clean = [f for f in real_files if "_generated" not in f]
    real_c = [f for f in real_clean if f.endswith("_c")]
    real_p = [f for f in real_clean if f.endswith("_p")]

    random.seed(42)
    real_val_names = random.sample(real_c, 60) + random.sample(real_p, 60)
    val_base_ids = {base_id(f) for f in real_val_names}

    real_train_names = [f for f in real_clean if f not in real_val_names]
    gen_train = [f for f in gen_files if base_id(f) not in val_base_ids]
    train_names = real_train_names + gen_train

    # 2) Groups & y -----------------------------------------------------------
    groups = np.array([base_id(f) for f in train_names])

    # Cargar features **y etiquetas reales**
    X_train, y_train_labels = load_data_raw(train_names, label_dict, audio_dir, TARGET_LEN)
    X_test,  y_test_labels  = load_data_raw(real_val_names, label_dict, audio_dir, TARGET_LEN)

    enc = OneHotEncoder(sparse_output=False)
    y_onehot = enc.fit_transform(y_train_labels.reshape(-1, 1))

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_onehot, dtype=torch.float32)

    # 3) Cross-validation -----------------------------------------------------
    for model_name in all_model_names:
        print(f"\n--- Modelo: {model_name} ---")
        model_dir = os.path.join(exp_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        out_base = os.path.join(model_dir, f"{AUDIO_DIR}_{exp_id}")

        best_acc, best_path = 0.0, ""
        metrics = {"acc": [], "prec": [], "rec": [], "f1": []}

        for fold, (tr_idx, va_idx) in enumerate(k_fold_split(X_tensor, y_tensor, groups)):
            print(f"Fold {fold+1}: {len(tr_idx)} train – {len(va_idx)} val")
            X_tr, X_va = X_tensor[tr_idx], X_tensor[va_idx]
            y_tr, y_va = y_tensor[tr_idx], y_tensor[va_idx]

            train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
            val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=bs)

            if model_name == "inception":
                model = InceptionTimeEnsemble(n_models=5, input_channels=1, nb_classes=2).to(device)
            else:
                model = get_model(model_name, input_size=X_tr.shape[2], output_size=2, device=device)

            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
            sched = ReduceLROnPlateau(opt, mode="min", factor=0.4, patience=15)
            save_fold = os.path.join(model_dir, f"{model_name}_fold{fold+1}.pth")

            model, hist = train_loop(
                model, train_loader, val_loader, nn.CrossEntropyLoss(),
                opt, sched, device, model_name,
                epochs=500, patience=15,
                save_path=save_fold,
                plot_path=f"{out_base}_fold{fold+1}.png",
            )

            acc, pr, rc, f1 = evaluate_metrics(model, X_va, y_va, device, model_name)
            for k, v in zip(metrics.keys(), [acc, pr, rc, f1]):
                metrics[k].append(v)

            if hist["val_acc"][-1] > best_acc:
                best_acc, best_path = hist["val_acc"][-1], save_fold

        # 4) Resumen CV -------------------------------------------------------
        def summarize(name: str, vals: List[float]):
            mu, sd = np.mean(vals), np.std(vals)
            return f"{name}: {mu:.4f} ± {sd:.4f} (CV {sd / mu * 100:.2f}%)"

        print("\nResultados k-fold")
        for m, v in metrics.items():
            print(summarize(m, v))
        with open(f"{out_base}_kfold.txt", "w") as fh:
            for m, v in metrics.items():
                fh.write(summarize(m, v) + "\n")

        # 5) Test ciego ------------------------------------------------------
        print("\nEvaluación ciega")
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.to(device).eval()

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(enc.transform(y_test_labels.reshape(-1, 1)), dtype=torch.float32)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=16)

        probs, preds, labels = validate_real_data(model, test_loader, device, model_name)
        plot_classification_report(labels, preds, ["Control", "Patológico"], out_base)
        plot_confusion_matrix(labels, preds, ["Control", "Patológico"], out_base)
        plot_roc_curve(labels, probs, out_base)

print("\nEntrenamiento completado noLeak (GroupKFold + etiquetas correctas)")
