# ─── 1. Carga y utilidades ──────────────────────────────────────
from typing import List, Tuple
import librosa, os, numpy as np


def read_audios_dir(audio_dir: str) -> Tuple[List[str], List[str], dict]:
    """
    Escanea la carpeta y crea:
      • real_files       - ficheros .wav reales      (lista de str)
      • generated_files  - ficheros .wav generados   (lista de str)
      • label_dict       - {basename: etiqueta 0/1}
    """
    label_dict, real_files, generated_files = {}, [], []

    for fn in os.listdir(audio_dir):
        if not fn.endswith(".wav"):
            continue
        base = os.path.splitext(fn)[0]

        # Etiquetas
        if "_p" in base:
            label_dict[base] = 1
        elif "_c" in base:
            label_dict[base] = 0

        # Tipo de archivo
        if "generated" in fn:
            generated_files.append(base)
        else:
            real_files.append(base)

    print(f"Total reales: {len(real_files)}, generados: {len(generated_files)}")
    return real_files, generated_files, label_dict


def read_audios_og(audio_dir: str):
    real, _, labels = read_audios_dir(audio_dir)
    return real, labels



def load_waveform(file_path: str,
                  target_len: int,
                  duration: float | None = None) -> np.ndarray:

    y, sr = librosa.load(file_path, sr=None, duration=duration)
    if len(y) < target_len:                      # padding con ceros
        y = np.pad(y, (0, target_len - len(y)))
    else:                                        # recorte
        y = y[:target_len]
    return y.astype(np.float32)                  # (target_len,)


def load_data_raw(file_list: List[str],
                  label_dict: dict,
                  audio_dir: str = "",
                  target_len: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga cada wav como onda 1-D y lo normaliza a longitud fija = target_len.
    Salida:
        X → shape (num_samples, target_len, 1)   # canal univariante
        y → shape (num_samples, 1)
    """
    X_local, y_local = [], []
    for fname in file_list:
        wav = load_waveform(os.path.join(audio_dir, fname + ".wav"), target_len)
        X_local.append(wav[:, None])          
        y_local.append(label_dict[fname])

    return np.array(X_local), np.array(y_local).reshape(-1, 1)