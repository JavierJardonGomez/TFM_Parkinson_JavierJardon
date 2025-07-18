import os
import librosa
import numpy as np
import statistics

# Carpeta donde se encuentran los archivos WAV
input_folder = "../pre_processed_input_wavs_e/"

# Lista para almacenar las duraciones de los audios
durations = []

# Procesar los archivos WAV
for file_name in os.listdir(input_folder):
    if file_name.endswith(".wav"):
        file_path = os.path.join(input_folder, file_name)
        
        # Cargar audio con librosa
        y, sr = librosa.load(file_path, sr=None)  # sr=None preserva la tasa de muestreo original
        
        # Calcular duración en segundos
        duration = librosa.get_duration(y=y, sr=sr)
        durations.append(duration)

if durations:
    # Estadísticas básicas
    min_duration = np.min(durations)
    max_duration = np.max(durations)
    mean_duration = np.mean(durations)
    median_duration = np.median(durations)
    mode_duration = statistics.mode(durations) if len(set(durations)) > 1 else "No hay una moda única"
    std_dev_duration = np.std(durations)

    # Mostrar los resultados
    print("Análisis Estadístico de las Duraciones de los Audios:")
    print(f"Duración mínima: {min_duration:.2f} segundos")
    print(f"Duración máxima: {max_duration:.2f} segundos")
    print(f"Duración media: {mean_duration:.2f} segundos")
    print(f"Duración mediana: {median_duration:.2f} segundos")
    print(f"Moda de duración: {mode_duration}")
    print(f"Desviación estándar: {std_dev_duration:.2f} segundos")
    
else:
    print("No se encontraron archivos de audio en la carpeta.")
