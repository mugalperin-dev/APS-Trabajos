from scipy.io import loadmat
from scipy.interpolate import CubicSpline
from scipy.signal import  find_peaks, butter, filtfilt
from scipy.ndimage import median_filter
from scipy.stats import median_abs_deviation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargar ECG
data = loadmat('./ECG_TP4.mat')
ecg = np.array(data['ecg_lead']).flatten()
qrs_ref = np.array(data['qrs_detections']).flatten().astype(int)
qrs_pattern = np.array(data['qrs_pattern1']).flatten()

fs = 1000
t = np.arange(len(ecg)) / fs
    
#%% Punto 1:

# Ventanas en muestras
T1 = int (0.200 * fs)   # 200 ms
T2 = int (0.600 * fs)   # 600 ms

# Aplicar mediana de 200 ms
b = median_filter(ecg, size=T1, mode='reflect')

# Aplicar mediana de 600 ms sobre el resultado
base = median_filter(b, size=T2, mode='reflect')

# ECG sin línea de base (método mediana)
ecg_filt = ecg - base

plt.figure(figsize=(12,4))
plt.plot(t, ecg, label='ECG original')
plt.plot(t, base, label='Línea de base calculada por medianas', linewidth=2)
#plt.xlim(0, 1000)
plt.legend()
plt.xlabel('Tiempo [s]')
plt.title('Estimación de línea de base')
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
plt.plot(t, ecg_filt, label='ECG filtrado')
plt.xlim(0, 1000)
plt.legend()
plt.xlabel('Tiempo [s]')
plt.title('ECG sin movimiento de línea de base')
plt.grid(True)
plt.show()

#%% Punto 2:
    
# Elegís n0 como adelanto respecto al QRS para caer en el segmento PQ
# (puede ser algo entre 80 y 150 ms, lo vas ajustando)
n0_ms = 120  # en ms
n0 = int(n0_ms * fs / 1000)  # pasar a muestras

# Evitar índices negativos
m = qrs_ref[qrs_ref > n0] - n0  # tiempos mi
s_m = ecg[m]                    # valores s(mi)

# Interpolación spline cúbica
spline = CubicSpline(m, s_m)

n = np.arange(len(ecg))
b_hat_spline = spline(n)

# ECG sin línea de base (método spline)
ecg_f_spline = ecg - b_hat_spline

# Graficar comparación
plt.figure(figsize=(12,4))
plt.plot(t, ecg, label='ECG original')
plt.plot(t, b_hat_spline, label='Línea de base')
#plt.xlim(0, 1000)
plt.legend()
plt.xlabel('Tiempo [s]')
plt.title('Estimación de línea de base con splines cúbicos')
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
plt.plot(t, ecg_f_spline, label='ECG filtrado (spline)')
plt.xlim(0, 1000)
plt.legend()
plt.xlabel('Tiempo [s]')
plt.title('ECG sin movimiento de línea de base (spline)')
plt.grid(True)
plt.show()

#%% Punto 3:
"""
# 1) Construir filtro adaptado (respuesta al impulso = patrón dado vuelta)
h = qrs_pattern[::-1]

# 2) Filtrar el ECG (equivalente a correlación con el patrón)
# Usamos lfilter o np.convolve; con 'same' queda alineado con la señal original
y = np.convolve(ecg, h, mode='same')

# 3) Detección de picos sobre la salida del filtro adaptado
#    - distance: periodo refractario (~250 ms)
#    - height: umbral relativo (lo podés ir ajustando)
min_distance = int(0.25 * fs)   # 250 ms
threshold = 0.4 * np.max(y)     # 40% del máximo, ajustable

peaks, props = find_peaks(y, distance=min_distance, height=threshold)
qrs_detected = peaks  # índices de muestra de los latidos detectados

# 4) Mostrar un segmento para ver las detecciones
t = np.arange(len(ecg)) / fs

plt.figure(figsize=(12,4))
plt.plot(t, ecg, label='ECG')
plt.plot(t[peaks], ecg[peaks], 'o', label='Detecciones (matched filter)')
#plt.xlim(0, 10)
plt.legend()
plt.xlabel('Tiempo [s]')
plt.title('Detección de QRS con filtro adaptado')
plt.grid(True)
plt.show()

# 5) Comparar con qrs_detections de la consigna
# Definimos una tolerancia de ventana para considerar que una detección coincide
tol = int(0.05 * fs)  # 50 ms

detected_used = np.zeros_like(qrs_detected, dtype=bool)
TP = 0  # verdaderos positivos
FN = 0  # falsos negativos
FP = 0  # falsos positivos

# Para cada latido "verdadero" buscamos una detección cerca
for q in qrs_ref:
    # diferencias con todas las detecciones
    diffs = np.abs(qrs_detected - q)
    # índice de la detección más cercana
    if len(diffs) == 0:
        FN += 1
        continue
    idx_min = np.argmin(diffs)
    if diffs[idx_min] <= tol:
        TP += 1
        detected_used[idx_min] = True
    else:
        FN += 1

# Cualquier detección que no se usó es un falso positivo
FP = np.sum(~detected_used)

# Métricas
sensibilidad = TP / (TP + FN) if (TP + FN) > 0 else 0
ppv = TP / (TP + FP) if (TP + FP) > 0 else 0  # Valor predictivo positivo

print(f"TP = {TP}, FN = {FN}, FP = {FP}")
print(f"Sensibilidad = {sensibilidad*100:.2f} %")
print(f"Valor predictivo positivo (PPV) = {ppv*100:.2f} %")
"""

#%%
def bandpass(x, fs, low=5, high=25, order=3):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def compute_metrics(detected, reference, fs, tol_ms):
    """
    Computar metricas:
        Si coincide con un latido real dentro de la tolerancia -> True Positive
        Sino coincide con un latido real -> False Positive
        Diferencia entre latidos reales y TP -> False Negative
        
    """
    tol = int((tol_ms/1000) * fs)
    used = np.zeros(len(detected), dtype=bool)
    TP = 0
    FN = 0
    for q in reference:
        if len(detected) == 0:
            FN += 1
            continue
        dif = np.abs(detected - q)
        i = np.argmin(dif)
        if dif[i] <= tol:
            TP += 1
            used[i] = True
        else:
            FN += 1
    FP = np.sum(~used)
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    ppv  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    return {"TP": TP, "FN": FN, "FP": int(FP), "sensitivity": sens, "ppv": ppv}

# Filtrar y normalizar ECG
ecg_f = bandpass(ecg, fs)
ecg_f = (ecg_f - np.mean(ecg_f)) / (np.std(ecg_f) + 1e-12)

# Normalizar el patron de referencia 
qrs_pattern = (qrs_pattern - np.mean(qrs_pattern)) / (np.std(qrs_pattern) + 1e-12)

# Inversion del patron y convolucion
h = qrs_pattern[::-1]
y = np.convolve(ecg_f, h, mode='same')

# Umbral de deteccion de los picos
thr = np.percentile(y, 90) + 3 * median_abs_deviation(y)

# Minima distancia entre picos
min_dist = int(0.25 * fs)

peaks, props = find_peaks(y, height=thr, distance=min_dist)
qrs_detected = peaks

metrics = compute_metrics(qrs_detected, qrs_ref, fs, tol_ms=25)

TP = metrics["TP"]
FN = metrics["FN"]
FP = metrics["FP"]
TN = 0  # No existe TN en detección de eventos

# Construir la matriz de confusión
conf_matrix = pd.DataFrame(
    [[TP, FN],
     [FP, TN]],
    index=["Real P", "Real N"],
    columns=["Predicho P", "Predicho N"]
)

print(f"Sensibilidad = {metrics['sensitivity']*100:.2f} %")
print(f"PPV = {metrics['ppv']*100:.2f} %")

print(conf_matrix)


