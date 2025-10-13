# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 17:03:10 2025

@author: Usuario
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

from lectura_sigs import ecg_one_lead, fs_ecg, ppg, fs_ppg, fs_audio, wav_data

# %%

plt.figure()
plt.plot(ecg_one_lead)

N = len(ecg_one_lead)


cantidad_promedios = 10
nperseg = N // cantidad_promedios
f, Pxx = sig.welch(ecg_one_lead, fs=fs_ecg, nperseg = nperseg, window = 'hann', nfft = 2 * N)


total_power = np.sum(Pxx)
cumsum = np.cumsum(Pxx) / total_power

# niveles inferior y superior
percentil = 0.95
low = (1 - percentil) / 2
high = 1 - low

# buscar índices
indiceLow = np.searchsorted(cumsum, low)
indiceHigh = np.searchsorted(cumsum, high)

bandWith = f[indiceLow:indiceHigh]

print (f"frecuencia baja: {bandWith[0]:.2f}, frecuencia alta {bandWith[-1]:.2f}")
print (f"ancho de banda {bandWith[-1] - bandWith[0]:.2f}")

# Graficar
#ymin, ymax = plt.ylim()   # límites actuales del eje y

plt.figure(figsize=(10,5))
plt.plot(f, Pxx)
plt.title("PDS del ECG sin ruido - Método de Welch")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud ")
plt.xlim(0,60)
#plt.ylim(0,50)
plt.axvspan(bandWith[0], bandWith[-1],color = "grey", alpha=0.3, label = f" ancho de banda al {percentil *  100:.0f} % de energia")
plt.legend()
plt.grid(True)
plt.show()

# %%

plt.figure()
plt.plot(ppg)
plt.grid(True)

N = len(ppg)
cantidad_promedios = 7
nperseg = N // cantidad_promedios
f, Pxx = sig.welch(ecg_one_lead, fs=fs_ppg, nperseg = nperseg, window = 'flattop', nfft = 2 * N)


total_power = np.sum(Pxx)
cumsum = np.cumsum(Pxx) / total_power

# niveles inferior y superior
percentil = 0.95
low = (1 - percentil) / 2
high = 1 - low

# buscar índices
indiceLow = np.searchsorted(cumsum, low)
indiceHigh = np.searchsorted(cumsum, high)

bandWith = f[indiceLow:indiceHigh]

print (f"frecuencia baja: {bandWith[0]:.2f}, frecuencia alta {bandWith[-1]:.2f}")

# Graficar
ymin, ymax = plt.ylim()   # límites actuales del eje y

plt.figure(figsize=(10,5))
plt.plot(f, Pxx)   
plt.xlim(0,20)
plt.title("PSD de PPG sin ruido - Método de Welch")
plt.xlabel("Frecuencia [Hz]")
plt.axvspan(bandWith[0], bandWith[-1],color = "grey", alpha=0.3, label = f" ancho de banda al {percentil *  100:.0f} % de energia")
plt.legend()
plt.grid(True)
plt.show()
# %%
plt.figure()
plt.plot(wav_data)
plt.grid(True)


N = len(wav_data)
cantidad_promedios = 9
nperseg = N // cantidad_promedios
f, Pxx = sig.welch(ecg_one_lead, fs=fs_audio, nperseg = nperseg, window = 'flattop', nfft = 5 * N)


total_power = np.sum(Pxx)
cumsum = np.cumsum(Pxx) / total_power

# niveles inferior y superior
percentil = 0.95
low = (1 - percentil) / 2
high = 1 - low

# buscar índices
indiceLow = np.searchsorted(cumsum, low)
indiceHigh = np.searchsorted(cumsum, high)

bandWith = f[indiceLow:indiceHigh]

print (f"frecuencia baja: {bandWith[0]:.2f}, frecuencia alta {bandWith[-1]:.2f}")
print (f"ancho de banda {bandWith[-1] - bandWith[0]:.2f}")

# Graficar
ymin, ymax = plt.ylim()   # límites actuales del eje y

plt.figure(figsize=(10,5))
plt.plot(f, Pxx)   
plt.xlim(0,1500)
plt.title("PSD de PPG sin ruido - Método de Welch")
plt.xlabel("Frecuencia [Hz]")
plt.axvspan(bandWith[0], bandWith[-1],color = "grey", alpha=0.3, label = f" ancho de banda al {percentil *  100:.0f} % de energia")
plt.legend()
plt.grid(True)
plt.show()