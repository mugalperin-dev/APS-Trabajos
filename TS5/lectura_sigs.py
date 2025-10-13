#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:55:30 2023

@author: mariano
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write


#%%

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
# sio.whosmat('ECG_TP4.mat')
# mat_struct = sio.loadmat('./ECG_TP4.mat')

# ecg_one_lead = np.squeeze(mat_struct['ecg_lead'])
# N = len(ecg_one_lead)

# # hb_1 = mat_struct['heartbeat_pattern1']
# # hb_2 = mat_struct['heartbeat_pattern2']

# plt.figure()
# plt.plot(ecg_one_lead[5000:12000])

# plt.figure()
# plt.plot(hb_1)

# plt.figure()
# plt.plot(hb_2)

##################
## ECG sin ruido
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy')

# plt.figure()
# plt.plot(ecg_one_lead)

# N = len(ecg_one_lead)


# cantidad_promedios = 15
# nperseg = N // cantidad_promedios
# f, Pxx = sig.welch(ecg_one_lead, fs=fs_ecg, nperseg = nperseg, window = 'hann', nfft = 2 * N)


# total_power = np.sum(Pxx)
# cumsum = np.cumsum(Pxx) / total_power

# # niveles inferior y superior
# percentil = 0.95
# low = (1 - percentil) / 2
# high = 1 - low

# # buscar índices
# indiceLow = np.searchsorted(cumsum, low)
# indiceHigh = np.searchsorted(cumsum, high)

# bandWith = f[indiceLow:indiceHigh]

# print (f"frecuencia baja: {bandWith[0]:.2f}, frecuencia alta {bandWith[-1]:.2f}")
# print (f"ancho de banda {bandWith[-1] - bandWith[0]:.2f}")

# # Graficar
# #ymin, ymax = plt.ylim()   # límites actuales del eje y

# plt.figure(figsize=(10,5))
# plt.plot(f, Pxx)
# plt.title("PDS del ECG sin ruido - Método de Welch")
# plt.xlabel("Frecuencia [Hz]")
# plt.ylabel("Amplitud ")
# plt.xlim(0,60)
# #plt.ylim(0,50)
# plt.axvspan(bandWith[0], bandWith[-1],color = "grey", alpha=0.3, label = f" ancho de banda al {percentil *  100:.0f} % de energia")
# plt.legend()
# plt.grid(True)
# plt.show()

#%%

####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

##################
## PPG con ruido
##################

# # Cargar el archivo CSV como un array de NumPy
# ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe


##################
## PPG sin ruido
##################

ppg = np.load('ppg_sin_ruido.npy')

# %%
# plt.figure()
# plt.plot(ppg)
# plt.grid(True)

# N = len(ppg)
# cantidad_promedios = 25
# nperseg = N // cantidad_promedios
# f, Pxx = sig.welch(ecg_one_lead, fs=fs_ppg, nperseg = nperseg, window = 'hann', nfft = 5 * N)


# total_power = np.sum(Pxx)
# cumsum = np.cumsum(Pxx) / total_power

# # niveles inferior y superior
# low = 0.00
# high = 0.95

# # buscar índices
# indiceLow = np.searchsorted(cumsum, low)
# indiceHigh = np.searchsorted(cumsum, high)

# BandWith = f[indiceLow:indiceHigh]

# print (f"frecuencia baja: {BandWith[0]:.2f}, frecuencia alta {BandWith[-1]:.2f}")

# # Graficar
# ymin, ymax = plt.ylim()   # límites actuales del eje y

# plt.figure(figsize=(10,5))
# plt.plot(f, Pxx)   
# plt.xlim(0,50)
# plt.title("Densidad Espectral de Potencia - Método de Welch")
# plt.xlabel("Frecuencia [Hz]")
# plt.axvspan(BandWith[0], BandWith[-1],color = "grey", alpha=0.3)
# plt.grid(True)
# plt.show()
# # %%

#%%

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# fs_audio, wav_data = sio.wavfile.read('silbido.wav')

#
# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
# import sounddevice as sd
# sd.play(wav_data, fs_audio)

