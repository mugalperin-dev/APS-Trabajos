import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Definición de transferencias
T = {
    "T_a": ([1, 1, 1, 1], [1, 0, 0, 0]),          # z^3 + z^2 + z + 1 / z^3
    "T_b": ([1, 1, 1, 1, 1], [1, 0, 0, 0, 0]),    # z^4 + z^3 + z^2 + z + 1 / z^4
    "T_c": ([1, 1], [1, 0]),                      # z + 1 / z
    "T_d": ([1, 0, 1], [1, 0, 0])                 # z^2 + 1 / z^2
}

# Grafco de polos y ceros
def plot_pz(num, den, title):
    z, p, _ = signal.tf2zpk(num, den)
    plt.figure(figsize=(4,4))
    plt.axhline(0, color='0.7')
    plt.axvline(0, color='0.7')
    
    # Círculo unitario
    circle = plt.Circle((0, 0), 1, color='black', fill=False, ls='dotted')
    plt.gca().add_artist(circle)
    
    # Ceros (o) y Polos (x)
    plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='b', label='Ceros')
    plt.scatter(np.real(p), np.imag(p), marker='x', color='r', label='Polos')
    
    plt.title(f"Diagrama de Polos y Ceros - {title}")
    plt.xlabel("Parte real")
    plt.ylabel("Parte imaginaria")
    plt.xlim(-1.25,1.25)
    plt.ylim(-1.25,1.25)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_freq_response(num, den, title):
    # Calcula la respuesta en frecuencia 
    w, h = signal.freqz(num, den, worN=1024)

    # Magnitud normalizada (en dB)
    mag = 20 * np.log10(np.abs(h) / np.max(np.abs(h)))

    phase = np.unwrap(np.angle(h))

    # --- Gráfico del módulo ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(w, mag)
    plt.axhline(y=-3, linestyle='--', color ='black', alpha = 0.7, label='-3 dB')
    plt.title(f"Módulo de {title}")
    plt.xlabel("ω [rad/muestra]")
    plt.ylabel("Magnitud [dB]")
    plt.xlim(0, np.pi)
    plt.legend()                   
    plt.ylim(-70, 5)                     
    plt.grid(True)

    # --- Gráfico de la fase ---
    plt.subplot(1, 2, 2)
    plt.plot(w, phase)
    plt.title(f"Fase de {title}")
    plt.xlabel("ω [rad/muestra]")
    plt.ylabel("Fase [rad]")
    plt.xlim(0, np.pi)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


plot_pz(*T["T_a"], "T_a")
# plot_pz(*T["T_b"], "T_b")
# plot_pz(*T["T_c"], "T_c")
# plot_pz(*T["T_d"], "T_d")


plot_freq_response(*T["T_a"], "T_a")
# plot_freq_response(*T["T_b"], "T_b")
# plot_freq_response(*T["T_c"], "T_c")
# plot_freq_response(*T["T_d"], "T_d")