import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Datos del Experimento
frecuencias = np.array([
    100, 120, 145, 170, 200, 235, 270, 310, 355, 405,
    460, 520, 585, 655, 730, 810, 895, 985, 1080, 1180,
    1290, 1410, 1540, 1680, 1830, 1990, 2160, 2340, 2530, 2730
], dtype=float)

impedancias = np.array([
    152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 133.6,
    132.7, 131.9, 131.4, 131.1, 130.9, 131.2, 131.6, 132.1, 132.9, 133.9,
    135.2, 136.9, 138.9, 141.1, 143.4, 146.0, 149.0, 152.4, 156.1, 160.5
], dtype=float)

print("=== PARTE B2: INTERPOLACIÓN POR SPLINES EN PYCHARM ===")

# Inicialización obligatoria con bc_type='natural' para forzar S''(extremos)=0
spline_natural = CubicSpline(frecuencias, impedancias, bc_type='natural')

# Evaluación matemática directa en 1000 Hz
z_spline_1000 = spline_natural(1000)
print(f"Magnitud de Impedancia exacta en f = 1000 Hz: {z_spline_1000:.4f} Ohm\n")

# Graficación fluida del Spline
f_continuo = np.linspace(100, 2730, 500)
plt.figure(figsize=(9, 5))
plt.scatter(frecuencias, impedancias, color='#e74c3c', label='Nodos de Control (Datos)')
plt.plot(f_continuo, spline_natural(f_continuo), color='#2980b9', linewidth=2.5, label='Spline Cúbico Natural Continuo')
plt.title('Parte B2: Ajuste Perfecto y Suavizado por Splines Cúbicos Naturales', fontsize=12, fontweight='bold', color='#1a5276')
plt.xlabel('Frecuencia (f) [Hz]')
plt.ylabel('Impedancia |Z| [Ohmios]')
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()