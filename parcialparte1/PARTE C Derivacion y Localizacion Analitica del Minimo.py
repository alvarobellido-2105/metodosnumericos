import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import bisect

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

print("=== PARTE C: DERIVACIÓN NUMÉRICA Y OPTIMIZACIÓN EN PYCHARM ===")

spline_natural = CubicSpline(frecuencias, impedancias, bc_type='natural')

# Extracción analítica infinitesimal de las derivadas de SciPy
primera_derivada_spline = spline_natural.derivative(nu=1)
segunda_derivada_spline = spline_natural.derivative(nu=2)

# Buscamos la raíz de la derivada en el intervalo de confinamiento verificado [700, 800]
f_min_analitico = bisect(primera_derivada_spline, 700, 800, xtol=1e-6)
z_min_analitico = spline_natural(f_min_analitico)
valor_segunda_derivada = segunda_derivada_spline(f_min_analitico)

print(f"Frecuencia Crítica Exacta del Mínimo (f_mín): {f_min_analitico:.4f} Hz")
print(f"Impedancia Mínima Calculada Absoluta (|Z|_mín): {z_min_analitico:.4f} Ohm")
print(f"Valor de la Segunda Derivada S''(f_mín): {valor_segunda_derivada:+.8f} Ohm/Hz^2")

if valor_segunda_derivada > 0:
    print("Validación de Estabilidad: Curvatura positiva confirmada. El punto es un MÍNIMO LOCAL ESTABLE.\n")

# --- GRÁFICA CON ACERCAMIENTO DINÁMICO (ZOOM EN EL VALLE) ---
f_zoom = np.linspace(550, 950, 300)
plt.figure(figsize=(9, 5))
plt.plot(f_zoom, spline_natural(f_zoom), color='#1a5276', linewidth=2, label='Curva del Spline S(f)')
plt.scatter(frecuencias[(frecuencias >= 550) & (frecuencias <= 950)],
            impedancias[(frecuencias >= 550) & (frecuencias <= 950)],
            color='#e74c3c', s=50, zorder=4, label='Muestras Locales')
plt.scatter(f_min_analitico, z_min_analitico, color='#f1c40f', edgecolor='#d4ac0d',
            marker='^', s=130, zorder=5, label=f'Mínimo Analítico ({f_min_analitico:.2f} Hz)')
plt.title('Parte C: Optimización y Acercamiento Matemático en el Mínimo Absoluto', fontsize=12, fontweight='bold', color='#1a5276')
plt.xlabel('Frecuencia (f) [Hz]')
plt.ylabel('Impedancia |Z| [Ohmios]')
plt.xlim(550, 950)
plt.ylim(130.5, 132.5)
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()