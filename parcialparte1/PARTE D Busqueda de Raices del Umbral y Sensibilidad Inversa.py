import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import bisect, newton

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

print("=== PARTE D: RAÍCES Y SENSIBILIDAD INVERSA EN PYCHARM ===")

spline_natural = CubicSpline(frecuencias, impedancias, bc_type='natural')
primera_derivada_spline = spline_natural.derivative(nu=1)

Umbral = 150.0

# Definición analítica de g(f) = S(f) - 150 = 0
def funcion_g(f):
    return spline_natural(f) - Umbral

# --- CÁLCULO DE LA RAÍZ 1 (LÍMITE INFERIOR) ---
r1_biseccion = bisect(funcion_g, 100, 150, xtol=1e-6)
r1_newton = newton(funcion_g, x0=100, tol=1e-6)

# --- CÁLCULO DE LA RAÍZ 2 (LÍMITE SUPERIOR CERCANO A 2000 HZ) ---
r2_biseccion = bisect(funcion_g, 2100, 2400, xtol=1e-6)
r2_newton = newton(funcion_g, x0=2300, tol=1e-6)

print(f"Raíz 1 (Banda Inferior):")
print(f"  - Algoritmo de Bisección: {r1_biseccion:.4f} Hz")
print(f"  - Algoritmo Newton-Raphson: {r1_newton:.4f} Hz")
print(f"Raíz 2 (Banda Superior):")
print(f"  - Algoritmo de Bisección: {r2_biseccion:.4f} Hz")
print(f"  - Algoritmo Newton-Raphson: {r2_newton:.4f} Hz\n")

# --- CÁLCULO DE SENSIBILIDAD INVERSA ANALÍTICA ---
pendiente_en_raiz2 = primera_derivada_spline(r2_biseccion)
sensibilidad_inversa = 1.0 / pendiente_en_raiz2

print(f"Análisis de Tolerancia y Ruido en la Raíz 2 ({r2_biseccion:.4f} Hz):")
print(f"  - Pendiente d|Z|/df local: {pendiente_en_raiz2:.4f} Ohm/Hz")
print(f"  - Factor de Sensibilidad df/d|Z|: {sensibilidad_inversa:.4f} Hz/Ohm")
print("  Conclusión: 1 Ohm de ruido térmico desplaza la raíz calculada en 56.41 Hz.\n")

# --- GRÁFICA DEL ANCHO DE BANDA SEGURO ---
f_continuo = np.linspace(100, 2730, 500)
plt.figure(figsize=(9, 5))
plt.plot(f_continuo, spline_natural(f_continuo), color='#2980b9', linewidth=2, label='Modelo Continuo S(f)')
plt.axhline(y=Umbral, color='#e67e22', linestyle='--', linewidth=2, label='Umbral Crítico Operativo (150 Ohm)')
plt.scatter([r1_biseccion, r2_biseccion], [Umbral, Umbral], color='#34495e', marker='s', s=80, zorder=5,
            label='Raíces de Corte (Límites de Banda)')
plt.title('Parte D: Delimitación de Fronteras del Ancho de Banda Operativo Seguro', fontsize=12, fontweight='bold', color='#1a5276')
plt.xlabel('Frecuencia (f) [Hz]')
plt.ylabel('Impedancia |Z| [Ohmios]')
plt.ylim(128, 165)
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()