import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# --- 1. DATOS DE ENTRADA ---
f = np.array([10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 57.5, 60.0, 62.5, 65.0, 67.5, 70.0, 72.5, 75.0, 77.5, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5])
V = np.array([0.842, 0.911, 0.986, 1.062, 1.143, 1.227, 1.314, 1.401, 1.482, 1.551, 1.216, 1.048, 0.866, 0.689, 0.521, 0.364, 0.223, 0.103, 0.012, -0.041, -0.057, -0.034, 0.018, 0.096, 0.197, 0.318, 0.452, 0.579, 0.700, 0.809, 0.611, 0.688, 0.756, 0.811, 0.856, 0.894, 0.926, 0.954, 0.980, 1.004])

# Spline para comparación
cs_V = CubicSpline(f, V, bc_type='natural')
derivada_spline = cs_V.derivative()
h = 2.5

# Fórmulas de diferencias finitas
def dif_centrada_o2(idx, y): return (y[idx+1] - y[idx-1]) / (2*h)
def dif_centrada_o4(idx, y): return (-y[idx+2] + 8*y[idx+1] - 8*y[idx-1] + y[idx-2]) / (12*h)
def dif_progresiva_o2(idx, y): return (-3*y[idx] + 4*y[idx+1] - y[idx+2]) / (2*h)

# --- 2. RESULTADOS NUMÉRICOS ---
print("--- RESULTADOS PARTE 2: DERIVACIÓN NUMÉRICA ---")
puntos = [10.0, 40.0, 70.0, 100.0]
resultados_diff = []

for pt in puntos:
    idx = np.where(f == pt)[0][0]
    der_spline = derivada_spline(pt)
    if pt == 10.0:
        der_prog2 = dif_progresiva_o2(idx, V)
        resultados_diff.append(der_prog2)
        print(f"f = {pt} kHz | Prog. O2: {der_prog2:.5f} | Spline: {der_spline:.5f}")
    else:
        der_c2 = dif_centrada_o2(idx, V)
        der_c4 = dif_centrada_o4(idx, V)
        resultados_diff.append(der_c2)
        print(f"f = {pt} kHz | Cent. O2: {der_c2:.5f} | Cent. O4: {der_c4:.5f} | Spline: {der_spline:.5f}")

# --- 3. GRÁFICA 2 ---
f_fine = np.linspace(10.0, 107.5, 1000)

plt.figure(figsize=(9, 4.5))
plt.plot(f_fine, derivada_spline(f_fine), 'b-', label='dV/df (spline)', linewidth=2)
plt.plot(puntos, derivada_spline(puntos), 'ro', markersize=8, label='Puntos evaluados (spline)')
plt.plot(puntos, resultados_diff, 's', color='orange', markersize=7, label='Diferencias Finitas (O2)')
plt.axhline(0, color='gray', linestyle='--')

for fx, dy in zip(puntos, resultados_diff):
    plt.text(fx, dy + 0.008, f'f={fx}', fontsize=9, ha='center', fontweight='bold')

plt.title('Derivada numérica dV/df vs Frecuencia', fontweight='bold')
plt.xlabel('Frecuencia f (kHz)')
plt.ylabel('dV/df (V/kHz)')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()