import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# --- 1. DATOS DE ENTRADA ---
f = np.array([10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 57.5, 60.0, 62.5, 65.0, 67.5, 70.0, 72.5, 75.0, 77.5, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5])
V = np.array([0.842, 0.911, 0.986, 1.062, 1.143, 1.227, 1.314, 1.401, 1.482, 1.551, 1.216, 1.048, 0.866, 0.689, 0.521, 0.364, 0.223, 0.103, 0.012, -0.041, -0.057, -0.034, 0.018, 0.096, 0.197, 0.318, 0.452, 0.579, 0.700, 0.809, 0.611, 0.688, 0.756, 0.811, 0.856, 0.894, 0.926, 0.954, 0.980, 1.004])

# Función continua para evaluar la bisección
cs_V = CubicSpline(f, V, bc_type='natural')

def biseccion_spline(a, b, tol=1e-6):
    iteracion = 0
    while (b - a) / 2 > tol:
        iteracion += 1
        c = (a + b) / 2
        if cs_V(c) == 0:
            break
        elif cs_V(a) * cs_V(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2, iteracion

# --- 2. RESULTADOS NUMÉRICOS ---
print("--- RESULTADOS PARTE 3: RAÍCES POR BISECCIÓN ---")
# 1er intervalo de cambio de signo: [55.0, 57.5]
raiz1, iter1 = biseccion_spline(55.0, 57.5)
print(f"1ra Raíz (Cruce Descendente): {raiz1:.4f} kHz | Hallada en {iter1} iteraciones")

# 2do intervalo de cambio de signo: [62.5, 65.0]
raiz2, iter2 = biseccion_spline(62.5, 65.0)
print(f"2da Raíz (Cruce Ascendente):  {raiz2:.4f} kHz | Hallada en {iter2} iteraciones")


# --- 3. GRÁFICA 3 (ZOOM EN RAÍCES) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Zoom Raíz 1
f_z1 = np.linspace(53, 59, 200)
ax1.plot(f_z1, cs_V(f_z1), 'b-', linewidth=2, label='V(f) Spline')
ax1.axhline(0, color='gray', linestyle='-')
ax1.axvline(raiz1, color='red', linestyle='--', label=f'Raíz Spline: {raiz1:.3f} kHz')
ax1.axvline(55.5660, color='orange', linestyle='--', label='Raíz Lineal: 55.566 kHz')
ax1.fill_between([55.0, 57.5], -0.06, 0.02, color='green', alpha=0.1, label='Intervalo [55.0, 57.5]')
ax1.plot([55.0, 57.5], [cs_V(55.0), cs_V(57.5)], 'ko', markersize=5)
ax1.set_title('1ª Raíz (Cruce Descendente)', fontweight='bold')
ax1.set_xlabel('Frecuencia (kHz)')
ax1.set_ylabel('V (V)')
ax1.set_xlim(54, 58)
ax1.set_ylim(-0.06, 0.03)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=9)

# Zoom Raíz 2
f_z2 = np.linspace(61, 67, 200)
ax2.plot(f_z2, cs_V(f_z2), 'b-', linewidth=2, label='V(f) Spline')
ax2.axhline(0, color='gray', linestyle='-')
ax2.axvline(raiz2, color='red', linestyle='--', label=f'Raíz Spline: {raiz2:.3f} kHz')
ax2.axvline(64.1346, color='orange', linestyle='--', label='Raíz Lineal: 64.135 kHz')
ax2.fill_between([62.5, 65.0], -0.04, 0.02, color='green', alpha=0.1, label='Intervalo [62.5, 65.0]')
ax2.plot([62.5, 65.0], [cs_V(62.5), cs_V(65.0)], 'ko', markersize=5)
ax2.set_title('2ª Raíz (Cruce Ascendente)', fontweight='bold')
ax2.set_xlabel('Frecuencia (kHz)')
ax2.set_xlim(61.5, 66)
ax2.set_ylim(-0.04, 0.03)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.show()