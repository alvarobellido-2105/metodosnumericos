import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, lagrange

# --- 1. DATOS DE ENTRADA ---
f = np.array([10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 57.5, 60.0, 62.5, 65.0, 67.5, 70.0, 72.5, 75.0, 77.5, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5])
V = np.array([0.842, 0.911, 0.986, 1.062, 1.143, 1.227, 1.314, 1.401, 1.482, 1.551, 1.216, 1.048, 0.866, 0.689, 0.521, 0.364, 0.223, 0.103, 0.012, -0.041, -0.057, -0.034, 0.018, 0.096, 0.197, 0.318, 0.452, 0.579, 0.700, 0.809, 0.611, 0.688, 0.756, 0.811, 0.856, 0.894, 0.926, 0.954, 0.980, 1.004])
Z = np.array([182.4, 178.9, 175.1, 171.0, 166.8, 162.7, 158.9, 155.4, 152.0, 149.0, 146.1, 145.2, 145.8, 147.3, 149.9, 153.5, 158.0, 163.2, 168.9, 174.8, 180.5, 186.2, 191.5, 196.2, 200.1, 203.1, 205.2, 206.3, 206.1, 204.7, 198.0, 194.4, 190.9, 187.8, 185.1, 183.0, 181.6, 180.8, 180.6, 180.9])

# Splines Cúbicos Naturales
cs_V = CubicSpline(f, V, bc_type='natural')
cs_Z = CubicSpline(f, Z, bc_type='natural')

# Función para Lagrange de 3 puntos (Grado 2)
def calcular_lagrange(f_target, f_data, y_data):
    idx = np.argsort(np.abs(f_data - f_target))[:3]
    poly = lagrange(f_data[idx], y_data[idx])
    return poly(f_target)

# --- 2. RESULTADOS NUMÉRICOS ---
print("--- RESULTADOS PARTE 1: INTERPOLACIÓN ---")
print(f"V(41.0 kHz)  -> Lagrange: {calcular_lagrange(41.0, f, V):.4f} V | Spline: {cs_V(41.0):.4f} V")
print(f"|Z|(41.0 kHz)-> Lagrange: {calcular_lagrange(41.0, f, Z):.3f} Ω | Spline: {cs_Z(41.0):.3f} Ω")
print(f"V(73.0 kHz)  -> Lagrange: {calcular_lagrange(73.0, f, V):.4f} V | Spline: {cs_V(73.0):.4f} V")
print(f"|Z|(73.0 kHz)-> Lagrange: {calcular_lagrange(73.0, f, Z):.3f} Ω | Spline: {cs_Z(73.0):.3f} Ω")

# --- 3. GRÁFICA 1 ---
f_fine = np.linspace(10.0, 107.5, 1000)

plt.figure(figsize=(10, 8))
# Subplot Voltaje
plt.subplot(2, 1, 1)
plt.plot(f_fine, cs_V(f_fine), 'b-', label='Spline cúbico V(f)', linewidth=2)
plt.plot(f, V, 'ko', markersize=4, label='Datos medidos')
plt.plot([41.0, 73.0], [cs_V(41.0), cs_V(73.0)], 'ro', markersize=8, label='Puntos interpolados')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Voltaje V(f) e Impedancia |Z|(f) vs Frecuencia', fontweight='bold')
plt.ylabel('Voltaje V (V)')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot Impedancia
plt.subplot(2, 1, 2)
plt.plot(f_fine, cs_Z(f_fine), 'r-', label='Spline cúbico |Z|(f)', linewidth=2)
plt.plot(f, Z, 'ko', markersize=4, label='Datos medidos')
plt.plot([41.0, 73.0], [cs_Z(41.0), cs_Z(73.0)], 'bo', markersize=8, label='Puntos interpolados')
plt.xlabel('Frecuencia f (kHz)')
plt.ylabel('Impedancia |Z| (Ω)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

