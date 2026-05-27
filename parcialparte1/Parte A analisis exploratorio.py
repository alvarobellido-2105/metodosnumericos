import numpy as np
import matplotlib.pyplot as plt

# 30 Nodos experimentales registrados a 37°C
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

# Búsqueda indexada del mínimo crudo experimental
idx_min_crudo = np.argmin(impedancias)
f_min_cruda = frecuencias[idx_min_crudo]
z_min_cruda = impedancias[idx_min_crudo]

print("=== PARTE A: ANÁLISIS EXPLORATORIO EN PYCHARM ===")
print(f"Mínimo visual detectado en la serie discreta:")
print(f"Frecuencia: {f_min_cruda} Hz | Impedancia: {z_min_cruda:.4f} Ohm\n")

# Construcción de la gráfica de dispersión
plt.figure(figsize=(9, 5))
plt.scatter(frecuencias, impedancias, color='#e74c3c', edgecolors='#c0392b', s=55, label='Muestras Experimentales')
plt.title('Parte A: Espacio de Dispersión Bidimensional (Curva en U)', fontsize=12, fontweight='bold', color='#1a5276')
plt.xlabel('Frecuencia (f) [Hz]', fontweight='bold')
plt.ylabel('Magnitud de Impedancia |Z| [Ohmios]', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper center')
plt.tight_layout()
plt.show()  # Recuerda cerrar la ventana flotante en PyCharm para finalizar el script