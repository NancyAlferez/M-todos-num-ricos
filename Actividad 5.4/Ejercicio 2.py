import numpy as np
import matplotlib.pyplot as plt

# Datos del ejercicio
x = np.array([0, 2, 4, 6, 8])
y = np.array([100, 92, 85, 78, 71])

# Cálculo de los coeficientes
n = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x**2)

# Fórmulas de regresión lineal
b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
a = (sum_y - b * sum_x) / n

print(f"Coeficientes de la regresión:")
print(f"a (intercepto) = {a:.4f}")
print(f"b (pendiente) = {b:.4f}")

# Estimación en x = 5 cm
x_eval = 5
y_eval = a + b * x_eval
print(f"Temperatura estimada en x = {x_eval} cm: {y_eval:.2f} °C")

# Predicción usando el modelo
y_pred = a + b * x

# Gráfica
plt.figure(figsize=(8,6))
plt.plot(x, y, 'o', label='Datos')
plt.plot(x, y_pred, '-', label=f'Ajuste lineal: y = {a:.2f} + {b:.2f}x', color='red')
plt.xlabel('Posición (cm)')
plt.ylabel('Temperatura (°C)')
plt.title('Regresión Lineal')
plt.legend()
plt.grid(True)
plt.savefig('regresion_lineal.png', dpi=300)
plt.show()
