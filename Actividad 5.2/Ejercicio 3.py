import numpy as np
import matplotlib.pyplot as plt

def newton_divided_diff(x, y):
    """ Calcula la tabla de diferencias divididas de Newton """
    n = len(x)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i+1, j-1] - coef[i, j-1]) / (x[i+j] - x[i])
    
    return coef[0, :]

def newton_interpolation(x_data, y_data, x):
    """ Evalúa el polinomio de Newton en los puntos x """
    coef = newton_divided_diff(x_data, y_data)
    n = len(x_data)
    
    y_interp = np.zeros_like(x, dtype=float)  # corregido para evitar ceros
    
    for i in range(len(x)):
        term = coef[0]
        product = 1
        for j in range(1, n):
            product *= (x[i] - x_data[j-1])
            term += coef[j] * product
        y_interp[i] = term
    
    return y_interp

# 🔹 Datos del ejercicio
x_data = np.array([10, 20, 30, 40, 50, 60])
y_data = np.array([0.32, 0.30, 0.28, 0.27, 0.26, 0.25])

# 🔹 Evaluar en V = 35 m/s
x_eval = np.array([35.0])
y_eval = newton_interpolation(x_data, y_data, x_eval)
print(f"El coeficiente Cd en V = 35 m/s es: {y_eval[0]}")

# Puntos para graficar
x_vals = np.linspace(min(x_data), max(x_data), 100)
y_interp = newton_interpolation(x_data, y_data, x_vals)

# Gráfica
plt.figure(figsize=(8, 6))
plt.plot(x_data, y_data, 'ro', label='Datos')
plt.plot(x_vals, y_interp, 'b-', label='Interpolación de Newton')
plt.xlabel('Velocidad (m/s)')
plt.ylabel('Coeficiente Cd')
plt.legend()
plt.title('Interpolación de Newton')
plt.grid(True)
plt.savefig("newton_interpolacion.png")
plt.show()
