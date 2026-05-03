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
    
    y_interp = np.zeros_like(x, dtype=float)  # evitar error de enteros
    for i in range(len(x)):
        term = coef[0]
        product = 1
        for j in range(1, n):
            product *= (x[i] - x_data[j-1])
            term += coef[j] * product
        y_interp[i] = term
    
    return y_interp

# 🔹 Datos del ejercicio
x_data = np.array([200, 250, 300, 350, 400])
y_data = np.array([30, 35, 40, 46, 53])

# 🔹 Evaluar en T = 275 °C
x_eval = np.array([275.0])
y_eval = newton_interpolation(x_data, y_data, x_eval)
print(f"La eficiencia para T = 275 °C es: {y_eval[0]} %")

# Puntos para graficar
x_vals = np.linspace(min(x_data), max(x_data), 100)
y_interp = newton_interpolation(x_data, y_data, x_vals)

# Gráfica
plt.figure(figsize=(8, 6))
plt.plot(x_data, y_data, 'ro', label='Datos')
plt.plot(x_vals, y_interp, 'b-', label='Interpolación de Newton')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Eficiencia (%)')
plt.legend()
plt.title('Interpolación de Newton')
plt.grid(True)
plt.savefig("newton_interpolacion.png")
plt.show()
