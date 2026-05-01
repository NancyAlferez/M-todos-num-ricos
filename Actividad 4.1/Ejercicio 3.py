import numpy as np
import matplotlib.pyplot as plt

# Funcion a derivar
def f(x):
    return x**3 - 2*x**2 + x

# Derivada analitica
def df_analytical(x):
    return 3*x**2 - 4*x + 1

# Metodo diferencias hacia adelante
def forward_diff(f, x, h=0.2):
    return (f(x + h) - f(x)) / h

# Metodo diferencias hacia atras
def backward_diff(f, x, h=0.2):
    return (f(x) - f(x - h)) / h

# Metodo diferencias centradas
def central_diff(f, x, h=0.2):
    return (f(x + h) - f(x - h)) / (2*h)

# Intervalo [-1, 2] con paso 0.2
a = -1.0
b = 2.0
h = 0.2
x_vals = np.arange(a, b, h)

# Derivada exacta
df_exact = df_analytical(x_vals)

# Aproximaciones numericas
df_forward = forward_diff(f, x_vals, h)
df_backward = backward_diff(f, x_vals, h)
df_central = central_diff(f, x_vals, h)

# Errores
error_forward = np.abs(df_forward - df_exact)
error_backward = np.abs(df_backward - df_exact)
error_central = np.abs(df_central - df_exact)

# Graficar funcion y derivadas
plt.figure(figsize=(10, 6))
plt.plot(x_vals, f(x_vals), '-', label='Funcion')
plt.plot(x_vals, df_exact, 'k-', label='Derivada Analitica')
plt.plot(x_vals, df_forward, 'r--', label='Hacia adelante')
plt.plot(x_vals, df_backward, 'g-.', label='Hacia atras')
plt.plot(x_vals, df_central, 'b:', label='Centrada')
plt.xlabel('x')
plt.ylabel("Valor")
plt.legend()
plt.title("Comparacion de Metodos de Diferenciacion Numerica")
plt.grid()
plt.show()

# Graficar errores
plt.figure(figsize=(10, 6))
plt.plot(x_vals, error_forward, 'r--', label='Error Hacia adelante')
plt.plot(x_vals, error_backward, 'g-.', label='Error Hacia atras')
plt.plot(x_vals, error_central, 'b:', label='Error Centrada')
plt.xlabel('x')
plt.ylabel("Error absoluto")
plt.legend()
plt.title("Errores en Diferenciacion Numerica")
plt.grid()
plt.show()
