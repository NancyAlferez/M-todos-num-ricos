#   Codigo que implementa el calculo de errores
#   en operaciones numericas
# 
#           Autor:
#   Dr. Ivan de Jesus May-Cen
#   imaycen@hotmail.com
#   Version 1.0 : 29/01/2025
#
import matplotlib.pyplot as plt   # ← AÑADIDO

def calcular_errores(x, y, valor_real):
    diferencia = x - y
    error_abs = abs(valor_real - diferencia)
    error_rel = error_abs / abs(valor_real)
    error_pct = error_rel * 100
    print(f"Diferencia: {diferencia}")
    print(f"Error absoluto: {error_abs}")
    print(f"Error relativo: {error_rel}")
    print(f"Error porcentual: {error_pct}%")
    return error_abs, error_rel

valores = [(1.0000001, 1.0000000, 0.0000001), (1.000000000000001, 1.000000000000000, 0.000000000000001)]

errores_abs = []   # ← AÑADIDO
errores_rel = []   # ← AÑADIDO
casos = []         # ← AÑADIDO

for i, (x, y, real) in enumerate(valores, start=1):
    print(f"\nPara x={x}, y={y}:")
    error_abs, error_rel = calcular_errores(x, y, real)

    errores_abs.append(error_abs)   # ← AÑADIDO
    errores_rel.append(error_rel)   # ← AÑADIDO
    casos.append(f"Caso {i}")       # ← AÑADIDO

# --- GRAFICA (AÑADIDO) ---
plt.figure()
plt.plot(casos, errores_abs, marker='o', label="Error absoluto")
plt.plot(casos, errores_rel, marker='o', label="Error relativo")
plt.yscale("log")
plt.xlabel("Casos")
plt.ylabel("Error")
plt.title("Errores en operaciones numéricas")
plt.legend()
plt.show()