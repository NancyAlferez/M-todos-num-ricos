import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # ← AÑADIDO: Importar pandas

epsilon = 1.0
iteracion = 0

iteraciones = []   # ← AÑADIDO
epsilons = []      # ← AÑADIDO

while 1.0 + epsilon != 1.0:
    epsilon /= 2
    iteracion = iteracion + 1
    print(f"Iteracion: {iteracion}, Precisión de máquina: {epsilon}")

    iteraciones.append(iteracion)  # ← AÑADIDO
    epsilons.append(epsilon)    # Añadido

epsilon *= 2
print(f"Precisión de máquina: {epsilon}")

# --- AÑADIDO PARA GENERAR EL EXCEL ---
df = pd.DataFrame({'Iteracion': iteraciones, 'Precision': epsilons})
df.to_excel('precision_maquina.xlsx', index=False)
# -------------------------------------

plt.plot(iteraciones, epsilons)    # ← AÑADIDO
plt.xlabel("Iteraciones")          # ← AÑADIDO
plt.ylabel("Precisión de máquina") # ← AÑADIDO
plt.yscale("log")                  # ← AÑADIDO
plt.title("Cálculo de la precisión de máquina")  # ← AÑADIDO
plt.show()              