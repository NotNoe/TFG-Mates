import matplotlib.pyplot as plt
import numpy as np

def get_sub(i):
    return str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))

fig, ax = plt.subplots()  # Se usa el tamaño por defecto
ax.set_aspect('equal')
plt.grid(True)

# Creamos una cuadrícula de 5x5
x_vals = np.arange(0, 5)   # 0, 1, 2, 3, 4
y_vals = np.arange(0, 5)   # 0, 1, 2, 3, 4
X, Y = np.meshgrid(x_vals, y_vals)
X = X.flatten()
Y = Y.flatten()

# Los objetivos (puntos verdes) son los que NO están en y=0, ni en los bordes laterales (x=0 o x=4)
green_mask = (Y != 0) & (X != 0) & (X != 4)
red_mask = ~green_mask  # Los del borde

# Graficamos los puntos con sus colores y etiquetas originales
ax.scatter(X[red_mask], Y[red_mask], c='red', 
           label="Condiciones iniciales\no de contorno", zorder=2)
ax.scatter(X[green_mask], Y[green_mask], c='green', 
           label="Objetivos", zorder=2)

# Graficamos los vectores con sus etiquetas
ax.quiver(0, 0, 0, 1, color='purple', scale=1, scale_units='xy', 
          zorder=2, label=r'$\Delta t$')
ax.quiver(0, 0, 1, 0, color='blue', scale=1, scale_units='xy', 
          zorder=2, label=r'$\Delta x$')

# Ajustamos la posición del eje para dejar suficiente espacio a la derecha para la leyenda
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Etiquetas de los ejes con subíndices:
# En el eje x: se indica "a=x₀", luego "x₁", "x₂", "x₃" y "b=x₄"
plt.xticks(np.arange(0, 5), 
           ["a=x%c" % get_sub(0)] + ["x%c" % get_sub(i) for i in range(1, 4)] + ["b=x%c" % get_sub(4)])
# En el eje y: "0=t₀", "t₁", "t₂", "t₃" y "T=t₄"
plt.yticks(np.arange(0, 5), 
           ["0=t%c" % get_sub(0)] + ["t%c" % get_sub(i) for i in range(1, 4)] + ["T=t%c" % get_sub(4)])

# Superponemos los marcadores especiales sin modificar la leyenda:
# (1,3) se muestra con marcador "x" (en verde)
ax.scatter(2, 1, c='green', marker='*', s=100, zorder=3, label='_nolegend_')
# (0,2), (0,3) y (0,4) se muestran con marcador "x" (en rojo)
special_red = np.array([[1, 0], [2, 0], [3, 0]])
ax.scatter(special_red[:, 0], special_red[:, 1], c='red', marker='x', s=100, zorder=3, label='_nolegend_')

plt.show()
