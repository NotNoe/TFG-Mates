import matplotlib.pyplot as plt
import numpy as np

# Función para generar subíndices (aunque para números negativos usaremos notación LaTeX directamente)
def get_sub(i):
    return str(i).translate(str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉₋"))

fig, ax = plt.subplots(figsize=(11, 8))  # Se usa el tamaño por defecto
ax.set_aspect('equal')
plt.grid(True)

x_vals = np.arange(0, 11)
y_vals = np.arange(0, 5)
X, Y = np.meshgrid(x_vals, y_vals)
X = X.flatten()
Y = Y.flatten()

valid_mask = ((Y <= X) & (Y <= 10 - X))
red_mask = ((Y == 0) | (Y == 1)) & valid_mask
green_mask = valid_mask & ~ red_mask & (X > 3) & (X < 7)
blue_mask = valid_mask & ~ red_mask & ~ green_mask

ax.scatter(X[red_mask], Y[red_mask], c='red', label="Condiciones iniciales", zorder=2)
ax.scatter(X[green_mask], Y[green_mask], c='green', label="Objetivos", zorder=2)
ax.scatter(X[blue_mask], Y[blue_mask], c='blue', label="Puntos intermedios", zorder=2)


ax.quiver(1, 0, 1, 0, angles='xy', scale_units='xy', scale=1, 
            color='blue', label=r'$\Delta x$', zorder=3, width=0.003)
ax.quiver(1, 0, 0, 1, angles='xy', scale_units='xy', scale=1, 
            color='purple', label=r'$\Delta y$', zorder=3, width=0.003)
    

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)


# Configurar los ticks del eje x: desde -4 hasta 6 (todos)
xticks = np.arange(0, 11)
xtick_labels = [f"$x_{{{i-4}}}$" for i in xticks]
xtick_labels[4] = r"$a=x_{0}$"
xtick_labels[6] = r"$b=x_{2}$"
ax.set_xticks(xticks)
ax.set_xticklabels(xtick_labels)

# Configurar los ticks del eje t: de 0 a 4, con notación similar
yticks = np.arange(0, 5)
ytick_labels = [r"$0=t_{0}$", r"$t_{1}$", r"$t_{2}$", r"$t_{3}$", r"$T=t_{4}$"]
ax.set_yticks(yticks)
ax.set_yticklabels(ytick_labels)

# Ajustar límites para ver todos los puntos y ticks
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 4.5)

ax.scatter(5, 2, c='green', marker='*', s=100, zorder=3, label='_nolegend_')
special_red = np.array([[5, 1], [5, 0], [4, 1], [6, 1]])
ax.scatter(special_red[:, 0], special_red[:, 1], c='red', marker='x', s=100, zorder=3, label='_nolegend_')


plt.tight_layout()
plt.show()
