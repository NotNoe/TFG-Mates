import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 16})

def get_sub(i):
    return str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))

# Generamos la malla 5x5x3
x_vals = np.arange(0, 5)
y_vals = np.arange(0, 5)
t_vals = np.arange(0, 3)

X, Y, T = np.meshgrid(x_vals, y_vals, t_vals, indexing='ij')
X = X.flatten()
Y = Y.flatten()
T = T.flatten()

unique_T = np.unique(T)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, t_val in enumerate(unique_T):
    ax = axs[i]
    ax.set_aspect('equal')
    ax.grid(True)
    
    mask = (T == t_val)
    x_slice = X[mask]
    y_slice = Y[mask]
    
    if t_val == 0:
        green_mask = np.full_like(x_slice, False, dtype=bool)
        red_mask = np.full_like(x_slice, True, dtype=bool)
    else:
        green_mask = (x_slice != 0) & (x_slice != 4) & (y_slice != 0) & (y_slice != 4)
        red_mask = ~green_mask

    ax.scatter(x_slice[red_mask], y_slice[red_mask], c='red', 
               label="Condiciones iniciales o de contorno", zorder=2)
    ax.scatter(x_slice[green_mask], y_slice[green_mask], c='green', 
               label="Objetivos", zorder=2)
    
    # Vectores de malla
    ax.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, 
              color='blue', label=r'$\Delta x$', zorder=3)
    ax.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, 
              color='purple', label=r'$\Delta y$', zorder=3)
    
    # Marcadores especiales del esquema de onda 2D
    if t_val == 0:
        ax.scatter(2, 2, c='red', marker='x', s=150, zorder=3, label='_nolegend_')
    if t_val == 1:
        special_red = np.array([[2, 2], [2, 3], [2, 1], [1, 2], [3, 2]])
        ax.scatter(special_red[:, 0], special_red[:, 1], c='green', 
                   marker='x', s=150, zorder=3, label='_nolegend_')
    if t_val == 2:
        ax.scatter(2, 2, c='green', marker='*', s=150, zorder=3, label='_nolegend_')

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xticks(np.arange(0, 5))
    ax.set_xticklabels(["a=x"+get_sub(0)] + ["x"+get_sub(i) for i in range(1, 4)] + ["b=x"+get_sub(4)])
    ax.set_yticks(np.arange(0, 5))
    ax.set_yticklabels(["c=y"+get_sub(0)] + ["y"+get_sub(i) for i in range(1, 4)] + ["d=y"+get_sub(4)])
    
    if t_val == 0:
        t_label = "t=0"
    elif t_val == 1:
        t_label = "t=t"+get_sub(1) + "=" + '$\Delta t$'
    else:
        t_label = "t=T"
    ax.set_title(t_label)

# Leyenda global
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
