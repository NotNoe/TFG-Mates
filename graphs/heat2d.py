import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 16})

def get_sub(i):
    return str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))

# Generamos la malla 5x5x3
x_vals = np.arange(0, 5)   # x: 0,1,2,3,4
y_vals = np.arange(0, 5)   # y: 0,1,2,3,4
t_vals = np.arange(0, 3)   # t (tiempo): 0,1,2

# Malla 3D (usando indexing='ij' para que X[i,j,k]=x_vals[i], etc.)
X, Y, T = np.meshgrid(x_vals, y_vals, t_vals, indexing='ij')
X = X.flatten()
Y = Y.flatten()
T = T.flatten()

# Obtenemos los niveles únicos de T (0,1,2)
unique_T = np.unique(T)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))


# Para cada slice (nivel de T) se generan las visualizaciones 2D
for i, t_val in enumerate(unique_T):
    ax = axs[i]
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Seleccionar puntos del slice actual (T = t_val)
    mask = (T == t_val)
    x_slice = X[mask]
    y_slice = Y[mask]
    
    # Clasificación de puntos:
    # En T=0, todos se consideran de contorno (rojos).
    # En T>0, los "Objetivos" (verdes) son los puntos que NO están en los bordes (x=0 o 4, y=0 o 4).
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
    
    # Dibujar vectores en cada slice:
    # Δx: vector horizontal (de (0,0) a (1,0)) en azul
    ax.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, 
              color='blue', label=r'$\Delta x$', zorder=3)
    # Δy: vector vertical (de (0,0) a (0,1)) en púrpura
    ax.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, 
              color='purple', label=r'$\Delta y$', zorder=3)
    
    # Sobrescribir marcadores especiales:
    # En T=1, el punto (3,3) se dibuja como estrella (verde)
    if t_val == 1:
        ax.scatter(2, 2, c='green', marker='*', s=150, zorder=3, label='_nolegend_')
    # En T=0, se sobrescriben los puntos (3,3), (2,3), (4,3), (3,2) y (3,4) como cruces (rojos)
    if t_val == 0:
        special_red = np.array([[2, 2], [2, 3], [2, 1], [1, 2], [3, 2]])
        ax.scatter(special_red[:, 0], special_red[:, 1], c='red', 
                   marker='x', s=150, zorder=3, label='_nolegend_')
    
    # Configuración de ejes y etiquetas:
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xticks(np.arange(0, 5))
    ax.set_xticklabels(["a=x"+get_sub(0)] + ["x"+get_sub(i) for i in range(1, 4)] + ["b=x"+get_sub(4)])
    ax.set_yticks(np.arange(0, 5))
    ax.set_yticklabels(["c=y"+get_sub(0)] + ["y"+get_sub(i) for i in range(1, 4)] + ["d=y"+get_sub(4)])
    
    # Título que indica el nivel de T
    if t_val == 0:
        t_label = "t=0"
    elif t_val == 1:
        t_label = "t=t"+get_sub(1) + "=" + '$\Delta t$'
    else:
        t_label = "t=T"
    ax.set_title(t_label)

# Crear una única leyenda global centrada.
# Se obtienen los handles y labels del primer subplot (son idénticos en todos).
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
