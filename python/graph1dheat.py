import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def get_sub(i):
    return str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
fig, ax = plt.subplots()

#plt.rcParams.update({'font.size':20})
plt.grid(True)


ax.scatter([0]*5 + [3]*5 + [1,2,3], [i for i in range(5)]*2 + [0]*3, c='red', label="Valores iniciales",zorder=2)
ax.scatter([1,2,3],[4]*3, c='green', label="Objetivos",zorder=2)
ax.scatter([1,2,3]*3,[1]*3+[2]*3+[3]*3, c='grey', label="Puntos intermedios",zorder=2)



# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(np.arange(0,4,1), ["a=x%c" % get_sub(0)] + ["x%c" % get_sub(i) for i in range(1,3,1)] + ['b=x%c' % get_sub(3)])
plt.yticks(np.arange(0,5,1), ["0=t%c" % get_sub(0)] + ["t%c" % get_sub(i) for i in range(1,4,1)] + ['T=t%c' % get_sub(4)])
ax.quiver([0,0],[0,0],[0,1],[1,0],color=['purple','blue'], scale=1, scale_units='xy',zorder=2)
#plt.scatter([0,1,2,1],[0,0,0,1], c=col, s=100)

plt.show()