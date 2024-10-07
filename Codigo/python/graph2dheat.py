import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def get_sub(i):
    return str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x1,y1,t1 = [],[],[]
x2,y2,t2 = [],[],[]
x3,y3,t3 = [],[],[]
for i in range(4):
    for j in range(4):
        for k in range(4):
            if k == 0 or i == 0 or i == 3 or j == 3 or j == 0:
                x1.append(i)
                y1.append(j)
                t1.append(k)
            elif k == 3:
                x2.append(i)
                y2.append(j)
                t2.append(k)
            else:
                x3.append(i)
                y3.append(j)
                t3.append(k)
ax.scatter(x1,y1,t1, c='darkorange',label="Valores iniciales")
ax.scatter(x2,y2,t2, c='royalblue',label="Objetivos")
ax.scatter(x3,y3,t3,c='black',label="Puntos intermedios")
ax.legend()
ax.quiver([0,0,0], [0,0,0],[0,0,0],[0,0,1],[0,1,0],[1,0,0], colors=['fuchsia', 'blue', 'purple','fuchsia', 'fuchsia', 'blue','blue', 'purple','purple'])
plt.xticks(np.arange(0,4,1), ["a=x%c" % get_sub(0)] + ["x%c" % get_sub(i) for i in range(1,3,1)] + ['b=x%c' % get_sub(3)])
plt.yticks(np.arange(0,4,1), ["c=y%c" % get_sub(0)] + ["y%c" % get_sub(i) for i in range(1,3,1)] + ['d=y%c' % get_sub(3)])
ax.set_zticks(np.arange(0,4,1), ["0=t%c" % get_sub(0)] + ["t%c" % get_sub(i) for i in range(1,3,1)] + ['T=t%c' % get_sub(3)])

plt.show()
