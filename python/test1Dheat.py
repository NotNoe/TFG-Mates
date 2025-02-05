from heat1D import calor1D
from heat1Dgpu import calor1D as calor1DGPU
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import  numpy as np
import math


def f(x):
    return 0
def alpha(t):
    return -60
def beta(t):
    return math.exp(t)

intervalo = [0,1]
nx = 10
nt = 1000
t_obj = 3
eps = 0.0001 #Sensibilidad de que se considera "igual"


resultado1 = calor1D(intervalo, f, alpha, beta, t_obj, nt, nx)
resultado2 = calor1DGPU(intervalo, f, alpha, beta, t_obj, nt, nx)


#Comprobamos si son iguales (con sensibilidad eps)
igual = True
for i in range(len(resultado1)):
    for j in range(len(resultado1[0])):
        if(abs(resultado1[i][j]-resultado2[i][j]) > eps): igual = False
print(igual)

M = max(resultado1.max(), resultado2.max())
m = min(resultado1.min(), resultado2.min())
x = np.linspace(intervalo[0], intervalo[1], nx + 2)
t = np.linspace(0, t_obj, 100)
y = [0,0.1]
X,Y = np.meshgrid(x,y)

#Grafica resultado 1
fig, ax = plt.subplots()
z = np.array([resultado1[0][i] for j in y for i in np.arange(len(x))])
Z = z.reshape(2,nx + 2)
plt.imshow(Z, origin="lower", interpolation="bilinear", cmap="coolwarm", vmin=m, vmax=M)
ax.set_ylim(0,0.1)
ax.get_yaxis().set_visible(False)
plt.colorbar().set_label("Temperatura(ºC)")




def update1(frame):
    k = int(frame // (t_obj/nt))
    z = np.array([resultado1[k][i] for j in y for i in np.arange(len(x))])
    Z = z.reshape(2,nx + 2)
    plt.imshow(Z, origin="lower", interpolation="bilinear", cmap="coolwarm", vmin=m, vmax=M)
    ax.set_ylim(0,0.1)
    ax.get_yaxis().set_visible(False)
    #ax.set_xlim(x[0], x[-1])
    ax.set_title("Resultado calculado en CPU\nt = " + str(round(frame, 3)))
    return ax


ani1 = animation.FuncAnimation(fig=fig, func=update1, frames=t, interval=60, repeat=False)
#ani.save("./out/1dheat.gif")

plt.show()

#Grafica resultado 2
fig, ax = plt.subplots()
z = np.array([resultado1[0][i] for j in y for i in np.arange(len(x))])
Z = z.reshape(2,nx + 2)
plt.imshow(Z, origin="lower", interpolation="bilinear", cmap="coolwarm", vmin=m, vmax=M)
ax.set_ylim(0,0.1)
ax.get_yaxis().set_visible(False)
plt.colorbar().set_label("Temperatura(ºC)")


def update2(frame):
    k = int(frame // (t_obj/nt))
    z = np.array([resultado2[k][i] for j in y for i in np.arange(len(x))])
    Z = z.reshape(2,nx + 2)
    plt.imshow(Z, origin="lower", interpolation="bilinear", cmap="coolwarm", vmin=m, vmax=M)
    ax.set_ylim(0,0.1)
    ax.get_yaxis().set_visible(False)
    #ax.set_xlim(x[0], x[-1])
    ax.set_title("Resultado calculado en GPU\nt = " + str(round(frame, 3)))
    return ax


ani2 = animation.FuncAnimation(fig=fig, func=update2, frames=t, interval=60, repeat=False)

plt.show()
