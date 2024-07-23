from heat1D_py import calor1D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import  numpy as np


def f(x):
    pow(2,x)+15
    return 0
def alpha(t):
    return pow(2,t)
def beta(t):
    return pow(2,t)

intervalo = [0,1]
nx = 10
nt = 10000
t_obj = 3

(resultado, M, m) = calor1D(intervalo, f, alpha, beta, t_obj, nt, nx)
fig, ax = plt.subplots()
x = np.linspace(intervalo[0], intervalo[1], nx + 2)
t = np.linspace(0, t_obj, 100)


m *= 0.5
M *= 1.5


def update(frame):
    i = int(frame // (t_obj/nt))
    ax.cla()
    ax.set_ylim(m,M)
    ax.set_ylabel("Temperatura")
    ax.set_xlabel("Posición")
    ax.set_title("t = " + str(round(frame, 3)))
    ax.plot(x, resultado[i])
    return ax


ani = animation.FuncAnimation(fig=fig, func=update, frames=t, interval=60)


plt.show()
