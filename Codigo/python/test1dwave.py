from wave1D import wave1D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import  numpy as np
from math import ceil,cos,sin


def f(x):
    return sin(x)
def g(x):
    return cos(x)

t_obj = 4
nt = 1000
c = 1
n_puntos = 1000
x0 = 0

resultado = wave1D(f, g, c, nt, x0, t_obj, n_puntos)
M, m = (resultado.max(), resultado.min())
fig, ax = plt.subplots()
t = np.linspace(0, t_obj, 100)
dx = c*t_obj/(nt + 1)
x = np.arange(start=x0, stop=x0+dx*n_puntos, step=dx)


def update(frame):
    i = int(frame // (t_obj/len(resultado)))-1
    ax.cla()
    ax.set_ylim(m,M)
    ax.set_ylabel("u(x,t)")
    ax.set_xlabel("x")
    ax.set_title("t = " + str(round(frame, 3)))
    ax.plot(x, resultado[i])
    return ax


ani = animation.FuncAnimation(fig=fig, func=update, frames=t, interval=60)


plt.show()
