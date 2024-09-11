import numpy as np
import pandas as pd
from math import sqrt, ceil
import warnings


def wave1D(f, g, c, nt, a=0, t_obj=1, n_puntos=1):
    """Solves wave equation in 1D. t_0 is assumed to be 0.
    a: The point where we want to know the solution
    f: The wave function for t=0
    g: The value of the derivative at t=0
    t_obj: Final time
    nt: Number of time slices (non counting 0)
    c: Constant of the equation
    Returns an np.matrix with the values in the grid.
    """
    dt = t_obj/(nt + 1)
    dx = c*dt
    nx = n_puntos + 2*nt #Esto son los puntos que va a tener la matriz, luego no devolveremos todos

    resultado = np.zeros(shape=(nt + 1, nx)) #En realidad la matriz tendra columnas auxiliares que no devolveremos
    #Casos iniciales
    for j in range(nx):
        x = a+(j-nt)*dx 
        resultado[0][j] = f(x) + dt*g(x)
    for j in range(1, nx - 1):
        x = a+(j-nt)*dx
        resultado[1][j] = f(x)

    #Rellenamos el resto de la matriz
    for i in range(2,nt):
        for j in range(i, nx-i): #Lo calculamos solo en los que tiene sentido
            resultado[i][j]  = resultado[i-1][j+1] + resultado[i-1][j-1] - resultado[i-2][j]
    
    return resultado[:,nt:nx-nt]
    