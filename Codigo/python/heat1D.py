import numpy as np
import pandas as pd
from math import sqrt
import warnings


def calor1D(intervalo, f, alpha, beta, t_obj, nt, nx):
    """Solves heat equation in 1D. t_0 is assumed to be 0.
    intervalo: (a,b) where a and b are the bounds of the dim
    f: The heat function for t=0
    alpha: Heat function in a for t>0
    beta: Heat function in b for t>0
    t_obj: Final time
    nt: Number of points in time, not including t=0
    nx: Number of internal points in space
    Returns an np.matrix with the values in the grid.
    If lambda is greater than 0.5, a message will be shown but  
    """

    dt = t_obj/nt #La matriz va de 0 a nt
    dx = (intervalo[1] - intervalo[0])/(nx + 1) #La matriz va de 0 a nx + 1
    lam = dt/pow(dx, 2) #Calculamos lambda
    if(lam > 1/2):
        print("Warning: dt/dx^2 must be less or equal than 0.5, and it's ", lam, ", the method might not converge.")
        print("dt = ", dt)
        print("dx = ",dx)
    
    resultado = np.zeros(shape=(nt + 1, nx + 2))
    #Rellenamos datos iniciales
    for i in range(nt + 1):
        t = i*dt
        resultado[i][0] = alpha(t)
        resultado[i][-1] = beta(t)
    for j in range(nx + 2):
        x = intervalo[0]+j*dx
        resultado[0][j] = f(x)
        

    #Rellenamos el resto de la matriz
    for i in range(1,nt+1):
        for j in range(1, nx+1):
            resultado[i][j]  = (1-2*lam)*resultado[i-1][j] + (resultado[i-1][j-1] + resultado[i-1][j+1])*lam
    
    return resultado
    