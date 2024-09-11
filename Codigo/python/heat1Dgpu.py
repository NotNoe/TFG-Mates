import numpy as np
import pandas as pd
from math import sqrt
import warnings
import generateMod as generateMod
import pycuda.driver as cuda


def calor1D(intervalo, f, alpha, beta, t_obj, nt, nx):
    """Solves heat equation in 1D on the GPU. t_0 is assumed to be 0.
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
    
    resultado = np.zeros(shape=(nt + 1, nx + 2)).astype(np.float32)
    #Rellenamos datos iniciales
    for i in range(nt + 1):
        t = i*dt
        resultado[i][0] = alpha(t)
        resultado[i][-1] = beta(t)
    for j in range(nx + 2):
        x = intervalo[0]+j*dx
        resultado[0][j] = f(x)

    #Iniciamos pycuda
    mod = generateMod.init(["heat1D.cu"])
    heat1D = mod.get_function("heat1D")
    #Reservamos memoria y copiamos
    resultado_gpu = cuda.mem_alloc(resultado.nbytes)
    cuda.memcpy_htod(resultado_gpu, resultado)
    #Calculamos cuantos hilos y bloques necesitamos
    b = nx if nx < 1024 else 1024 #El bloque va a ser de 1024 (siempre que n > 1024)
    g = nx // 1024 + 1 #Tantos grids como haga falta


    #Rellenamos el resto de la matriz en la gpu
    for i in range(1,nt+1): #Para cada t
        heat1D(resultado_gpu, np.int32(i), np.float32(lam), np.int32(nx), grid=(g,1,1), block=(b,1,1))
        cuda.Context.synchronize() #No podemos avanzar hasta que no hayamos terminado la linea
        
    cuda.memcpy_dtoh(resultado, resultado_gpu)

    return resultado
    