import generateMod as generateMod
import pycuda.driver as cuda
import numpy as np
import sys


mod = generateMod.init(["HelloWorld.cu"])
hello_world = mod.get_function("hello_world")
a = [1.0, 2.0]
a = np.asarray(a).astype("float32") #Esta es la memoria que gestionaremos en host, arrays de numpy
a_gpu = cuda.mem_alloc(a.nbytes) #Reservamos la memoria en GPU
cuda.memcpy_htod(a_gpu, a)
hello_world(a_gpu, block = (2,1,1))