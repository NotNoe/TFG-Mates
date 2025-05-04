import numpy as np
import utils.generateMod as generateMod
import pycuda.driver as cuda
from timeit import default_timer as timer

mod = generateMod.init(["demos/vectorAdd.cu"])
vectorAdd = mod.get_function("vectorAdd")
def add_random_vects(n, out):
    #Tiempo en CPU (Contando reservar memoria)
    start = timer()
    a = np.random.randn(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    c2 = np.zeros(n).astype(np.float32)
    np.add(a,b,out=c2)
    CPUt = timer()-start
    #Tiempo en GPU (Contando reservar y copiar los datos al dispositivo)
    start = timer()
    c1 = np.zeros(n).astype(np.float32)
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c1.nbytes)
    n_gpu = cuda.mem_alloc(np.int32(n).nbytes)
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)
    b = n if n < 1024 else 1024 #el bloque va a ser de 1024 (siempre que n > 1024)
    g = n // 1024 + 1 #Tantos grids como haga falta
    vectorAdd(a_gpu, b_gpu, c_gpu, np.int32(n), grid=(g,1,1), block = (b,1,1))
    cuda.memcpy_dtoh(c1, c_gpu)
    GPUt = timer() - start
    
    if (c1 == c2).all():
        text = f"""------------------N = {n}------------------
        Los resultados coinciden:
        Tiempo en GPU {GPUt}
        Tiempo en CPU: {CPUt}
        La GPU es un {round(CPUt/GPUt*100,1)}% mas rapida"""
        print(text, file=out)
        print(text)
    else:
        print("Ha habido un error en el computo en GPU.\n", file=out)
        print("Ha habido un error en el computo en GPU.\n")

if __name__ == "__main__":
    with open("../out/vectorAdd.out", "w") as out:
        for n in [10**(2*e) for e in range(1,5)]:
            add_random_vects(n, out)