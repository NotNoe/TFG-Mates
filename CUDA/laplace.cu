__device__ int index(int i, int j, int n_x) {
    return j * n_x + i;
}

__device__ int get_i() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int get_j() {
    return blockIdx.y * blockDim.y + threadIdx.y;
}

__global__ void resolve(const float* __restrict__ u, float* __restrict__ u_new, unsigned int* diff_max_bits, int n_x, int n_y, float lam_x, float lam_y){ //n es el tamanno de cada fila

    //Calculamos que indice corresponde a nuestro hilo
    int i = get_i();
    int j = get_j();
    if (i <= 0 || i >= n_x - 1 || j <= 0 || j >= n_y - 1) return;

    int idx = index(i, j, n_x);
    float oldv = u[idx];
    float newv = (lam_x * (u[index(i-1, j, n_x)] + u[index(i+1, j, n_x)]) +
                  lam_y * (u[index(i, j-1, n_x)] + u[index(i, j+1, n_x)]));
    u_new[idx] = newv;
    //Calculamos la mejora local
    float d = fabsf(newv - oldv);
    //Convertimos a bits
    unsigned int d_bits =  __float_as_uint(d);
    //Utilizamos atomicMax, que permite (de manera atomica) comparar el valor actual con el nuevo y guardar el maximo
    atomicMax(diff_max_bits, d_bits);
}