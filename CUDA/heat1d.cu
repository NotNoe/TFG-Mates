__device__ int get_i(){
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void resolve(const float* __restrict__ u_old, float* __restrict__ u_new, const float* __restrict__ g, const float* __restrict__ h, float lam, int n, int n_x){ //n es el tamanno de cada fila
    //Calculamos que indice corresponde a nuestro hilo
    int i = get_i();
    if (i < 0 || i >= n_x) return; // Si el hilo no corresponde a un indice valido, salimos
    if (i == 0) u_new[i] = g[n];
    else if (i == n_x - 1) u_new[i] = h[n];
    else u_new[i] = (1-2*lam)*u_old[i] + (u_old[i-1] + u_old[i+1])*lam;
}