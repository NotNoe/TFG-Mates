__device__ int get_i(){
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void resolve(const float* __restrict__ u_old,
                        const float* __restrict__ u_older,
                        float* __restrict__ u_new,
                        int n, float lam2, int n_x){
    //Calculamos que indice corresponde a nuestro hilo
    int i = get_i();
    if (i < 1 + n || i > n_x - 2 - n) return;

    u_new[i] = 2*(1-lam2)*u_old[i] + lam2*(u_old[i-1] + u_old[i+1]) - u_older[i];
}