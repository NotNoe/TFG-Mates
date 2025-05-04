__device__ int index(int i, int j, int n_y) {
    return i * n_y + j;
}

__device__ int get_i() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int get_j() {
    return blockIdx.y * blockDim.y + threadIdx.y;
}

__global__ void resolve(const float* __restrict__ u_old,
                        const float* __restrict__ u_older,
                        float* __restrict__ u_new,
                        float lam_x2, float lam_y2, int n_x, int n_y){ //n es el tamanno de cada fila

    //Calculamos que indice corresponde a nuestro hilo
    int i = get_i();
    int j = get_j();
    if (i < 0 || i > n_x - 1 || j < 0 || j > n_y - 1) return;
    if (i == 0 || i == n_x - 1 || j == 0 || j == n_y - 1) u_new[index(i, j, n_y)] = 0;
    else{
        int idx = index(i, j, n_y);
        float center = u_old[index(i, j, n_y)];
        float left = u_old[index(i - 1, j, n_y)];
        float right = u_old[index(i + 1, j, n_y)];
        float down = u_old[index(i, j - 1, n_y)];
        float up = u_old[index(i, j + 1, n_y)];

        u_new[idx] = 2*(1 - lam_x2 - lam_y2) * center + (left + right) * lam_x2 + (down + up) * lam_y2 - u_older[idx];
    }
}