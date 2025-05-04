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
                        float* __restrict__ u_new,
                        const float* __restrict__ g_left,
                        const float* __restrict__ g_right,
                        const float* __restrict__ g_bottom,
                        const float* __restrict__ g_top,
                        int n, float lam_x, float lam_y,
                        int n_x, int n_y, int n_t){ //n es el tamanno de cada fila

    //Calculamos que indice corresponde a nuestro hilo
    int i = get_i();
    int j = get_j();
    if (i < 0 || i > n_x - 1 || j < 0 || j > n_y - 1) return;
    if (i == 0) u_new[index(i,j,n_y)] = g_left[n];
    else if (i == n_x - 1) u_new[index(i,j,n_y)] = g_right[n];
    else if (j == 0) u_new[index(i,j,n_y)] = g_bottom[n];
    else if (j == n_y - 1) u_new[index(i,j,n_y)] = g_top[n];
    else {
        int idx_next = index(i, j, n_y);
        float center = u_old[index(i, j, n_y)];
        float left = u_old[index(i - 1, j, n_y)];
        float right = u_old[index(i + 1, j, n_y)];
        float down = u_old[index(i, j - 1, n_y)];
        float up = u_old[index(i, j + 1, n_y)];

        u_new[idx_next] = (1 - 2 * lam_x - 2 * lam_y) * center + (left + right) * lam_x + (down + up) * lam_y;
    }
}