__device__ int index(int i, int j, int nx) { //Hace la conversion de i j a indice
    return i*(nx+2)+j;
}

__global__ void heat1D(float* resultado, int i, float lam, int nx){ //n es el tamanno de cada fila
    //Implementa la formula: resultado[i][j]  = (1-2*lam)*resultado[i-1][j] + (resultado[i-1][j-1] + resultado[i-1][j+1])*lam
    int j = blockIdx.x * 1024 + threadIdx.x + 1; //El +1 es porque la columna 0 no la hacemos
    if(1 <= j && j <= nx + 1){ //No queremos que calcule el ultimo ni el primero
        resultado[index(i,j,nx)] = (1-2*lam)*resultado[index(i-1,j,nx)] + (resultado[index(i-1,j-1,nx)] + resultado[index(i-1,j+1,nx)])*lam;
    }
}