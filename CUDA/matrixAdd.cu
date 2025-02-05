__global__ void matrixAdd(float* a, float* b, float* c){
    int i = blockIdx.x; int j = threadIdx.x;
    int index = i + j*blockDim.x;
    c[index] = a[index] + b[index];
}