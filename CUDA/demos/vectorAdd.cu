__global__ void vectorAdd(float* a, float* b, float* c, int n){
    int i = blockIdx.x * 1024 + threadIdx.x; //Calculamos el indice teniendo en cuenta que cada bloque tiene 1024 elementos
    if(i<n){
        c[i] = a[i] + b[i];
    }
}

