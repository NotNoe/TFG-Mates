
__global__ void hello_world(float* a){
    int id = threadIdx.x;
    printf("Hello World, my id is %i and my number is \"%f\".\n", id, a[id]);
}