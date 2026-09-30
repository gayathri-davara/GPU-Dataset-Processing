#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloGPU()
{
    printf("Hello from CUDA thread %d, block %d\n",
           threadIdx.x, blockIdx.x);
}

int main()
{
    printf("CPU: Starting CUDA test...\n");

    helloGPU<<<2, 4>>>();

    cudaError_t error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    printf("CPU: CUDA kernel completed successfully.\n");

    return 0;
}