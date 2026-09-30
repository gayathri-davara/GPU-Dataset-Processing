#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <windows.h>
#include <cuda_runtime.h>

#define N 1000000
#define THREADS_PER_BLOCK 256
#define BLOCKS 256

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t error = call;                                 \
    if (error != cudaSuccess) {                               \
        fprintf(stderr, "CUDA ERROR: %s\n",                  \
                cudaGetErrorString(error));                  \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)


// ============================================================
// CUDA reduction kernel
// ============================================================

__global__ void reduce_stats(
    const float* data,
    float* block_sum,
    float* block_min,
    float* block_max)
{
    __shared__ float shared_sum[THREADS_PER_BLOCK];
    __shared__ float shared_min[THREADS_PER_BLOCK];
    __shared__ float shared_max[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;

    float local_sum = 0.0f;
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    // Grid-stride loop
    for (int i = global_id; i < N; i += blockDim.x * gridDim.x)
    {
        float value = data[i];

        local_sum += value;

        if (value < local_min)
            local_min = value;

        if (value > local_max)
            local_max = value;
    }

    shared_sum[tid] = local_sum;
    shared_min[tid] = local_min;
    shared_max[tid] = local_max;

    __syncthreads();


    // Block reduction
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2)
    {
        if (tid < stride)
        {
            shared_sum[tid] += shared_sum[tid + stride];

            if (shared_min[tid + stride] < shared_min[tid])
                shared_min[tid] = shared_min[tid + stride];

            if (shared_max[tid + stride] > shared_max[tid])
                shared_max[tid] = shared_max[tid + stride];
        }

        __syncthreads();
    }


    // One result per block
    if (tid == 0)
    {
        block_sum[blockIdx.x] = shared_sum[0];
        block_min[blockIdx.x] = shared_min[0];
        block_max[blockIdx.x] = shared_max[0];
    }
}


// ============================================================
// CPU statistics
// ============================================================

void cpu_stats(
    const float* data,
    double& sum,
    float& min_val,
    float& max_val)
{
    sum = 0.0;
    min_val = FLT_MAX;
    max_val = -FLT_MAX;

    for (int i = 0; i < N; i++)
    {
        float value = data[i];

        sum += value;

        if (value < min_val)
            min_val = value;

        if (value > max_val)
            max_val = value;
    }
}


// ============================================================
// Main
// ============================================================

int main()
{
    printf("============================================\n");
    printf("       GPU DATASET PROCESSING - CUDA\n");
    printf("============================================\n");

    printf("Dataset elements : %d\n", N);
    printf("Threads per block: %d\n", THREADS_PER_BLOCK);
    printf("Number of blocks : %d\n\n", BLOCKS);


    // --------------------------------------------------------
    // Allocate host memory
    // --------------------------------------------------------

    float* h_data =
        (float*)malloc(N * sizeof(float));

    if (!h_data)
    {
        printf("ERROR: Host memory allocation failed.\n");
        return 1;
    }


    // --------------------------------------------------------
    // Load dataset
    // --------------------------------------------------------

    FILE* file =
        fopen("data/dataset_1M.bin", "rb");

    if (!file)
    {
        printf("ERROR: Could not open dataset.\n");
        free(h_data);
        return 1;
    }

    size_t elements_read =
        fread(
            h_data,
            sizeof(float),
            N,
            file
        );

    fclose(file);

    if (elements_read != N)
    {
        printf(
            "ERROR: Expected %d elements, got %zu.\n",
            N,
            elements_read
        );

        free(h_data);
        return 1;
    }

    printf("Dataset loaded successfully.\n");


    // --------------------------------------------------------
    // CPU processing
    // --------------------------------------------------------

    double cpu_sum;
    float cpu_min;
    float cpu_max;

    LARGE_INTEGER cpu_frequency;
    LARGE_INTEGER cpu_start;
    LARGE_INTEGER cpu_end;

    QueryPerformanceFrequency(&cpu_frequency);
    QueryPerformanceCounter(&cpu_start);

    cpu_stats(
        h_data,
        cpu_sum,
        cpu_min,
        cpu_max
    );

    QueryPerformanceCounter(&cpu_end);

    double cpu_time =
        (double)(cpu_end.QuadPart - cpu_start.QuadPart)
        * 1000.0
        / cpu_frequency.QuadPart;


    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------

    float* d_data;
    float* d_block_sum;
    float* d_block_min;
    float* d_block_max;

    CUDA_CHECK(
        cudaMalloc(
            &d_data,
            N * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_block_sum,
            BLOCKS * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_block_min,
            BLOCKS * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_block_max,
            BLOCKS * sizeof(float)
        )
    );


    // --------------------------------------------------------
    // Copy dataset CPU -> GPU
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            d_data,
            h_data,
            N * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    // --------------------------------------------------------
    // CUDA timing
    // --------------------------------------------------------

    cudaEvent_t gpu_start;
    cudaEvent_t gpu_stop;

    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_stop));

    CUDA_CHECK(cudaEventRecord(gpu_start));


    // --------------------------------------------------------
    // Launch CUDA kernel
    // --------------------------------------------------------

    reduce_stats<<<BLOCKS, THREADS_PER_BLOCK>>>(
        d_data,
        d_block_sum,
        d_block_min,
        d_block_max
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(gpu_stop));
    CUDA_CHECK(cudaEventSynchronize(gpu_stop));


    float gpu_time;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &gpu_time,
            gpu_start,
            gpu_stop
        )
    );


    // --------------------------------------------------------
    // Copy block results back to CPU
    // --------------------------------------------------------

    float h_block_sum[BLOCKS];
    float h_block_min[BLOCKS];
    float h_block_max[BLOCKS];

    CUDA_CHECK(
        cudaMemcpy(
            h_block_sum,
            d_block_sum,
            BLOCKS * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            h_block_min,
            d_block_min,
            BLOCKS * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            h_block_max,
            d_block_max,
            BLOCKS * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // --------------------------------------------------------
    // Combine block results
    // --------------------------------------------------------

    double gpu_sum = 0.0;
    float gpu_min = FLT_MAX;
    float gpu_max = -FLT_MAX;

    for (int i = 0; i < BLOCKS; i++)
    {
        gpu_sum += h_block_sum[i];

        if (h_block_min[i] < gpu_min)
            gpu_min = h_block_min[i];

        if (h_block_max[i] > gpu_max)
            gpu_max = h_block_max[i];
    }


    // --------------------------------------------------------
    // Calculate averages
    // --------------------------------------------------------

    double cpu_average =
        cpu_sum / N;

    double gpu_average =
        gpu_sum / N;


    // --------------------------------------------------------
    // Display CPU results
    // --------------------------------------------------------

    printf("\n============================================\n");
    printf("CPU RESULTS\n");
    printf("============================================\n");

    printf("SUM     : %.6f\n", cpu_sum);
    printf("MIN     : %.6f\n", cpu_min);
    printf("MAX     : %.6f\n", cpu_max);
    printf("AVERAGE : %.6f\n", cpu_average);
    printf("CPU TIME: %.6f ms\n", cpu_time);


    // --------------------------------------------------------
    // Display GPU results
    // --------------------------------------------------------

    printf("\n============================================\n");
    printf("GPU RESULTS\n");
    printf("============================================\n");

    printf("SUM     : %.6f\n", gpu_sum);
    printf("MIN     : %.6f\n", gpu_min);
    printf("MAX     : %.6f\n", gpu_max);
    printf("AVERAGE : %.6f\n", gpu_average);
    printf("GPU TIME: %.6f ms\n", gpu_time);


    // --------------------------------------------------------
    // Validation
    // --------------------------------------------------------

    double sum_difference =
        fabs(cpu_sum - gpu_sum);

    float min_difference =
        fabs(cpu_min - gpu_min);

    float max_difference =
        fabs(cpu_max - gpu_max);

    double average_difference =
        fabs(cpu_average - gpu_average);


    printf("\n============================================\n");
    printf("VALIDATION\n");
    printf("============================================\n");

    printf(
        "SUM difference     : %.6f\n",
        sum_difference
    );

    printf(
        "MIN difference     : %.6f\n",
        min_difference
    );

    printf(
        "MAX difference     : %.6f\n",
        max_difference
    );

    printf(
        "AVERAGE difference : %.6f\n",
        average_difference
    );


    // --------------------------------------------------------
    // Performance
    // --------------------------------------------------------

    double speedup =
        cpu_time / gpu_time;


    printf("\n============================================\n");
    printf("PERFORMANCE\n");
    printf("============================================\n");

    printf(
        "CPU time : %.6f ms\n",
        cpu_time
    );

    printf(
        "GPU time : %.6f ms\n",
        gpu_time
    );

    printf(
        "Speedup  : %.2fx\n",
        speedup
    );


    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_block_sum));
    CUDA_CHECK(cudaFree(d_block_min));
    CUDA_CHECK(cudaFree(d_block_max));

    CUDA_CHECK(cudaEventDestroy(gpu_start));
    CUDA_CHECK(cudaEventDestroy(gpu_stop));

    free(h_data);

    printf("\nCUDA processing completed successfully.\n");

    return 0;
}