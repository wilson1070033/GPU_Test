#include "benchmark/parallelism.h"
#include "benchmark/common.h"

namespace gpu_benchmark {

__global__ void parallelismKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        atomicAdd(&data[0], val * 0.000001f);
    }
}

float testParallelism(int numThreads) {
    printf("\n==== Parallelism Test ====\n");
    printf("Threads: %d\n", numThreads);

    size_t bytes = numThreads * sizeof(float);
    float* h_data = (float*)malloc(bytes);
    for (int i = 0; i < numThreads; ++i) h_data[i] = 1.0f;

    float* d_data;
    GPU_CHECK(cudaMalloc((void**)&d_data, bytes));
    GPU_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

    parallelismKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, numThreads);
    GPU_CHECK(cudaDeviceSynchronize());

    double start = get_time_ms();
    parallelismKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, numThreads);
    GPU_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    double elapsed = (end - start) / 1000.0;
    double threadsPerSec = numThreads / (elapsed * 1.0e6);
    printf("Threads per second: %.2f million\n", threadsPerSec);

    GPU_CHECK(cudaFree(d_data));
    free(h_data);

    return threadsPerSec;
}

} // namespace gpu_benchmark
