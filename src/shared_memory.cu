#include "benchmark/shared_memory.h"
#include "benchmark/common.h"

namespace gpu_benchmark {

__global__ void sharedMemoryBandwidthKernel(float* data, int iterations) {
    __shared__ float sharedData[4096];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int i = tid; i < 4096; i += stride) {
        sharedData[i] = data[i];
    }
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        int idx = (tid + i) % 4096;
        sum += sharedData[idx];
        sharedData[idx] = sum;
    }
    __syncthreads();

    data[tid] = sum + sharedData[tid];
}

float testSharedMemoryBandwidth() {
    printf("\n==== Shared Memory Bandwidth Test ====\n");

    int dataSize = 4096;
    size_t bytes = dataSize * sizeof(float);

    float* h_data = (float*)malloc(bytes);
    for (int i = 0; i < dataSize; ++i) {
        h_data[i] = rand() / (float)RAND_MAX;
    }

    float* d_data;
    GPU_CHECK(cudaMalloc((void**)&d_data, bytes));
    GPU_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    int iterations = 10000;
    int blockSize = 256;

    sharedMemoryBandwidthKernel<<<1, blockSize>>>(d_data, 10);
    GPU_CHECK(cudaDeviceSynchronize());

    double start = get_time_ms();
    sharedMemoryBandwidthKernel<<<1, blockSize>>>(d_data, iterations);
    GPU_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    double elapsed = (end - start) / 1000.0;
    double totalBytes = (double)iterations * blockSize * 2.0 * sizeof(float);
    double bandwidth = totalBytes / (elapsed * 1.0e9);

    printf("Shared Memory Bandwidth: %.2f GB/s\n", bandwidth);

    GPU_CHECK(cudaFree(d_data));
    free(h_data);

    return bandwidth;
}

} // namespace gpu_benchmark
