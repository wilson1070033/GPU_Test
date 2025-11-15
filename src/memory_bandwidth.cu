#include "benchmark/memory_bandwidth.h"
#include "benchmark/common.h"
#include <math.h>

namespace gpu_benchmark {

// Optimized version using vector loads (float4 for better memory coalescing)
__global__ void memoryBandwidthKernel(float* input, float* output, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < n) {
        // Vector load/store for better bandwidth
        float4 data = *reinterpret_cast<float4*>(&input[idx]);
        *reinterpret_cast<float4*>(&output[idx]) = data;
    } else if (idx < n) {
        // Handle remainder
        for (int i = idx; i < n && i < idx + 4; ++i) {
            output[i] = input[i];
        }
    }
}

float testMemoryBandwidth(int size) {
    int numElements = size / sizeof(float);
    size_t bytes = numElements * sizeof(float);

    printf("==== Memory Bandwidth Test ====\n");
    printf("Data Size: %.2f MB\n", bytes / (1024.0 * 1024.0));

    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    for (int i = 0; i < numElements; ++i) {
        h_input[i] = rand() / (float)RAND_MAX;
    }

    float *d_input, *d_output;
    GPU_CHECK(cudaMalloc((void**)&d_input, bytes));
    GPU_CHECK(cudaMalloc((void**)&d_output, bytes));
    GPU_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    // Each thread processes 4 elements
    int blocksPerGrid = ((numElements / 4) + threadsPerBlock - 1) / threadsPerBlock;

    memoryBandwidthKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements);
    GPU_CHECK(cudaDeviceSynchronize());

    int iterations = 20;
    double start = get_time_ms();
    for (int i = 0; i < iterations; ++i) {
        memoryBandwidthKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements);
    }
    GPU_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    double elapsed = (end - start) / 1000.0;
    double bandwidth = (2.0 * bytes * iterations) / (elapsed * 1.0e9);
    printf("Bandwidth: %.2f GB/s\n", bandwidth);

    GPU_CHECK(cudaFree(d_input));
    GPU_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return bandwidth;
}

} // namespace gpu_benchmark
