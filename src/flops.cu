#include "benchmark/flops.h"
#include "benchmark/common.h"

namespace gpu_benchmark {

__global__ void flopsKernel(float* data, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = data[idx];
    float y = x;
    float z = x;
    float w = x;

    for (int i = 0; i < iterations; ++i) {
        x = x + y * z;
        y = y + z * w;
        z = z + w * x;
        w = w + x * y;
    }

    data[idx] = x + y + z + w;
}

float testFLOPSThroughput() {
    printf("\n==== Floating-Point Throughput Test ====\n");

    int numElements = 1000000;
    size_t bytes = numElements * sizeof(float);

    float* h_data = (float*)malloc(bytes);
    for (int i = 0; i < numElements; ++i) {
        h_data[i] = 1.0f + (rand() % 1000) / 1000.0f;
    }

    float* d_data;
    GPU_CHECK(cudaMalloc((void**)&d_data, bytes));
    GPU_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    int iterations = 1000;

    flopsKernel<<<gridSize, blockSize>>>(d_data, 10);
    GPU_CHECK(cudaDeviceSynchronize());

    double start = get_time_ms();
    flopsKernel<<<gridSize, blockSize>>>(d_data, iterations);
    GPU_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    double elapsed = (end - start) / 1000.0;
    double totalFlops = (double)numElements * iterations * 8;
    double tflops = totalFlops / (elapsed * 1.0e12);

    printf("FP Throughput: %.2f TFLOPS\n", tflops);

    GPU_CHECK(cudaFree(d_data));
    free(h_data);

    return tflops * 1000.0f; // Convert to GFLOPS
}

} // namespace gpu_benchmark
