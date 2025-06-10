#include "benchmark/compute_perf.h"
#include "benchmark/common.h"
#include <math.h>

namespace gpu_benchmark {

__global__ void computeIntensiveKernel(float* input, float* output, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        for (int i = 0; i < iterations; ++i) {
            val = sinf(val) * cosf(val) + sqrtf(fabsf(val)) + expf(val * 0.01f);
        }
        output[idx] = val;
    }
}

float testComputePerformance(int numElements, int computeIterations) {
    size_t bytes = numElements * sizeof(float);

    printf("\n==== Compute Performance Test ====\n");
    printf("Elements: %d, Iterations: %d\n", numElements, computeIterations);

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
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    computeIntensiveKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements, 1);
    GPU_CHECK(cudaDeviceSynchronize());

    double start = get_time_ms();
    computeIntensiveKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements, computeIterations);
    GPU_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    double elapsed = (end - start) / 1000.0;
    double gflops = (numElements * computeIterations * 5.0) / (elapsed * 1.0e9);
    printf("Performance: %.2f GFLOPS\n", gflops);

    GPU_CHECK(cudaFree(d_input));
    GPU_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return gflops;
}

} // namespace gpu_benchmark
