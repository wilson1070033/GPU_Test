#include "benchmark/stress_test.h"
#include "benchmark/common.h"
#include <math.h>

namespace gpu_benchmark {

__global__ void stressKernel(float* input, float* output, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        for (int i = 0; i < iterations; ++i) {
            val = sinf(val) * cosf(val) + sqrtf(fabsf(val)) + expf(val * 0.01f);
        }
        output[idx] = val;
    }
}

float stressTest(int durationSec) {
    printf("\n==== Stress Test (%d sec) ====\n", durationSec);

    int numElements = 10000000;
    size_t bytes = numElements * sizeof(float);

    float *d_input, *d_output;
    GPU_CHECK(cudaMalloc((void**)&d_input, bytes));
    GPU_CHECK(cudaMalloc((void**)&d_output, bytes));

    float* h_input = (float*)malloc(bytes);
    for (int i = 0; i < numElements; ++i) h_input[i] = rand() / (float)RAND_MAX;
    GPU_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    printf("Starting stress test...\n");
    double start = get_time_ms();
    double current = start;
    int iterations = 0;
    while ((current - start) < durationSec * 1000.0) {
        stressKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements, 100);
        GPU_CHECK(cudaDeviceSynchronize());
        float* tmp = d_input; d_input = d_output; d_output = tmp;
        iterations++;
        current = get_time_ms();
    }

    double elapsed = (current - start) / 1000.0;
    double iterPerSec = iterations / elapsed;

    printf("Iterations: %d\n", iterations);
    printf("Average iterations/sec: %.2f\n", iterPerSec);

    GPU_CHECK(cudaFree(d_input));
    GPU_CHECK(cudaFree(d_output));
    free(h_input);

    return iterPerSec;
}

} // namespace gpu_benchmark
