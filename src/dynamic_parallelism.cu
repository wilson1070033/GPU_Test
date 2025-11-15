#include "benchmark/dynamic_parallelism.h"
#include "benchmark/common.h"
#include <unistd.h>
#include <time.h>

namespace gpu_benchmark {

__global__ void dynamicParallelismKernel(int depth, int maxDepth, int* counter) {
    if (depth >= maxDepth) return;

    atomicAdd(counter, 1);

    if (depth < maxDepth - 1) {
        dim3 block(4);
        dim3 grid(2);
        dynamicParallelismKernel<<<grid, block>>>(depth + 1, maxDepth, counter);
    }
}

float testDynamicParallelism() {
    printf("\n==== Dynamic Parallelism Test ====\n");

    cudaDeviceProp prop;
    int device;
    GPU_CHECK(cudaGetDevice(&device));
    GPU_CHECK(cudaGetDeviceProperties(&prop, device));

    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5)) {
        printf("GPU Compute Capability: %d.%d, Dynamic Parallelism not supported (requires 3.5+)\n",
               prop.major, prop.minor);
        return 0.0f;
    }

    printf("Running Dynamic Parallelism test...\n");

    int* d_counter;
    GPU_CHECK(cudaMalloc((void**)&d_counter, sizeof(int)));
    GPU_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    dynamicParallelismKernel<<<1, 1>>>(0, 2, d_counter);
    GPU_CHECK(cudaDeviceSynchronize());
    GPU_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    int maxDepth = 3;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    dim3 grid(8);
    dim3 block(32);
    dynamicParallelismKernel<<<grid, block>>>(0, maxDepth, d_counter);

    float timeout = 5.0f;
    cudaError_t result = cudaEventRecord(stop);
    if (result != cudaSuccess) {
        printf("Dynamic Parallelism test error, skipping...\n");
        GPU_CHECK(cudaFree(d_counter));
        return 0.0f;
    }

    unsigned long startTime = (unsigned long)time(nullptr);
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        if ((unsigned long)time(nullptr) - startTime > timeout) {
            printf("Dynamic Parallelism test timeout, forcing stop...\n");
            cudaDeviceReset();
            return 0.0f;
        }
        usleep(10000);
    }

    float milliseconds = 0;
    result = cudaEventElapsedTime(&milliseconds, start, stop);
    if (result != cudaSuccess) {
        printf("Failed to get execution time, skipping...\n");
        GPU_CHECK(cudaFree(d_counter));
        return 0.0f;
    }

    double elapsed = milliseconds / 1000.0;

    int counter;
    GPU_CHECK(cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

    printf("Dynamic Parallelism count: %d\n", counter);
    printf("Execution time: %.4f s\n", elapsed);

    float kernelsPerSec = 0.0f;
    if (elapsed > 0) {
        kernelsPerSec = counter / (1.0e6 * elapsed);
        printf("Kernel launch rate: %.2f million/s\n", kernelsPerSec);
    } else {
        printf("Execution time too short for accurate measurement\n");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    GPU_CHECK(cudaFree(d_counter));

    return kernelsPerSec;
}

} // namespace gpu_benchmark
