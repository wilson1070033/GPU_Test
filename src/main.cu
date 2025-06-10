#include "benchmark/benchmark.h"
#include <stdio.h>

namespace gpu_benchmark {

static void showGPUInfo(int* computeCapability, int* coreCount, float* memoryGB) {
    cudaDeviceProp prop;
    int device;
    GPU_CHECK(cudaGetDevice(&device));
    GPU_CHECK(cudaGetDeviceProperties(&prop, device));

    *computeCapability = prop.major * 10 + prop.minor;
    *coreCount = prop.multiProcessorCount;
    *memoryGB = prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f);

    printf("\n==== GPU Info ====\n");
    printf("Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Memory: %.2f GB\n", *memoryGB);
}

static void calculateGPUScore(float memBandwidth, float computePerf, float parallelPerf, float texturePerf, float stressPerf,
                              int computeCapability, int coreCount, float memoryGB) {
    float baseScore = 100.0f;
    float score = baseScore * (
        0.2f * (memBandwidth / 300.0f) +
        0.25f * (computePerf / 10000.0f) +
        0.2f * (parallelPerf / 5.0f) +
        0.15f * (texturePerf / 1000.0f) +
        0.2f * (stressPerf / 5.0f));

    float bonus = 1.0f;
    if (computeCapability >= 70) bonus += 0.1f;
    if (coreCount > 30) bonus += 0.1f;
    if (memoryGB > 8.0f) bonus += 0.1f;

    score *= bonus;

    printf("\nTotal Score: %.1f\n", score);
}

} // namespace gpu_benchmark

int main() {
    using namespace gpu_benchmark;
    int computeCapability = 0;
    int coreCount = 0;
    float memoryGB = 0.0f;

    showGPUInfo(&computeCapability, &coreCount, &memoryGB);

    float memBandwidth = testMemoryBandwidth(512 * 1024 * 1024);
    float computePerf = testComputePerformance(5000000, 1000);
    float parallelPerf = testParallelism(10000000);
    float texturePerf = testTexturePerformance(1024, 1024);
    float stressPerf = stressTest(10);

    calculateGPUScore(memBandwidth, computePerf, parallelPerf, texturePerf, stressPerf,
                      computeCapability, coreCount, memoryGB);

    return 0;
}
