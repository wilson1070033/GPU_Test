#include "benchmark/benchmark.h"
#include <stdio.h>
#include <math.h>

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
    printf("Clock Rate: %d MHz\n", prop.clockRate / 1000);
    printf("Memory: %.2f GB\n", *memoryGB);
    printf("Memory Clock: %d MHz\n", prop.memoryClockRate / 1000);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("\n");
}

static void calculateGPUScore(float memBandwidth, float computePerf, float parallelPerf,
                              float stressPerf, float texturePerf, float dynamicParallelismPerf,
                              float sharedMemBandwidth, float flopsThroughput, float tensorCorePerf,
                              int computeCapability, int coreCount, float memoryGB) {
    printf("\n==== GPU Performance Score (RTX 4090 Baseline) ====\n");

    // Test weights
    float memBandwidthWeight = 0.20f;
    float computePerfWeight = 0.15f;
    float parallelPerfWeight = 0.05f;
    float stressPerfWeight = 0.05f;
    float texturePerfWeight = 0.10f;
    float dynamicParallelismWeight = 0.05f;
    float sharedMemBandwidthWeight = 0.10f;
    float flopsThroughputWeight = 0.15f;
    float tensorCoreWeight = 0.15f;

    // Baseline values (RTX 4090)
    float memBandwidthBaseline = 1000.0f;
    float computePerfBaseline = 80000.0f;
    float parallelPerfBaseline = 300.0f;
    float stressPerfBaseline = 50.0f;
    float texturePerfBaseline = 10000.0f;
    float dynamicParallelismBaseline = 10.0f;
    float sharedMemBandwidthBaseline = 40000.0f;
    float flopsThroughputBaseline = 80000.0f;
    float tensorCoreBaseline = 330000.0f;

    // Calculate individual scores (max 100)
    float memScore = fminf(100.0f, (memBandwidth / memBandwidthBaseline) * 100.0f);
    float computeScore = fminf(100.0f, (computePerf / computePerfBaseline) * 100.0f);
    float parallelScore = fminf(100.0f, (parallelPerf / parallelPerfBaseline) * 100.0f);
    float stressScore = fminf(100.0f, (stressPerf / stressPerfBaseline) * 100.0f);
    float textureScore = fminf(100.0f, (texturePerf / texturePerfBaseline) * 100.0f);

    float dynamicParallelismScore = 0.0f;
    if (dynamicParallelismPerf > 0.0f) {
        dynamicParallelismScore = fminf(100.0f, (dynamicParallelismPerf / dynamicParallelismBaseline) * 100.0f);
    }

    float sharedMemScore = fminf(100.0f, (sharedMemBandwidth / sharedMemBandwidthBaseline) * 100.0f);
    float flopsScore = fminf(100.0f, (flopsThroughput / flopsThroughputBaseline) * 100.0f);

    float tensorCoreScore = 0.0f;
    if (tensorCorePerf > 0.0f) {
        tensorCoreScore = fminf(100.0f, (tensorCorePerf / tensorCoreBaseline) * 100.0f);
    }

    // Calculate weighted score
    float weightSum = 0.0f;
    float weightedScore = 0.0f;

    weightedScore += memBandwidthWeight * memScore;
    weightSum += memBandwidthWeight;

    weightedScore += computePerfWeight * computeScore;
    weightSum += computePerfWeight;

    weightedScore += parallelPerfWeight * parallelScore;
    weightSum += parallelPerfWeight;

    weightedScore += stressPerfWeight * stressScore;
    weightSum += stressPerfWeight;

    weightedScore += texturePerfWeight * textureScore;
    weightSum += texturePerfWeight;

    if (dynamicParallelismPerf > 0.0f) {
        weightedScore += dynamicParallelismWeight * dynamicParallelismScore;
        weightSum += dynamicParallelismWeight;
    }

    weightedScore += sharedMemBandwidthWeight * sharedMemScore;
    weightSum += sharedMemBandwidthWeight;

    weightedScore += flopsThroughputWeight * flopsScore;
    weightSum += flopsThroughputWeight;

    if (tensorCorePerf > 0.0f) {
        weightedScore += tensorCoreWeight * tensorCoreScore;
        weightSum += tensorCoreWeight;
    }

    float totalScore = weightedScore / weightSum;

    // Hardware bonus
    float hwBonus = 1.0f;
    if (computeCapability >= 90) hwBonus += 0.25f;      // Ada Lovelace (RTX 40)
    else if (computeCapability >= 86) hwBonus += 0.20f; // Ampere (RTX 30)
    else if (computeCapability >= 75) hwBonus += 0.15f; // Turing (RTX 20)
    else if (computeCapability >= 70) hwBonus += 0.10f; // Volta
    else if (computeCapability >= 60) hwBonus += 0.05f; // Pascal

    if (coreCount > 100) hwBonus += 0.15f;
    else if (coreCount > 70) hwBonus += 0.10f;
    else if (coreCount > 40) hwBonus += 0.05f;

    if (memoryGB > 20.0f) hwBonus += 0.15f;
    else if (memoryGB > 12.0f) hwBonus += 0.10f;
    else if (memoryGB > 8.0f) hwBonus += 0.05f;

    totalScore *= hwBonus;
    totalScore = fminf(100.0f, totalScore);

    // Display scores
    printf("Memory Bandwidth: %.1f / 100 (%.2f GB/s)\n", memScore, memBandwidth);
    printf("Compute Performance: %.1f / 100 (%.2f GFLOPS)\n", computeScore, computePerf);
    printf("Parallel Performance: %.1f / 100 (%.2f M threads/s)\n", parallelScore, parallelPerf);
    printf("Stress Performance: %.1f / 100 (%.2f iter/s)\n", stressScore, stressPerf);
    printf("Texture Performance: %.1f / 100 (%.2f M accesses/s)\n", textureScore, texturePerf);

    if (dynamicParallelismPerf > 0.0f) {
        printf("Dynamic Parallelism: %.1f / 100 (%.2f M kernels/s)\n", dynamicParallelismScore, dynamicParallelismPerf);
    } else {
        printf("Dynamic Parallelism: Not Supported\n");
    }

    printf("Shared Memory Bandwidth: %.1f / 100 (%.2f GB/s)\n", sharedMemScore, sharedMemBandwidth);
    printf("FP Throughput: %.1f / 100 (%.2f GFLOPS)\n", flopsScore, flopsThroughput);

    if (tensorCorePerf > 0.0f) {
        printf("TensorCore: %.1f / 100 (%.2f GFLOPS)\n", tensorCoreScore, tensorCorePerf);
    } else {
        printf("TensorCore: Not Supported\n");
    }

    printf("Hardware Bonus: %.2fx\n", hwBonus);
    printf("\n");

    printf("Total Score: %.1f\n", totalScore);

    // Performance tier
    printf("Performance Tier: ");
    if (totalScore >= 90.0f) {
        printf("S++ Top Flagship (Ultra High-End Professional/AI Research)\n");
    } else if (totalScore >= 80.0f) {
        printf("S+ Top Flagship (High-End Professional)\n");
    } else if (totalScore >= 70.0f) {
        printf("S High-End (High-End Gaming/Professional)\n");
    } else if (totalScore >= 60.0f) {
        printf("A+ Premium (Mid-High Gaming/Professional)\n");
    } else if (totalScore >= 50.0f) {
        printf("A Good (Mid-Range Gaming/General Professional)\n");
    } else if (totalScore >= 40.0f) {
        printf("B+ Medium (Entry Gaming/General)\n");
    } else if (totalScore >= 30.0f) {
        printf("B Basic (Light Gaming/General)\n");
    } else {
        printf("C Entry (General/Basic Display)\n");
    }
}

} // namespace gpu_benchmark

int main() {
    using namespace gpu_benchmark;

    printf("====================================================\n");
    printf("         CUDA GPU Performance Benchmark             \n");
    printf("====================================================\n");
    printf("This program will run comprehensive GPU tests.\n");
    printf("Note: GPU will run at full load, ensure good cooling.\n");
    printf("====================================================\n\n");

    int computeCapability = 0;
    int coreCount = 0;
    float memoryGB = 0.0f;

    showGPUInfo(&computeCapability, &coreCount, &memoryGB);

    // Run all tests
    float memBandwidth = testMemoryBandwidth(512 * 1024 * 1024);
    float computePerf = testComputePerformance(5000000, 1000);
    float parallelPerf = testParallelism(10000000);
    float texturePerf = testTexturePerformance(2048, 2048);
    float sharedMemBandwidth = testSharedMemoryBandwidth();
    float flopsThroughput = testFLOPSThroughput();
    float tensorCorePerf = testTensorCore();
    float dynamicParallelismPerf = testDynamicParallelism();
    float stressPerf = stressTest(10);

    calculateGPUScore(memBandwidth, computePerf, parallelPerf, stressPerf,
                      texturePerf, dynamicParallelismPerf, sharedMemBandwidth,
                      flopsThroughput, tensorCorePerf, computeCapability,
                      coreCount, memoryGB);

    printf("\nAll tests completed!\n");

    return 0;
}
