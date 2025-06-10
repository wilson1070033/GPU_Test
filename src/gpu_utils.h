#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include "benchmark_common.h" // For CHECK_CUDA_ERROR and common types

// Shows GPU information
void showGPUInfo(int *computeCapability, int *coreCount, float *memoryGB);

#endif // GPU_UTILS_H
