#ifndef COMPUTE_TESTS_H
#define COMPUTE_TESTS_H

#include "benchmark_common.h" // For CHECK_CUDA_ERROR, CHECK_CUBLAS_ERROR, get_time_ms, cudaDeviceProp, etc.

// Test compute performance (general)
float testComputePerformance(int numElements, int computeIterations);

// Test FLOPs throughput
float testFLOPSThroughput();

// Test TensorCore performance
float testTensorCore();

// Forward declaration for the kernel if it's not intended to be public,
// but used by a function within compute_tests.cu that might be called from elsewhere later.
// However, kernels are typically static to the .cu file or anonymous namespace.
// For now, we assume kernels are only called by functions within the same .cu file.

__global__ void computeIntensiveTest(float *input, float *output, int n, int iterations);

#endif // COMPUTE_TESTS_H
