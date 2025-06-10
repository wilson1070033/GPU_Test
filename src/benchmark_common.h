#ifndef BENCHMARK_COMMON_H
#define BENCHMARK_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For memset
#include <unistd.h> // For usleep
#include <time.h>   // For time()
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h> // For cuBLAS types if needed by macros, or specific test headers
#include <cuda_fp16.h> // For half type if needed by macros, or specific test headers
#include <sys/time.h>

// Error checking macro - CUDA
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Error checking macro - cuBLAS
#define CHECK_CUBLAS_ERROR(call) \
do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error in %s:%d - Error Code: %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Get time in milliseconds
double get_time_ms();

#endif // BENCHMARK_COMMON_H
