#include "benchmark/tensor_core.h"
#include "benchmark/common.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <string.h>

#define CHECK_CUBLAS_ERROR(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d - code: %d\n", __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace gpu_benchmark {

float testTensorCore() {
    printf("\n==== TensorCore Test (Neural Network Engine) ====\n");

    cudaDeviceProp prop;
    int device;
    GPU_CHECK(cudaGetDevice(&device));
    GPU_CHECK(cudaGetDeviceProperties(&prop, device));

    if (prop.major < 7) {
        printf("GPU Compute Capability: %d.%d, TensorCore not supported (requires 7.0+)\n",
               prop.major, prop.minor);
        return 0.0f;
    }

    printf("Running TensorCore test...\n");

    int M = 4096;
    int N = 4096;
    int K = 4096;

    size_t sizeA = M * K * sizeof(half);
    size_t sizeB = K * N * sizeof(half);
    size_t sizeC = M * N * sizeof(float);

    if (sizeA > 1024*1024*1024 || sizeB > 1024*1024*1024 || sizeC > 1024*1024*1024) {
        printf("Matrix too large, using smaller size...\n");
        M = N = K = 2048;
        sizeA = M * K * sizeof(half);
        sizeB = K * N * sizeof(half);
        sizeC = M * N * sizeof(float);
    }

    half* h_A = (half*)malloc(sizeA);
    half* h_B = (half*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.01f);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.01f);
    }
    memset(h_C, 0, sizeC);

    half *d_A, *d_B;
    float *d_C;
    GPU_CHECK(cudaMalloc((void**)&d_A, sizeA));
    GPU_CHECK(cudaMalloc((void**)&d_B, sizeB));
    GPU_CHECK(cudaMalloc((void**)&d_C, sizeC));

    GPU_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K,
                                &alpha,
                                d_B, CUDA_R_16F, N,
                                d_A, CUDA_R_16F, K,
                                &beta,
                                d_C, CUDA_R_32F, N,
                                CUDA_R_32F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    const int iterations = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < iterations; ++i) {
        CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    N, M, K,
                                    &alpha,
                                    d_B, CUDA_R_16F, N,
                                    d_A, CUDA_R_16F, K,
                                    &beta,
                                    d_C, CUDA_R_32F, N,
                                    CUDA_R_32F,
                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f / iterations;

    double flops = 2.0 * M * N * K;
    double tflops = (flops / seconds) / 1e12;

    printf("TensorCore Performance: %.2f TFLOPS\n", tflops);
    printf("Matrix Size: %dx%d x %dx%d = %dx%d\n", M, K, K, N, M, N);
    printf("Time per operation: %.2f ms\n", seconds * 1000.0f);

    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    GPU_CHECK(cudaFree(d_A));
    GPU_CHECK(cudaFree(d_B));
    GPU_CHECK(cudaFree(d_C));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_A);
    free(h_B);
    free(h_C);

    return tflops * 1000.0f; // Convert to GFLOPS
}

} // namespace gpu_benchmark
