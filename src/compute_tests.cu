#include "compute_tests.h"
__global__ void computeIntensiveTest(float *input, float *output, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];

        // 執行大量浮點運算以測試計算能力
        for (int i = 0; i < iterations; i++) {
            val = sinf(val) * cosf(val) + sqrtf(fabs(val)) + exp(val * 0.01f);
        }

        output[idx] = val;
    }
}

float testComputePerformance(int numElements, int computeIterations) {
    size_t bytes = numElements * sizeof(float);

    printf("\n==== 計算效能測試 ====\n");
    printf("元素數量: %d, 每個元素計算迭代: %d\n", numElements, computeIterations);

    // 主機記憶體分配
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);

    // 初始化資料
    for (int i = 0; i < numElements; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }

    // 設備記憶體分配
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, bytes));

    // 複製資料到設備
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // 計算網格和區塊大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // 預熱GPU
    computeIntensiveTest<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements, 1);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 測試計算效能
    double startTime = get_time_ms();

    computeIntensiveTest<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements, computeIterations);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    double endTime = get_time_ms();

    double elapsedTime = (endTime - startTime) / 1000.0; // 秒
    double gflops = (numElements * computeIterations * 5.0) / (elapsedTime * 1.0e9); // GFLOPS (假設每次迭代大約5個浮點運算)

    printf("計算效能: %.2f GFLOPS\n", gflops);
    printf("執行時間: %.2f 秒\n", elapsedTime);

    // 釋放記憶體
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return gflops;
}

__global__ void flopsTest(float *data, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = data[idx];
    float y = x;
    float z = x;
    float w = x;

    // 大量浮點運算
    for (int i = 0; i < iterations; i++) {
        // 混合加法、乘法、平方根、三角函數來測試不同指令
        x = x + y * z;
        y = y + z * w;
        z = z + w * x;
        w = w + x * y;
    }

    data[idx] = x + y + z + w;
}

float testFLOPSThroughput() {
    printf("\n==== 浮點指令吞吐量測試 ====\n");

    int numElements = 1000000;
    size_t bytes = numElements * sizeof(float);

    // 分配和初始化數據
    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < numElements; i++) {
        h_data[i] = 1.0f + (rand() % 1000) / 1000.0f;
    }

    float *d_data;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // 設定執行參數
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    int iterations = 1000;

    // 預熱
    flopsTest<<<gridSize, blockSize>>>(d_data, 10);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 測量效能
    double startTime = get_time_ms();

    flopsTest<<<gridSize, blockSize>>>(d_data, iterations);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    double endTime = get_time_ms();

    double elapsedTime = (endTime - startTime) / 1000.0; // 秒

    // 每次迭代有 8 個浮點運算 (4個加法和4個乘法)
    double totalFlops = (double)numElements * iterations * 8;
    double tflops = totalFlops / (elapsedTime * 1.0e12); // TFLOPS

    printf("浮點指令吞吐量: %.2f TFLOPS\n", tflops);

    // 釋放資源
    CHECK_CUDA_ERROR(cudaFree(d_data));
    free(h_data);

    return tflops * 1000; // 轉換為 GFLOPS 以保持一致性
}

float testTensorCore() {
    printf("\n==== TensorCore測試 (類神經網路引擎) ====\n");

    // 檢查是否支援TensorCore (CUDA計算能力7.0及以上)
    cudaDeviceProp prop;
    int device;
    CHECK_CUDA_ERROR(cudaGetDevice(&device));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));

    if (prop.major < 7) {
        printf("您的GPU計算能力為 %d.%d，不支援TensorCore (需要 7.0+)\n", prop.major, prop.minor);
        return 0.0f;
    }

    printf("執行TensorCore測試...\n");

    // 矩陣大小 (適合TensorCore操作的尺寸)
    int M = 4096;  // 矩陣A的行數
    int N = 4096;  // 矩陣B的列數
    int K = 4096;  // 矩陣A的列數和矩陣B的行數

    // 分配主機記憶體
    size_t sizeA = M * K * sizeof(half);
    size_t sizeB = K * N * sizeof(half);
    size_t sizeC = M * N * sizeof(float);

    // 使用標準浮點測試
    if (sizeA > 1024*1024*1024 || sizeB > 1024*1024*1024 || sizeC > 1024*1024*1024) {
        printf("矩陣太大，使用較小的矩陣尺寸...\n");
        M = N = K = 2048;
        sizeA = M * K * sizeof(half);
        sizeB = K * N * sizeof(half);
        sizeC = M * N * sizeof(float);
    }

    // 分配主機記憶體
    half *h_A = (half*)malloc(sizeA);
    half *h_B = (half*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    // 初始化數據 (使用float轉換為half)
    for (int i = 0; i < M * K; i++) {
        h_A[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.01f);
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.01f);
    }
    memset(h_C, 0, sizeC);

    // 分配設備記憶體
    half *d_A;
    half *d_B;
    float *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, sizeA));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, sizeB));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, sizeC));

    // 複製數據到設備
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice));

    // 初始化cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    // 設置使用TensorCore
    CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // 預熱
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

    // 測量效能
    const int iterations = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < iterations; i++) {
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

    // 計算TFLOPS (每次矩陣乘法有2*M*N*K次浮點運算)
    double flops = 2.0 * M * N * K;
    double tflops = (flops / seconds) / 1e12;

    printf("TensorCore性能: %.2f TFLOPS\n", tflops);
    printf("矩陣尺寸: %dx%d x %dx%d = %dx%d\n", M, K, K, N, M, N);
    printf("每次運算時間: %.2f ms\n", seconds * 1000.0f);

    // 清理資源
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_A);
    free(h_B);
    free(h_C);

    return tflops * 1000.0f; // 轉為GFLOPS保持一致性
}
