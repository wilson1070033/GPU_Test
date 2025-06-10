#include "memory_tests.h"
__global__ void memoryBandwidthTest(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

float testMemoryBandwidth(int size) {
    int numElements = size / sizeof(float);
    size_t bytes = numElements * sizeof(float);

    printf("==== 記憶體頻寬測試 ====\n");
    printf("資料大小: %.2f MB\n", bytes / (1024.0 * 1024.0));

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
    memoryBandwidthTest<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 測試記憶體讀取頻寬
    int iterations = 20;
    double startTime = get_time_ms();

    for (int i = 0; i < iterations; i++) {
        memoryBandwidthTest<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements);
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    double endTime = get_time_ms();

    double elapsedTime = (endTime - startTime) / 1000.0; // 秒
    double bandwidth = (2.0 * bytes * iterations) / (elapsedTime * 1.0e9); // GB/s (2倍因為讀和寫)

    printf("記憶體頻寬: %.2f GB/s\n", bandwidth);

    // 釋放記憶體
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return bandwidth;
}

__global__ void sharedMemoryBandwidthTest(float *data, int iterations) {
    __shared__ float sharedData[4096]; // 16 KB (或接近最大共享記憶體大小)

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // 初始化共享記憶體
    for (int i = tid; i < 4096; i += stride) {
        sharedData[i] = data[i];
    }
    __syncthreads();

    // 執行多次讀寫操作來測量頻寬
    float sum = 0.0f;
    for (int i = 0; i < iterations; i++) {
        int idx = (tid + i) % 4096;
        sum += sharedData[idx];
        sharedData[idx] = sum;
    }
    __syncthreads();

    // 寫回以防止編譯器優化
    data[tid] = sum + sharedData[tid];
}

float testSharedMemoryBandwidth() {
    printf("\n==== 共享記憶體頻寬測試 ====\n");

    int dataSize = 4096; // 元素數量
    size_t bytes = dataSize * sizeof(float);

    // 分配和初始化數據
    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < dataSize; i++) {
        h_data[i] = rand() / (float)RAND_MAX;
    }

    float *d_data;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // 執行測試
    int iterations = 10000;
    int blockSize = 256;

    // 預熱
    sharedMemoryBandwidthTest<<<1, blockSize>>>(d_data, 10);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 測量效能
    double startTime = get_time_ms();

    sharedMemoryBandwidthTest<<<1, blockSize>>>(d_data, iterations);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    double endTime = get_time_ms();

    double elapsedTime = (endTime - startTime) / 1000.0; // 秒

    // 估算共享記憶體頻寬 (每次迭代有 2 次存取 - 讀和寫)
    // 每個線程訪問多個元素
    double totalBytes = (double)iterations * blockSize * 2.0 * sizeof(float);
    double bandwidth = totalBytes / (elapsedTime * 1.0e9); // GB/s

    printf("共享記憶體頻寬: %.2f GB/s\n", bandwidth);

    // 釋放資源
    CHECK_CUDA_ERROR(cudaFree(d_data));
    free(h_data);

    return bandwidth;
}
