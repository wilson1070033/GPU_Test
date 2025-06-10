/*
 * GPU基準測試程式 - 校準版
 * 使用RTX 4090作為頂級基準
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <sys/time.h>

// 錯誤檢查巨集 - CUDA
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA 錯誤在 %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 錯誤檢查巨集 - cuBLAS
#define CHECK_CUBLAS_ERROR(call) \
do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS 錯誤在 %s:%d - 錯誤碼: %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 取得時間（毫秒）
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

// 記憶體頻寬測試核函數
__global__ void memoryBandwidthTest(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

// 計算密集型測試核函數（大量浮點運算）
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

// 高密度並行測試核函數
__global__ void parallelismTest(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        
        // 大量原子操作以測試並行效能
        atomicAdd(&data[0], val * 0.000001f);
    }
}

// 測試記憶體頻寬
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

// 測試計算效能
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

// 測試GPU並行效能
float testParallelism(int numThreads) {
    printf("\n==== 並行效能測試 ====\n");
    printf("執行緒數量: %d\n", numThreads);
    
    size_t bytes = numThreads * sizeof(float);
    
    // 主機記憶體分配
    float *h_data = (float*)malloc(bytes);
    
    // 初始化資料
    for (int i = 0; i < numThreads; i++) {
        h_data[i] = 1.0f;
    }
    
    // 設備記憶體分配
    float *d_data;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, bytes));
    
    // 複製資料到設備
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    
    // 計算網格和區塊大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
    
    // 預熱GPU
    parallelismTest<<<blocksPerGrid, threadsPerBlock>>>(d_data, numThreads);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // 測試並行效能
    double startTime = get_time_ms();
    
    parallelismTest<<<blocksPerGrid, threadsPerBlock>>>(d_data, numThreads);
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    double endTime = get_time_ms();
    
    double elapsedTime = (endTime - startTime) / 1000.0; // 秒
    double threadsPerSec = numThreads / (elapsedTime * 1.0e6); // 每秒百萬執行緒
    
    printf("並行執行時間: %.2f 秒\n", elapsedTime);
    printf("每秒執行的執行緒: %.2f 百萬\n", threadsPerSec);
    
    // 釋放記憶體
    CHECK_CUDA_ERROR(cudaFree(d_data));
    free(h_data);
    
    return threadsPerSec;
}

// 紋理存取測試核函數
__global__ void textureAccessTest(float *output, cudaTextureObject_t texObj, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float u = x / (float)width;
        float v = y / (float)height;
        
        // 執行紋理存取操作
        float4 texValue = tex2D<float4>(texObj, u, v);
        
        // 寫入輸出
        int idx = y * width + x;
        output[idx] = texValue.x + texValue.y + texValue.z + texValue.w;
    }
}

// 測試紋理存取效能
float testTexturePerformance(int width, int height) {
    printf("\n==== 紋理存取測試 ====\n");
    printf("紋理尺寸: %d x %d\n", width, height);
    
    size_t texelSize = 4 * sizeof(float);  // RGBA
    size_t texSize = width * height * texelSize;
    
    // 分配和初始化紋理數據
    float4 *h_texData = (float4*)malloc(texSize);
    for (int i = 0; i < width * height; i++) {
        h_texData[i] = make_float4(
            rand() / (float)RAND_MAX, 
            rand() / (float)RAND_MAX, 
            rand() / (float)RAND_MAX, 
            rand() / (float)RAND_MAX
        );
    }
    
    // 分配設備記憶體並複製紋理數據
    float4 *d_texData;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_texData, texSize));
    CHECK_CUDA_ERROR(cudaMemcpy(d_texData, h_texData, texSize, cudaMemcpyHostToDevice));
    
    // 創建CUDA陣列
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaArray_t cuArray;
    CHECK_CUDA_ERROR(cudaMallocArray(&cuArray, &channelDesc, width, height));
    
    // 使用cudaMemcpy2DToArray替代棄用的cudaMemcpyToArray
    cudaMemcpy2DToArray(cuArray, 0, 0, h_texData, width * texelSize, 
                         width * texelSize, height, cudaMemcpyHostToDevice);
    
    // 設定紋理參數
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.normalizedCoords = 1;
    
    // 創建紋理物件
    cudaTextureObject_t texObj = 0;
    CHECK_CUDA_ERROR(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    
    // 輸出結果緩衝區
    float *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, width * height * sizeof(float)));
    
    // 設定核函數參數
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // 預熱GPU
    textureAccessTest<<<gridSize, blockSize>>>(d_output, texObj, width, height);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // 測量效能
    int iterations = 100;
    double startTime = get_time_ms();
    
    for (int i = 0; i < iterations; i++) {
        textureAccessTest<<<gridSize, blockSize>>>(d_output, texObj, width, height);
    }
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    double endTime = get_time_ms();
    double elapsedTime = (endTime - startTime) / 1000.0; // 秒
    
    // 計算紋理存取速率（每秒百萬紋理獲取）
    double texelAccessRate = (width * height * iterations) / (elapsedTime * 1.0e6);
    printf("紋理存取速率: %.2f 百萬次/秒\n", texelAccessRate);
    
    // 釋放資源
    CHECK_CUDA_ERROR(cudaDestroyTextureObject(texObj));
    CHECK_CUDA_ERROR(cudaFreeArray(cuArray));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_texData));
    free(h_texData);
    
    return texelAccessRate;
}

// 動態平行化測試核函數 (簡化版本)
__global__ void dynamicParallelismTest(int depth, int maxDepth, int *counter) {
    if (depth >= maxDepth) return;
    
    // 原子計數
    atomicAdd(counter, 1);
    
    // 啟動子核函數 (減少啟動的線程數)
    if (depth < maxDepth - 1) {
        dim3 block(4);  // 減少為4個線程
        dim3 grid(2);   // 保持2個區塊
        dynamicParallelismTest<<<grid, block>>>(depth + 1, maxDepth, counter);
    }
}

// 測試動態平行化效能 (需要計算能力 3.5 或更高)
float testDynamicParallelism() {
    printf("\n==== 動態平行化測試 ====\n");
    
    // 檢查計算能力
    cudaDeviceProp prop;
    int device;
    CHECK_CUDA_ERROR(cudaGetDevice(&device));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));
    
    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5)) {
        printf("您的GPU計算能力為 %d.%d，不支援動態平行化 (需要 3.5+)\n", prop.major, prop.minor);
        return 0.0f;
    }
    
    printf("執行動態平行化測試...\n");
    
    // 計數器
    int *d_counter;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_counter, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_counter, 0, sizeof(int)));
    
    // 預熱 (使用小一點的參數)
    dynamicParallelismTest<<<1, 1>>>(0, 2, d_counter);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemset(d_counter, 0, sizeof(int)));
    
    // 執行測試 (降低參數)
    int maxDepth = 3;  // 減少深度
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    dim3 grid(8);      // 減少網格大小
    dim3 block(32);    // 減少區塊大小
    dynamicParallelismTest<<<grid, block>>>(0, maxDepth, d_counter);
    
    // 設置超時檢測
    float timeout = 5.0f; // 5秒超時
    cudaError_t result = cudaEventRecord(stop);
    if (result != cudaSuccess) {
        printf("動態平行化測試出錯，跳過...\n");
        CHECK_CUDA_ERROR(cudaFree(d_counter));
        return 0.0f;
    }
    
    // 等待完成，但設置超時
    unsigned long startTime = (unsigned long)time(NULL);
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        if ((unsigned long)time(NULL) - startTime > timeout) {
            printf("動態平行化測試超時，強制停止...\n");
            // 重置設備
            cudaDeviceReset();
            return 0.0f;
        }
        // 短暫休眠避免忙等待
        usleep(10000); // 10ms
    }
    
    float milliseconds = 0;
    result = cudaEventElapsedTime(&milliseconds, start, stop);
    if (result != cudaSuccess) {
        printf("獲取執行時間失敗，跳過測試...\n");
        CHECK_CUDA_ERROR(cudaFree(d_counter));
        return 0.0f;
    }
    
    double elapsedTime = milliseconds / 1000.0;
    
    // 獲取計數器值
    int counter;
    CHECK_CUDA_ERROR(cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("動態平行化計數: %d\n", counter);
    printf("執行時間: %.4f 秒\n", elapsedTime);
    
    // 計算每秒啟動的核心數 (每秒百萬核心)
    float kernelsPerSec = 0.0f;
    if (elapsedTime > 0) {
        kernelsPerSec = counter / (1.0e6 * elapsedTime);
        printf("核心啟動速率: %.2f 百萬/秒\n", kernelsPerSec);
    } else {
        printf("執行時間太短，無法計算準確速率\n");
    }
    
    // 釋放資源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK_CUDA_ERROR(cudaFree(d_counter));
    
    return kernelsPerSec;
}

// 共享記憶體存取測試核函數
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

// 測試共享記憶體頻寬
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

// 浮點指令吞吐量測試核函數
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

// 測試浮點指令吞吐量
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

// 測試TensorCore性能 (僅適用於Volta架構以上的GPU)
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

// 強負載測試
float stressTest(int durationSec) {
    printf("\n==== 強負載測試 (%d秒) ====\n", durationSec);
    
    int numElements = 10000000; // 1千萬元素
    size_t bytes = numElements * sizeof(float);
    
    // 設備記憶體分配
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, bytes));
    
    // 初始化資料
    float *h_input = (float*)malloc(bytes);
    for (int i = 0; i < numElements; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // 計算網格和區塊大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("開始強負載測試，持續%d秒...\n", durationSec);
    
    double startTime = get_time_ms();
    double currentTime = startTime;
    int iterations = 0;
    
    // 連續運行多個密集型核函數
    while ((currentTime - startTime) < durationSec * 1000) {
        computeIntensiveTest<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements, 100);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // 交換輸入和輸出
        float *temp = d_input;
        d_input = d_output;
        d_output = temp;
        
        iterations++;
        currentTime = get_time_ms();
    }
    
    double elapsedTime = (currentTime - startTime) / 1000.0;
    double iterPerSec = iterations / elapsedTime;
    
    printf("完成強負載測試\n");
    printf("總迭代次數: %d\n", iterations);
    printf("平均每秒迭代次數: %.2f\n", iterPerSec);
    
    // 釋放記憶體
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    free(h_input);
    
    return iterPerSec;
}

// 顯示GPU資訊
void showGPUInfo(int *computeCapability, int *coreCount, float *memoryGB) {
    cudaDeviceProp prop;
    int device;
    
    CHECK_CUDA_ERROR(cudaGetDevice(&device));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));
    
    *computeCapability = prop.major * 10 + prop.minor;
    *coreCount = prop.multiProcessorCount;
    *memoryGB = prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
    
    printf("\n==== GPU資訊 ====\n");
    printf("設備名稱: %s\n", prop.name);
    printf("計算能力: %d.%d\n", prop.major, prop.minor);
    printf("核心數量: %d\n", prop.multiProcessorCount);
    printf("時脈頻率: %d MHz\n", prop.clockRate / 1000);
    printf("記憶體總量: %.2f GB\n", *memoryGB);
    printf("記憶體時脈頻率: %d MHz\n", prop.memoryClockRate / 1000);
    printf("記憶體匯流排寬度: %d bits\n", prop.memoryBusWidth);
    printf("L2快取大小: %d KB\n", prop.l2CacheSize / 1024);
    printf("最大執行緒數/區塊: %d\n", prop.maxThreadsPerBlock);
    printf("最大執行緒維度: [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("最大網格維度: [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("最大共享記憶體/區塊: %lu KB\n", prop.sharedMemPerBlock / 1024);
    printf("暫存器數量/區塊: %d\n", prop.regsPerBlock);
    printf("線程束大小: %d\n", prop.warpSize);
    printf("\n");
}

// 計算顯卡評分 (以RTX 4090為標準的校準版本)
void calculateGPUScore(float memBandwidth, float computePerf, float parallelPerf, float stressPerf, 
                      float texturePerf, float dynamicParallelismPerf, float sharedMemBandwidth, 
                      float flopsThroughput, float tensorCorePerf, int computeCapability, 
                      int coreCount, float memoryGB) {
    printf("\n==== 顯卡效能評分 (以RTX 4090為標準) ====\n");
    
    // 各項測試權重 (相對重要性)
    float memBandwidthWeight = 0.20f;
    float computePerfWeight = 0.15f;
    float parallelPerfWeight = 0.05f;
    float stressPerfWeight = 0.05f;
    float texturePerfWeight = 0.10f;
    float dynamicParallelismWeight = 0.05f;
    float sharedMemBandwidthWeight = 0.10f;
    float flopsThroughputWeight = 0.15f;
    float tensorCoreWeight = 0.15f;
    
    // 標準基準值 (以RTX 4090為基準)
    float memBandwidthBaseline = 1000.0f; // GB/s (RTX 4090約1000-1200GB/s)
    float computePerfBaseline = 80000.0f; // GFLOPS (RTX 4090約83 TFLOPS)
    float parallelPerfBaseline = 300.0f;  // Million threads/sec
    float stressPerfBaseline = 50.0f;     // iterations/sec
    float texturePerfBaseline = 10000.0f; // Million accesses/sec
    float dynamicParallelismBaseline = 10.0f; // Million kernels/sec
    float sharedMemBandwidthBaseline = 40000.0f; // GB/s
    float flopsThroughputBaseline = 80000.0f; // GFLOPS (RTX 4090約83 TFLOPS)
    float tensorCoreBaseline = 330000.0f; // GFLOPS (RTX 4090約330 TFLOPS)
    
    // 計算各項得分 (最高限制為100分)
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
    
    // 計算總評分 (加權平均)
    float weightSum = 0.0f;
    float weightedScore = 0.0f;
    
    // 記憶體頻寬
    weightedScore += memBandwidthWeight * memScore;
    weightSum += memBandwidthWeight;
    
    // 計算效能
    weightedScore += computePerfWeight * computeScore;
    weightSum += computePerfWeight;
    
    // 並行效能
    weightedScore += parallelPerfWeight * parallelScore;
    weightSum += parallelPerfWeight;
    
    // 強負載效能
    weightedScore += stressPerfWeight * stressScore;
    weightSum += stressPerfWeight;
    
    // 紋理存取效能
    weightedScore += texturePerfWeight * textureScore;
    weightSum += texturePerfWeight;
    
    // 動態平行化
    if (dynamicParallelismPerf > 0.0f) {
        weightedScore += dynamicParallelismWeight * dynamicParallelismScore;
        weightSum += dynamicParallelismWeight;
    }
    
    // 共享記憶體頻寬
    weightedScore += sharedMemBandwidthWeight * sharedMemScore;
    weightSum += sharedMemBandwidthWeight;
    
    // 浮點指令吞吐量
    weightedScore += flopsThroughputWeight * flopsScore;
    weightSum += flopsThroughputWeight;
    
    // TensorCore
    if (tensorCorePerf > 0.0f) {
        weightedScore += tensorCoreWeight * tensorCoreScore;
        weightSum += tensorCoreWeight;
    }
    
    // 計算正規化的加權平均分數 - 修正此處的計算
    float totalScore = (weightedScore / weightSum) ; // 不需要乘數
    
    // 硬體評分加成
    float hwBonus = 1.0f;
    if (computeCapability >= 90) hwBonus += 0.25f; // Ada Lovelace (RTX 40系列)
    else if (computeCapability >= 86) hwBonus += 0.20f; // Ampere (RTX 30系列)
    else if (computeCapability >= 75) hwBonus += 0.15f; // Turing (RTX 20系列)
    else if (computeCapability >= 70) hwBonus += 0.10f; // Volta
    else if (computeCapability >= 60) hwBonus += 0.05f; // Pascal
    
    if (coreCount > 100) hwBonus += 0.15f; // 高核心數加成 (RTX 4090: 128 SM)
    else if (coreCount > 70) hwBonus += 0.10f; // 中高核心數加成
    else if (coreCount > 40) hwBonus += 0.05f; // 中核心數加成
    
    if (memoryGB > 20.0f) hwBonus += 0.15f; // 超大記憶體加成 (RTX 4090: 24GB)
    else if (memoryGB > 12.0f) hwBonus += 0.10f; // 大記憶體加成
    else if (memoryGB > 8.0f) hwBonus += 0.05f; // 中大記憶體加成
    
    totalScore *= hwBonus;
    
    // 限制最終分數不超過100
    totalScore = fminf(100.0f, totalScore);
    
    // 顯示各項校準後得分
    printf("記憶體頻寬得分: %.1f / 100 (%.2f GB/s)\n", memScore, memBandwidth);
    printf("計算效能得分: %.1f / 100 (%.2f GFLOPS)\n", computeScore, computePerf);
    printf("並行效能得分: %.1f / 100 (%.2f M threads/s)\n", parallelScore, parallelPerf);
    printf("強負載效能得分: %.1f / 100 (%.2f iter/s)\n", stressScore, stressPerf);
    printf("紋理存取效能得分: %.1f / 100 (%.2f M accesses/s)\n", textureScore, texturePerf);
    
    if (dynamicParallelismPerf > 0.0f) {
        printf("動態平行化得分: %.1f / 100 (%.2f M kernels/s)\n", dynamicParallelismScore, dynamicParallelismPerf);
    } else {
        printf("動態平行化得分: 不支援\n");
    }
    
    printf("共享記憶體頻寬得分: %.1f / 100 (%.2f GB/s)\n", sharedMemScore, sharedMemBandwidth);
    printf("浮點指令吞吐量得分: %.1f / 100 (%.2f GFLOPS)\n", flopsScore, flopsThroughput);
    
    if (tensorCorePerf > 0.0f) {
        printf("TensorCore得分: %.1f / 100 (%.2f GFLOPS)\n", tensorCoreScore, tensorCorePerf);
    } else {
        printf("TensorCore得分: 不支援\n");
    }
    printf("硬體規格加成: %.2fx\n", hwBonus);
    printf("\n");
    
    // 總評分與等級 (經過校準的評級)
    printf("總評分: %.1f\n", totalScore);
    
    // 評級 (校準後)
    printf("效能等級: ");
    if (totalScore >= 90.0f) {
        printf("S++ 頂尖旗艦 (超高階專業/AI研究用途)\n");
    } else if (totalScore >= 80.0f) {
        printf("S+ 頂級旗艦 (高階專業用途)\n");
    } else if (totalScore >= 70.0f) {
        printf("S 高階 (高階遊戲/專業用途)\n");
    } else if (totalScore >= 60.0f) {
        printf("A+ 優質 (中高階遊戲/專業用途)\n");
    } else if (totalScore >= 50.0f) {
        printf("A 良好 (中階遊戲/一般專業用途)\n");
    } else if (totalScore >= 40.0f) {
        printf("B+ 中等 (入門遊戲/一般應用)\n");
    } else if (totalScore >= 30.0f) {
        printf("B 基本 (輕度遊戲/一般應用)\n");
    } else {
        printf("C 入門 (一般應用/基本顯示)\n");
    }
    
    // 用途建議 (根據評級)
    printf("\n適用場景建議: \n");
    if (totalScore >= 90.0f) {
        printf("✓ 頂尖AI研究與大規模深度學習訓練\n");
        printf("✓ 8K/16K影片編輯與實時渲染\n");
        printf("✓ 高精度科學模擬與大數據分析\n");
        printf("✓ 光線追蹤與高階路徑追蹤渲染\n");
        printf("✓ 4K/8K HDR遊戲與VR/AR開發\n");
        printf("✓ 多顯示器8K+環境\n");
    } else if (totalScore >= 80.0f) {
        printf("✓ 專業深度學習訓練與研究\n");
        printf("✓ 4K/8K視訊編輯與動畫渲染\n");
        printf("✓ 大型科學模擬與計算\n");
        printf("✓ 高品質光線追蹤渲染\n");
        printf("✓ 4K高幀率/8K遊戲體驗\n");
    } else if (totalScore >= 70.0f) {
        printf("✓ 中小規模深度學習訓練\n");
        printf("✓ 4K視訊編輯與專業3D建模\n");
        printf("✓ 中等規模科學計算\n");
        printf("✓ 1440p/4K高幀率遊戲體驗\n");
        printf("✓ 光線追蹤效果遊戲\n");
    } else if (totalScore >= 60.0f) {
        printf("✓ 深度學習推論與小型模型訓練\n");
        printf("✓ 1080p/4K視訊編輯\n");
        printf("✓ 3D建模與渲染\n");
        printf("✓ 1080p高幀率/1440p遊戲\n");
        printf("✓ 中等複雜度科學計算\n");
    } else if (totalScore >= 50.0f) {
        printf("✓ AI模型推論與資料分析\n");
        printf("✓ 1080p視訊編輯與圖像處理\n");
        printf("✓ 1080p中高畫質遊戲\n");
        printf("✓ 基礎3D建模與渲染\n");
        printf("✓ 一般科學計算\n");
    } else if (totalScore >= 40.0f) {
        printf("✓ 輕量AI推論\n");
        printf("✓ 1080p基本視訊編輯\n");
        printf("✓ 1080p中等畫質遊戲\n");
        printf("✓ 基礎圖像處理\n");
    } else if (totalScore >= 30.0f) {
        printf("✓ 基本圖像處理\n");
        printf("✓ 720p/1080p低畫質遊戲\n");
        printf("✓ 一般辦公與多媒體應用\n");
    } else {
        printf("✓ 基本辦公應用\n");
        printf("✓ 網頁瀏覽與媒體播放\n");
        printf("✓ 簡單2D圖形處理\n");
        printf("✓ 輕度遊戲\n");
    }
}


// 向用戶顯示測試說明
void showTestDescription() {
    printf("====================================================\n");
    printf("           CUDA GPU 效能測試程式 (校準版)           \n");
    printf("====================================================\n");
    printf("此程式將進行以下測試:\n");
    printf("1. 記憶體頻寬測試 - 測量GPU的記憶體讀寫速度\n");
    printf("2. 計算效能測試 - 測量GPU的浮點運算能力\n");
    printf("3. 並行效能測試 - 測量GPU處理大量並行任務的能力\n");
    printf("4. 紋理存取測試 - 測量GPU的紋理存取性能\n");
    printf("5. 共享記憶體頻寬測試 - 測量共享記憶體存取速度\n");
    printf("6. 浮點指令吞吐量測試 - 測量原始浮點運算能力\n");
    printf("7. TensorCore測試 - 測量神經網路加速引擎性能 (僅支援計算能力 ≥ 7.0)\n");
    printf("8. 動態平行化測試 - 測量核函數啟動其他核函數的能力 (僅支援計算能力 ≥ 3.5)\n");
    printf("9. 強負載測試 - 持續60秒的高強度測試\n\n");
    printf("測試完成後，將根據您的GPU性能給出綜合評分和適用場景建議。\n");
    printf("注意：測試過程中GPU將滿負荷運行，可能會導致溫度升高，請確保散熱良好。\n");
    printf("====================================================\n\n");
    printf("按Enter鍵開始測試...");
    getchar();
    printf("\n");
}

// 主函數
int main() {
    // 顯示測試說明
    showTestDescription();
    
    int computeCapability = 0;
    int coreCount = 0;
    float memoryGB = 0.0f;
    
    // 顯示GPU資訊
    showGPUInfo(&computeCapability, &coreCount, &memoryGB);
    
    // 測試記憶體頻寬 (512 MB)
    float memBandwidth = testMemoryBandwidth(512 * 1024 * 1024);
    
    // 測試計算效能
    float computePerf = testComputePerformance(5000000, 1000);
    
    // 測試並行效能
    float parallelPerf = testParallelism(10000000);
    
    // 測試紋理存取效能
    float texturePerf = testTexturePerformance(2048, 2048);
    
    // 測試共享記憶體頻寬
    float sharedMemBandwidth = testSharedMemoryBandwidth();
    
    // 測試浮點指令吞吐量
    float flopsThroughput = testFLOPSThroughput();
    
    // 測試TensorCore性能 (如果支援)
    float tensorCorePerf = testTensorCore();
    
    // 測試動態平行化 (如果支援)
    float dynamicParallelismPerf = testDynamicParallelism();
    
    // 強負載測試60秒
    float stressPerf = stressTest(60);
    
    // 計算顯卡評分 (使用所有測試結果)
    calculateGPUScore(memBandwidth, computePerf, parallelPerf, stressPerf, 
                      texturePerf, dynamicParallelismPerf, sharedMemBandwidth, 
                      flopsThroughput, tensorCorePerf, computeCapability, 
                      coreCount, memoryGB);
    
    printf("\n所有測試完成！\n");
    
    return 0;
}
