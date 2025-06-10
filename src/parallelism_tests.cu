#include "parallelism_tests.h"
__global__ void parallelismTest(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];

        // 大量原子操作以測試並行效能
        atomicAdd(&data[0], val * 0.000001f);
    }
}

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
