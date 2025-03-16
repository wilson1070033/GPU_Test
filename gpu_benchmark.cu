#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// 錯誤檢查巨集
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA 錯誤在 %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
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

// 計算顯卡評分
void calculateGPUScore(float memBandwidth, float computePerf, float parallelPerf, float stressPerf, 
                       int computeCapability, int coreCount, float memoryGB) {
    printf("\n==== 顯卡效能評分 ====\n");
    
    // 基礎評分分數
    float baseScore = 100.0f;
    
    // 各項測試權重
    float memBandwidthWeight = 0.25f;
    float computePerfWeight = 0.30f;
    float parallelPerfWeight = 0.20f;
    float stressPerfWeight = 0.25f;
    
    // 標準基準值（以中高階顯卡為標準）
    float memBandwidthBaseline = 300.0f; // GB/s
    float computePerfBaseline = 10000.0f; // GFLOPS
    float parallelPerfBaseline = 5.0f; // Million threads/sec
    float stressPerfBaseline = 5.0f; // iterations/sec
    
    // 計算各項得分
    float memScore = (memBandwidth / memBandwidthBaseline) * 100.0f;
    float computeScore = (computePerf / computePerfBaseline) * 100.0f;
    float parallelScore = (parallelPerf / parallelPerfBaseline) * 100.0f;
    float stressScore = (stressPerf / stressPerfBaseline) * 100.0f;
    
    // 計算總評分
    float totalScore = baseScore * (
        memBandwidthWeight * (memBandwidth / memBandwidthBaseline) +
        computePerfWeight * (computePerf / computePerfBaseline) +
        parallelPerfWeight * (parallelPerf / parallelPerfBaseline) +
        stressPerfWeight * (stressPerf / stressPerfBaseline)
    );
    
    // 硬體評分加成
    float hwBonus = 1.0f;
    if (computeCapability >= 70) hwBonus += 0.15f; // 針對較新的架構給予獎勵
    if (coreCount > 30) hwBonus += 0.1f; // 多核心加成
    if (memoryGB > 8.0f) hwBonus += 0.1f; // 大記憶體加成
    
    totalScore *= hwBonus;
    
    // 顯示各項得分
    printf("記憶體頻寬得分: %.1f / 100\n", memScore);
    printf("計算效能得分: %.1f / 100\n", computeScore);
    printf("並行效能得分: %.1f / 100\n", parallelScore);
    printf("強負載效能得分: %.1f / 100\n", stressScore);
    printf("硬體規格加成: %.2fx\n", hwBonus);
    printf("\n");
    
    // 總評分與等級
    printf("總評分: %.1f\n", totalScore);
    
    // 評級
    printf("效能等級: ");
    if (totalScore >= 300.0f) {
        printf("S+ 頂級旗艦 (超高階專業用途)\n");
    } else if (totalScore >= 250.0f) {
        printf("S 頂級 (高階遊戲/專業用途)\n");
    } else if (totalScore >= 200.0f) {
        printf("A+ 優秀 (中高階遊戲/專業用途)\n");
    } else if (totalScore >= 150.0f) {
        printf("A 良好 (中階遊戲/一般專業用途)\n");
    } else if (totalScore >= 100.0f) {
        printf("B+ 中等 (入門遊戲/一般應用)\n");
    } else if (totalScore >= 70.0f) {
        printf("B 基本 (輕度遊戲/一般應用)\n");
    } else if (totalScore >= 40.0f) {
        printf("C 普通 (一般應用/輕量工作)\n");
    } else {
        printf("D 較弱 (基礎顯示/簡單應用)\n");
    }
    
    // 用途建議
    printf("\n適用場景建議: \n");
    if (totalScore >= 250.0f) {
        printf("✓ 4K/8K 視訊編輯\n");
        printf("✓ 高階深度學習與AI訓練\n");
        printf("✓ 複雜3D渲染與動畫製作\n");
        printf("✓ 高效能科學計算\n");
        printf("✓ 4K高幀率遊戲體驗\n");
        printf("✓ 多螢幕設置\n");
    } else if (totalScore >= 150.0f) {
        printf("✓ 1080p/4K 視訊編輯\n");
        printf("✓ 中階深度學習與AI推論\n");
        printf("✓ 3D建模與渲染\n");
        printf("✓ 1080p/1440p高幀率遊戲\n");
        printf("✓ 一般科學計算\n");
    } else if (totalScore >= 70.0f) {
        printf("✓ 1080p視訊編輯\n");
        printf("✓ 簡單AI模型推論\n");
        printf("✓ 1080p中等設定遊戲\n");
        printf("✓ 基礎3D建模\n");
    } else {
        printf("✓ 基本辦公應用\n");
        printf("✓ 網頁瀏覽與多媒體播放\n");
        printf("✓ 簡單2D圖形處理\n");
        printf("✓ 輕度遊戲\n");
    }
}

// 主函數
int main() {
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
    
    // 強負載測試20秒
    float stressPerf = stressTest(20);
    
    // 計算顯卡評分
    calculateGPUScore(memBandwidth, computePerf, parallelPerf, stressPerf, 
                       computeCapability, coreCount, memoryGB);
    
    printf("\n所有測試完成！\n");
    
    return 0;
}
