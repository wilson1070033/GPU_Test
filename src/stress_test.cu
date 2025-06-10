#include "stress_test.h"
#include "compute_tests.h" // For computeIntensiveTest kernel declaration

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
