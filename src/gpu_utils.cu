#include "gpu_utils.h"
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
