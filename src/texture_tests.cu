#include "texture_tests.h"
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
