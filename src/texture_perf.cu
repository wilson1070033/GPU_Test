#include "benchmark/texture_perf.h"
#include "benchmark/common.h"
#include <string.h>

namespace gpu_benchmark {

__global__ void textureAccessKernel(float* output, cudaTextureObject_t texObj, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float u = x / (float)width;
        float v = y / (float)height;
        float4 texValue = tex2D<float4>(texObj, u, v);
        int idx = y * width + x;
        output[idx] = texValue.x + texValue.y + texValue.z + texValue.w;
    }
}

float testTexturePerformance(int width, int height) {
    printf("\n==== Texture Test ====\n");
    printf("Texture size: %d x %d\n", width, height);

    size_t texelSize = 4 * sizeof(float);
    size_t texSize = width * height * texelSize;

    float4* h_texData = (float4*)malloc(texSize);
    for (int i = 0; i < width * height; ++i) {
        h_texData[i] = make_float4(
            rand() / (float)RAND_MAX,
            rand() / (float)RAND_MAX,
            rand() / (float)RAND_MAX,
            rand() / (float)RAND_MAX);
    }

    float4* d_texData;
    GPU_CHECK(cudaMalloc((void**)&d_texData, texSize));
    GPU_CHECK(cudaMemcpy(d_texData, h_texData, texSize, cudaMemcpyHostToDevice));

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaArray_t cuArray;
    GPU_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));
    cudaMemcpy2DToArray(cuArray, 0, 0, h_texData, width * texelSize,
                         width * texelSize, height, cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    GPU_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    float* d_output;
    GPU_CHECK(cudaMalloc((void**)&d_output, width * height * sizeof(float)));

    dim3 block(16,16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    textureAccessKernel<<<grid, block>>>(d_output, texObj, width, height);
    GPU_CHECK(cudaDeviceSynchronize());

    int iterations = 100;
    double start = get_time_ms();
    for (int i = 0; i < iterations; ++i) {
        textureAccessKernel<<<grid, block>>>(d_output, texObj, width, height);
    }
    GPU_CHECK(cudaDeviceSynchronize());
    double end = get_time_ms();

    double elapsed = (end - start) / 1000.0;
    double texelRate = (width * height * iterations) / (elapsed * 1.0e6);
    printf("Texture access rate: %.2f M/s\n", texelRate);

    GPU_CHECK(cudaDestroyTextureObject(texObj));
    GPU_CHECK(cudaFreeArray(cuArray));
    GPU_CHECK(cudaFree(d_output));
    GPU_CHECK(cudaFree(d_texData));
    free(h_texData);

    return texelRate;
}

} // namespace gpu_benchmark
