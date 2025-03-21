# GPU Benchmark 編譯與執行指南

本文檔說明如何編譯與執行 `gpu_benchmark.cu` 程式。

## 先確認是否支援Cuda

```bash
nvcc --version
```

## 編譯程式

使用以下指令來編譯程式：

```bash
nvcc -o gpu_benchmark gpu_benchmark.cu -lm
```

```bash
nvcc -o gpu_benchmark gpu_benchmark_fix_score.cu -rdc=true -lm
```

## 執行程式

```bash
./gpu_benchmark
```
