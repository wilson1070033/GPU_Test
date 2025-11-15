# GPU Benchmark

A comprehensive CUDA GPU benchmark suite for measuring GPU performance across multiple dimensions.

## Features

### Comprehensive Testing Suite
- **Memory Bandwidth Test** - Measures GPU memory read/write speed with optimized vector loads
- **Compute Performance Test** - Measures floating-point computation capability
- **Parallelism Test** - Measures GPU's ability to handle massive parallel tasks
- **Texture Access Test** - Measures texture memory access performance
- **Shared Memory Bandwidth Test** - Measures shared memory access speed
- **Floating-Point Throughput Test** - Measures raw floating-point computation capability
- **TensorCore Test** - Measures neural network acceleration engine performance (Compute Capability ≥ 7.0)
- **Dynamic Parallelism Test** - Measures kernel launching capability (Compute Capability ≥ 3.5)
- **Stress Test** - Continuous high-intensity testing

### Modular Architecture
- Clean separation of concerns with header/source organization
- Easy to extend with new benchmark tests
- Namespace isolation for better code organization

### Advanced Scoring System
- Calibrated against RTX 4090 as baseline (100 points)
- Weighted scoring across all test dimensions
- Hardware-specific bonus system
- Performance tier classification (S++, S+, S, A+, A, B+, B, C)

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- GCC/G++ compiler
- Make build system

### Verify CUDA Installation

```bash
nvcc --version
```

## Building

### Quick Start (Recommended)

Build the modular version using Make:

```bash
make
```

This will create the `gpu_benchmark` executable with all optimizations enabled.

### Build Options

```bash
# Build modular version (recommended)
make

# Build legacy single-file versions
make legacy

# Clean build files
make clean

# Clean and rebuild
make rebuild

# Show help
make help
```

### Manual Compilation

If you prefer to compile manually:

```bash
# Create build directory
mkdir -p build

# Compile object files
nvcc -O3 -std=c++11 -arch=sm_50 -rdc=true -Iinclude -dc -o build/main.o src/main.cu
nvcc -O3 -std=c++11 -arch=sm_50 -rdc=true -Iinclude -dc -o build/compute_perf.o src/compute_perf.cu
nvcc -O3 -std=c++11 -arch=sm_50 -rdc=true -Iinclude -dc -o build/memory_bandwidth.o src/memory_bandwidth.cu
nvcc -O3 -std=c++11 -arch=sm_50 -rdc=true -Iinclude -dc -o build/parallelism.o src/parallelism.cu
nvcc -O3 -std=c++11 -arch=sm_50 -rdc=true -Iinclude -dc -o build/texture_perf.o src/texture_perf.cu
nvcc -O3 -std=c++11 -arch=sm_50 -rdc=true -Iinclude -dc -o build/stress_test.o src/stress_test.cu
nvcc -O3 -std=c++11 -arch=sm_50 -rdc=true -Iinclude -dc -o build/shared_memory.o src/shared_memory.cu
nvcc -O3 -std=c++11 -arch=sm_50 -rdc=true -Iinclude -dc -o build/flops.o src/flops.cu
nvcc -O3 -std=c++11 -arch=sm_50 -rdc=true -Iinclude -dc -o build/tensor_core.o src/tensor_core.cu
nvcc -O3 -std=c++11 -arch=sm_50 -rdc=true -Iinclude -dc -o build/dynamic_parallelism.o src/dynamic_parallelism.cu

# Link executable
nvcc -O3 -std=c++11 -arch=sm_50 -rdc=true -Iinclude -o gpu_benchmark build/*.o -lm -lcublas
```

### Legacy Versions

For the original single-file versions:

```bash
# Basic version
nvcc -o gpu_benchmark_basic gpu_benchmark.cu -lm

# Enhanced version with fixed scoring
nvcc -o gpu_benchmark_enhanced gpu_benchmark_fix_score.cu -rdc=true -lm -lcublas
```

## Running

```bash
./gpu_benchmark
```

**Note:** The benchmark will run the GPU at full load. Ensure adequate cooling is available.

## Project Structure

```
GPU_Test/
├── Makefile                    # Build system
├── README.md                   # This file
├── include/                    # Header files
│   └── benchmark/
│       ├── benchmark.h         # Main header
│       ├── common.h            # Common utilities
│       ├── compute_perf.h      # Compute performance test
│       ├── memory_bandwidth.h  # Memory bandwidth test
│       ├── parallelism.h       # Parallelism test
│       ├── texture_perf.h      # Texture performance test
│       ├── stress_test.h       # Stress test
│       ├── shared_memory.h     # Shared memory test
│       ├── flops.h             # FP throughput test
│       ├── tensor_core.h       # TensorCore test
│       └── dynamic_parallelism.h # Dynamic parallelism test
├── src/                        # Source files
│   ├── main.cu                 # Main program
│   ├── compute_perf.cu         # Compute performance implementation
│   ├── memory_bandwidth.cu     # Memory bandwidth implementation
│   ├── parallelism.cu          # Parallelism implementation
│   ├── texture_perf.cu         # Texture performance implementation
│   ├── stress_test.cu          # Stress test implementation
│   ├── shared_memory.cu        # Shared memory implementation
│   ├── flops.cu                # FP throughput implementation
│   ├── tensor_core.cu          # TensorCore implementation
│   └── dynamic_parallelism.cu  # Dynamic parallelism implementation
├── gpu_benchmark.cu            # Legacy basic version
└── gpu_benchmark_fix_score.cu  # Legacy enhanced version
```

## Understanding the Results

### Test Metrics

- **Memory Bandwidth** - Measured in GB/s, indicates data transfer speed
- **Compute Performance** - Measured in GFLOPS, indicates general compute capability
- **Parallel Performance** - Million threads per second
- **Texture Performance** - Million texture accesses per second
- **Shared Memory Bandwidth** - GB/s for on-chip memory
- **FP Throughput** - TFLOPS for raw floating-point operations
- **TensorCore** - GFLOPS for matrix multiplication (AI workloads)

### Performance Tiers

- **S++** (90-100): Top flagship GPUs for ultra high-end professional/AI research
- **S+** (80-90): Top flagship GPUs for high-end professional work
- **S** (70-80): High-end GPUs for gaming and professional use
- **A+** (60-70): Premium GPUs for mid-high gaming/professional
- **A** (50-60): Good GPUs for mid-range gaming/general professional
- **B+** (40-50): Medium GPUs for entry gaming/general use
- **B** (30-40): Basic GPUs for light gaming/general use
- **C** (0-30): Entry GPUs for general/basic display

### Baseline

All scores are calibrated against NVIDIA RTX 4090 as the baseline (100 points per test).

## Performance Optimizations

The modular version includes several optimizations:

1. **Vector Memory Access** - Uses `float4` vector loads for better memory coalescing
2. **Optimized Grid Sizes** - Tuned thread block sizes for maximum occupancy
3. **Separate Compilation** - Enables better optimization and faster incremental builds
4. **Hardware-Specific Paths** - Tests adapt to GPU compute capability

## Adding New Tests

To add a new benchmark test:

1. Create header file in `include/benchmark/your_test.h`
2. Create implementation in `src/your_test.cu`
3. Add to `include/benchmark/benchmark.h`
4. Add to `Makefile` SOURCES list
5. Call from `src/main.cu`

Example header:
```cpp
#pragma once
namespace gpu_benchmark {
    float testYourFeature();
}
```

Example implementation:
```cpp
#include "benchmark/your_test.h"
#include "benchmark/common.h"

namespace gpu_benchmark {
    float testYourFeature() {
        // Your test implementation
        return result;
    }
}
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source. See LICENSE file for details.
