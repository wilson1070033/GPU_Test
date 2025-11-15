# GPU Benchmark Makefile
# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -std=c++11 -arch=sm_50 -rdc=true
NVCC_FLAGS += -Iinclude

# Libraries
LIBS = -lm -lcublas

# Source files
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

# Source files
SOURCES = $(SRC_DIR)/main.cu \
          $(SRC_DIR)/compute_perf.cu \
          $(SRC_DIR)/memory_bandwidth.cu \
          $(SRC_DIR)/parallelism.cu \
          $(SRC_DIR)/texture_perf.cu \
          $(SRC_DIR)/stress_test.cu \
          $(SRC_DIR)/shared_memory.cu \
          $(SRC_DIR)/flops.cu \
          $(SRC_DIR)/tensor_core.cu \
          $(SRC_DIR)/dynamic_parallelism.cu

# Object files
OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SOURCES))

# Target executable
TARGET = gpu_benchmark

# Default target
all: $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $<

# Link executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

# Legacy single-file versions
legacy: gpu_benchmark_legacy gpu_benchmark_fix_score

gpu_benchmark_legacy: gpu_benchmark.cu
	$(NVCC) -O3 -o $@ $< -lm

gpu_benchmark_fix_score: gpu_benchmark_fix_score.cu
	$(NVCC) -O3 -rdc=true -o $@ $< -lm -lcublas

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(TARGET) gpu_benchmark_legacy gpu_benchmark_fix_score

# Clean and rebuild
rebuild: clean all

# Install (optional)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

# Help
help:
	@echo "GPU Benchmark Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the modular gpu_benchmark (default)"
	@echo "  legacy    - Build legacy single-file versions"
	@echo "  clean     - Remove build files"
	@echo "  rebuild   - Clean and rebuild"
	@echo "  install   - Install to /usr/local/bin"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make              # Build modular version"
	@echo "  make legacy       # Build legacy versions"
	@echo "  make clean        # Clean build files"

.PHONY: all legacy clean rebuild install help
