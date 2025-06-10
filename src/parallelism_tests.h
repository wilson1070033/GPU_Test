#ifndef PARALLELISM_TESTS_H
#define PARALLELISM_TESTS_H

#include "benchmark_common.h"

// Test GPU parallelism (atomic operations)
float testParallelism(int numThreads);

// Test dynamic parallelism performance
float testDynamicParallelism();

#endif // PARALLELISM_TESTS_H
