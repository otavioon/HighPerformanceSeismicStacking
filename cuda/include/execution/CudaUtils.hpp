#pragma once

#include <cuda.h>

#define CUDA_ASSERT(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0);

void cudaAssert(cudaError_t errorCode, const char *file, int line);
