#ifndef __GPUUTIL_H__
#define __GPUUTIL_H__

//#include <stdint.h>

#include <cuda_runtime.h>

#include "gpu_funcs.h"

// Macros to simplify error checking
#define CUDA_CHECK(val) cudaCheck((val), #val, __FILE__, __LINE__)
#define CUDA_LAST_ERROR() cudaLastError(__FILE__, __LINE__)
#define NPP_CHECK(val) nppCheck((val), #val, __FILE__, __LINE__)

// Fix to support both C and C++ compilers
#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cudaCheck(cudaError_t err, const char* const func, const char* const file, const int line);
cudaError_t cudaLastError(const char* const file, const int line);

int  GpuInit(void);
void GpuPinMemory(void *ptr, size_t size);
void GpuUnPinMemory(void *ptr);
int InitPinnedBuffers(int width, int height);
void FreePinnedBuffers(void);
void GpuSync(void);

#ifdef __cplusplus
}
#endif

#endif

