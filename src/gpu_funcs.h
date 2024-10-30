#ifndef __GPUFUNCS_H__
#define __GPUFUNCS_H__

#include <stdint.h>
#include <cuda_runtime.h>

#include "util.h"

// Structure to hold buffers and data for processing of one camera
typedef struct {
	int srcWidth;					// Source image data width
	int srcHeight;					// Source image data height
	int dstSize;					// Output image size (square)
	cudaStream_t stream;			// CUDA stream for processing
	float *lineData;				// Line data (4 floats per line)
	// PITCH FOR IMAGE BUFFERS
	// Note: pitch is in bytes
	size_t pitchInOrig;
	size_t pitchInFloat;
	size_t pitchAccum;
	size_t pitchOutput;
	// IMAGE BUFFERS
	uint8_t  *imgInOrig;	// Input image data
	float    *imgInFloat;	// Input image data as float
	float    *imgAccum;		// Accumulated data
	uint8_t  *imgOut;		// Output image
} gpuData_t;

// Fix to support both C and C++ compilers
#ifdef __cplusplus
extern "C" {
#endif

int  GpuInitBuffers(gpuData_t *gpuData);
void GpuFreeBuffers(gpuData_t *gpuData);
void GpuLoadLines(gpuData_t *gpuData, line_t *lines);
void GpuDrawLines(gpuData_t *gpuData);
void GpuOutConvert(uint8_t *hostDst, gpuData_t *gpuData);

#ifdef __cplusplus
}
#endif

#endif

