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
	// LINE DATA
	float *lineData;				// Line data (4 floats per line)
	// LINE COVERAGE DATA
	float *lineCoverage;			// Precalculated coverage of pixels vs distance/angle
	size_t pitchCoverage;			// Pitch in bytes
	// ERROR CALCULATION
	double *partialSums;			// Block sums of error
	double *sumResult;				// Total error
	// TEXTURES
	cudaTextureObject_t texImageIn;  // Texture for input image
	cudaTextureObject_t texWeights;  // Texture for image weights
	// PITCH FOR IMAGE BUFFERS
	// Note: pitch is in bytes
	size_t pitchIn;
	size_t pitchWeight;
	size_t pitchAccum;
	size_t pitchOutput;
	// IMAGE BUFFERS
	uint8_t  *imgIn;		// Input image data
	uint8_t  *imgWeight;	// Input weights
	float    *imgAccum;		// Accumulated data
	uint8_t  *imgOut;		// Output image
} gpuData_t;

// Fix to support both C and C++ compilers
#ifdef __cplusplus
extern "C" {
#endif

int GpuInitBuffers(gpuData_t *gpuData, int widthIn, int heightIn);
void GpuFreeBuffers(gpuData_t *gpuData);
void GpuLoadLines(gpuData_t *gpuData, line_t *lines);
void GpuDrawLines(gpuData_t *gpuData);
double GpucalculateImageError(gpuData_t *gpuData);
void GpuOutConvert(uint8_t *hostDst, gpuData_t *gpuData);

#ifdef __cplusplus
}
#endif

#endif

