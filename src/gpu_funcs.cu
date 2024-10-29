#include <stdio.h>
//#include <stdint.h>
//#include <float.h>

// CUDA Runtime
#include <cuda_runtime.h>

#include "gpu_funcs.h"
#include "gpu_util.h"
#include "settings.h"

// GPU buffers
extern gpuData_t gpuData;

// Initialise gpuImgData_t structure and allocate buffers
int GpuInitBuffers(gpuData_t *gpuData) {
	printf("GpuInitBuffers\n");

	// Clear the structure in case of errors part way through initialisation
	memset(gpuData, 0, sizeof(gpuData_t));

	gpuData->srcWidth  = IMG_WIDTH;
	gpuData->srcHeight = IMG_HEIGHT;
	gpuData->dstSize = DATA_SIZE;

	// Create stream for processing
	CUDA_CHECK(cudaStreamCreate(&gpuData->stream));

	// Global memory on GPU
	CUDA_CHECK(cudaMallocPitch(&gpuData->imgInOrig, &gpuData->pitchInOrig,
							gpuData->srcWidth*sizeof(uint8_t), gpuData->srcHeight));
	CUDA_CHECK(cudaMallocPitch(&gpuData->imgInFloat, &gpuData->pitchInFloat,
							gpuData->srcWidth*sizeof(float), gpuData->srcHeight));
	CUDA_CHECK(cudaMallocPitch(&gpuData->imgOut, &gpuData->pitchOutput,
							gpuData->dstSize*sizeof(float), gpuData->dstSize));

	printf("pitch inOrig:  %5lu (%lu) at %p\n", gpuData->pitchInOrig, 	gpuData->srcWidth*sizeof(uint8_t), gpuData->imgInOrig);
	printf("pitch inFloat: %5lu (%lu) at %p\n", gpuData->pitchInFloat, 	gpuData->srcWidth*sizeof(float), gpuData->imgInFloat);
	printf("pitch output:  %5lu (%lu) at %p\n", gpuData->pitchOutput, 	gpuData->srcWidth*sizeof(float), gpuData->imgOut);

	CUDA_CHECK(cudaDeviceSynchronize());

	return CUDA_LAST_ERROR();
}

// Cleanup gpuImgData_t structure and free buffers in GPU memory
void GpuFreeBuffers(gpuData_t *gpuData) {
	printf("GpuFreeBuffers\n");

	CUDA_CHECK(cudaStreamDestroy(gpuData->stream));

	// Free GPU memory
	if (gpuData->imgInOrig != NULL) 	CUDA_CHECK(cudaFree(gpuData->imgInOrig));
	if (gpuData->imgInFloat != NULL) 	CUDA_CHECK(cudaFree(gpuData->imgInFloat));
	if (gpuData->imgOut != NULL) 		CUDA_CHECK(cudaFree(gpuData->imgOut));
}
