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

	CUDA_CHECK(cudaMalloc(&gpuData->lineData, NUM_LINES*4*sizeof(float)));

	// Global memory on GPU
	CUDA_CHECK(cudaMallocPitch(&gpuData->imgInOrig, &gpuData->pitchInOrig,
							gpuData->srcWidth*sizeof(uint8_t), gpuData->srcHeight));
	CUDA_CHECK(cudaMallocPitch(&gpuData->imgInFloat, &gpuData->pitchInFloat,
							gpuData->srcWidth*sizeof(float), gpuData->srcHeight));
	CUDA_CHECK(cudaMallocPitch(&gpuData->imgAccum, &gpuData->pitchAccum,
							gpuData->dstSize*sizeof(float), gpuData->dstSize));
	CUDA_CHECK(cudaMallocPitch(&gpuData->imgOut, &gpuData->pitchOutput,
							gpuData->dstSize*sizeof(uint8_t), gpuData->dstSize));

	printf("pitch inOrig:  %5lu (%lu) at %p\n", gpuData->pitchInOrig, 	gpuData->srcWidth*sizeof(uint8_t), gpuData->imgInOrig);
	printf("pitch inFloat: %5lu (%lu) at %p\n", gpuData->pitchInFloat, 	gpuData->srcWidth*sizeof(float), gpuData->imgInFloat);
	printf("pitch Accum:   %5lu (%lu) at %p\n", gpuData->pitchAccum, 	gpuData->srcWidth*sizeof(float), gpuData->imgAccum);
	printf("pitch output:  %5lu (%lu) at %p\n", gpuData->pitchOutput, 	gpuData->srcWidth*sizeof(uint8_t), gpuData->imgOut);

	CUDA_CHECK(cudaDeviceSynchronize());

	return CUDA_LAST_ERROR();
}

// Cleanup gpuImgData_t structure and free buffers in GPU memory
void GpuFreeBuffers(gpuData_t *gpuData) {
	printf("GpuFreeBuffers\n");

	CUDA_CHECK(cudaStreamDestroy(gpuData->stream));

	// Free GPU memory
	if (gpuData->lineData != NULL) 		CUDA_CHECK(cudaFree(gpuData->lineData));
	if (gpuData->imgInOrig != NULL) 	CUDA_CHECK(cudaFree(gpuData->imgInOrig));
	if (gpuData->imgInFloat != NULL) 	CUDA_CHECK(cudaFree(gpuData->imgInFloat));
	if (gpuData->imgAccum != NULL) 		CUDA_CHECK(cudaFree(gpuData->imgAccum));
	if (gpuData->imgOut != NULL) 		CUDA_CHECK(cudaFree(gpuData->imgOut));
}


// Copy line data to GPU
void GpuLoadLines(gpuData_t *gpuData, line_t *lines) {

	CUDA_CHECK(cudaMemcpy(gpuData->lineData, lines, NUM_LINES*4*sizeof(float), cudaMemcpyHostToDevice));
	//CUDA_CHECK(cudaDeviceSynchronize());
}

// Compute the perpendicular distance from the pixel (x0, y0) to the line Ax + By + C = 0
__device__ float compute_distance(float x0, float y0, float A, float B, float C, float inv_denom) {
	float numerator = fabsf(fmaf(A, x0, fmaf(B, y0, C)));  // |Ax0 + By0 + C| using fmaf for fused multiply-add
	return numerator * inv_denom;  // Multiply instead of divide
}

// (GPU) Draw a line
__global__
void DrawLine_kernel(float *dataDst, size_t pitchDst, int width, int height, const float *lineData) {
	float A, B, C, inv_denom;
	float x, y, dist;
	float value;

	// Get line parameters (format Ax + By + C = 0)
	// The parameter 1/sqrt(A^2 + B^2) is also precalculated
	A = lineData[0];
	B = lineData[1];
	C = lineData[2];
	inv_denom = lineData[3];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	x = i;
	y = j;

	if ((i<width) && (j<height)) {
		dist = compute_distance(x, y, A, B, C, inv_denom);

		//if (dist < 1.0f) value = 0.7f;
		//else value = 0.2f;
		value = min(1.0f, max(0.0f, 1.5f - dist/2));

		// Convert and store value into output array
		if (value > 0.0f) dataDst[j*pitchDst + i] = value;
	}
}



// Draw a line
void GpuDrawLines(gpuData_t *gpuData) {
	int width = gpuData->dstSize;
	int height = gpuData->dstSize;

	// Threads per Block
	const dim3 blockSize(16,16,1);

	// Number of blocks
	const dim3 gridSize(ceil(width/(float)blockSize.x),
						ceil(height/(float)blockSize.y),
						1);

	DrawLine_kernel<<<gridSize, blockSize, 0, gpuData->stream>>>(gpuData->imgAccum, gpuData->pitchAccum/sizeof(float),
																width, height, gpuData->lineData);

	CUDA_LAST_ERROR(); // Clear previous non-sticky errors
}





// (GPU) Convert accumulator buffer to output format
__global__
void OutConvert_kernel(uint8_t *dataDst, size_t pitchDst, float *dataSrc, size_t pitchSrc, int width, int height) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	// Flip vertically
	int j_flip = height-1 - j;

	if ((i<width) && (j<height)) {
		// Convert and store value into output array
		dataDst[j*pitchDst + i] = round(dataSrc[j_flip*pitchSrc + i]*255.0f);
	}
}

// Set initial image stats
void GpuOutConvert(uint8_t *hostDst, gpuData_t *gpuData) {
	int width = gpuData->dstSize;
	int height = gpuData->dstSize;

	// Threads per Block
	const dim3 blockSize(16,16,1);

	// Number of blocks
	const dim3 gridSize(ceil(width/(float)blockSize.x),
						ceil(height/(float)blockSize.y),
						1);

	OutConvert_kernel<<<gridSize, blockSize, 0, gpuData->stream>>>(gpuData->imgOut, gpuData->pitchOutput/sizeof(uint8_t),
																gpuData->imgAccum, gpuData->pitchAccum/sizeof(float), width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Copy output image for display
	CUDA_CHECK(cudaMemcpy2DAsync(
		hostDst, width * sizeof(uint8_t),
		gpuData->imgOut, gpuData->pitchOutput,
		width * sizeof(uint8_t), height, cudaMemcpyDeviceToHost, gpuData->stream));

	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_LAST_ERROR(); // Clear previous non-sticky errors
}
