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
int GpuInitBuffers(gpuData_t *gpuData, int widthIn, int heightIn) {
	printf("GpuInitBuffers\n");

	// Clear the structure in case of errors part way through initialisation
	memset(gpuData, 0, sizeof(gpuData_t));

	gpuData->srcWidth  = widthIn;
	gpuData->srcHeight = heightIn;
	gpuData->dstSize = DATA_SIZE;

	// Create stream for processing
	CUDA_CHECK(cudaStreamCreate(&gpuData->stream));

	// Buffer for line data
	CUDA_CHECK(cudaMalloc(&gpuData->lineData, NUM_LINES*4*sizeof(float)));

	// Buffers for image difference calculation
	CUDA_CHECK(cudaMalloc((void**)&gpuData->partialSums, SUM_BLOCK_SIZE*SUM_BLOCK_SIZE*sizeof(double)));
	CUDA_CHECK(cudaMalloc((void**)&gpuData->sumResult, sizeof(double)));

	// Image buffers
	CUDA_CHECK(cudaMallocPitch(&gpuData->imgIn, &gpuData->pitchIn,
							gpuData->srcWidth*sizeof(uint8_t), gpuData->srcHeight));
	CUDA_CHECK(cudaMallocPitch(&gpuData->imgWeight, &gpuData->pitchWeight,
							gpuData->srcWidth*sizeof(uint8_t), gpuData->srcHeight));
	CUDA_CHECK(cudaMallocPitch(&gpuData->imgAccum, &gpuData->pitchAccum,
							gpuData->dstSize*sizeof(float), gpuData->dstSize));
	CUDA_CHECK(cudaMallocPitch(&gpuData->imgOut, &gpuData->pitchOutput,
							gpuData->dstSize*sizeof(uint8_t), gpuData->dstSize));

	printf("pitch imgIn:     %5lu (%lu) at %p\n", gpuData->pitchIn, 	gpuData->srcWidth*sizeof(uint8_t), gpuData->imgIn);
	printf("pitch imgWeight: %5lu (%lu) at %p\n", gpuData->pitchWeight, gpuData->srcWidth*sizeof(uint8_t), gpuData->imgWeight);
	printf("pitch Accum:     %5lu (%lu) at %p\n", gpuData->pitchAccum, 	gpuData->srcWidth*sizeof(float), gpuData->imgAccum);
	printf("pitch output:    %5lu (%lu) at %p\n", gpuData->pitchOutput, gpuData->srcWidth*sizeof(uint8_t), gpuData->imgOut);

	// Memory for 2D texture
	CUDA_CHECK(cudaMallocPitch(&gpuData->lineCoverage, &gpuData->pitchCoverage,
							LINE_TEX_ANGLE_SAMPLES*sizeof(float), LINE_TEX_DIST_SAMPLES));

	printf("pitch lineCoverage: %5lu (%lu) at %p\n", gpuData->pitchCoverage, LINE_TEX_ANGLE_SAMPLES*sizeof(float), gpuData->lineCoverage);

	CUDA_CHECK(cudaDeviceSynchronize());

	return CUDA_LAST_ERROR();
}

// Cleanup gpuImgData_t structure and free buffers in GPU memory
void GpuFreeBuffers(gpuData_t *gpuData) {
	printf("GpuFreeBuffers\n");

	CUDA_CHECK(cudaStreamDestroy(gpuData->stream));
	CUDA_CHECK(cudaDestroyTextureObject(gpuData->texCoverage));
	CUDA_CHECK(cudaDestroyTextureObject(gpuData->texImageIn));
	CUDA_CHECK(cudaDestroyTextureObject(gpuData->texWeights));

	// Free GPU memory
	if (gpuData->imgIn != NULL) 		CUDA_CHECK(cudaFree(gpuData->imgIn));
	if (gpuData->imgWeight != NULL) 	CUDA_CHECK(cudaFree(gpuData->imgWeight));
	if (gpuData->imgAccum != NULL) 		CUDA_CHECK(cudaFree(gpuData->imgAccum));
	if (gpuData->imgOut != NULL) 		CUDA_CHECK(cudaFree(gpuData->imgOut));

	if (gpuData->lineData != NULL) 		CUDA_CHECK(cudaFree(gpuData->lineData));
	if (gpuData->lineCoverage != NULL) 	CUDA_CHECK(cudaFree(gpuData->lineCoverage));

	if (gpuData->partialSums != NULL) 	CUDA_CHECK(cudaFree(gpuData->partialSums));
	if (gpuData->sumResult != NULL) 	CUDA_CHECK(cudaFree(gpuData->sumResult));
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

// Compute the angle of the line Ax + By + C = 0 relative to the x-axis
// Angle returned is in the range 0 to pi
__device__ float compute_angle(float A, float B) {
	// Calculate the angle in radians with respect to the x-axis
	float angle = atan2f(-A, B); // atan2(-A, B) ensures the correct quadrant

	// Convert the angle to the range [0, π] if necessary
	if (angle < 0) angle += (float)M_PI;

	return angle;
}

// (GPU) Draw many lines
__global__
void DrawLine_kernel(float *dataDst, size_t pitchDst, int width, int height, const float *lineData, float lineThickness, const cudaTextureObject_t tex) {
	float A, B, C, inv_denom;
	float dist, angle;
	float maxDist;
	float value;
	int line;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	// Flip vertically
	//int j_flip = height-1 - j;

	// Calculate the maximum distance at which the line overlaps a pixel
	maxDist = sqrtf(2)/2 + lineThickness/2;

	if ((i<width) && (j<height)) {
		value = 1.0f;

		for (line=0; line<NUM_LINES; line++) {
			// Get line parameters (format Ax + By + C = 0)
			// The parameter 1/sqrt(A^2 + B^2) is also precalculated
			A = lineData[4*line + 0];
			B = lineData[4*line + 1];
			C = lineData[4*line + 2];
			inv_denom = lineData[4*line + 3];

			// Calculate distance and angle for calculating partial coverage
			dist = compute_distance(i, j, A, B, C, inv_denom);
			angle = compute_angle(A, B);

			// Look up the coverage at this pixel and accumulate it to the total
			value *= (1.0f - tex2D<float>(tex, angle, dist/maxDist));
		}

		// Convert and store value into output array
		if (value < 1.0f) dataDst[j*pitchDst + i] = max(0.0f, value);
	}

}

// Draw many lines
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
																width, height, gpuData->lineData, STRING_THICKNESS, gpuData->texCoverage);

	CUDA_LAST_ERROR(); // Clear previous non-sticky errors
}


// Compute the sum of weighted errors for each block in the image
__global__ void computeBlockErrors_kernel(double* partialSums, const float* dataAccum, size_t pitchAccum, int width, int height,
									const cudaTextureObject_t texImage, const cudaTextureObject_t texWeight) {

	__shared__ double blockSum[SUM_BLOCK_SIZE*SUM_BLOCK_SIZE];

	float image, accum, weight;
	float diff;

	// Linear thread ID
	int tid = threadIdx.y * blockDim.x + threadIdx.x;

	// Pixel 2D position
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	// Normalised 2D coordinates for textures
	float u = (float)i / (float)(width - 1);
	float v = (float)j / (float)(height - 1);

	// Fetch the data for this pixel
	image  = tex2D<float>(texImage, u, v);  // Input image
	weight = tex2D<float>(texWeight, u, v); // Weight
	accum  = dataAccum[j*pitchAccum + i];   // Generated line image

	// Compute the weighted absolute difference for the pixel
	diff = fabsf(image - accum)*weight;

	// Store the result in shared memory
	blockSum[tid] = diff;
	__syncthreads();

	// Perform reduction sum within the block
	for (int stride = (SUM_BLOCK_SIZE*SUM_BLOCK_SIZE) / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			blockSum[tid] += blockSum[tid + stride];
		}
		__syncthreads();
	}

	// Write the block's sum to the global array
	if (tid == 0) {
		partialSums[blockIdx.y * gridDim.x + blockIdx.x] = blockSum[0];
	}

}

// Sum the block errors into a total error
__global__ void reducePartialSums_kernel(double* result, const double* partialSums, int numElements) {
	__shared__ double blockSum[SUM_BLOCK_SIZE*SUM_BLOCK_SIZE];

	int tx = threadIdx.x;
	int index = blockIdx.x * blockDim.x + tx;

	double sum = (index < numElements) ? partialSums[index] : 0.0;

	blockSum[tx] = sum;
	__syncthreads();

	// Perform reduction within the block
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tx < stride) {
			blockSum[tx] += blockSum[tx + stride];
		}
		__syncthreads();
	}

	// Write the block's sum to the global result
	if (tx == 0) {
		atomicAdd(result, blockSum[0]);
	}
}

// Compute the total weighted error between the original image and the generated lines
double GpucalculateImageError(gpuData_t *gpuData) {
	double h_result;

	int width = gpuData->dstSize;
	int height = gpuData->dstSize;

	// Threads per Block
	const dim3 blockSize(SUM_BLOCK_SIZE, SUM_BLOCK_SIZE, 1);

	// Number of blocks (should alwyas be an integer multiple)
	const dim3 gridSize(width/blockSize.x, height/blockSize.y, 1);

	size_t numBlocks = gridSize.x * gridSize.y;

	// Clear the total sum in GPU memory
	cudaMemset(gpuData->sumResult, 0, sizeof(double));
	CUDA_CHECK(cudaDeviceSynchronize());//??

	// Launch the first kernel to compute block partial sums
	computeBlockErrors_kernel<<<gridSize, blockSize, 0, gpuData->stream>>>(gpuData->partialSums, gpuData->imgAccum, gpuData->pitchAccum/sizeof(float), width, height,
									gpuData->texImageIn, gpuData->texWeights);

	//cudaStreamSynchronize(gpuData->stream);  // Synchronize after first kernel??

	// Launch the second kernel to reduce partial sums
	int threadsPerBlock = 256;
	int blocksPerGrid = (numBlocks + threadsPerBlock - 1) / threadsPerBlock; // 16
	reducePartialSums_kernel<<<blocksPerGrid, threadsPerBlock, 0, gpuData->stream>>>(gpuData->sumResult, gpuData->partialSums, numBlocks);

	CUDA_CHECK(cudaDeviceSynchronize());//??

	// Retrieve the final result
	cudaMemcpyAsync(&h_result, gpuData->sumResult, sizeof(double), cudaMemcpyDeviceToHost, gpuData->stream);

	return h_result;
}


// (GPU) Convert accumulator buffer to output format
__global__
void OutConvert_kernel(uint8_t *dataDst, size_t pitchDst, float *dataSrc, size_t pitchSrc, int width, int height) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	// Flip vertically
	int j_flip = j;//height-1 - j;

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
