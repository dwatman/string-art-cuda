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

	// Buffers for line data
	CUDA_CHECK(cudaMalloc(&gpuData->lineData_A, NUM_LINES*sizeof(float)));
	CUDA_CHECK(cudaMalloc(&gpuData->lineData_B, NUM_LINES*sizeof(float)));
	CUDA_CHECK(cudaMalloc(&gpuData->lineData_C, NUM_LINES*sizeof(float)));
	CUDA_CHECK(cudaMalloc(&gpuData->lineData_inv_denom, NUM_LINES*sizeof(float)));

	// Buffers for image difference calculation
	int numBlocks = (DATA_SIZE/SUM_BLOCK_SIZE)*(DATA_SIZE/SUM_BLOCK_SIZE);
	CUDA_CHECK(cudaMalloc((void**)&gpuData->partialSums, numBlocks*sizeof(double)));
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

	// Memory for line coverage data
	CUDA_CHECK(cudaMallocPitch(&gpuData->lineCoverage, &gpuData->pitchCoverage,
							LINE_TEX_DIST_SAMPLES*sizeof(float), LINE_TEX_ANGLE_SAMPLES));

	printf("pitch lineCoverage: %5lu (%lu) at %p\n", gpuData->pitchCoverage, LINE_TEX_DIST_SAMPLES*sizeof(float), gpuData->lineCoverage);

	CUDA_CHECK(cudaDeviceSynchronize());

	return CUDA_LAST_ERROR();
}

// Cleanup gpuImgData_t structure and free buffers in GPU memory
void GpuFreeBuffers(gpuData_t *gpuData) {
	printf("GpuFreeBuffers\n");

	CUDA_CHECK(cudaStreamDestroy(gpuData->stream));
	CUDA_CHECK(cudaDestroyTextureObject(gpuData->texImageIn));
	CUDA_CHECK(cudaDestroyTextureObject(gpuData->texWeights));

	// Free GPU memory
	if (gpuData->imgIn != NULL) 		CUDA_CHECK(cudaFree(gpuData->imgIn));
	if (gpuData->imgWeight != NULL) 	CUDA_CHECK(cudaFree(gpuData->imgWeight));
	if (gpuData->imgAccum != NULL) 		CUDA_CHECK(cudaFree(gpuData->imgAccum));
	if (gpuData->imgOut != NULL) 		CUDA_CHECK(cudaFree(gpuData->imgOut));

	if (gpuData->lineData_A != NULL) 	CUDA_CHECK(cudaFree(gpuData->lineData_A));
	if (gpuData->lineData_B != NULL) 	CUDA_CHECK(cudaFree(gpuData->lineData_B));
	if (gpuData->lineData_C != NULL) 	CUDA_CHECK(cudaFree(gpuData->lineData_C));
	if (gpuData->lineData_inv_denom != NULL) 	CUDA_CHECK(cudaFree(gpuData->lineData_inv_denom));

	if (gpuData->lineCoverage != NULL) 	CUDA_CHECK(cudaFree(gpuData->lineCoverage));

	if (gpuData->partialSums != NULL) 	CUDA_CHECK(cudaFree(gpuData->partialSums));
	if (gpuData->sumResult != NULL) 	CUDA_CHECK(cudaFree(gpuData->sumResult));
}

// Copy line data to GPU
void GpuLoadLines(gpuData_t *gpuData, lineArray_t *lineList) {

	CUDA_CHECK(cudaMemcpyAsync(gpuData->lineData_A, lineList->A, NUM_LINES*sizeof(float), cudaMemcpyHostToDevice, gpuData->stream));
	CUDA_CHECK(cudaMemcpyAsync(gpuData->lineData_B, lineList->B, NUM_LINES*sizeof(float), cudaMemcpyHostToDevice, gpuData->stream));
	CUDA_CHECK(cudaMemcpyAsync(gpuData->lineData_C, lineList->C, NUM_LINES*sizeof(float), cudaMemcpyHostToDevice, gpuData->stream));
	CUDA_CHECK(cudaMemcpyAsync(gpuData->lineData_inv_denom, lineList->inv_denom, NUM_LINES*sizeof(float), cudaMemcpyHostToDevice, gpuData->stream));
	CUDA_CHECK(cudaDeviceSynchronize());
}

#define DIST_SCALE (float)((LINE_TEX_DIST_SAMPLES - 1) / MAX_DIST)
#define ANGLE_SCALE (float)((LINE_TEX_ANGLE_SAMPLES - 1) / (float)M_PI)

// (GPU) Draw many lines
__global__
void DrawLine_kernel(float *dataDst, size_t pitchDst, int width, int height, const float *lineA, const float *lineB, const float *lineC, const float *lineInvDenom,  const float *coverage, size_t coveragePitch) {
	// Shared memory for storing the coverage array and line parameters per block
	__shared__ float sharedCoverage[LINE_TEX_ANGLE_SAMPLES*LINE_TEX_DIST_SAMPLES];
	__shared__ float sharedLineParams[LINE_CHUNK_SIZE][4];
	__shared__ uint8_t sharedAngleIndex[LINE_CHUNK_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x * blockDim.x;
	int by = blockIdx.y * blockDim.y;

	int i = bx + tx;
	int j = by + ty;

	// Thread's contribution to the output
	float pixelValue = 1.0f;

	// Copy coverage array into shared memory
	for (int linearIdx = threadIdx.x + threadIdx.y * blockDim.x; linearIdx < LINE_TEX_ANGLE_SAMPLES * LINE_TEX_DIST_SAMPLES; linearIdx += blockDim.x * blockDim.y) {
		if (linearIdx < LINE_TEX_ANGLE_SAMPLES * LINE_TEX_DIST_SAMPLES) { // Bounds checking
			int angleIndex = linearIdx / LINE_TEX_DIST_SAMPLES;
			int distIndex = linearIdx % LINE_TEX_DIST_SAMPLES;

			sharedCoverage[linearIdx] = __ldg(&coverage[angleIndex * coveragePitch + distIndex]);
		}
	}
	__syncthreads();

	// Process lines in chunks of LINE_CHUNK_SIZE
	for (int lineChunkStart = 0; lineChunkStart < NUM_LINES; lineChunkStart += LINE_CHUNK_SIZE) {
		int localLineIdx = ty * blockDim.x + tx;

		// Each thread loads a line parameter set into shared memory (if within bounds)
		if (localLineIdx < LINE_CHUNK_SIZE && (lineChunkStart + localLineIdx) < NUM_LINES) {
			sharedLineParams[localLineIdx][0] = lineA[lineChunkStart + localLineIdx];
			sharedLineParams[localLineIdx][1] = lineB[lineChunkStart + localLineIdx];
			sharedLineParams[localLineIdx][2] = lineC[lineChunkStart + localLineIdx];
			sharedLineParams[localLineIdx][3] = lineInvDenom[lineChunkStart + localLineIdx];

			// Precompute angle index and store it for reuse
			float A = sharedLineParams[localLineIdx][0];
			float B = sharedLineParams[localLineIdx][1];

			float angle = atan2f(-A, B); 			// Calculate the angle in radians with respect to the x-axis
			if (angle < 0) angle += (float)M_PI; 	// Convert the angle to the range [0, PI] if necessary

			//sharedAngleIndex[localLineIdx] = roundf(angle / (float)M_PI * (LINE_TEX_ANGLE_SAMPLES - 1));
			sharedAngleIndex[localLineIdx] = roundf(angle * ANGLE_SCALE);
		}
		__syncthreads();

		// Calculate pixel contribution for the loaded lines
		if (i < width && j < height) {
			for (int lineIdx = 0; lineIdx < LINE_CHUNK_SIZE; ++lineIdx) {
				if (lineChunkStart + lineIdx >= NUM_LINES) break;

				// Extract line parameters
				float A = sharedLineParams[lineIdx][0];
				float B = sharedLineParams[lineIdx][1];
				float C = sharedLineParams[lineIdx][2];
				float inv_denom = sharedLineParams[lineIdx][3];

				// Validate inv_denom to prevent undefined behavior
				if (inv_denom == 0.0f) continue;

				// Compute distance (distance is per-pixel, angle is precomputed per-line)
				//float dist = fabsf(A*i + B*j + C) * inv_denom;
				float dist = fabsf(fmaf(A, i, fmaf(B, j, C)) * inv_denom); // Use fused operations

				//const float distScale = (LINE_TEX_DIST_SAMPLES - 1) / MAX_DIST;
				int distIndex = roundf(dist * DIST_SCALE);

				// Calculate indices for coverage lookup
				//int distIndex = roundf(dist / MAX_DIST * (LINE_TEX_DIST_SAMPLES - 1));
				int angleIndex = sharedAngleIndex[lineIdx];

				// Skip calculation if lines will not contribute to the image
				if (dist > MAX_DIST) continue;

				// Accumulate coverage contribution
				pixelValue *= 1.0f - sharedCoverage[angleIndex * LINE_TEX_DIST_SAMPLES + distIndex];
			}
		}
		__syncthreads();
	}

	// Store final pixel value to output
	if (i < width && j < height) {
		dataDst[j * pitchDst + i] = pixelValue;
	}
}

// Draw many lines
void GpuDrawLines(gpuData_t *gpuData) {
	int width = gpuData->dstSize;
	int height = gpuData->dstSize;

	// Temporary for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Threads per Block
	const dim3 blockSize(16,16,1);

	// Number of blocks
	const dim3 gridSize(ceil(width/(float)blockSize.x),
						ceil(height/(float)blockSize.y),
						1);

	cudaEventRecord(start, gpuData->stream);	// Temporary for timing
	DrawLine_kernel<<<gridSize, blockSize, 0, gpuData->stream>>>(gpuData->imgAccum, gpuData->pitchAccum/sizeof(float),
																width, height, gpuData->lineData_A, gpuData->lineData_B, gpuData->lineData_C, gpuData->lineData_inv_denom,
																gpuData->lineCoverage, gpuData->pitchCoverage/sizeof(float));
	// Temporary for timing
	cudaEventRecord(stop, gpuData->stream);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("    DrawLine_kernel: %f ms\n", milliseconds);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	CUDA_LAST_ERROR(); // Clear previous non-sticky errors
}

// Compute the sum of weighted errors for each block in the image
__global__ void computeBlockErrors_kernel(double* partialSums, const float* dataAccum, size_t pitchAccum, int width, int height,
									const cudaTextureObject_t texImage, const cudaTextureObject_t texWeight) {

	extern __shared__ double blockSum[];

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
	extern __shared__ double blockSum[];

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
	size_t sharedMemSize;
	double h_result;

	int width = gpuData->dstSize;
	int height = gpuData->dstSize;

	// Threads per Block
	const dim3 blockSize(SUM_BLOCK_SIZE, SUM_BLOCK_SIZE, 1);

	// Number of blocks (should alwyas be an integer multiple)
	const dim3 gridSize(width/blockSize.x, height/blockSize.y, 1);

	size_t numBlocks = gridSize.x*gridSize.y;

	// Clear the total sum in GPU memory
	CUDA_CHECK(cudaMemsetAsync(gpuData->sumResult, 0, sizeof(double), gpuData->stream));

	// Launch the first kernel to compute block partial sums
	sharedMemSize = SUM_BLOCK_SIZE*SUM_BLOCK_SIZE*sizeof(double);

	computeBlockErrors_kernel<<<gridSize, blockSize, sharedMemSize, gpuData->stream>>>(gpuData->partialSums, gpuData->imgAccum, gpuData->pitchAccum/sizeof(float), width, height,
									gpuData->texImageIn, gpuData->texWeights);

	// Launch the second kernel to reduce partial sums
	int threadsPerBlock = 256;
	int blocksPerGrid = (numBlocks + threadsPerBlock - 1) / threadsPerBlock; // 16
	sharedMemSize = threadsPerBlock * sizeof(double);

	reducePartialSums_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize, gpuData->stream>>>(gpuData->sumResult, gpuData->partialSums, numBlocks);

	// Retrieve the final result
	CUDA_CHECK(cudaMemcpyAsync(&h_result, gpuData->sumResult, sizeof(double), cudaMemcpyDeviceToHost, gpuData->stream));
	CUDA_CHECK(cudaDeviceSynchronize());

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
