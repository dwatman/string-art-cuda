#include <stdio.h>

#include "gpu_util.h"
#include "settings.h"

// CPU memory buffers
extern uint8_t *h_imageIn;
extern uint8_t *h_weights;
extern uint8_t *h_imageOut;
extern float   *h_lineCoverage;

// GPU memory buffers
extern gpuData_t gpuData;

// Check enclosed function for CUDA errors
cudaError_t cudaCheck(cudaError_t err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		fprintf(stderr,"ERROR in CUDA Runtime at: %s line %d\n", file, line);
		fprintf(stderr,"    (%d) %s\n", err, cudaGetErrorString(err));
		// Don't exit on error
		// exit(EXIT_FAILURE);
	}
	return err;
}

// Check last recorded CUDA status for errors and clear non-sticky error flag
cudaError_t cudaLastError(const char* const file, const int line) {
	cudaError_t const err{cudaGetLastError()};

	if (err != cudaSuccess) {
		fprintf(stderr,"ERROR in CUDA Runtime at: %s line %d\n", file, line);
		fprintf(stderr,"    (%d) %s\n", err, cudaGetErrorString(err));
		// Don't exit on error
		// exit(EXIT_FAILURE);
	}
	return err;
}

// Initialise GPU and buffers
int GpuInit(void) {
	int err;
	cudaError_t cudaStatus;
	cudaDeviceProp deviceProp;
	int deviceCount = 0;
	int driverVersion = 0, runtimeVersion = 0;

	// Check for available GPUs (returns 0 if there are no CUDA capable devices)
	cudaStatus = CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
	if (cudaStatus != cudaSuccess) {
		exit(EXIT_FAILURE);
	}

	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
		exit(EXIT_FAILURE);
	} else {
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	// Use first device
	CUDA_CHECK(cudaSetDevice(0));
	CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
	CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
	CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));

	printf("Device %d: \"%s\"\n", 0, deviceProp.name);
	printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
		driverVersion / 1000, (driverVersion % 100) / 10,
		runtimeVersion / 1000, (runtimeVersion % 100) / 10);
	printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
		deviceProp.major, deviceProp.minor);

	printf("texturePitchAlignment: %lu\n", deviceProp.texturePitchAlignment);

	err = CUDA_LAST_ERROR();
	printf("CUDA initialized\n\n");

	return err;
}

// Page-lock/pin CPU memory for faster access by GPU
void GpuPinMemory(void *ptr, size_t size) {
	CUDA_CHECK(cudaHostRegister(ptr, size, 0));
}

// Unpin CPU memory before freeing
void GpuUnPinMemory(void *ptr) {
	CUDA_CHECK(cudaHostUnregister(ptr));
}

// Allocate aligned memory on CPU and pin it for fast GPU access
int AllocAndAlignPinned(void **buf, size_t size) {
	size_t sizeAligned;

	// Use fixed 4096 byte alignment to match page size, as GPU pins whole pages (probably)
	size_t alignment = 4096;

	// Make sure the buffer is a multiple of the alignment size
	sizeAligned  = (((alignment-1) + size) / alignment) * alignment;

	//printf("size:  %lu (%lu aligned)\n", size, sizeAligned);

	// Allocate aligned memory on CPU
	*buf = aligned_alloc(alignment, sizeAligned);

	if (*buf == NULL) {
		printf("Error in AllocAndAlignPinned, could not allocate aligned buffer\n");
		return -1;
	}

	// Pin aligned memory for faster GPU access
	GpuPinMemory(*buf, sizeAligned);

	return CUDA_LAST_ERROR();
}

// Initialise pinned host memory
int InitPinnedBuffers(gpuData_t *gpuData) {
	printf("InitPinnedBuffers\n");

	// imageIn is already allocated so pin it only
	GpuPinMemory(h_imageIn, gpuData->srcWidth * gpuData->srcHeight * sizeof(uint8_t));
	//AllocAndAlignPinned((void **)&h_imageIn, gpuData->srcWidth * gpuData->srcHeight * sizeof(uint8_t));

	AllocAndAlignPinned((void **)&h_imageOut, DATA_SIZE * DATA_SIZE * sizeof(uint8_t));
	AllocAndAlignPinned((void **)&h_lineCoverage, LINE_TEX_DIST_SAMPLES * LINE_TEX_ANGLE_SAMPLES * sizeof(float));

	printf("h_imageIn at      %p\n", h_imageIn);
	printf("h_imageOut at     %p\n", h_imageOut);
	printf("h_lineCoverage at %p\n", h_lineCoverage);

	return CUDA_LAST_ERROR();
}

// Free pinned host memory
void FreePinnedBuffers(void) {
	printf("FreePinnedBuffers\n");

	if (h_imageIn != NULL) {
		GpuUnPinMemory(h_imageIn);
		free(h_imageIn);
	}

	if (h_weights != NULL) {
		//GpuUnPinMemory(h_weights);
		free(h_weights);
	}

	if (h_imageOut != NULL) {
		GpuUnPinMemory(h_imageOut);
		free(h_imageOut);
	}

	if (h_lineCoverage != NULL) {
		GpuUnPinMemory(h_lineCoverage);
		free(h_lineCoverage);
	}
}

// Create a bindless texture for the input image
void InitImageInTexture(gpuData_t *deviceData) {
	cudaResourceDesc texRes;
	cudaTextureDesc texDescr;

	// Clear resource descriptors
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	// Set up the 2D texture parameters
	texRes.resType = cudaResourceTypePitch2D;
	texRes.res.pitch2D.devPtr = (void *)deviceData->imgIn;
	texRes.res.pitch2D.desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	texRes.res.pitch2D.width = deviceData->srcWidth;
	texRes.res.pitch2D.height = deviceData->srcHeight;
	texRes.res.pitch2D.pitchInBytes = deviceData->pitchIn;

	// Set up the way the texture is accessed
	texDescr.normalizedCoords = 1;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeNormalizedFloat;

	CUDA_CHECK(cudaCreateTextureObject(&deviceData->texImageIn, &texRes, &texDescr, NULL));
}

// Create a bindless texture for the image weights
void InitWeightsTexture(gpuData_t *deviceData) {
	cudaResourceDesc texRes;
	cudaTextureDesc texDescr;

	// Clear resource descriptors
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	// Set up the 2D texture parameters
	texRes.resType = cudaResourceTypePitch2D;
	texRes.res.pitch2D.devPtr = (void *)deviceData->imgWeight;
	texRes.res.pitch2D.desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	texRes.res.pitch2D.width = deviceData->srcWidth;
	texRes.res.pitch2D.height = deviceData->srcHeight;
	texRes.res.pitch2D.pitchInBytes = deviceData->pitchWeight;

	// Set up the way the texture is accessed
	texDescr.normalizedCoords = 1;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeNormalizedFloat;

	CUDA_CHECK(cudaCreateTextureObject(&deviceData->texWeights, &texRes, &texDescr, NULL));
}

// Copy line coverage data to GPU
void GpuUpdateCoverage(gpuData_t *deviceData, const float *hostData) {
	CUDA_CHECK(cudaMemcpy2DAsync(
		deviceData->lineCoverage, deviceData->pitchCoverage,
		hostData, LINE_TEX_DIST_SAMPLES*sizeof(float),
		LINE_TEX_DIST_SAMPLES*sizeof(float), LINE_TEX_ANGLE_SAMPLES,
		cudaMemcpyHostToDevice, deviceData->stream));
}

// Copy input image to GPU
void GpuUpdateImageIn(gpuData_t *deviceData, const uint8_t *hostData) {
	CUDA_CHECK(cudaMemcpy2DAsync(
		deviceData->imgIn, deviceData->pitchIn,
		hostData, deviceData->srcWidth*sizeof(uint8_t),
		deviceData->srcWidth*sizeof(uint8_t), deviceData->srcHeight,
		cudaMemcpyHostToDevice, deviceData->stream));
}

// Copy weighting mask to GPU
void GpuUpdateWeights(gpuData_t *deviceData, const uint8_t *hostData) {
	CUDA_CHECK(cudaMemcpy2DAsync(
		deviceData->imgWeight, deviceData->pitchWeight,
		hostData, deviceData->srcWidth*sizeof(uint8_t),
		deviceData->srcWidth*sizeof(uint8_t), deviceData->srcHeight,
		cudaMemcpyHostToDevice, deviceData->stream));
}

// Wait for GPU to finish before accessing on host
void GpuSync(void) {
	cudaDeviceSynchronize();

	CUDA_LAST_ERROR(); // Clear previous non-sticky errors
}

// (CUDA) Fill buffer with a value
__global__
void fill_kernel(float *dataDst, size_t pitchDst, float value, int width, int height) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i<width) && (j<height)) {
		// Store value into output array
		dataDst[j*pitchDst + i] = value;
	}
}

// Clear the accumulator buffer
void ClearAccumBuffer(gpuData_t *gpuData) {
	int width = gpuData->dstSize;
	int height = gpuData->dstSize;

	// Threads per Block
	const dim3 blockSize(16,16,1);

	// Number of blocks
	const dim3 gridSize(ceil(width/(float)blockSize.x),
						ceil(height/(float)blockSize.y),
						1);

	// Set accumulator to white
	fill_kernel<<<gridSize, blockSize, 0, gpuData->stream>>>(gpuData->imgAccum, gpuData->pitchAccum/sizeof(float), 1.0f, width, height);

	CUDA_CHECK(cudaDeviceSynchronize());
}
