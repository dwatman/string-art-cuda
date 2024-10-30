#include <stdio.h>
#include <stdlib.h> // for atexit()
//#include <string.h>

#include "settings.h"
#include "gpu_funcs.h"
#include "gpu_util.h"
#include "image_io.h"

void cleanup(void);
void GpuCleanup(void);

// CPU buffers
float *h_imageIn = NULL;
float *h_imageOut = NULL;

// GPU buffers
gpuData_t gpuData;

int main(int argc, char* argv[]) {
	int err;

	printf("Start\n");

	// Initialise GPU device and buffers
	GpuInit(); // Set up CUDA device
	atexit(GpuCleanup); // set cleanup function for GPU memory
	GpuInitBuffers(&gpuData); // Initialise GPU buffers

	// Allocate global buffers
	atexit(cleanup); // set cleanup function for CPU memory
	InitPinnedBuffers(IMG_WIDTH, IMG_HEIGHT);

	// Write image data
	write_png("out.png", h_imageOut, DATA_SIZE, DATA_SIZE, 8);

	printf("Finished\n");
	return 0;
}

// Clean up CPU resources on exit
void cleanup(void) {
	printf("Cleaning up main...\n");

	FreePinnedBuffers();

	printf("Cleanup main done\n");
}

// Clean up GPU resources on exit
void GpuCleanup(void) {
	printf("GpuCleanup\n");

	GpuFreeBuffers(&gpuData);
}
