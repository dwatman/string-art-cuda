#include <stdio.h>
#include <stdlib.h> // for atexit(), rand()
//#include <string.h>
#include <time.h> // to init rand()

#include "settings.h"
#include "gpu_funcs.h"
#include "gpu_util.h"
#include "image_io.h"
#include "util.h"

void cleanup(void);
void GpuCleanup(void);

// CPU buffers
uint8_t *h_imageIn = NULL;
uint8_t *h_imageOut = NULL;
line_t *lines = NULL;

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
	InitPinnedBuffers();

	// Clear GPU buffers
	clearBuffers(&gpuData);

	lines = (line_t *)malloc(NUM_LINES*sizeof(line_t));

	srand(time(NULL));   // Initialise RNG

	point_t p0, p1;
	int i;

	for (i=0; i<NUM_LINES; i++) {
		p0.x = rand() % DATA_SIZE;
		p0.y = rand() % DATA_SIZE;
		p1.x = rand() % DATA_SIZE;
		p1.y = rand() % DATA_SIZE;

		lines[i] = pointsToLine(p0, p1);

		//printf("Line (%5.1f, %5.1f)-(%5.1f, %5.1f)", p0.x, p0.y, p1.x, p1.y);
		//printf(" -> %f %f %f (%f)\n", lines[i].A, lines[i].B, lines[i].C, lines[i].inv_denom);
	}

	GpuLoadLines(&gpuData, lines);

	GpuDrawLines(&gpuData);

	// Convert data to int and write to CPU buffer
	GpuOutConvert(h_imageOut, &gpuData);

	// Write image data to disk
	write_png("out.png", h_imageOut, DATA_SIZE, DATA_SIZE, 8);

	printf("Finished\n");
	return 0;
}

// Clean up CPU resources on exit
void cleanup(void) {
	printf("Cleaning up main...\n");

	FreePinnedBuffers();

	if (lines != NULL) free(lines);

	printf("Cleanup main done\n");
}

// Clean up GPU resources on exit
void GpuCleanup(void) {
	printf("GpuCleanup\n");

	GpuFreeBuffers(&gpuData);
}
