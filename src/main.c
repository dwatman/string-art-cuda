#include <stdio.h>
#include <stdlib.h> // for atexit(), rand()
//#include <string.h>
#include <time.h> // to init rand()
#include <math.h> // for abs

#include "settings.h"
#include "gpu_funcs.h"
#include "gpu_util.h"
#include "image_io.h"
#include "util.h"

void cleanup(void);
void GpuCleanup(void);

point_t nails[NUM_NAILS];
int pointList[NUM_LINES+1];

// CPU buffers
uint8_t *h_imageIn = NULL;
uint8_t *h_imageOut = NULL;
line_t *lines = NULL;

// GPU buffers
gpuData_t gpuData;

int main(int argc, char* argv[]) {
	int err;
	int i;

	printf("Start\n");

	// Initialise GPU device and buffers
	GpuInit(); // Set up CUDA device
	atexit(GpuCleanup); // set cleanup function for GPU memory
	GpuInitBuffers(&gpuData); // Initialise GPU buffers

	// Allocate global buffers
	atexit(cleanup); // set cleanup function for CPU memory
	InitPinnedBuffers();

	// Clear GPU buffers
	ClearBuffers(&gpuData);

	lines = (line_t *)malloc(NUM_LINES*sizeof(line_t));


	InitNailPositions(nails, NUM_NAILS);

/*
	for (i=0; i<NUM_NAILS; i++) {
		printf("Nail %2u: (%8.3f, %8.3f)\n", i, nails[i].x, nails[i].y);
	}
*/

	srand(time(NULL));   // Initialise RNG

	point_t p0, p1;
	int n0, n1;

	// First nail
	pointList[0] = rand() % NUM_NAILS;

	for (i=0; i<NUM_LINES; i++) {
		// Select next nail
		do {
			pointList[i+1] = rand() % NUM_NAILS;
		} while (ValidateNextNail(pointList[i], pointList[i+1], MIN_DIST) == 0);

		p0.x = nails[pointList[i]].x;
		p0.y = nails[pointList[i]].y;
		p1.x = nails[pointList[i+1]].x;
		p1.y = nails[pointList[i+1]].y;
		lines[i] = PointsToLine(p0, p1);

		//printf("Nail %3u to %3u\n", pointList[i], pointList[i+1]);
		//printf("Line (%5.1f, %5.1f)-(%5.1f, %5.1f)", p0.x, p0.y, p1.x, p1.y);
		//printf(" -> %f %f %f (%f)\n", lines[i].A, lines[i].B, lines[i].C, lines[i].inv_denom);
	}

	/*for (i=0; i<=NUM_LINES; i++) {
		printf("Point %3u = %3u\n", i, pointList[i]);
	}*/

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
