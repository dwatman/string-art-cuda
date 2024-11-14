#include <stdio.h>
#include <stdlib.h> // for atexit(), rand()
#include <string.h> // for memset
#include <time.h> // to init rand()
#include <math.h> // for abs

#include "settings.h"
#include "gpu_funcs.h"
#include "gpu_util.h"
#include "image_io.h"
#include "util.h"
#include "geometry.h"

void cleanup(void);
void GpuCleanup(void);

point_t nails[NUM_NAILS];
int pointList[NUM_LINES+1];

// CPU buffers
uint8_t *h_imageIn = NULL;
uint8_t *h_weights = NULL;
uint8_t *h_imageOut = NULL;
float   *h_lineCoverage = NULL;
line_t *lines = NULL;

// GPU buffers
gpuData_t gpuData;

char filenameInput[] = "test.png";

int main(int argc, char* argv[]) {
	int err;
	int i, j;
	float lineRatio;

	printf("Start\n");

	// Initialise GPU device and buffers
	GpuInit(); // Set up CUDA device

	printf("Running with %u points, %u lines\n", NUM_NAILS, NUM_LINES);

	// Check how the number of lines compares to the max possible unique connections
	lineRatio = (float)NUM_LINES/((NUM_NAILS-1)*NUM_NAILS/2);
	printf("Line ratio is %.2f%% of maximum connections\n", lineRatio*100);

	if (lineRatio >= 1.0f) {
		printf("ERROR: Number of lines exceeds the number of possible connections (%u)\n", (NUM_NAILS-1)*NUM_NAILS/2);
		return -1;
	}
	else if (lineRatio > 0.25) {
		printf("WARNING: Number of lines exceeds 25%% of all possible connections\n");
		printf("    Recommend reducing the number of lines to below %u\n", (NUM_NAILS-1)*NUM_NAILS/2/4);
	}
	printf("\n");

	int widthIn, heightIn;
	// Load input image
	err = load_greyscale_png(filenameInput, &h_imageIn, &widthIn, &heightIn);
	if (err == 0) {
		printf("Loaded input %s (%dx%d)\n", filenameInput, widthIn, heightIn);
	}
	else {
		printf("Failed to load image %s\n", filenameInput);
		return -2;
	}

	if (widthIn != heightIn)
		printf("WARNING: Input image is not square, it will be stretched to fit\n");

	atexit(GpuCleanup); // set cleanup function for GPU memory
	GpuInitBuffers(&gpuData, widthIn, heightIn); // Initialise GPU buffers

	// Allocate global buffers
	atexit(cleanup); // set cleanup function for CPU memory
	InitPinnedBuffers(&gpuData);heightIn*sizeof(uint8_t);

	// Clear GPU buffers
	ClearBuffers(&gpuData);


	// Set nail positions in a circle (for now)
	InitNailPositions(nails, NUM_NAILS);

	// for (i=0; i<NUM_NAILS; i++) {
	// 	printf("Nail %2u: (%8.3f, %8.3f)\n", i, nails[i].x, nails[i].y);
	// }

	// Allocate CPU buffer for lines
	lines = (line_t *)malloc(NUM_LINES*sizeof(line_t));

	// Create a mask, not pinned (TODO: optionally load from image)
	// check cleanup order
	h_weights = malloc(widthIn*heightIn*sizeof(uint8_t));

	// Fill the weights with maximum value as there is no image
	memset(h_weights, 255, widthIn*heightIn*sizeof(uint8_t));

	// Clear areas outside the border of nails with black to ignore it
	for (j=0; j<heightIn; j++) {
		for (i=0; i<widthIn; i++) {
			if (inside_poly(nails, NUM_NAILS, i*4, j*4) == 0)
				h_weights[j*widthIn + i] = 0;
		}
	}
	//write_png("mask.png", h_weights, widthIn, heightIn, 8);


	// Load the input image into a GPU texture
	GpuUpdateImageIn(&gpuData, h_imageIn);
	InitImageInTexture(&gpuData);


	// Calculate the coverage of lines over pixels
	CalcLineCoverage(h_lineCoverage, STRING_THICKNESS);

	// Copy the coverage data to GPU memory
	GpuUpdateCoverage(&gpuData, h_lineCoverage);

	// Map the coverage data to a texture for fast lookup
	InitCoverageTexture(&gpuData);

	// Reset the map of line connections between nails
	ResetConnections();


	srand(time(NULL));   // Initialise RNG

	point_t p0, p1;
	int n0, n1;

	// First nail
	pointList[0] = rand() % NUM_NAILS;

	// Create lines
	for (i=0; i<NUM_LINES; i++) {
		int retries = 0;

		// Select next nail
		pointList[i+1] = rand() % NUM_NAILS;

		// If the selected nail is not valid (too close, already connected)
		// Choose another until the limit is reached or a suitable nail is found
		while ((retries < RETRY_LIMIT) && (ValidateNextNail(pointList[i], pointList[i+1], MIN_LINE_DIST) == 0)) {
			retries++;
			pointList[i+1] = rand() % NUM_NAILS;
			//printf("Retry %u: %u -> %u\n", retries, pointList[i], pointList[i+1]);
		}

		if (retries == RETRY_LIMIT) {
			printf("ERROR: Retry limit reached for line %u\n", i);
			break;
		}

		SetConnection(pointList[i], pointList[i+1]);

		p0.x = nails[pointList[i]].x;
		p0.y = nails[pointList[i]].y;
		p1.x = nails[pointList[i+1]].x;
		p1.y = nails[pointList[i+1]].y;
		lines[i] = PointsToLine(p0, p1);

		//printf("Nail %3u to %3u\n", pointList[i], pointList[i+1]);
		//printf("Line (%5.1f, %5.1f)-(%5.1f, %5.1f)", p0.x, p0.y, p1.x, p1.y);
		//printf(" -> %f %f %f (%f)\n", lines[i].A, lines[i].B, lines[i].C, lines[i].inv_denom);
	}

	// Check if lines were unable to be completed
	if (i < NUM_LINES) {
		printf("ERROR: Unable to initialise all lines\n");
		return -1;
	}

	// for (i=0; i<=NUM_LINES; i++) {
	// 	printf("Point %3u = %3u\n", i, pointList[i]);
	// }

	// Copy line data to GPU memory
	GpuLoadLines(&gpuData, lines);

	// Draw the set of lines in the GPU image buffer
	GpuDrawLines(&gpuData);

	// Convert the image to uint and write to CPU buffer
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
