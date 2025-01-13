#include <stdio.h>
#include <stdlib.h> // for atexit(), rand()
#include <string.h> // for memset
#include <time.h> // to init rand()
#include <math.h> // for abs
#include <pthread.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "settings.h"
#include "gpu_funcs.h"
#include "gpu_util.h"
#include "image_io.h"
#include "util.h"
#include "geometry.h"
#include "display.h"

void CpuCleanup(void);
void GpuCleanup(void);
void *computationThreadFunc(void *arg);

// CPU buffers
uint8_t *h_imageIn = NULL;
uint8_t *h_weights = NULL;
uint8_t *h_imageOut = NULL;
float   *h_lineCoverage = NULL;

// GPU buffers
gpuData_t gpuData;

char filenameInput[] = "test.png";
int widthIn, heightIn;

SharedParameters_t parameters = {0.0f, 0};
volatile uint8_t running = 1;
volatile uint8_t update_image = 0;

pthread_t computation_thread;
pthread_mutex_t param_mutex = PTHREAD_MUTEX_INITIALIZER;

int main(int argc, char* argv[]) {
	int err;
	float lineRatio;

	// Set initial parameter values
	parameters.linesToMove = sqrt(NUM_LINES)/4;
	parameters.maxMoveDist = NUM_NAILS/2;
	parameters.update_needed = 1;

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

	printf("MAX_DIST: %f\n", MAX_DIST);

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
	atexit(CpuCleanup); // set cleanup function for CPU memory
	InitPinnedBuffers(&gpuData);


	// Initialize GLUT and GLEW
	initGL(&argc, argv, 512, 512);

	// Create computation thread
	if (pthread_create(&computation_thread, NULL, computationThreadFunc, NULL) != 0) {
		printf("ERROR: Could not create computation thread\n");
		return -3;
	}

	// Start OpenGL main loop
	glutMainLoop();

	printf("Finished\n");
	return 0;
}

// Clean up CPU resources on exit
void CpuCleanup(void) {
	printf("CpuCleanup\n");

	FreePinnedBuffers();

	printf("CpuCleanup done\n");
}

// Clean up GPU resources on exit
void GpuCleanup(void) {
	printf("GpuCleanup\n");

	GpuFreeBuffers(&gpuData);

	printf("GpuCleanup done\n");
}

// CUDA computation thread function
void *computationThreadFunc(void *arg) {
	int err;
	int i, j;

	// Local copies of the parameters
	int moveLines;
	int maxDist;

	point_t nails[NUM_NAILS];
	int pointList[NUM_LINES+1];
	lineArray_t lines;
	uint64_t connections[LINE_BIT_ARRAY_SIZE]; // Bit array to track which nails are connected by lines
	double totalLength;
	double imageError;

	// Storage for best result
	int bestPoints[NUM_LINES+1];
	lineArray_t bestLines;
	uint64_t bestConnections[LINE_BIT_ARRAY_SIZE];
	double bestError;

	printf("computationThreadFunc started\n");

	srand(time(NULL));   // Initialise RNG
	//srand(1234567);   // Initialise RNG to fixed seed for testing

	// Set nail positions in a circle (for now)
	InitNailPositions(nails, NUM_NAILS);

	// for (i=0; i<NUM_NAILS; i++) {
	// 	printf("Nail %2u: (%8.3f, %8.3f)\n", i, nails[i].x, nails[i].y);
	// }

	// Create a mask, not pinned (TODO: optionally load from image)
	// check cleanup order
	h_weights = (uint8_t *)malloc(widthIn*heightIn*sizeof(uint8_t));

	// Fill the weights with maximum value as there is no image
	memset(h_weights, 255, widthIn*heightIn*sizeof(uint8_t));

	// Clear areas outside the border of nails with black to ignore it
	for (j=0; j<heightIn; j++) {
		for (i=0; i<widthIn; i++) {
			if (inside_poly(nails, NUM_NAILS, i*(DATA_SIZE/widthIn), j*(DATA_SIZE/heightIn)) == 0)
				h_weights[j*widthIn + i] = 0;
		}
	}

	// Load the input image into a GPU texture
	GpuUpdateImageIn(&gpuData, h_imageIn);
	InitImageInTexture(&gpuData);

	// Load the weights into a GPU texture
	GpuUpdateWeights(&gpuData, h_weights);
	InitWeightsTexture(&gpuData);

	// Calculate the coverage of lines over pixels
	CalcLineCoverage(h_lineCoverage, STRING_THICKNESS);

	// Copy the coverage data to GPU memory
	GpuUpdateCoverage(&gpuData, h_lineCoverage);

	// Generate an initial random line pattern
	err = GenerateRandomPattern(bestConnections, &bestLines, bestPoints, nails);
	if (err) {
		running = 0; // Indicate failure to the main thread
		return NULL;
	}

	bestError = 10000000.0;

	while (running) {
		// Lock parameter access
		pthread_mutex_lock(&param_mutex);
		if (parameters.update_needed) {
			moveLines = parameters.linesToMove;
			maxDist = parameters.maxMoveDist;
			parameters.update_needed = 0;
		}
		pthread_mutex_unlock(&param_mutex);

		// Run optimisation

		// Start with the best result
		memcpy(pointList, bestPoints, (NUM_LINES+1)*sizeof(int));
		memcpy(&lines, &bestLines, sizeof(lineArray_t));
		memcpy(connections, bestConnections, LINE_BIT_ARRAY_SIZE*sizeof(uint64_t));

		// Move some points around
		for (j=0; j<moveLines; j++) {
			MovePattern(connections, &lines, pointList, nails, maxDist);
		}

		// Copy line data to GPU memory
		GpuLoadLines(&gpuData, &lines);

		// Draw the set of lines in the GPU image buffer
		GpuDrawLines(&gpuData);

		// Compute error between original and generated images
		imageError = GpucalculateImageError(&gpuData);
		//printf("#%i imageError: %f", i, imageError);

		totalLength = CalcTotalLength(pointList, nails);
		//printf("  length  %5.1f", totalLength);

		// Divide error by total line length to reduce bias for short connections
		imageError /= totalLength;
		//printf("  (%f)", imageError);

		if (imageError < bestError) {
			bestError = imageError;

			// Update best pattern
			memcpy(bestPoints, pointList, (NUM_LINES+1)*sizeof(int));
			memcpy(&bestLines, &lines, sizeof(lineArray_t));
			memcpy(bestConnections, connections, LINE_BIT_ARRAY_SIZE*sizeof(uint64_t));

			//printf("  (best)");
		}
		//printf("\n");
		CUDA_CHECK(cudaDeviceSynchronize());

		// Update the image for display at the display rate
		if (update_image) {
			update_image = 0;
			// printf("#%i imageError: %f", i, imageError);
			// printf("  length  %5.1f", totalLength);
			// printf("  (%f)", imageError);
			// printf("\n");
			GpuOutConvert(h_imageOut, &gpuData);// Convert the image to uint and write to CPU buffer
		}
	}

	printf("bestError: %f\n", bestError);

	// Clear areas outside the border of nails
	for (j=0; j<DATA_SIZE; j++) {
		for (i=0; i<DATA_SIZE; i++) {
			if (inside_poly(nails, NUM_NAILS, i, j) == 0)
				h_imageOut[j*DATA_SIZE + i] = 128;
		}
	}

	// Write image data to disk
	write_png("out.png", h_imageOut, DATA_SIZE, DATA_SIZE, 8);

	printf("computationThreadFunc stopped\n");

	return NULL;
}
