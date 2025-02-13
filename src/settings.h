#ifndef __SETTINGS_H__
#define __SETTINGS_H__

// Nail and line counts
#define NUM_NAILS 300
#define NUM_LINES 2000
#define SQUARE_SHAPE 0 // Round if 0, square if 1. NUM_NAILS must be divisible by 4 if square

// Output option
#define OUTPUT_VIDEO 1 			// Output a sequence of images during optimisation
#define VIDEO_DIR "video" 		// Directory name
#define VIDEO_FILENAME "img_" 	// File name, number will be appended

// String parameters
#define STRING_THICKNESS 0.25f // Thickness of the string in pixels (float)
#define MAX_DIST (sqrtf(2)/2 + STRING_THICKNESS/2)

// Settings for finding the next nail
#define MIN_LINE_DIST 2 // Minimum interval between nails that a line is allowed
#define RETRY_LIMIT 20  // How many times to retry selecting the next valid nail

// Array to track connections and prevent lines repeating the same path
#define LINE_BIT_ARRAY_SIZE ((NUM_NAILS * NUM_NAILS) / 64 + 1) // Number of uint64_t elements needed

// For line coverage lookup texture
#define LINE_TEX_DIST_SAMPLES 32
#define LINE_TEX_ANGLE_SAMPLES 32

// Number of lines to process in each chunk
#define LINE_CHUNK_SIZE 256

// Block size for error summing (only 16 or 32)
#define SUM_BLOCK_SIZE 16

// Output size (square for now)
#define DATA_SIZE 512

// OpenGL window refresh rate (ms)
#define REFRESH_DELAY 1/10.0

#endif
