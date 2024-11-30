#ifndef __SETTINGS_H__
#define __SETTINGS_H__

// Nail and line counts
#define NUM_NAILS 12
#define NUM_LINES 1

// String parameters
#define STRING_THICKNESS 2.5f // Thickness of the string in pixels (float)

// Settings for finding the next nail
#define MIN_LINE_DIST 2 // Minimum interval between nails that a line is allowed
#define RETRY_LIMIT 20 // How many times to retry selecting the next valid nail

// Array to track connections and prevent lines repeating the same path
#define LINE_BIT_ARRAY_SIZE ((NUM_NAILS * NUM_NAILS) / 64 + 1) // Number of uint64_t elements needed

// For line coverage lookup texture
#define LINE_TEX_ANGLE_SAMPLES 32
#define LINE_TEX_DIST_SAMPLES 32

// Block size for error summing (only 16 or 32)
#define SUM_BLOCK_SIZE 16

// Output size (square for now)
#define DATA_SIZE 1024

#endif
