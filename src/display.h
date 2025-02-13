#ifndef __DISPLAY_H__
#define __DISPLAY_H__

//#include <stdint.h>
#include <cuda_runtime.h>

// Shared parameters structure
typedef struct {
	float acceptThresh;
	int maxMoveDist;
	uint8_t auto_mode;		// Flag to indicate if "cooling" is being adjusted automatically
	uint8_t update_needed;	// Flag to indicate if parameters were updated
} SharedParameters_t;

// Fix to support both C and C++ compilers
#ifdef __cplusplus
extern "C" {
#endif

//void GpuOutConvert(uint8_t *hostDst, gpuData_t *gpuData);
void initGL(int *argc, char **argv, int width, int height);

#ifdef __cplusplus
}
#endif

#endif
