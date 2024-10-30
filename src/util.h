#ifndef __UTIL_H__
#define __UTIL_H__

#include <math.h>

// 2D point coordinate
typedef struct {
	float x;
	float y;
} point_t;

// Coefficients for a line in Ax + By + C = 0 format
// The parameter 1/sqrt(A^2 + B^2) is also stored for efficiency
typedef struct {
	float A;
	float B;
	float C;
	float inv_denom;
} line_t;

line_t pointsToLine(point_t p1, point_t p2);

#endif
