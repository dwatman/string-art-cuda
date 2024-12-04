#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// 2D point coordinate
typedef struct {
	float x;
	float y;
} point_t;

typedef struct {
	double x;
	double y;
} pointd_t;

// Coefficients for a line in Ax + By + C = 0 format
// The parameter 1/sqrt(A^2 + B^2) is also stored for efficiency
typedef struct {
	float A;
	float B;
	float C;
	float inv_denom;
} line_t;

void InitNailPositions(point_t *nails, int numNails);
int ValidateNextNail(int first, int next, int thresh, uint64_t *connections);
line_t PointsToLine(point_t p1, point_t p2);
void CalcLineParams(line_t *lines, const int *pointList, const point_t *nails, int pointIndex);
line_t DistAngleToLine(float dist, float angle);
void ResetConnections(uint64_t *connections);
void SetConnection(int i, int j, uint64_t *connections);
void ClearConnection(int i, int j, uint64_t *connections);
int IsConnected(int i, int j, uint64_t *connections);
double CalcTotalLength(const int *pointList, const point_t *nails);

#endif
