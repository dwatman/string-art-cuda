#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdlib.h>
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
int ValidateNextNail(int first, int next, int thresh);
line_t PointsToLine(point_t p1, point_t p2);
line_t DistAngleToLine(float dist, float angle);
void ResetConnections(void);
void SetConnection(int i, int j);
void ClearConnection(int i, int j);
int IsConnected(int i, int j);
double CalcTotalLength(const int *pointList, const point_t *nails);

#endif
