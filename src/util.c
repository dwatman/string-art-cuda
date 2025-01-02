#include <stdint.h> // for uint64_t

#include "util.h"
#include "settings.h"

// Initialise nail positions in a cirlce (for now)
void InitNailPositions(point_t *nails, int numNails) {
	float x, y;
	float angle;
	int i;

	// Arrange nails in a circle
	for (i=0; i<numNails; i++) {
		angle = i*2*M_PI/numNails;

		x = (1.0 + cos(angle)) * (DATA_SIZE-1)/2.0;
		y = (1.0 - sin(angle)) * (DATA_SIZE-1)/2.0;

		nails[i].x = x;
		nails[i].y = y;
	}
}

// Check if the next proposed nail position is valid
// Prevents short connections and repeated lines
int ValidateNextNail(int first, int next, int thresh, uint64_t *connections) {
	int diff_direct, diff_wrap, diff_min;

	diff_direct = abs(next - first);
	diff_wrap = NUM_NAILS - diff_direct;
	diff_min = diff_direct < diff_wrap ? diff_direct : diff_wrap;

	if ((diff_min <= thresh) || IsConnected(first, next, connections))
		return 0; // Not valid
	else
		return 1; // Valid
}

// Calculate line parameters from from two points
lineParam_t PointsToLine(point_t p1, point_t p2) {
	lineParam_t line;

	// Calculate coefficients A, B, and C
	line.A = p2.y - p1.y;
	line.B = p1.x - p2.x;
	line.C = (p1.y * p2.x) - (p2.y * p1.x);

	// Calculate 1/sqrt(A^2 + B^2) for later efficient use
	line.inv_denom = 1.0f / sqrtf(line.A*line.A + line.B*line.B);

	return line;
}

// Calculate the line parameters between each sequential pair in the point list
void CalcLineParams(lineArray_t *lineList, const int *pointList, const point_t *nails, int pointIndex) {
	point_t p0, p1;
	lineParam_t line;

	// Make sure the index is valid
	if ((pointIndex < 0) || (pointIndex > NUM_LINES))
		return;

	p0.x = nails[pointList[pointIndex]].x;
	p0.y = nails[pointList[pointIndex]].y;
	p1.x = nails[pointList[pointIndex+1]].x;
	p1.y = nails[pointList[pointIndex+1]].y;

	line = PointsToLine(p0, p1);
	lineList->A[pointIndex] = line.A;
	lineList->B[pointIndex] = line.B;
	lineList->C[pointIndex] = line.C;
	lineList->inv_denom[pointIndex] = line.inv_denom;
}

// Calculate line parameters from from a distance and an angle
lineParam_t DistAngleToLine(float dist, float angle) {
	lineParam_t line;

	// Calculate coefficients A, B, and C
	line.A = cos(angle);
	line.B = -sin(angle);
	line.C = -dist;

	// Calculate 1/sqrt(A^2 + B^2) for efficiency
	line.inv_denom = 1.0f / sqrtf(line.A*line.A + line.B*line.B);

	return line;
}

// Caclulate bit index in connections array
static inline int get_bit_index(int i, int j) {
	return i * NUM_NAILS + j;
}

// Clear all recorded connections, and preset invalid links
void ResetConnections(uint64_t *connections) {
	int i;

	// Clear all connections
	for (i=0; i<LINE_BIT_ARRAY_SIZE; i++)
		connections[i] = 0;

	// Set points as connected to themselves
	for (i=0; i<NUM_NAILS; i++)
		SetConnection(i, i, connections);
}

// Mark a connection (in both directions)
void SetConnection(int i, int j, uint64_t *connections) {
	// Prevent out of array access
	if ((i >= NUM_NAILS) || (j >= NUM_NAILS))
		return;

	int bit_index = get_bit_index(i, j);
	connections[bit_index / 64] |= ((uint64_t)1 << (bit_index % 64));

	if (i != j) { // For non-diagonal entries, set the reverse as well
		bit_index = get_bit_index(j, i);
		connections[bit_index / 64] |= ((uint64_t)1 << (bit_index % 64));
	}
}

// Clear a connection (in both directions)
void ClearConnection(int i, int j, uint64_t *connections) {
	// Prevent out-of-array access
	if ((i >= NUM_NAILS) || (j >= NUM_NAILS))
		return;

	int bit_index = get_bit_index(i, j);
	connections[bit_index / 64] &= ~((uint64_t)1 << (bit_index % 64));

	if (i != j) { // For non-diagonal entries, clear the reverse as well
		bit_index = get_bit_index(j, i);
		connections[bit_index / 64] &= ~((uint64_t)1 << (bit_index % 64));
	}
}

// Check if two nails are connected
int IsConnected(int i, int j, uint64_t *connections) {
	// Prevent out of array access
	if ((i >= NUM_NAILS) || (j >= NUM_NAILS))
		return 1; // Say out of range points are connected

	int bit_index = get_bit_index(i, j);
	return (connections[bit_index / 64] & ((uint64_t)1 << (bit_index % 64))) ? 1 : 0;
}

//#include <stdio.h>
// Calculate the total length of string
double CalcTotalLength(const int *pointList, const point_t *nails) {
	int i;
	float dx, dy, dist;
	double totalLength = 0.0;

	for (i=0; i<NUM_LINES; i++) {
		dx = nails[pointList[i]].x - nails[pointList[i+1]].x;
		dy = nails[pointList[i]].y - nails[pointList[i+1]].y;
		dist = hypot(dx, dy);
		// printf("Nail %3u to %3u\n", pointList[i], pointList[i+1]);
		// printf("    diffXY %5.1f, %5.1f\n", dx, dy);
		// printf("    dist %5.1f\n", dist);
		totalLength += dist;
	}
	// printf("totalLength %5.1f\n", totalLength);
	return totalLength;
}
