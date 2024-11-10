#include <stdint.h> // for uint64_t

#include "util.h"
#include "settings.h"

// Bit array to track which nails are connected by lines
static uint64_t connections[LINE_BIT_ARRAY_SIZE] = {0};

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

int ValidateNextNail(int first, int next, int thresh) {
	int diff_direct, diff_wrap, diff_min;

	diff_direct = abs(next - first);
	diff_wrap = NUM_NAILS - diff_direct;
	diff_min = diff_direct < diff_wrap ? diff_direct : diff_wrap;

	if (diff_min <= thresh)
		return 0;
	else
		return 1;
}

// Calculate line parameters from from two points
line_t PointsToLine(point_t p1, point_t p2) {
	line_t line;

	// Calculate coefficients A, B, and C
	line.A = p2.y - p1.y;
	line.B = p1.x - p2.x;
	line.C = (p1.y * p2.x) - (p2.y * p1.x);

	// Calculate 1/sqrt(A^2 + B^2) for efficiency
	line.inv_denom = 1.0f / sqrtf(line.A*line.A + line.B*line.B);

	return line;
}

// Calculate line parameters from from a distance and an angle
line_t DistAngleToLine(float dist, float angle) {
	line_t line;

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

void SetConnection(int i, int j) {
	int bit_index_ij = get_bit_index(i, j);
	int bit_index_ji = get_bit_index(j, i);
	connections[bit_index_ij / 64] |= (1ULL << (bit_index_ij % 64));
	connections[bit_index_ji / 64] |= (1ULL << (bit_index_ji % 64));
}

int IsConnected(int i, int j) {
	int bit_index_ij = get_bit_index(i, j);
	return connections[bit_index_ij / 64] & (1ULL << (bit_index_ij % 64));
}
