#include "util.h"
#include "settings.h"

void InitNailPositions(point_t *nails, int numNails) {
	float x, y;
	float angle;
	int i;

	// Arrange nails in a circle
	for (i=0; i<numNails; i++) {
		angle = i*2*M_PI/numNails;

		x = (1.0 + sin(angle)) * (DATA_SIZE-1)/2.0;
		y = (1.0 + cos(angle)) * (DATA_SIZE-1)/2.0;

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
