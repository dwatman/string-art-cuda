#include "util.h"

line_t pointsToLine(point_t p1, point_t p2) {
	line_t line;

	// Calculate coefficients A, B, and C
	line.A = p2.y - p1.y;
	line.B = p1.x - p2.x;
	line.C = (p1.y * p2.x) - (p2.y * p1.x);

	// Calculate 1/sqrt(A^2 + B^2) for efficiency
	line.inv_denom = 1.0f / sqrtf(line.A*line.A + line.B*line.B);

	return line;
}
