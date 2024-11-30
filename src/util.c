#include <stdint.h> // for uint64_t

#include "util.h"
#include "settings.h"

// Bit array to track which nails are connected by lines
static uint64_t connections[LINE_BIT_ARRAY_SIZE] = {0};

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
int ValidateNextNail(int first, int next, int thresh) {
	int diff_direct, diff_wrap, diff_min;

	diff_direct = abs(next - first);
	diff_wrap = NUM_NAILS - diff_direct;
	diff_min = diff_direct < diff_wrap ? diff_direct : diff_wrap;

	if ((diff_min <= thresh) || IsConnected(first, next))
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

	// Calculate 1/sqrt(A^2 + B^2) for later efficient use
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

// Clear all recorded connections, and preset invalid links
void ResetConnections(void) {
	int i;

	// Clear all connections
	for (i=0; i<LINE_BIT_ARRAY_SIZE; i++)
		connections[i] = 0;

	// Set points as connected to themselves
	for (i=0; i<NUM_NAILS; i++)
		SetConnection(i, i);
}

// Mark a connection (in both directions)
void SetConnection(int i, int j) {
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
void ClearConnection(int i, int j) {
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
int IsConnected(int i, int j) {
	// Prevent out of array access
	if ((i >= NUM_NAILS) || (j >= NUM_NAILS))
		return 1; // Say out of range points are connected

	int bit_index = get_bit_index(i, j);
	return (connections[bit_index / 64] & ((uint64_t)1 << (bit_index % 64))) ? 1 : 0;
}
