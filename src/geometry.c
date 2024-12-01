//#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "geometry.h"
#include "settings.h"

// Global variable to store the centroid for sorting
static pointd_t centroid;

// Helper function to check if a point is within square bounds
int in_square_bounds(double x, double y) {
	return (x >= -0.5 && x <= 0.5 && y >= -0.5 && y <= 0.5);
}

// Check if a point (x, y) is within the thick line region
int point_in_thick_line(double A, double B, double C, double t, double x, double y) {
	double distance = fabs(A * x + B * y + C) / sqrt(A * A + B * B);
	return distance <= t / 2;
}

// Check intersection of a line with each edge of the square
void line_square_intersections(double A, double B, double C, pointd_t *intersections, int *count) {
	*count = 0;  // Reset count at the start
	double x, y;

	if (fabs(A) < 1e-6) {  // Horizontal line case (A ≈ 0)
		y = -C / B;  // Line equation for horizontal line (A = 0)
		if (in_square_bounds(-0.5, y)) intersections[(*count)++] = (pointd_t){-0.5, y};  // Left edge
		if (in_square_bounds(0.5, y)) intersections[(*count)++] = (pointd_t){0.5, y};    // Right edge
	}
	else if (fabs(B) < 1e-6) {  // Vertical line case (B ≈ 0)
		x = -C / A;  // Line equation for vertical line (B = 0)
		if (in_square_bounds(x, -0.5)) intersections[(*count)++] = (pointd_t){x, -0.5};  // Bottom edge
		if (in_square_bounds(x, 0.5)) intersections[(*count)++] = (pointd_t){x, 0.5};    // Top edge
	}
	else {  // General line case
		// Intersection with top edge (y = 0.5)
		x = -(B * 0.5 + C) / A;
		if (in_square_bounds(x, 0.5)) intersections[(*count)++] = (pointd_t){x, 0.5};

		// Intersection with bottom edge (y = -0.5)
		x = -(B * -0.5 + C) / A;
		if (in_square_bounds(x, -0.5)) intersections[(*count)++] = (pointd_t){x, -0.5};

		// Intersection with left edge (x = -0.5)
		y = -(A * -0.5 + C) / B;
		if (in_square_bounds(-0.5, y)) intersections[(*count)++] = (pointd_t){-0.5, y};

		// Intersection with right edge (x = 0.5)
		y = -(A * 0.5 + C) / B;
		if (in_square_bounds(0.5, y)) intersections[(*count)++] = (pointd_t){0.5, y};
	}
}

// Main function to find intersections for both offset lines
int find_intersections(double A, double B, double C, float t, pointd_t *intersections) {
	int count = 0;
	double norm = sqrt(A * A + B * B);

	// Offset distances for the thick line boundaries
	double C1 = C + (t * norm / 2);
	double C2 = C - (t * norm / 2);

	// Collect intersections for both boundary lines
	int current_count;
	line_square_intersections(A, B, C1, intersections, &current_count);
	count += current_count;
	// printf("count1 (C1 = %+.3f) %i\n", C1, current_count);

	line_square_intersections(A, B, C2, intersections + count, &current_count);
	count += current_count;
	// printf("count2 (C2 = %+.3f) %i\n", C2, current_count);

	// int i;
	// for (i=0; i<count; i++)
	// 	printf("    point %i: %+6.4f, %+6.4f\n", i, intersections[i].x, intersections[i].y);

	return count;
}

// Helper function to compute the centroid for sorting
pointd_t compute_centroid(pointd_t *points, int count) {
	pointd_t centroid = {0.0, 0.0};
	for (int i = 0; i < count; i++) {
		centroid.x += points[i].x;
		centroid.y += points[i].y;
	}
	centroid.x /= count;
	centroid.y /= count;
	return centroid;
}

// Comparator function for qsort
int compare_points(const void *a, const void *b) {
	pointd_t *p1 = (pointd_t *)a;
	pointd_t *p2 = (pointd_t *)b;

	double angle1 = atan2(p1->y - centroid.y, p1->x - centroid.x);
	double angle2 = atan2(p2->y - centroid.y, p2->x - centroid.x);

	return (angle1 > angle2) - (angle1 < angle2);  // Ascending order of angle
}

// Sort the points in counterclockwise order
void sort_points(pointd_t *points, int count) {
	centroid = compute_centroid(points, count);  // Set global centroid for sorting
	qsort(points, count, sizeof(pointd_t), compare_points);
}

// Calculate the area of a polygon using the shoelace formula
double polygon_area(pointd_t *points, int count) {
	double area = 0.0;
	for (int i = 0; i < count; i++) {
		int j = (i + 1) % count;  // Next vertex (wraps around)
		area += points[i].x * points[j].y - points[i].y * points[j].x;
	}
	return fabs(area) / 2.0;
}

// Function to determine the relevant corner for a triangle formation
void find_square_corner(pointd_t *points, int *count) {
	// Calculate the approximate midpoint between the two intersection points
	double mid_x = (points[0].x + points[1].x) / 2;
	double mid_y = (points[0].y + points[1].y) / 2;

	// Determine which corner of the square is closest to the midpoint
	pointd_t corner;
	corner.x = (mid_x > 0) ? 0.5 : -0.5;
	corner.y = (mid_y > 0) ? 0.5 : -0.5;

	// Add the corner point to complete the triangle
	points[(*count)++] = corner;
}

// Calculate the area of a unit square covered by a line
float line_area_fraction(line_t line, float t) {
	float area;
	pointd_t intersections[8]; // Maximum number of intersections possible is 8

	int count = find_intersections(line.A, line.B, line.C, t, intersections);
	//printf("%i intersections\n", count);

	// Check each corner of the square
	pointd_t corners[4] = {
		{-0.5, -0.5},
		{0.5, -0.5},
		{0.5, 0.5},
		{-0.5, 0.5}
	};
	int corners_in_line = 0;
	for (int i = 0; i < 4; i++) {
		if (point_in_thick_line(line.A, line.B, line.C, t, corners[i].x, corners[i].y)) {
			intersections[count++] = corners[i];  // Include corner in polygon
			corners_in_line++;
		}
	}

	// If all four corners are within the thick line, the area is the entire square
	if (corners_in_line == 4) {
		return 1.0f;
	}

	// Sort points in counterclockwise order to form a polygon
	sort_points(intersections, count);

	// Calculate the area of the polygon formed by intersection points
	area = polygon_area(intersections, count);
	//printf("area = %f\n", area);

	return area;
}

// Check if a point is inside a polygon
// From https://wrfranklin.org/Research/Short_Notes/pnpoly.html
int inside_poly(point_t *vert, int nvert, float testx, float testy) {
	int i, j, inside = 0;

	for (i = 0, j = nvert-1; i < nvert; j = i++) {
		if ( ((vert[i].y>testy) != (vert[j].y>testy)) &&
		(testx < (vert[j].x-vert[i].x) * (testy-vert[i].y) / (vert[j].y-vert[i].y) + vert[i].x) )
			inside = !inside;
	}

	return inside;
}

// Calculate the coverage of a line over a unit square at various angles and distances
// X axis represents the angle between 0 and pi
// Y axis represents the distance from the centre of the pixel to the corner distance + thickness/2
void CalcLineCoverage(float *map, float lineWidth) {
	int i, j;
	float angle, dist, area, maxval;
	line_t line;

	float maxAngle = M_PI;
	float maxDist = sqrt(2)/2 + lineWidth/2;

	if (map == NULL) {
		printf("ERROR in CalcLineCoverage, map not initialised\n");
		return;
	}

	maxval = 0;
	for (j=0; j<LINE_TEX_DIST_SAMPLES; j++) {
		for (i=0; i<LINE_TEX_ANGLE_SAMPLES; i++) {
			angle = ((float)i/LINE_TEX_ANGLE_SAMPLES)*maxAngle;
			dist = ((float)j/LINE_TEX_DIST_SAMPLES)*maxDist;

			//printf("d %f, a %f\n", dist, angle);
			line = DistAngleToLine(dist, angle);
			area = line_area_fraction(line, lineWidth);

			// Keep track of the maximum value
			if (area > maxval) maxval = area;

			map[LINE_TEX_ANGLE_SAMPLES*j + i] = area;
		}
	}
	printf("maxval %f\n", maxval);

}

// Generate a random (but valid) list of points, and calculate line parameters
// TODO: separate line calculation?
int GenerateRandomPattern(line_t *lines, int *pointList, const point_t *nails) {
	point_t p0, p1;
	int n0, n1;
	int i;

	// Reset the map of line connections between nails
	ResetConnections();

	// Select first nail
	pointList[0] = rand() % NUM_NAILS;

	// Create lines
	for (i=0; i<NUM_LINES; i++) {
		int retries = 0;

		// Select next nail
		pointList[i+1] = rand() % NUM_NAILS;

		// If the selected nail is not valid (too close, already connected)
		// Choose another until the limit is reached or a suitable nail is found
		while ((retries < RETRY_LIMIT) && (ValidateNextNail(pointList[i], pointList[i+1], MIN_LINE_DIST) == 0)) {
			retries++;
			pointList[i+1] = rand() % NUM_NAILS;
			//printf("Retry %u: %u -> %u\n", retries, pointList[i], pointList[i+1]);
		}

		if (retries == RETRY_LIMIT) {
			printf("ERROR: Retry limit reached for line %u\n", i);
			break;
		}

		SetConnection(pointList[i], pointList[i+1]);

		p0.x = nails[pointList[i]].x;
		p0.y = nails[pointList[i]].y;
		p1.x = nails[pointList[i+1]].x;
		p1.y = nails[pointList[i+1]].y;
		lines[i] = PointsToLine(p0, p1);

		printf("step #%i %3u to %3u\n", i, pointList[i], pointList[i+1]);
		//printf("Line (%5.1f, %5.1f)-(%5.1f, %5.1f)", p0.x, p0.y, p1.x, p1.y);
		//printf(" -> %f %f %f (%f)\n", lines[i].A, lines[i].B, lines[i].C, lines[i].inv_denom);
	}

	// Check if lines were unable to be completed
	if (i < NUM_LINES) {
		printf("ERROR: Unable to initialise all lines\n");
		return -1;
	}

	// for (i=0; i<=NUM_LINES; i++) {
	// 	printf("Point %3u = %3u\n", i, pointList[i]);
	// }
	return 0;
}

// Ensure an index is between 0 and NUM_NAILS
static int FixNailIndex(int p) {
	if (p < 0) return p + NUM_NAILS;
	if (p >= NUM_NAILS) return p - NUM_LINES;
	return p;
}

// Move the line(s) at a nail to another vaild nail within maxDist
int MovePattern(line_t *lines, int *pointList, const point_t *nails, int maxDist) {
	int pointFrom, pointTo;
	int changeDist;
	int i;

	printf("        ");
	for (i=0; i<=NUM_LINES; i++)
		printf("%2i ", i);
	printf("\npoints: ");
	for (i=0; i<=NUM_LINES; i++)
		printf("%2i ", pointList[i]);
	printf("\n");

	// Select a point to move
	pointFrom = rand() % (NUM_LINES+1);

	// Decide how far to move the point (1 to maxDist)
	changeDist = 1 + rand() % maxDist;

	// Choose sign
	if (rand() % 2) changeDist *= -1;

	pointTo = FixNailIndex(pointList[pointFrom] + changeDist);

	printf("point #%i from %i by %+i to %i\n", pointFrom, pointList[pointFrom], changeDist, pointTo);
	printf("nail %i to %i\n", pointList[pointFrom], pointTo);

	if (pointFrom == 0) {
		printf("FIRST\n");
		printf("line %i to %i -> ", pointList[0], pointList[1]);
		printf("line %i to %i\n", pointTo, pointList[1]);

		if (ValidateNextNail(pointTo, pointList[1], MIN_LINE_DIST) == 0)
			printf("INVALID\n");
		else
			printf("VALID\n");
	}
	else if (pointFrom == NUM_LINES) {
		printf("LAST\n");
		printf("line %i to %i -> ", pointList[NUM_LINES-1], pointList[NUM_LINES]);
		printf("line %i to %i\n", pointList[NUM_LINES-1], pointTo);

		if (ValidateNextNail(pointList[NUM_LINES-1], pointTo, MIN_LINE_DIST) == 0)
			printf("INVALID\n");
		else
			printf("VALID\n");
	}
	else {
		printf("MID\n");
		printf("line %i to %i -> ", pointList[pointFrom-1], pointList[pointFrom]);
		printf("line %i to %i\n", pointList[pointFrom-1], pointTo);

		if (ValidateNextNail(pointList[pointFrom-1], pointTo, MIN_LINE_DIST) == 0)
			printf("INVALID\n");
		else
			printf("VALID\n");

		printf("line %i to %i -> ", pointList[pointFrom], pointList[pointFrom+1]);
		printf("line %i to %i\n", pointTo, pointList[pointFrom+1]);

		if (ValidateNextNail(pointTo, pointList[pointFrom+1], MIN_LINE_DIST) == 0)
			printf("INVALID\n");
		else
			printf("VALID\n");
	}



	return 0;
}
