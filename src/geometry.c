//#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "geometry.h"

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
