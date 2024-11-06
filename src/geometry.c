//#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "geometry.h"

typedef struct {
	double x, y;
} Point;

// Global variable to store the centroid for sorting
Point centroid;

// Helper function to check if a point is within square bounds
int in_square_bounds(double x, double y) {
	return (x >= -0.5 && x <= 0.5 && y >= -0.5 && y <= 0.5);
}

// Check intersection of a line with each edge of the square
void line_square_intersections(double A, double B, double C, Point *intersections, int *count) {
	*count = 0;  // Reset count at the start
	double x, y;

	// Intersection with top edge (y = 0.5)
	if (fabs(B) > 1e-6) {  // Avoid division by zero
		x = -(B * 0.5 + C) / A;  // Solving for x when y = 0.5
		if (in_square_bounds(x, 0.5)) {
			intersections[(*count)++] = (Point){x, 0.5};
		}
	}

	// Intersection with bottom edge (y = -0.5)
	if (fabs(B) > 1e-6) {
		x = -(B * -0.5 + C) / A;  // Solving for x when y = -0.5
		if (in_square_bounds(x, -0.5)) {
			intersections[(*count)++] = (Point){x, -0.5};
		}
	}

	// Intersection with left edge (x = -0.5)
	if (fabs(A) > 1e-6) {  // Avoid division by zero
		y = -(A * -0.5 + C) / B;  // Solving for y when x = -0.5
		if (in_square_bounds(-0.5, y)) {
			intersections[(*count)++] = (Point){-0.5, y};
		}
	}

	// Intersection with right edge (x = 0.5)
	if (fabs(A) > 1e-6) {
		y = -(A * 0.5 + C) / B;  // Solving for y when x = 0.5
		if (in_square_bounds(0.5, y)) {
			intersections[(*count)++] = (Point){0.5, y};
		}
	}
}

// Main function to find intersections for both offset lines
int find_intersections(double A, double B, double C, double t, Point *intersections) {
	int count = 0;
	double norm = sqrt(A * A + B * B);

	// Offset distances for the thick line boundaries
	double C1 = C + (t * norm / 2);
	double C2 = C - (t * norm / 2);

	// Collect intersections for both boundary lines
	int current_count;
	line_square_intersections(A, B, C1, intersections, &current_count);
	count += current_count;
printf("count1 (C1 = %+.3f) %i\n", C1, current_count);

	line_square_intersections(A, B, C2, intersections + count, &current_count);
	count += current_count;
printf("count2 (C2 = %+.3f) %i\n", C2, current_count);

int i;
for (i=0; i<count; i++)
	printf("    point %i: %+6.4f, %+6.4f\n", i, intersections[i].x, intersections[i].y);

	return count;
}

// Helper function to compute the centroid for sorting
Point compute_centroid(Point *points, int count) {
	Point centroid = {0.0, 0.0};
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
	Point *p1 = (Point *)a;
	Point *p2 = (Point *)b;

	double angle1 = atan2(p1->y - centroid.y, p1->x - centroid.x);
	double angle2 = atan2(p2->y - centroid.y, p2->x - centroid.x);

	return (angle1 > angle2) - (angle1 < angle2);  // Ascending order of angle
}

// Sort the points in counterclockwise order
void sort_points(Point *points, int count) {
	centroid = compute_centroid(points, count);  // Set global centroid for sorting
	qsort(points, count, sizeof(Point), compare_points);
}

// Calculate the area of a polygon using the shoelace formula
double polygon_area(Point *points, int count) {
	double area = 0.0;
	for (int i = 0; i < count; i++) {
		int j = (i + 1) % count;  // Next vertex (wraps around)
		area += points[i].x * points[j].y - points[i].y * points[j].x;
	}
	return fabs(area) / 2.0;
}

// Function to determine the relevant corner for a triangle formation
void find_square_corner(Point *points, int *count) {
	// Calculate the approximate midpoint between the two intersection points
	double mid_x = (points[0].x + points[1].x) / 2;
	double mid_y = (points[0].y + points[1].y) / 2;

	// Determine which corner of the square is closest to the midpoint
	Point corner;
	corner.x = (mid_x > 0) ? 0.5 : -0.5;
	corner.y = (mid_y > 0) ? 0.5 : -0.5;

	// Add the corner point to complete the triangle
	points[(*count)++] = corner;
}

double compute_area_fraction(double A, double B, double C, double t) {
	// Maximum number of intersections possible is 8
	Point intersections[8];

	int count = find_intersections(A, B, C, t, intersections);
printf("%i intersections\n", count);


	// If fewer than 2 intersections, no area is covered
	if (count < 2) {
		return 0.0;
	}

	// If exactly 2 intersections, add the square corner to form a triangle
	if (count == 2) {
		find_square_corner(intersections, &count);  // Find the correct corner
	}

	// Sort points in counterclockwise order to form a polygon
	sort_points(intersections, count);

	// Calculate the area of the polygon formed by intersection points
	double area = polygon_area(intersections, count);
printf("area = %f\n", area);
	// Area fraction is the polygon area divided by the unit square area (1x1 square)
	return area;
}
