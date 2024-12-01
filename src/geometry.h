#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include "util.h"

float line_area_fraction(line_t line, float t);
int inside_poly(point_t *vert, int nvert, float testx, float testy);
void CalcLineCoverage(float *map, float lineWidth);
int GenerateRandomPattern(line_t *lines, int *pointList, const point_t *nails);
int MovePattern(line_t *lines, int *pointList, const point_t *nails, int maxDist);

#endif
