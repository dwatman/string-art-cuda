#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include "util.h"

double line_area_fraction(double A, double B, double C, double t);
int inside_poly(point_t *vert, int nvert, float testx, float testy);

#endif
