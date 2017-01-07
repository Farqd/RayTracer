#ifndef UTILS_H
#define UTILS_H

#include "structures.h"

#include <algorithm>
#include <vector>

double vectorlen(Vector const& vec);

double dotProduct(Vector const& a, Vector const& b);

bool pointInShadow(Point const& point, Point const& light, Sphere const& sphere);

bool pointInShadow(Point const& point, Point const& light, Plane const& plane);

void normalize(Vector& vec);

double distance(Point const& a, Point const& b);

// Returns true if intersection exists
// If two points of intersection exist closest is returned
std::pair<bool, Point> intersection(Segment segment, Sphere sphere);

std::pair<bool, Point> intersection(Segment segment, Plane plane);

Segment reflection(Segment const& segment, Sphere const& sphere);
Segment reflection(Segment const& segment, Plane const& plane);

#endif // UTILS_H
