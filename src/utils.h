#ifndef UTILS_H
#define UTILS_H

#include "structures.h"

#include <vector>
#include <algorithm> 

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

#endif // UTILS_H
