#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <algorithm>
#include <vector>

#include "common/structures.h"

float vectorlen(Vector const& vec);

float dotProduct(Vector const& a, Vector const& b);

bool pointInShadow(Point const& point, Point const& light, Sphere const& sphere);

bool pointInShadow(Point const& point, Point const& light, Plane const& plane);

void normalize(Vector& vec);

float distance(Point const& a, Point const& b);

// Returns true if intersection exists
// If two points of intersection exist closest is returned
std::pair<bool, Point> intersection(Segment segment, Sphere sphere);

std::pair<bool, Point> intersection(Segment segment, Plane plane);

Segment reflection(Segment const& segment, Sphere const& sphere);

Segment reflection(Segment const& segment, Plane const& plane);

template <typename T>
bool isCloseToZero(T x)
{
  return std::abs(x) < std::numeric_limits<T>::epsilon();
}

#endif // COMMON_UTILS_H
