#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <algorithm>
#include <vector>

#include "common/Structures.h"
#include "common/StructuresOperators.h"

float dotProduct(Vector const& a, Vector const& b);

float vectorLen(Vector const& vec);

Vector normalize(Vector const& vec);

float distance(Point const& a, Point const& b);

template <typename T>
bool pointInShadow(Point const& point, Point const& light, T const& object)
{
  Segment seg = {point, light};
  auto const& res = intersection(seg, object);
  return res.first && distance(point, res.second) < distance(point, light);
}

template <typename Iterator>
bool pointInShadow(Point const& point, Point const& light, Iterator begin, Iterator end)
{
  for (Iterator it = begin; it != end; ++it)
  {
    if (pointInShadow(point, light, *it))
      return true;
  }
  return false;
}

// Returns true if intersection exists
// If two points of intersection exist closest is returned
std::pair<bool, Point> intersection(Segment const& segment, Sphere const& sphere);

std::pair<bool, Point> intersection(Segment const& segment, Plane const& plane);

Segment reflection(Segment const& segment, Sphere const& sphere);

Segment reflection(Segment const& segment, Plane const& plane);

template <typename T>
bool isCloseToZero(T x)
{
  return std::abs(x) < std::numeric_limits<T>::epsilon();
}

template <typename T>
std::pair<int, Point> findClosestIntersection(std::vector<T> const& objects, Segment const& seg)
{
  Point closestPoint{};
  int sphereIndex = -1;
  float closestDistance = std::numeric_limits<float>::max();

  for (size_t i = 0; i < objects.size(); i++)
  {
    auto const& res = intersection(seg, objects[i]);
    if (!res.first)
      continue;

    float dist = distance(seg.a, res.second);
    if (dist < closestDistance)
    {
      closestDistance = dist;
      closestPoint = res.second;
      sphereIndex = i;
    }
  }
  return {sphereIndex, closestPoint};
}

inline RGB calculateColorFromReflection(RGB currentColor, RGB reflectedColor,
                                        float reflectionCoefficient)
{
  return currentColor * (1.0f - reflectionCoefficient) + reflectedColor * reflectionCoefficient;
}


// New functions, consider changing return types if others are more convinient
std::pair<bool, Point> intersection(Segment const& segment, Triangle const& triangle);

// We assume point is on triangle
RGB colorOfPoint(Point const& point, Triangle const& triangle);

bool intersection(Segment const& segment, BoundingBox const& box);


#endif // COMMON_UTILS_H
