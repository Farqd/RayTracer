#include "Utils.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "common/Structures.h"

float dotProduct(Vector const& a, Vector const& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

float vectorLen(Vector const& vec)
{
  return std::sqrt(dotProduct(vec, vec));
}

Vector normalize(Vector const& vec)
{
  return vec / vectorLen(vec);
}

float distance(Point const& a, Point const& b)
{
  return vectorLen(b - a);
}

std::pair<bool, Point> intersection(Segment const& segment, Sphere const& sphere)
{
  float x0 = segment.a.x;
  float y0 = segment.a.y;
  float z0 = segment.a.z;

  float x1 = segment.b.x;
  float y1 = segment.b.y;
  float z1 = segment.b.z;

  float dx = x1 - x0;
  float dy = y1 - y0;
  float dz = z1 - z0;

  float cx = sphere.center.x;
  float cy = sphere.center.y;
  float cz = sphere.center.z;

  float a = dx * dx + dy * dy + dz * dz;
  float b = 2 * dx * (x0 - cx) + 2 * dy * (y0 - cy) + 2 * dz * (z0 - cz);
  float c = cx * cx + cy * cy + cz * cz + x0 * x0 + y0 * y0 + z0 * z0
            - 2 * (cx * x0 + cy * y0 + cz * z0) - sphere.radius * sphere.radius;

  float discriminant = b * b - 4 * a * c;
  if (!isCloseToZero(discriminant) && discriminant < 0)
    return {false, {}};

  float t = (-b - std::sqrt(discriminant)) / (2 * a);
  if (t < 0)
    return {false, {}};

  return {true, {x0 + t * dx, y0 + t * dy, z0 + t * dz}};
}

Segment reflection(Segment const& segment, Sphere const& sphere)
{
  Point normalVector = normalize((segment.b - sphere.center) / sphere.radius);

  Vector ri = normalize(segment.b - segment.a);
  float dot = dotProduct(ri, normalVector);
  ri -= normalVector * (2 * dot);

  return {segment.b, segment.b + ri};
}

std::pair<bool, Point> intersection(Segment const& segment, Plane const& plane)
{
  Vector V = segment.b - segment.a;
  float x = dotProduct(V, plane.normal);
  if (x == 0)
    return {false, {}};

  float t = -(dotProduct(segment.a, plane.normal) + plane.d) / x;
  if (t < 0 || isCloseToZero(t))
    return {false, {}};

  return {true, segment.a + V * t};
}

Segment reflection(Segment const& segment, Plane const& plane)
{
  Vector ri = segment.b - segment.a;
  ri -= plane.normal * (2 * dotProduct(ri, plane.normal));
  return {segment.b, segment.b + ri};
}


// TODO

std::pair<bool, Point> intersection(Segment const& segment, Triangle const& triangle)
{
  return {};
}

// We assume point is on triangle
RGB colorOfPoint(Point const& point, Triangle const& triangle)
{
  return {};
}

bool intersection(Segment const& segment, BoundingBox const& box)
{
  return false;
}