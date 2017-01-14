#include "common/Utils.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "common/Structures.h"

float vectorLen(Vector const& vec)
{
  return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

Vector normalize(Vector vec)
{
  float len = vectorLen(vec);
  vec.x = vec.x / len;
  vec.y = vec.y / len;
  vec.z = vec.z / len;
  return vec;
}

float dotProduct(Vector const& a, Vector const& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

float distance(Point const& a, Point const& b)
{
  return std::sqrt(std::pow(b.x - a.x, 2) + std::pow(b.y - a.y, 2) + std::pow(b.z - a.z, 2));
}

std::pair<bool, Point> intersection(Segment segment, Sphere sphere)
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
  Segment result;
  result.a = segment.b;
  Point normalVector = normalize({(segment.b.x - sphere.center.x) / sphere.radius,
                                  (segment.b.y - sphere.center.y) / sphere.radius,
                                  (segment.b.z - sphere.center.z) / sphere.radius});

  Vector ri =
      normalize({segment.b.x - segment.a.x, segment.b.y - segment.a.y, segment.b.z - segment.a.z});

  float dot = dotProduct(ri, normalVector);
  ri.x = ri.x - 2 * normalVector.x * dot;
  ri.y = ri.y - 2 * normalVector.y * dot;
  ri.z = ri.z - 2 * normalVector.z * dot;

  result.b.x = result.a.x + ri.x;
  result.b.y = result.a.y + ri.y;
  result.b.z = result.a.z + ri.z;

  return result;
}

std::pair<bool, Point> intersection(Segment segment, Plane plane)
{
  Vector V = {segment.b.x - segment.a.x, segment.b.y - segment.a.y, segment.b.z - segment.a.z};
  float x = dotProduct(V, plane.normal);
  if (x == 0)
    return {false, {}};

  float t = -(dotProduct(segment.a, plane.normal) + plane.d) / x;
  if (t < 0 || isCloseToZero(t))
    return {false, {}};

  Point result;
  result.x = segment.a.x + t * V.x;
  result.y = segment.a.y + t * V.y;
  result.z = segment.a.z + t * V.z;

  return {true, result};
}

Segment reflection(Segment const& segment, Plane const& plane)
{
  Segment result;
  result.a = segment.b;
  Point normalVector = plane.normal;

  Vector ri = {segment.b.x - segment.a.x, segment.b.y - segment.a.y, segment.b.z - segment.a.z};

  float dot = dotProduct(ri, normalVector);
  ri.x = ri.x - 2 * normalVector.x * dot;
  ri.y = ri.y - 2 * normalVector.y * dot;
  ri.z = ri.z - 2 * normalVector.z * dot;

  result.b.x = result.a.x + ri.x;
  result.b.y = result.a.y + ri.y;
  result.b.z = result.a.z + ri.z;

  return result;
}
