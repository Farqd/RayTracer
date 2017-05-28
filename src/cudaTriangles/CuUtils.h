#ifndef CUDA_TRIANGLES_CUUTILS_H
#define CUDA_TRIANGLES_CUUTILS_H

#include <cfloat>
#include <cstdint>
#include <cstdio>

#include <device_launch_parameters.h>
#include <host_defines.h>
#include <math_functions.h>

#include "common/Structures.h"

__device__ inline Point& operator/=(Point& p, float value)
{
  p.x /= value;
  p.y /= value;
  p.z /= value;
  return p;
}

__device__ inline Point operator/(Point p, float value)
{
  return p /= value;
}

__device__ inline Point& operator*=(Point& p, float value)
{
  p.x *= value;
  p.y *= value;
  p.z *= value;
  return p;
}

__device__ inline Point operator*(Point p, float value)
{
  return p *= value;
}

__device__ inline Point& operator-=(Point& a, Point const& b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

__device__ inline Point operator-(Point a, Point const& b)
{
  return a -= b;
}

__device__ inline Point& operator+=(Point& a, Point const& b)
{
  a.y += b.y;
  a.z += b.z;
  a.x += b.x;
  return a;
}

__device__ inline Point operator+(Point a, Point const& b)
{
  return a += b;
}

__device__ inline RGB& operator*=(RGB& rgb, float times)
{
  rgb.r *= times;
  rgb.g *= times;
  rgb.b *= times;
  return rgb;
}

__device__ inline RGB operator*(RGB rgb, float times)
{
  return rgb *= times;
}

__device__ inline RGB& operator+=(RGB& lhs, RGB rhs)
{
  lhs.r += rhs.r;
  lhs.g += rhs.g;
  lhs.b += rhs.b;
  return lhs;
}

__device__ inline RGB operator+(RGB lhs, RGB rhs)
{
  return lhs += rhs;
}

extern "C" {

__device__ bool isCloseToZero(float x)
{
  return abs(x) < DBL_EPSILON;
}

__device__ float distance(Point const& a, Point const& b)
{
  return sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2) + pow(b.z - a.z, 2));
}

__device__ float vectorLen(Point const& vec)
{
  return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__device__ float dotProduct(Point const& a, Point const& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ Vector crossProduct(Vector const& a, Vector const& b)
{
  Vector vec;
  vec.x = a.y * b.z - b.y * a.z;
  vec.y = a.z * b.x - a.x * b.z;
  vec.z = a.x * b.y - a.y * b.x;
  return vec;
}

__device__ Point normalize(Point vec)
{
  float len = vectorLen(vec);
  vec.x = vec.x / len;
  vec.y = vec.y / len;
  vec.z = vec.z / len;
  return vec;
}

__device__ bool intersectsBoundingBoxN(Segment const& segment, BoundingBox const& box)
{
  Vector dir = segment.b - segment.a;
  float dirfracX = 1.0f / dir.x;
  float dirfracY = 1.0f / dir.y;
  float dirfracZ = 1.0f / dir.z;

  float t1 = (box.vMin.x - segment.a.x) * dirfracX;
  float t2 = (box.vMax.x - segment.a.x) * dirfracX;
  float t3 = (box.vMin.y - segment.a.y) * dirfracY;
  float t4 = (box.vMax.y - segment.a.y) * dirfracY;
  float t5 = (box.vMin.z - segment.a.z) * dirfracZ;
  float t6 = (box.vMax.z - segment.a.z) * dirfracZ;

  float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
  float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

  return 0 <= tmax && tmin <= tmax;
}
__device__ bool intersectsBoundingBox(Segment const& segment, BoundingBox const& box)
{
  Vector dir = normalize(segment.b - segment.a);
  float dirfracX = 1.0f / dir.x;
  float dirfracY = 1.0f / dir.y;
  float dirfracZ = 1.0f / dir.z;

  float t1 = (box.vMin.x - segment.a.x) * dirfracX;
  float t2 = (box.vMax.x - segment.a.x) * dirfracX;
  float t3 = (box.vMin.y - segment.a.y) * dirfracY;
  float t4 = (box.vMax.y - segment.a.y) * dirfracY;
  float t5 = (box.vMin.z - segment.a.z) * dirfracZ;
  float t6 = (box.vMax.z - segment.a.z) * dirfracZ;

  float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
  float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

  return 0 <= tmax && tmin <= tmax;
}

__device__ RGB calculateColorFromReflection(RGB currentColor, RGB reflectedColor,
                                            float reflectionCoefficient)
{
  return currentColor * (1.0f - reflectionCoefficient) + reflectedColor * reflectionCoefficient;
}

// We assume point is on triangle
__device__ RGB colorOfPoint(Point const& point, Triangle const& triangle)
{
  float u;
  float v;

  Point v0 = triangle.x;
  Point v1 = triangle.y;
  Point v2 = triangle.z;

  Vector v0v1 = v1 - v0;
  Vector v0v2 = v2 - v0;

  Vector N = crossProduct(v0v1, v0v2);
  float denom = dotProduct(N, N);

  Vector C;

  Vector edge1 = v2 - v1;
  Vector vp1 = point - v1;
  C = crossProduct(edge1, vp1);
  u = dotProduct(N, C);

  Vector edge2 = v0 - v2;
  Vector vp2 = point - v2;
  C = crossProduct(edge2, vp2);
  v = dotProduct(N, C);

  u /= denom;
  v /= denom;

  return triangle.colorX * u + triangle.colorY * v + triangle.colorZ * (1 - u - v);
}

__device__ Segment reflection(Segment const& segment, Triangle const& triangle)
{
  Vector ri = segment.b - segment.a;
  

  Vector N = triangle.normal;

  ri -= N * (2 * dotProduct(ri, N));
  return {segment.b, segment.b + ri};
}

struct pairip
{
  int first;
  Point second;
};

struct IntersectionResult
{
  bool intersects;
  Point intersectionPoint;
};

__device__ IntersectionResult intersection(Segment const& segment, Triangle const& triangle)
{
  Vector const& V1 = triangle.x;
  Vector const& V2 = triangle.y;
  Vector const& V3 = triangle.z;

  Vector const& O = segment.a;
  Vector const& D = normalize(segment.b - segment.a);

  Vector e1, e2;
  Vector P, Q, T;
  float det, inv_det, u, v;

  e1 = V2 - V1;
  e2 = V3 - V1;

  P = crossProduct(D, e2);
  det = dotProduct(e1, P);

  if (isCloseToZero(det))
    return {false, {}};

  inv_det = 1.f / det;
  T = O - V1;
  u = dotProduct(T, P) * inv_det;
  if (u < 0.f || u > 1.f)
    return {false, {}};

  Q = crossProduct(T, e1);
  v = dotProduct(D, Q) * inv_det;

  if (v < 0.f || u + v > 1.f)
    return {false, {}};

  float t = dotProduct(e2, Q) * inv_det;
  if (t > FLT_EPSILON)
  {
    Point res = segment.a + D * t;
    return {true, res};
  }

  return {false, {}};
}
__device__ IntersecRes intersectionT(Segment const& segment, Triangle const& triangle)
{
  Vector const& V1 = triangle.x;
  Vector const& V2 = triangle.y;
  Vector const& V3 = triangle.z;

  Vector const& O = segment.a;
  Vector const& D = normalize(segment.b - segment.a);

  Vector e1, e2;
  Vector P, Q, T;
  float det, inv_det, u, v;

  e1 = V2 - V1;
  e2 = V3 - V1;

  P = crossProduct(D, e2);
  det = dotProduct(e1, P);

  if (isCloseToZero(det))
    return {false, {}};

  inv_det = 1.f / det;
  T = O - V1;
  u = dotProduct(T, P) * inv_det;
  if (u < 0.f || u > 1.f)
    return {false, {}};

  Q = crossProduct(T, e1);
  v = dotProduct(D, Q) * inv_det;

  if (v < 0.f || u + v > 1.f)
    return {false, {}};

  float t = dotProduct(e2, Q) * inv_det;
  if (t > FLT_EPSILON)
  {
    Point res = segment.a + D * t;
    return {true, res, t};
  }

  return {false, {}};
}

__device__ IntersectionResult intersectionN(Segment const& segment, Triangle const& triangle)
{
  Vector const& V1 = triangle.x;
  Vector const& V2 = triangle.y;
  Vector const& V3 = triangle.z;

  Vector const& O = segment.a;
  Vector const& D = segment.b - segment.a;

  Vector e1, e2;
  Vector P, Q, T;
  float det, inv_det, u, v;

  e1 = V2 - V1;
  e2 = V3 - V1;

  P = crossProduct(D, e2);
  det = dotProduct(e1, P);

  if (isCloseToZero(det))
    return {false, {}};

  inv_det = 1.f / det;
  T = O - V1;
  u = dotProduct(T, P) * inv_det;
  if (u < 0.f || u > 1.f)
    return {false, {}};

  Q = crossProduct(T, e1);
  v = dotProduct(D, Q) * inv_det;

  if (v < 0.f || u + v > 1.f)
    return {false, {}};

  float t = dotProduct(e2, Q) * inv_det;
  if (t > FLT_EPSILON)
  {
    Point res = segment.a + D * t;
    return {true, res};
  }

  return {false, {}};
}
};
#endif // CUDA_TRIANGLES_CUUTILS_H
