#ifndef COMMON_STRUCTURES_OPERATORS_H
#define COMMON_STRUCTURES_OPERATORS_H

#include <iostream>

#include "common/Structures.h"

inline Point& operator/=(Point& p, float value)
{
  p.x /= value;
  p.y /= value;
  p.z /= value;
  return p;
}

inline Point operator/(Point p, float value)
{
  return p /= value;
}

inline Point& operator*=(Point& p, float value)
{
  p.x *= value;
  p.y *= value;
  p.z *= value;
  return p;
}

inline Point operator*(Point p, float value)
{
  return p *= value;
}

inline Point& operator-=(Point& a, Point const& b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

inline Point operator-(Point a, Point const& b)
{
  return a -= b;
}

inline Point& operator+=(Point& a, Point const& b)
{
  a.y += b.y;
  a.z += b.z;
  a.x += b.x;
  return a;
}

inline Point operator+(Point a, Point const& b)
{
  return a += b;
}

inline RGB& operator*=(RGB& rgb, float times)
{
  rgb.r *= times;
  rgb.g *= times;
  rgb.b *= times;
  return rgb;
}

inline RGB operator*(RGB rgb, float times)
{
  return rgb *= times;
}

inline RGB& operator+=(RGB& lhs, RGB rhs)
{
  lhs.r += rhs.r;
  lhs.g += rhs.g;
  lhs.b += rhs.b;
  return lhs;
}

inline RGB operator+(RGB lhs, RGB rhs)
{
  return lhs += rhs;
}

std::ostream& operator<<(std::ostream& outs, RGB rgb);

std::ostream& operator<<(std::ostream& outs, Point const& point);

std::ostream& operator<<(std::ostream& outs, Segment const& segment);

std::ostream& operator<<(std::ostream& outs, Sphere const& sphere);

std::ostream& operator<<(std::ostream& outs, Plane const& plane);

std::ostream& operator<<(std::ostream& outs, BoundingBox const& bb);

#endif // COMMON_STRUCTURES_OPERATORS_H
