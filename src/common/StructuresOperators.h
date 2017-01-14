#ifndef COMMON_STRUCTURES_OPERATORS_H
#define COMMON_STRUCTURES_OPERATORS_H

#include <iostream>

#include "common/Structures.h"

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

#endif // COMMON_STRUCTURES_OPERATORS_H
