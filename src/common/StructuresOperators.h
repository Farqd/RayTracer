#ifndef COMMON_STRUCTURES_OPERATORS_H
#define COMMON_STRUCTURES_OPERATORS_H

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>

#include "common/Structures.h"

inline RGB operator*(RGB rgb, float times)
{
  rgb.r *= times;
  rgb.g *= times;
  rgb.b *= times;

  return rgb;
}

std::ostream& operator<<(std::ostream& outs, RGB rgb);

std::ostream& operator<<(std::ostream& outs, Point const& point);

std::ostream& operator<<(std::ostream& outs, Segment const& segment);

std::ostream& operator<<(std::ostream& outs, Sphere const& sphere);

#endif // COMMON_STRUCTURES_OPERATORS_H
