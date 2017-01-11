#ifndef COMMON_STRUCTURES_H
#define COMMON_STRUCTURES_H

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>

struct RGB
{
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

struct Point
{
  double x;
  double y;
  double z;
};

using Vector = Point;

struct Sphere
{
  Point center;
  double radius;
  RGB color;
  double reflectionCoefficient = 0.0;
};

struct Segment
{
  Point a;
  Point b;
};

// p * normal + d = 0
struct Plane
{
  Point P;
  Vector normal;
  double d;
  RGB color;
  double reflectionCoefficient = 0.0;
};

#endif // COMMON_STRUCTURES_H
