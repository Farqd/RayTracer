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
  float x;
  float y;
  float z;
};

using Vector = Point;

struct Sphere
{
  Point center;
  float radius = 0;
  RGB color;
  float reflectionCoefficient = 0;
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
  float d = 0;
  RGB color;
  float reflectionCoefficient = 0;
};

struct Triangle
{
  Point x;
  Point y;
  Point z;
  RGB colorX;
  RGB colorY;
  RGB colorZ;
};

struct BoundingBox
{
  Point vMin;
  Point vMax;
};
#endif // COMMON_STRUCTURES_H
