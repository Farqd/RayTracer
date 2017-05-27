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
  bool operator==(Point const& other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct IntersecRes
{
  bool exists;
  Point point;
  float t;
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
  RGB colorX = {};
  RGB colorY = {};
  RGB colorZ = {};
  Vector normal = {};
  float reflectionCoefficient = 0.1;
  bool operator==(Triangle const& other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct BoundingBox
{
  Point vMin;
  Point vMax;
};

struct BaseConfig
{
  int antiAliasing = 2;
  int maxRecursionLevel = 1;
  float ambientCoefficient = 0.1;

  int imageX;
  int imageY;
  int imageZ;
  Point observer;
  Point light;
  RGB background = {0, 0, 0};
};

#endif // COMMON_STRUCTURES_H
