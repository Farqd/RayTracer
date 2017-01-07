#ifndef STRUCTURES_H
#define STRUCTURES_H

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

inline RGB operator*(RGB rgb, double const& times)
{
  rgb.r *= times;
  rgb.g *= times;
  rgb.b *= times;

  return rgb;
}

std::ostream& operator<<(std::ostream& outs, RGB const& rgb);

struct Point
{
  double x;
  double y;
  double z;
};

using Vector = Point;

std::ostream& operator<<(std::ostream& outs, Point const& point);

struct Sphere
{
  Point center;
  double radius;
  RGB color;
};
std::ostream& operator<<(std::ostream& outs, Sphere const& sphere);

struct Segment
{
  Point a;
  Point b;
};
std::ostream& operator<<(std::ostream& outs, Segment const& segment);

// p * normal + d = 0
struct Plane
{
  Point P;
  Vector normal;
  double d;
  RGB color;
};

template <typename T>
bool isCloseToZero(T x)
{
  return std::abs(x) < std::numeric_limits<T>::epsilon();
}



#endif
