#include "common/StructuresOperators.h"

#include <iostream>

std::ostream& operator<<(std::ostream& outs, RGB rgb)
{
  outs << static_cast<int>(rgb.r) << " " << static_cast<int>(rgb.g) << " "
       << static_cast<int>(rgb.b);
  return outs;
}

std::ostream& operator<<(std::ostream& outs, Point const& point)
{
  outs << "{ " << point.x << " " << point.y << " " << point.z << "} ";
  return outs;
}

std::ostream& operator<<(std::ostream& outs, Segment const& segment)
{
  outs << "A: " << segment.a << " B: " << segment.b;
  return outs;
}

std::ostream& operator<<(std::ostream& outs, Sphere const& sphere)
{
  outs << "Center: " << sphere.center << " R: " << sphere.radius;
  return outs;
}
