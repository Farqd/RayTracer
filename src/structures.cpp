#include "structures.h"

#include "utils.h"

#include <cmath>
#include <limits>
#include <utility>

std::ostream& operator<<(std::ostream& outs, RGB const& rgb)
{
  outs << static_cast<int16_t>(rgb.r) << " " << static_cast<int16_t>(rgb.g) << " "
       << static_cast<int16_t>(rgb.b);
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
