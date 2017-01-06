#include "structures.h"

#include <cmath>
#include <limits>
#include <utility>

std::pair<bool, std::pair<Point, double>> intersection(Segment segment, Sphere sphere)
{
  double x0 = segment.a.x;
  double y0 = segment.a.y;
  double z0 = segment.a.z;

  double x1 = segment.b.x;
  double y1 = segment.b.y;
  double z1 = segment.b.z;

  double dx = x1 - x0;
  double dy = y1 - y0;
  double dz = z1 - z0;

  double cx = sphere.center.x;
  double cy = sphere.center.y;
  double cz = sphere.center.z;

  double a = dx * dx + dy * dy + dz * dz;
  double b = 2 * dx * (x0 - cx) + 2 * dy * (y0 - cy) + 2 * dz * (z0 - cz);
  double c = cx * cx + cy * cy + cz * cz + x0 * x0 + y0 * y0 + z0 * z0
             - 2 * (cx * x0 + cy * y0 + cz * z0) - sphere.radius * sphere.radius;

  double discriminant = b * b - 4 * a * c;
  if (!isCloseToZero(discriminant) && discriminant < 0)
    return {false, {}};

  double t = (-b - std::sqrt(discriminant)) / (2 * a);
  if (t < 0)
    return {false, {}};
  return {true, {{x0 + t * dx, y0 + t * dy, z0 + t * dz}, t}};
}


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
