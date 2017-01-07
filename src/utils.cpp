#include "utils.h"

#include "structures.h"

#include <vector>
#include <algorithm> 
#include <cmath>

double vectorlen(Vector const& vec)
{
  return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

double dotProduct(Vector const& a, Vector const& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

bool pointInShadow(Point const& point, Point const& light, Sphere const& sphere)
{
  Segment seg = {point, light};
  auto const& res = intersection(seg, sphere);
  if(res.first && distance(point, res.second) < distance(point, light))
    return true;
  return false;
}

bool pointInShadow(Point const& point, Point const& light, Plane const& plane)
{
  Segment seg = {point, light};
  auto const& res = intersection(seg, plane);
  if(res.first && distance(point, res.second) < distance(point, light))
    return true;
  return false;
}

void normalize(Vector& vec)
{
  double len = vectorlen(vec);
  vec.x = vec.x / len;
  vec.y = vec.y / len;
  vec.z = vec.z / len;
}

double distance(Point const& a, Point const& b)
{
  return std::sqrt(std::pow(b.x-a.x, 2) + std::pow(b.y-a.y, 2) + std::pow(b.z-a.z, 2));
}

std::pair<bool, Point> intersection(Segment segment, Sphere sphere)
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

  return {true, {x0 + t * dx, y0 + t * dy, z0 + t * dz}};
}

std::pair<bool, Point> intersection(Segment segment, Plane plane)
{
  Vector V = {segment.b.x - segment.a.x, segment.b.y - segment.a.y, segment.b.z - segment.a.z};
  double x = dotProduct(V, plane.normal);
  if(x == 0)
    return {false, {}};

  double t = -(dotProduct(segment.a, plane.normal) + plane.d)/x;
  //std::cerr<<"T "<<t<<std::endl;
  if(t < 0 || isCloseToZero(t))
    return {false, {}};

  Point result;
  result.x = segment.a.x + t*V.x;
  result.y = segment.a.y + t*V.y;
  result.z = segment.a.z + t*V.z;
  
  return {true, result};
}
