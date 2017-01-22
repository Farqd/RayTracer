#ifndef COMMON_RAYTRACERCONFIG_H
#define COMMON_RAYTRACERCONFIG_H

#include <vector>

#include "common/Structures.h"

struct RayTracerConfig
{
  int antiAliasing = 2;
  int maxRecursionLevel = 1;
  float ambientCoefficient = 0.1;

  int imageX;
  int imageY;
  int imageZ;
  Point observer;
  Point light;

  std::vector<Sphere> spheres;
  std::vector<Plane> planes;
  std::vector<Triangle> triangles;

  static RayTracerConfig fromFile(std::string const& path);
  static RayTracerConfig fromPlyFile(std::string const& path);
  static RayTracerConfig defaultConfig();
  void scaleTriangles();

  friend std::ostream& operator<<(std::ostream& out, RayTracerConfig const& config);
};

#endif // COMMON_RAYTRACERCONFIG_H
