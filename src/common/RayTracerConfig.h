#ifndef COMMON_RAYTRACERCONFIG_H
#define COMMON_RAYTRACERCONFIG_H

#include <vector>

#include "common/Structures.h"

struct RayTracerConfig : BaseConfig
{
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
