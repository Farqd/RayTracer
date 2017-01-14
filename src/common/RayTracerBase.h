#ifndef COMMON_RAYTRACERBASE_H
#define COMMON_RAYTRACERBASE_H

#include <iostream>
#include <vector>

#include "common/DynamicArray2D.h"
#include "common/RayTracerConfig.h"
#include "common/Structures.h"

struct RayTracerBase
{
  RayTracerBase(RayTracerConfig const& config)
    : config(config)
    , bitmap(2 * config.imageY, 2 * config.imageZ)
  {
  }

  RayTracerConfig const config;
  DynamicArray2D<RGB> bitmap;
  std::vector<Sphere> spheres;
  std::vector<Plane> planes;

  void printBitmap(std::ostream& out);
};

#endif // COMMON_RAYTRACERBASE_H
