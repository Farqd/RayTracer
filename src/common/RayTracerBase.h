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
    , bitmap(config.imageY * config.antiAliasing, config.imageZ * config.antiAliasing)
  {
  }

  RayTracerConfig config;
  DynamicArray2D<RGB> bitmap;

  void printBitmap(std::ostream& out);
};

#endif // COMMON_RAYTRACERBASE_H
