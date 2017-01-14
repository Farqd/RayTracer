#ifndef COMMON_RAYTRACERCONFIG_H
#define COMMON_RAYTRACERCONFIG_H

#include "common/Structures.h"

struct RayTracerConfig
{
  int antiAliasing = 2;
  int maxRecursionLevel = 1;
  float diffuseCoefficient = 0.9;
  float ambientCoefficient = 0.1;

  int imageX;
  int imageY;
  int imageZ;
  Point imageCenter;
  Point observer;
  Point light;
};

#endif // COMMON_RAYTRACERCONFIG_H
