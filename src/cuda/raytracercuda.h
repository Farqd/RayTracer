#ifndef COMMON_RAYTRACERCUDA_H
#define COMMON_RAYTRACERCUDA_H

#include "common/raytracer.h"

class RayTracerCuda : public RayTracer
{
  RGB background = {100, 100, 200};

public:
  void processPixelsCuda();
};


#endif // COMMON_RAYTRACERCUDA_H
