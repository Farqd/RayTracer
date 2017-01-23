#ifndef COMMON_RAYTRACERCUDA_H
#define COMMON_RAYTRACERCUDA_H

#include <vector>

#include "common/RayTracerBase.h"
#include "common/RayTracerConfig.h"

struct RayTracerCudaTriangles : public RayTracerBase
{
  RayTracerCudaTriangles(RayTracerConfig const& config)
    : RayTracerBase(config)
  {
  }

  void processPixelsCuda();

private:
  RGB const background = {100, 100, 200};
};


#endif // COMMON_RAYTRACERCUDA_H
