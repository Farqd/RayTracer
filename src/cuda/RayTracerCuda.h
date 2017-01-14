#ifndef COMMON_RAYTRACERCUDA_H
#define COMMON_RAYTRACERCUDA_H

#include <vector>

#include "common/RayTracerBase.h"
#include "common/RayTracerConfig.h"

struct RayTracerCuda : public RayTracerBase
{
  RayTracerCuda(RayTracerConfig const& config)
    : RayTracerBase(config)
  {
  }

  void processPixelsCuda();

private:
  RGB const background = {100, 100, 200};
};


#endif // COMMON_RAYTRACERCUDA_H
