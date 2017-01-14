#ifndef COMMON_RAYTRACER_H
#define COMMON_RAYTRACER_H

#include <thread>
#include <vector>

#include "common/DynamicArray2D.h"
#include "common/RayTracerBase.h"
#include "common/RayTracerConfig.h"
#include "common/Structures.h"

struct RayTracer : public RayTracerBase
{
  RayTracer(RayTracerConfig const& config)
    : RayTracerBase(config)
    , threadNumber(std::thread::hardware_concurrency())
  {
    std::cerr << threadNumber << " threads available\n";
  }

  void processPixels();

private:
  // We assume threadNumber < imageY
  int const threadNumber;

  void processPixelsThreads(int threadId);
  RGB processPixel(Segment const& ray, int recursionLevel);
  RGB processPixelOnBackground();
  RGB processPixelOnSphere(Point const& rayBeg, Point const& pointOnSphere,
                           std::vector<Sphere>::const_iterator sphereIt, int recursionLevel);
  RGB processPixelOnPlane(Point const& rayBeg, Point const& pointOnPlane,
                          std::vector<Plane>::const_iterator planeIt, int recursionLevel);
  RGB calculateColorInShadow(RGB currentColor, Vector const& normalVec, Vector const& unitVec);
};

#endif // COMMON_RAYTRACER_H
