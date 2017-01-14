#ifndef COMMON_RAYTRACER_H
#define COMMON_RAYTRACER_H

#include <thread>
#include <vector>

#include "common/DynamicArray2D.h"
#include "common/RayTracerBase.h"
#include "common/raytracerconfig.h"
#include "common/structures.h"

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

  RGB processPixel(Segment const& ray, int recursionLevel);
  RGB processPixelOnBackground();
  RGB processPixelOnSphere(Point const& rayBeg, Point const& pointOnSphere, size_t sphereIndex,
                           int recursionLevel);
  RGB processPixelOnPlane(Point const& rayBeg, Point const& pointOnPlane, size_t planeIndex,
                          int recursionLevel);
  std::pair<int, Point> findClosestSphereIntersection(Segment const& seg);
  std::pair<int, Point> findClosestPlaneIntersection(Segment const& seg);
  void processPixelsThreads(int threadId);
};

#endif // COMMON_RAYTRACER_H
