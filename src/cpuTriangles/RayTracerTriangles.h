#ifndef COMMON_RAYTRACERTRIANGLES_H
#define COMMON_RAYTRACERTRIANGLES_H

#include <thread>
#include <vector>

#include "common/DynamicArray2D.h"
#include "common/RayTracerBase.h"
#include "common/Structures.h"
#include "cpuTriangles/KdNode.h"

class RayTracerTriangles : public RayTracerBase
{
public:
  RayTracerTriangles(RayTracerConfig const& config)
    : RayTracerBase(config)
    , threadNumber(std::thread::hardware_concurrency())
  {
    std::cerr << threadNumber << " threads available\n";
    kdTree = KdNode::build(const_cast<RayTracerConfig&>(config).triangles);
  }

  void processPixels();

private:
  // We assume threadNumber < imageY
  int const threadNumber = 1;

  KdNode* kdTree = nullptr;

  void processPixelsThreads(int threadId);
  RGB processPixel(Segment const& ray, int recursionLevel);
  RGB processPixelOnTriangle(Point const& rayBeg, Point const& pointOnTriangle,
                             Triangle const& triangle, int recursionLevel);
  RGB processPixelOnBackground();
  RGB calculateColorInLight(Point const& pointOnTriangle, Triangle const& triangle, RGB color);
  bool pointInShadow(Point const& point, Triangle const& triangle);
};

#endif // COMMON_RAYTRACERTRIANGLES_H
