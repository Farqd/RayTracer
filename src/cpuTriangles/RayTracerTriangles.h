#ifndef COMMON_RAYTRACERTRIANGLES_H
#define COMMON_RAYTRACERTRIANGLES_H

#include <thread>
#include <vector>

#include "common/DynamicArray2D.h"
#include "common/RayTracerBase.h"
#include "common/Structures.h"
#include "common/Utils.h"
#include "cpuTriangles/KdNode.h"

class RayTracerTriangles : public RayTracerBase
{
public:
  RayTracerTriangles(RayTracerConfig const& config)
    : RayTracerBase(config)
    , threadNumber(std::thread::hardware_concurrency())
  {
    std::cerr << threadNumber << " threads available\n";
    std::vector<float> ranges;
    if(config.triangles.size() > 0)
    {
      ranges.push_back(config.triangles[0].x.x);
      ranges.push_back(config.triangles[0].x.x);
      ranges.push_back(config.triangles[0].x.y);
      ranges.push_back(config.triangles[0].x.y);
      ranges.push_back(config.triangles[0].x.z);
      ranges.push_back(config.triangles[0].x.z);
    }
    for(auto const& triangle : config.triangles)
    {
      Point pMin = getMinPoint(triangle);
      Point pMax = getMaxPoint(triangle);

       ranges[0] = std::min(ranges[0], pMin.x); 
       ranges[1] = std::max(ranges[1], pMax.x); 
       ranges[2] = std::min(ranges[2], pMin.y); 
       ranges[3] = std::max(ranges[3], pMax.y); 
       ranges[4] = std::min(ranges[4], pMin.z); 
       ranges[5] = std::max(ranges[5], pMax.z); 
    }
    kdTree = KdNode::build(const_cast<RayTracerConfig&>(config).triangles, ranges, 0);
    std::cerr << "building complete" << std::endl;
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
