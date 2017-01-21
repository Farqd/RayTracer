#ifndef COMMON_RAYTRACERTRIANGLES_H
#define COMMON_RAYTRACERTRIANGLES_H

#include <thread>
#include <vector>

#include "common/DynamicArray2D.h"
#include "common/RayTracerBase.h"
#include "common/Structures.h"
#include "cpuTriangles/KdNode.h"

class RayTracerTriangles
{
  RayTracerTriangles(RayTracerConfig const& config, std::vector<Triangle>& triangles);

  void processPixels();

private:
  RayTracerConfig const config;

  // We assume threadNumber < imageY
  int const threadNumber = 2;


  DynamicArray2D<RGB> bitmap;
  KdNode* kdTree = nullptr;

  float reflectionCoefficient = 0.1;

  void processPixelsThreads(int threadId);
  RGB processPixel(Segment const& ray, int recursionLevel);
  RGB processPixelOnTriangle(Point const& rayBeg, Point const& pointOnTriangle,
                             Triangle const& triangle, int recursionLevel);
  RGB processPixelOnBackground();
  RGB calculateColorInLight(Point const& rayBeg, Point const& pointOnTriangle,
                            Triangle const& triangle);
  bool pointInShadow(Point const& point);
};

#endif // COMMON_RAYTRACERTRIANGLES_H
