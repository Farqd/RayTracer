#include "cpuTriangles/RayTracerTriangles.h"


#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#include "common/StructuresOperators.h"
#include "cpu/Utils.h"
#include "cpu/Utils.h"

RayTracerTriangles::RayTracerTriangles(RayTracerConfig const& config,
                                       std::vector<Triangle>& triangles)
  : config(config)
  , bitmap(config.imageY * config.antiAliasing, config.imageZ * config.antiAliasing)
{
  kdTree = KdNode::build(triangles);
}

RGB RayTracerTriangles::processPixelOnBackground()
{
  return {0, 0, 0};
}


bool RayTracerTriangles::pointInShadow(Point const& point)
{
  Segment seg = {point, config.light};
  auto const& res = kdTree->find(seg);

  return res.exists && distance(point, res.point) < distance(point, config.light);
}

RGB RayTracerTriangles::processPixelOnTriangle(Point const& rayBeg, Point const& pointOnTriangle,
                                               Triangle const& triangle, int recursionLevel)
{
  bool const isInShadow = pointInShadow(pointOnTriangle);

  RGB color = colorOfPoint(pointOnTriangle, triangle);

  RGB resultCol;
  if (isInShadow)
    resultCol = color * config.ambientCoefficient;
  else
    resultCol = calculateColorInLight(rayBeg, pointOnTriangle, triangle);

  if (recursionLevel >= config.maxRecursionLevel || isCloseToZero(reflectionCoefficient))
    return resultCol;

  Segment refl = reflection({rayBeg, pointOnTriangle}, triangle);
  RGB reflectedColor = processPixel(refl, recursionLevel + 1);

  return calculateColorFromReflection(resultCol, reflectedColor, reflectionCoefficient);
}


RGB RayTracerTriangles::calculateColorInLight(Point const& rayBeg, Point const& pointOnTriangle,
                                              Triangle const& triangle)
{
  // TODO
  return RGB{};
  // float dot = dotProduct(normalVec, normalize(lightVec));
  // return currentColor
  //        * (std::max(0.0f, (1 - config.ambientCoefficient) * dot) + config.ambientCoefficient);
}

RGB RayTracerTriangles::processPixel(Segment const& ray, int recursionLevel)
{
  FindResult triangleIntersec = kdTree->find(ray);

  if (triangleIntersec.exists == false)
    return processPixelOnBackground();

  return processPixelOnTriangle(ray.a, triangleIntersec.point, triangleIntersec.triangle,
                                recursionLevel);
}

void RayTracerTriangles::processPixelsThreads(int threadId)
{
  int const shiftY = bitmap.rows / 2;
  int const shiftZ = bitmap.cols / 2;
  for (int y = -shiftY + threadId; y < shiftY; y += threadNumber)
    for (int z = -shiftZ; z < shiftZ; ++z)
    {
      RGB const& color = processPixel(
          {config.observer,
           {static_cast<float>(config.imageX), static_cast<float>(y) / config.antiAliasing,
            static_cast<float>(z) / config.antiAliasing}},
          0);

      bitmap(y + shiftY, z + shiftZ) = color;
    }
}

void RayTracerTriangles::processPixels()
{
  std::vector<std::thread> thVec;
  for (int i = 0; i < threadNumber - 1; i++)
    thVec.push_back(std::thread(&RayTracerTriangles::processPixelsThreads, this, i));

  processPixelsThreads(threadNumber - 1);

  for (int i = 0; i < threadNumber - 1; i++)
    thVec[i].join();
}
