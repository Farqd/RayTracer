#include "cpuTriangles/RayTracerTriangles.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#include "common/StructuresOperators.h"
#include "common/Utils.h"


RGB RayTracerTriangles::processPixelOnBackground()
{
  return config.background;
}

bool RayTracerTriangles::pointInShadow(Point const& point, Triangle const& triangle)
{
  Segment seg = {point, config.light};
  auto const& res = kdTree->find(seg, triangle, 0, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());

  return res.exists && distance(point, res.point) < distance(point, config.light);
}

RGB RayTracerTriangles::processPixelOnTriangle(Point const& rayBeg, Point const& pointOnTriangle,
                                               Triangle const& triangle, int recursionLevel)
{
  bool const isInShadow = pointInShadow(pointOnTriangle, triangle);
  RGB color = colorOfPoint(pointOnTriangle, triangle);

  RGB resultCol;
  if (isInShadow)
    resultCol = color * config.ambientCoefficient;
  else
    resultCol = calculateColorInLight(pointOnTriangle, triangle, color);

  if (recursionLevel >= config.maxRecursionLevel || isCloseToZero(triangle.reflectionCoefficient))
    return resultCol;

  Segment refl = randomReflection({rayBeg, pointOnTriangle}, triangle);
  RGB reflectedColor = processPixel(refl, recursionLevel + 1);

  return calculateColorFromReflection(resultCol, reflectedColor, triangle.reflectionCoefficient);
}


RGB RayTracerTriangles::calculateColorInLight(Point const& pointOnTriangle,
                                              Triangle const& triangle, RGB color)
{
  Vector v0v1 = triangle.y - triangle.x;
  Vector v0v2 = triangle.z - triangle.x;

  Vector N = normalize(crossProduct(v0v1, v0v2));

  Vector lightVec = config.light - pointOnTriangle;
  float dot = std::abs(dotProduct(N, normalize(lightVec)));
  return color
         * (std::max(0.0f, (1 - config.ambientCoefficient) * dot) + config.ambientCoefficient);
}

RGB RayTracerTriangles::processPixel(Segment const& ray, int recursionLevel)
{
  FindResult triangleIntersec;
  if (kdTree != nullptr)
    triangleIntersec = kdTree->find(ray, Triangle{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, 0, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
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
