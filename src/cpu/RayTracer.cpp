#include "RayTracer.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#include "common/StructuresOperators.h"
#include "common/Utils.h"
#include "common/Utils.h"

RGB RayTracer::processPixelOnBackground()
{
  return {0, 0, 0};
}

RGB RayTracer::processPixelOnSphere(Point const& rayBeg, Point const& pointOnSphere,
                                    std::vector<Sphere>::const_iterator sphereIt,
                                    int recursionLevel)
{
  bool const isInShadow =
      pointInShadow(pointOnSphere, config.light, config.spheres.cbegin(), sphereIt)
      || pointInShadow(pointOnSphere, config.light, sphereIt + 1, config.spheres.cend())
      || pointInShadow(pointOnSphere, config.light, config.planes.cbegin(), config.planes.cend());

  RGB resultCol;
  if (isInShadow)
    resultCol = sphereIt->color * config.ambientCoefficient;
  else
  {
    Vector const normalVec = (pointOnSphere - sphereIt->center) / sphereIt->radius;
    resultCol = calculateColorInLight(sphereIt->color, normalVec, config.light - pointOnSphere);
  }

  if (recursionLevel >= config.maxRecursionLevel || isCloseToZero(sphereIt->reflectionCoefficient))
    return resultCol;

  Segment refl = reflection({rayBeg, pointOnSphere}, *sphereIt);
  RGB reflectedColor = processPixel(refl, recursionLevel + 1);

  return calculateColorFromReflection(resultCol, reflectedColor, sphereIt->reflectionCoefficient);
}

RGB RayTracer::processPixelOnPlane(Point const& rayBeg, Point const& pointOnPlane,
                                   std::vector<Plane>::const_iterator planeIt, int recursionLevel)
{
  bool const isInShadow =
      pointInShadow(pointOnPlane, config.light, config.planes.cbegin(), planeIt)
      || pointInShadow(pointOnPlane, config.light, planeIt + 1, config.planes.cend())
      || pointInShadow(pointOnPlane, config.light, config.spheres.cbegin(), config.spheres.cend());

  RGB resultCol;
  if (isInShadow)
    resultCol = planeIt->color * config.ambientCoefficient;
  else
    resultCol = calculateColorInLight(planeIt->color, planeIt->normal, config.light - pointOnPlane);

  if (recursionLevel >= config.maxRecursionLevel || isCloseToZero(planeIt->reflectionCoefficient))
    return resultCol;

  Segment refl = reflection({rayBeg, pointOnPlane}, *planeIt);
  RGB reflectedColor = processPixel(refl, recursionLevel + 1);

  return calculateColorFromReflection(resultCol, reflectedColor, planeIt->reflectionCoefficient);
}

RGB RayTracer::calculateColorInLight(RGB currentColor, Vector const& normalVec,
                                     Vector const& lightVec)
{
  float dot = dotProduct(normalVec, normalize(lightVec));
  return currentColor
         * (std::max(0.0f, (1 - config.ambientCoefficient) * dot) + config.ambientCoefficient);
}

RGB RayTracer::processPixel(Segment const& ray, int recursionLevel)
{
  std::pair<int, Point> sphereIntersection = findClosestIntersection(config.spheres, ray);
  std::pair<int, Point> planeIntersection = findClosestIntersection(config.planes, ray);

  if (sphereIntersection.first == -1 && planeIntersection.first == -1)
    return processPixelOnBackground();

  auto closerIntersectionCmp = [&ray](std::pair<int, Point> const& i1,
                                      std::pair<int, Point> const& i2) {
    if (i1.first != -1 && i2.first != -1)
      return distance(ray.a, i1.second) < distance(ray.a, i2.second);
    return i1.first != -1;
  };

  if (closerIntersectionCmp(sphereIntersection, planeIntersection))
    return processPixelOnSphere(ray.a, sphereIntersection.second,
                                config.spheres.cbegin() + sphereIntersection.first, recursionLevel);
  else
    return processPixelOnPlane(ray.a, planeIntersection.second,
                               config.planes.cbegin() + planeIntersection.first, recursionLevel);
}

void RayTracer::processPixelsThreads(int threadId)
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

void RayTracer::processPixels()
{
  std::vector<std::thread> thVec;
  for (int i = 0; i < threadNumber - 1; i++)
    thVec.push_back(std::thread(&RayTracer::processPixelsThreads, this, i));

  processPixelsThreads(threadNumber - 1);

  for (int i = 0; i < threadNumber - 1; i++)
    thVec[i].join();
}
