#include "RayTracer.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#include "Utils.h"
#include "common/StructuresOperators.h"
#include "cpu/Utils.h"

RGB RayTracer::processPixelOnBackground()
{
  return {0, 0, 0};
}

RGB RayTracer::processPixelOnSphere(Point const& rayBeg, Point const& pointOnSphere,
                                    std::vector<Sphere>::const_iterator sphereIt,
                                    int recursionLevel)
{
  bool const isInShadow =
      pointInShadow(pointOnSphere, config.light, spheres.cbegin(), sphereIt)
      || pointInShadow(pointOnSphere, config.light, sphereIt + 1, spheres.cend())
      || pointInShadow(pointOnSphere, config.light, planes.cbegin(), planes.cend());

  RGB resultCol;

  if (isInShadow)
  {
    resultCol = sphereIt->color * config.ambientCoefficient;
  }
  else
  {
    Point const& center = sphereIt->center;
    float radius = sphereIt->radius;
    Point normalVector = {(pointOnSphere.x - center.x) / radius,
                          (pointOnSphere.y - center.y) / radius,
                          (pointOnSphere.z - center.z) / radius};
    Point unitVec = {config.light.x - pointOnSphere.x, config.light.y - pointOnSphere.y,
                     config.light.z - pointOnSphere.z};
    resultCol = calculateColorInShadow(sphereIt->color, normalVector, unitVec);
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
      pointInShadow(pointOnPlane, config.light, planes.cbegin(), planeIt)
      || pointInShadow(pointOnPlane, config.light, planeIt + 1, planes.cend())
      || pointInShadow(pointOnPlane, config.light, spheres.cbegin(), spheres.cend());

  RGB resultCol;
  if (isInShadow)
    resultCol = planeIt->color * config.ambientCoefficient;
  else
  {
    Point unitVec = {config.light.x - pointOnPlane.x, config.light.y - pointOnPlane.y,
                     config.light.z - pointOnPlane.z};
    resultCol = calculateColorInShadow(planeIt->color, planeIt->normal, unitVec);
  }

  if (recursionLevel >= config.maxRecursionLevel || isCloseToZero(planeIt->reflectionCoefficient))
    return resultCol;

  Segment refl = reflection({rayBeg, pointOnPlane}, *planeIt);

  RGB reflectedColor = processPixel(refl, recursionLevel + 1);

  return calculateColorFromReflection(resultCol, reflectedColor, planeIt->reflectionCoefficient);
}

RGB RayTracer::calculateColorInShadow(RGB currentColor, Vector const& normalVec,
                                      Vector const& unitVec)
{
  float shadow = dotProduct(normalVec, normalize(unitVec));
  return currentColor
         * (std::max(0.0f, config.diffuseCoefficient * shadow) + config.ambientCoefficient);
}

RGB RayTracer::processPixel(Segment const& ray, int recursionLevel)
{
  std::pair<int, Point> sphereIntersection = findClosestIntersection(spheres, ray);
  std::pair<int, Point> planeIntersection = findClosestIntersection(planes, ray);

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
                                spheres.cbegin() + sphereIntersection.first, recursionLevel);
  else
    return processPixelOnPlane(ray.a, planeIntersection.second,
                               planes.cbegin() + planeIntersection.first, recursionLevel);
}

void RayTracer::processPixelsThreads(int threadId)
{
  for (int y = -config.imageY + threadId; y < config.imageY; y += threadNumber)
    for (int z = -config.imageZ; z < config.imageZ; ++z)
    {
      RGB const& color = processPixel(
          {config.observer,
           {static_cast<float>(config.imageX), static_cast<float>(y) / config.antiAliasing,
            static_cast<float>(z) / config.antiAliasing}},
          0);

      bitmap(y + config.imageY, z + config.imageZ) = color;
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
