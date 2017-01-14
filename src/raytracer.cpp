#include "raytracer.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#include "common/structures_operators.h"
#include "common/utils.h"

RGB RayTracer::processPixelOnBackground()
{
  return {0, 0, 0};
}

std::pair<int, Point> RayTracer::findClosestSphereIntersection(Segment const& seg)
{
  bool foundAny = false;
  Point closestPoint{};
  int sphereIndex = -1;
  float closestDistance = std::numeric_limits<float>::max();

  for (size_t i = 0; i < spheres.size(); i++)
  {
    auto const& res = intersection(seg, spheres[i]);
    if (res.first)
    {
      if (!foundAny)
      {
        closestDistance = distance(seg.a, res.second);
        closestPoint = res.second;
        sphereIndex = i;
        foundAny = true;
      }
      else
      {
        float dist = distance(seg.a, res.second);
        if (dist < closestDistance)
        {
          closestDistance = dist;
          closestPoint = res.second;
          sphereIndex = i;
        }
      }
    }
  }
  return {sphereIndex, closestPoint};
}

std::pair<int, Point> RayTracer::findClosestPlaneIntersection(Segment const& seg)
{
  bool foundAny = false;
  Point closestPoint{};
  int planeIndex = -1;
  float closestDistance = std::numeric_limits<float>::max();

  for (size_t i = 0; i < planes.size(); i++)
  {
    auto const& res = intersection(seg, planes[i]);
    if (res.first)
    {
      if (!foundAny)
      {
        closestDistance = distance(seg.a, res.second);
        closestPoint = res.second;
        planeIndex = i;
        foundAny = true;
      }
      else
      {
        float dist = distance(seg.a, res.second);
        if (dist < closestDistance)
        {
          closestDistance = dist;
          closestPoint = res.second;
          planeIndex = i;
        }
      }
    }
  }

  return {planeIndex, closestPoint};
}


RGB RayTracer::processPixelOnSphere(Point const& rayBeg, Point const& pointOnSphere,
                                    size_t sphereIndex, int recursionLevel)
{
  RGB resultCol;

  Point const& center = spheres[sphereIndex].center;
  float radius = spheres[sphereIndex].radius;
  RGB basicColor = spheres[sphereIndex].color;

  bool isInShadow = false;

  for (size_t i = 0; i < spheres.size(); i++)
  {
    if (i != sphereIndex && pointInShadow(pointOnSphere, config.light, spheres[i]))
    {
      isInShadow = true;
      break;
    }
  }

  for (size_t i = 0; i < planes.size(); i++)
  {
    if (pointInShadow(pointOnSphere, config.light, planes[i]))
    {
      isInShadow = true;
      break;
    }
  }

  if (isInShadow)
  {
    resultCol = basicColor * config.ambientCoefficient;
  }
  else
  {
    Point normalVector = {(pointOnSphere.x - center.x) / radius,
                          (pointOnSphere.y - center.y) / radius,
                          (pointOnSphere.z - center.z) / radius};
    Point unitVec = {config.light.x - pointOnSphere.x, config.light.y - pointOnSphere.y,
                     config.light.z - pointOnSphere.z};
    normalize(unitVec);
    float dot = dotProduct(normalVector, unitVec);

    resultCol =
        basicColor * (std::max(0.0f, config.diffuseCoefficient * dot) + config.ambientCoefficient);
  }

  if (recursionLevel >= config.maxRecursionLevel
      || isCloseToZero(spheres[sphereIndex].reflectionCoefficient))
    return resultCol;

  Segment refl = reflection({rayBeg, pointOnSphere}, spheres[sphereIndex]);
  RGB reflectedColor = processPixel(refl, recursionLevel + 1);

  float refC = spheres[sphereIndex].reflectionCoefficient;
  resultCol = resultCol * (1.0 - refC);
  resultCol.r += reflectedColor.r * refC;
  resultCol.g += reflectedColor.g * refC;
  resultCol.b += reflectedColor.b * refC;

  return resultCol;
}

RGB RayTracer::processPixelOnPlane(Point const& rayBeg, Point const& pointOnPlane,
                                   size_t planeIndex, int recursionLevel)
{
  RGB resultCol;
  bool isInShadow = false;

  for (size_t i = 0; i < spheres.size(); i++)
  {
    if (pointInShadow(pointOnPlane, config.light, spheres[i]))
    {
      isInShadow = true;
      break;
    }
  }

  for (size_t i = 0; i < planes.size(); i++)
  {
    if (i != planeIndex && pointInShadow(pointOnPlane, config.light, planes[i]))
    {
      isInShadow = true;
      break;
    }
  }

  if (isInShadow)
    resultCol = planes[planeIndex].color * config.ambientCoefficient;
  else
  {
    Point unitVec = {config.light.x - pointOnPlane.x, config.light.y - pointOnPlane.y,
                     config.light.z - pointOnPlane.z};
    normalize(unitVec);
    float dot = dotProduct(planes[planeIndex].normal, unitVec);
    resultCol = planes[planeIndex].color
                * (std::max(0.0f, config.diffuseCoefficient * dot) + config.ambientCoefficient);
  }

  if (recursionLevel >= config.maxRecursionLevel
      || isCloseToZero(planes[planeIndex].reflectionCoefficient))
    return resultCol;

  Segment refl = reflection({rayBeg, pointOnPlane}, planes[planeIndex]);

  RGB reflectedColor = processPixel(refl, recursionLevel + 1);

  float refC = planes[planeIndex].reflectionCoefficient;
  resultCol = resultCol * (1.0 - refC);
  resultCol.r += reflectedColor.r * refC;
  resultCol.g += reflectedColor.g * refC;
  resultCol.b += reflectedColor.b * refC;

  return resultCol;
}

RGB RayTracer::processPixel(Segment const& ray, int recursionLevel)
{
  std::pair<int, Point> sphereIntersection = findClosestSphereIntersection(ray);
  std::pair<int, Point> planeIntersection = findClosestPlaneIntersection(ray);

  if (sphereIntersection.first != -1 && planeIntersection.first != -1)
  {
    if (distance(ray.a, sphereIntersection.second) < distance(ray.a, planeIntersection.second))
      return processPixelOnSphere(ray.a, sphereIntersection.second, sphereIntersection.first,
                                  recursionLevel);
    else
      return processPixelOnPlane(ray.a, planeIntersection.second, planeIntersection.first,
                                 recursionLevel);
  }

  if (sphereIntersection.first != -1)
    return processPixelOnSphere(ray.a, sphereIntersection.second, sphereIntersection.first,
                                recursionLevel);

  if (planeIntersection.first != -1)
    return processPixelOnPlane(ray.a, planeIntersection.second, planeIntersection.first,
                               recursionLevel);

  return processPixelOnBackground();
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
