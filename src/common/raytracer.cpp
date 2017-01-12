#include "common/raytracer.h"

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
  Point closestPoint;
  int sphereIndex = -1;
  double closestDistance = std::numeric_limits<double>::max();

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
        double dist = distance(seg.a, res.second);
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
  Point closestPoint;
  int planeIndex = -1;
  double closestDistance = std::numeric_limits<double>::max();

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
        double dist = distance(seg.a, res.second);
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
  double radius = spheres[sphereIndex].radius;
  RGB basicColor = spheres[sphereIndex].color;

  bool isInShadow = false;

  for (size_t i = 0; i < spheres.size(); i++)
  {
    if (i != sphereIndex && pointInShadow(pointOnSphere, light, spheres[i]))
    {
      isInShadow = true;
      break;
    }
  }

  for (size_t i = 0; i < planes.size(); i++)
  {
    if (pointInShadow(pointOnSphere, light, planes[i]))
    {
      isInShadow = true;
      break;
    }
  }

  if (isInShadow)
  {
    resultCol = basicColor * ambientCoefficient;
  }
  else
  {
    Point normalVector = {(pointOnSphere.x - center.x) / radius,
                          (pointOnSphere.y - center.y) / radius,
                          (pointOnSphere.z - center.z) / radius};
    Point unitVec = {light.x - pointOnSphere.x, light.y - pointOnSphere.y,
                     light.z - pointOnSphere.z};
    normalize(unitVec);
    double dot = dotProduct(normalVector, unitVec);

    resultCol = basicColor * (std::max(0.0, diffuseCoefficient * dot) + ambientCoefficient);
  }

  if (recursionLevel >= maxRecursionLevel
      || isCloseToZero(spheres[sphereIndex].reflectionCoefficient))
    return resultCol;

  Segment refl = reflection({rayBeg, pointOnSphere}, spheres[sphereIndex]);
  RGB reflectedColor = processPixel(refl, recursionLevel + 1);

  double refC = spheres[sphereIndex].reflectionCoefficient;
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
    if (pointInShadow(pointOnPlane, light, spheres[i]))
    {
      isInShadow = true;
      break;
    }
  }

  for (size_t i = 0; i < planes.size(); i++)
  {
    if (i != planeIndex && pointInShadow(pointOnPlane, light, planes[i]))
    {
      isInShadow = true;
      break;
    }
  }

  if (isInShadow)
    resultCol = planes[planeIndex].color * ambientCoefficient;
  else
  {
    Point unitVec = {light.x - pointOnPlane.x, light.y - pointOnPlane.y, light.z - pointOnPlane.z};
    normalize(unitVec);
    double dot = dotProduct(planes[planeIndex].normal, unitVec);
    resultCol =
        planes[planeIndex].color * (std::max(0.0, diffuseCoefficient * dot) + ambientCoefficient);
  }

  if (recursionLevel >= maxRecursionLevel
      || isCloseToZero(planes[planeIndex].reflectionCoefficient))
    return resultCol;

  Segment refl = reflection({rayBeg, pointOnPlane}, planes[planeIndex]);

  RGB reflectedColor = processPixel(refl, recursionLevel + 1);

  double refC = planes[planeIndex].reflectionCoefficient;
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
  for (int y = -imageY + threadId; y < imageY; y += threadNumber)
    for (int z = -imageZ; z < imageZ; ++z)
    {
      RGB const& color = processPixel(
          {observer,
           {imageX, static_cast<double>(y) / antiAliasing, static_cast<double>(z) / antiAliasing}},
          0);

      bitmap[y + imageY][z + imageZ] = color;
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

void RayTracer::printBitmap()
{
  // see https://en.wikipedia.org/wiki/Netpbm_format for format details

  std::cout << "P3" << std::endl;
  std::cout << imageZ * 2 / antiAliasing << " " << imageY * 2 / antiAliasing << std::endl
            << 255 << std::endl;
  for (int i = imageY * 2 - 1; i >= 0; i -= antiAliasing)
  {
    for (int j = 0; j < imageZ * 2; j += antiAliasing)
    {
      int r = 0;
      int g = 0;
      int b = 0;

      for (int ii = 0; ii < antiAliasing; ii++)
        for (int jj = 0; jj < antiAliasing; jj++)
        {
          r += bitmap[i - ii][j + jj].r;
          g += bitmap[i - ii][j + jj].g;
          b += bitmap[i - ii][j + jj].b;
        }

      RGB color = {0, 0, 0};
      color.r = r / (antiAliasing * antiAliasing);
      color.g = g / (antiAliasing * antiAliasing);
      color.b = b / (antiAliasing * antiAliasing);
      std::cout << color << " ";
    }
    std::cout << std::endl;
  }
}
