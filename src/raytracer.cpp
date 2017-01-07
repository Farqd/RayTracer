#include "raytracer.h"

#include "structures.h"
#include "utils.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <vector>


RGB RayTracer::processPixelOnBackground()
{
  return {0, 0, 0};
}

std::pair<int, Point> RayTracer::findClosestSphereIntersection(Segment const& seg)
{
  std::vector<std::pair<std::pair<Point, double>, size_t>> distanceIndex;
  for (size_t i = 0; i < spheres.size(); i++)
  {
    Sphere const& sphere = spheres[i];
    auto const& res = intersection(seg, sphere);
    if (res.first)
      distanceIndex.push_back({{res.second, distance(observer, res.second)}, i});
  }

  if(distanceIndex.empty())
    return {-1, {}};

  std::sort(distanceIndex.begin(), distanceIndex.end(),
              [](std::pair<std::pair<Point, double>, int> const& a,
                 std::pair<std::pair<Point, double>, int> const& b) {
                return a.first.second < b.first.second;
              });

  return {distanceIndex[0].second, distanceIndex[0].first.first};

}

std::pair<int, Point> RayTracer::findClosestPlaneIntersection(Segment const& seg)
{
  std::vector<std::pair<std::pair<Point, double>, size_t>> distanceIndex;
  for (size_t i = 0; i < planes.size(); i++)
  {
    Plane const& plane = planes[i];
    auto const& res = intersection(seg, plane);
    if (res.first)
      distanceIndex.push_back({{res.second, distance(observer, res.second)}, i});
  }

  if(distanceIndex.empty())
    return {-1, {}};

  std::sort(distanceIndex.begin(), distanceIndex.end(),
              [](std::pair<std::pair<Point, double>, int> const& a,
                 std::pair<std::pair<Point, double>, int> const& b) {
                return a.first.second < b.first.second;
              });

  return {distanceIndex[0].second, distanceIndex[0].first.first};

}


RGB RayTracer::processPixelOnSphere(Point const& pointOnSphere, size_t sphereIndex)
{
  Point const& center = spheres[sphereIndex].center;
  double radius = spheres[sphereIndex].radius;
  RGB rgb = spheres[sphereIndex].color;

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
    return rgb * ambientCoefficient;
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

    return rgb * (std::max(0.0, diffuseCoefficient * dot) + ambientCoefficient);
  }
}

RGB RayTracer::processPixelOnPlane(Point const& pointOnPlane, size_t planeIndex)
{
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
    return planes[planeIndex].color * ambientCoefficient;

  Point unitVec = {light.x - pointOnPlane.x, light.y - pointOnPlane.y,
                   light.z - pointOnPlane.z};
  normalize(unitVec);
  double dot = dotProduct(planes[planeIndex].normal, unitVec);
  
  // check if light is not from the other side - it probably should be done in a different way
  Point nor = planes[planeIndex].normal;
  nor.x = -nor.x;
  nor.y = -nor.y;
  nor.z = -nor.z;

  double dot2 = dotProduct(nor, unitVec);
  dot = std::max(dot, dot2);

  return planes[planeIndex].color * (std::max(0.0, diffuseCoefficient * dot) + ambientCoefficient);

}

RGB RayTracer::processPixel(Point const& point)
{

  Segment seg{observer, point};

  std::pair<int, Point> sphereIntersection = findClosestSphereIntersection(seg);
  std::pair<int, Point> planeIntersection = findClosestPlaneIntersection(seg);

  if(sphereIntersection.first != -1 && planeIntersection.first != -1)
  {
    if(distance(point, sphereIntersection.second) < distance(point, planeIntersection.second))
      return processPixelOnSphere(sphereIntersection.second, sphereIntersection.first);
    else
      return processPixelOnPlane(planeIntersection.second, planeIntersection.first);
  }

  if (sphereIntersection.first != -1)
    return processPixelOnSphere(sphereIntersection.second, sphereIntersection.first);
  
  if (planeIntersection.first != -1)
    return processPixelOnPlane(planeIntersection.second, planeIntersection.first);

  return processPixelOnBackground();
}

void RayTracer::processPixels()
{
  for (int y = -imageY; y < imageY; ++y)
    for (int z = -imageZ; z < imageZ; ++z)
    {

      RGB const& color = processPixel( {imageX, static_cast<double>(y) / antiAliasing,
                             static_cast<double>(z) / antiAliasing});
      bitmap[static_cast<double>(y) / antiAliasing + imageY][static_cast<double>(z) / antiAliasing + imageZ] = color;
    }
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
