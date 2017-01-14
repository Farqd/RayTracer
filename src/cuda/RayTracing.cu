#include <cfloat>
#include <cstdio>
#include <cstdint>

#include <host_defines.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#include "common/Structures.h"

struct IntersectionResult
{
  Point intersectionPoint;
  bool intersects;
};

struct pairip
{
  int first;
  Point second;
};

extern "C" {

__device__ RGB processPixel(Segment const& ray,
                            int recurstionLevel,
                            Sphere* spheres,
                            int spheresNum,
                            Plane* planes,
                            int planesNum,
                            int maxRecursionLevel,
                            float ambientCoefficient,
                            Point const& light,
                            RGB const& background);

__device__ bool isCloseToZero(float x)
{
  return abs(x) < DBL_EPSILON;
}

__device__ RGB& operator*=(RGB& rgb, float times)
{
  rgb.r *= times;
  rgb.g *= times;
  rgb.b *= times;
  return rgb;
}

__device__ RGB operator*(RGB rgb, float times)
{
  return rgb *= times;
}

__device__ RGB& operator+=(RGB& lhs, RGB rhs)
{
  lhs.r += rhs.r;
  lhs.g += rhs.g;
  lhs.b += rhs.b;
  return lhs;
}

__device__ RGB operator+(RGB lhs, RGB rhs)
{
  return lhs += rhs;
}

__device__ float distance(Point const& a, Point const& b)
{
  return sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2) + pow(b.z - a.z, 2));
}

__device__ IntersectionResult intersection(Segment segment, Sphere sphere)
{
  float x0 = segment.a.x;
  float y0 = segment.a.y;
  float z0 = segment.a.z;

  float x1 = segment.b.x;
  float y1 = segment.b.y;
  float z1 = segment.b.z;

  float dx = x1 - x0;
  float dy = y1 - y0;
  float dz = z1 - z0;

  float cx = sphere.center.x;
  float cy = sphere.center.y;
  float cz = sphere.center.z;

  float a = dx * dx + dy * dy + dz * dz;
  float b = 2 * dx * (x0 - cx) + 2 * dy * (y0 - cy) + 2 * dz * (z0 - cz);
  float c = cx * cx + cy * cy + cz * cz + x0 * x0 + y0 * y0 + z0 * z0
            - 2 * (cx * x0 + cy * y0 + cz * z0) - sphere.radius * sphere.radius;

  float discriminant = b * b - 4 * a * c;
  if (!isCloseToZero(discriminant) && discriminant < 0)
    return {{}, false};

  float t = (-b - sqrt(discriminant)) / (2 * a);
  if (t < 0)
    return {{}, false};
  return {{x0 + t * dx, y0 + t * dy, z0 + t * dz}, true};
}

__device__ float vectorLen(Point const& vec)
{
  return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__device__ float dotProduct(Point const& a, Point const& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ void normalize(Point& vec)
{
  float len = vectorLen(vec);
  vec.x = vec.x / len;
  vec.y = vec.y / len;
  vec.z = vec.z / len;
}

__device__ IntersectionResult intersectionP(Segment segment, Plane plane)
{
  Point V = {segment.b.x - segment.a.x, segment.b.y - segment.a.y, segment.b.z - segment.a.z};
  float x = dotProduct(V, plane.normal);
  if (x == 0)
    return {{}, false};

  float t = -(dotProduct(segment.a, plane.normal) + plane.d) / x;
  if (t < 0 || isCloseToZero(t))
    return {{}, false};

  Point result;
  result.x = segment.a.x + t * V.x;
  result.y = segment.a.y + t * V.y;
  result.z = segment.a.z + t * V.z;

  return {result, true};
}

__device__ Segment reflectionP(Segment const& segment, Plane const& plane)
{
  Segment result;
  result.a = segment.b;
  Point normalVector = plane.normal;

  Vector ri = {segment.b.x - segment.a.x, segment.b.y - segment.a.y, segment.b.z - segment.a.z};

  float dot = dotProduct(ri, normalVector);
  ri.x = ri.x - 2 * normalVector.x * dot;
  ri.y = ri.y - 2 * normalVector.y * dot;
  ri.z = ri.z - 2 * normalVector.z * dot;

  result.b.x = result.a.x + ri.x;
  result.b.y = result.a.y + ri.y;
  result.b.z = result.a.z + ri.z;

  return result;
}

__device__ Segment reflection(Segment const& segment, Sphere const& sphere)
{
  Segment result;
  result.a = segment.b;
  Point normalVector = {(segment.b.x - sphere.center.x) / sphere.radius,
                        (segment.b.y - sphere.center.y) / sphere.radius,
                        (segment.b.z - sphere.center.z) / sphere.radius};

  Point ri = {segment.b.x - segment.a.x, segment.b.y - segment.a.y, segment.b.z - segment.a.z};

  normalize(ri);
  normalize(normalVector);

  float dot = dotProduct(ri, normalVector);
  ri.x = ri.x - 2 * normalVector.x * dot;
  ri.y = ri.y - 2 * normalVector.y * dot;
  ri.z = ri.z - 2 * normalVector.z * dot;

  result.b.x = result.a.x + ri.x;
  result.b.y = result.a.y + ri.y;
  result.b.z = result.a.z + ri.z;

  return result;
}

__device__ bool pointInShadow(Point const& point, Point const& light, Sphere const& sphere)
{
  Segment seg = {point, light};
  IntersectionResult const& res = intersection(seg, sphere);
  return res.intersects && distance(point, res.intersectionPoint) < distance(point, light);
}

__device__ bool pointInShadowP(Point const& point, Point const& light, Plane const& plane)
{
  Segment seg = {point, light};
  IntersectionResult const& res = intersectionP(seg, plane);
  return res.intersects && distance(point, res.intersectionPoint) < distance(point, light);
}

__device__ RGB processPixelOnBackground(RGB const& background)
{
  return background;
}

__device__ pairip findClosestSphereIntersection(Segment const& seg, Sphere* spheres, int spheresNum)
{
  Point closestPoint{};
  int sphereIndex = -1;
  float closestDistance = FLT_MAX;

  for (size_t i = 0; i < spheresNum; i++)
  {
    IntersectionResult const& res = intersection(seg, spheres[i]);

    if (!res.intersects)
      continue;

    float dist = distance(seg.a, res.intersectionPoint);
    if (dist < closestDistance)
    {
      closestDistance = dist;
      closestPoint = res.intersectionPoint;
      sphereIndex = i;
    }
  }
  return {sphereIndex, closestPoint};
}

__device__ pairip findClosestPlaneIntersection(Segment const& seg, Plane* planes, int planesNum)
{
  Point closestPoint{};
  int planeIndex = -1;
  float closestDistance = FLT_MAX;

  for (size_t i = 0; i < planesNum; i++)
  {
    IntersectionResult const& res = intersectionP(seg, planes[i]);

    if (!res.intersects)
      continue;

    float dist = distance(seg.a, res.intersectionPoint);
    if (dist < closestDistance)
    {
      closestDistance = dist;
      closestPoint = res.intersectionPoint;
      planeIndex = i;
    }
  }

  return {planeIndex, closestPoint};
}

__device__ RGB calculateColorFromReflection(RGB currentColor, RGB reflectedColor,
                                            float reflectionCoefficient)
{
  return currentColor * (1.0f - reflectionCoefficient) + reflectedColor * reflectionCoefficient;
}


__device__ RGB processPixelOnSphere(Point const& rayBeg, Point const& pointOnSphere,
                                    size_t sphereIndex, int recursionLevel, int maxRecursionLevel,
                                    Sphere* spheres, int spheresNum, Plane* planes, int planesNum,
                                    float ambientCoefficient,
                                    Point const& light, RGB const& background)
{
  bool isInShadow = false;

  for (size_t i = 0; i < sphereIndex; i++)
    if (pointInShadow(pointOnSphere, light, spheres[i]))
    {
      isInShadow = true;
      break;
    }

  if (!isInShadow)
  {
    for (size_t i = sphereIndex + 1; i < spheresNum; i++)
      if (pointInShadow(pointOnSphere, light, spheres[i]))
      {
        isInShadow = true;
        break;
      }
  }

  if (!isInShadow)
  {
    for (size_t i = 0; i < planesNum; i++)
      if (pointInShadowP(pointOnSphere, light, planes[i]))
      {
        isInShadow = true;
        break;
      }
  }

  RGB resultCol;
  RGB basicColor = spheres[sphereIndex].color;
  if (isInShadow)
  {
    resultCol = basicColor * ambientCoefficient;
  }
  else
  {
    Point const& center = spheres[sphereIndex].center;
    float radius = spheres[sphereIndex].radius;
    Point normalVector = {(pointOnSphere.x - center.x) / radius,
                          (pointOnSphere.y - center.y) / radius,
                          (pointOnSphere.z - center.z) / radius};
    Point unitVec = {light.x - pointOnSphere.x, light.y - pointOnSphere.y,
                     light.z - pointOnSphere.z};
    normalize(unitVec);
    float dot = dotProduct(normalVector, unitVec);

    resultCol = basicColor * (max(0.0f, (1 - ambientCoefficient) * dot) + ambientCoefficient);
  }

  if (recursionLevel >= maxRecursionLevel
      || isCloseToZero(spheres[sphereIndex].reflectionCoefficient))
    return resultCol;

  Segment refl = reflection({rayBeg, pointOnSphere}, spheres[sphereIndex]);
  RGB reflectedColor = processPixel(refl, recursionLevel + 1, spheres, spheresNum, planes,
                                    planesNum, maxRecursionLevel,
                                    ambientCoefficient, light, background);

  return calculateColorFromReflection(resultCol, reflectedColor,
                                      spheres[sphereIndex].reflectionCoefficient);
}

__device__ RGB processPixelOnPlane(Point const& rayBeg, Point const& pointOnPlane,
                                   size_t planeIndex, int recursionLevel, int maxRecursionLevel,
                                   Sphere* spheres, int spheresNum, Plane* planes, int planesNum,
                                   float ambientCoefficient,
                                   Point const& light, RGB const& background)
{
  bool isInShadow = false;

  for (size_t i = 0; i < planeIndex; i++)
    if (pointInShadowP(pointOnPlane, light, planes[i]))
    {
      isInShadow = true;
      break;
    }

  if (!isInShadow)
  {
    for (size_t i = planeIndex + 1; i < planesNum; i++)
      if (pointInShadowP(pointOnPlane, light, planes[i]))
      {
        isInShadow = true;
        break;
      }
  }

  if (!isInShadow)
  {
    for (size_t i = 0; i < spheresNum; i++)
      if (pointInShadow(pointOnPlane, light, spheres[i]))
      {
        isInShadow = true;
        break;
      }
  }

  RGB resultCol;
  if (isInShadow)
    resultCol = planes[planeIndex].color * ambientCoefficient;
  else
  {
    Point unitVec = {light.x - pointOnPlane.x, light.y - pointOnPlane.y, light.z - pointOnPlane.z};
    normalize(unitVec);
    float dot = dotProduct(planes[planeIndex].normal, unitVec);
    resultCol =
        planes[planeIndex].color * (max(0.0f, (1 - ambientCoefficient) * dot) + ambientCoefficient);
  }

  if (recursionLevel >= maxRecursionLevel
      || isCloseToZero(planes[planeIndex].reflectionCoefficient))
    return resultCol;

  Segment refl = reflectionP({rayBeg, pointOnPlane}, planes[planeIndex]);

  RGB reflectedColor = processPixel(refl, recursionLevel + 1, spheres, spheresNum, planes,
                                    planesNum, maxRecursionLevel,
                                    ambientCoefficient, light, background);

  return calculateColorFromReflection(resultCol, reflectedColor,
                                      planes[planeIndex].reflectionCoefficient);
}
__device__ RGB processPixel(Segment const& ray,
                            int recursionLevel,
                            Sphere* spheres,
                            int spheresNum,
                            Plane* planes,
                            int planesNum,
                            int maxRecursionLevel,
                            float ambientCoefficient,
                            Point const& light,
                            RGB const& background)
{
  pairip sphereIntersection = findClosestSphereIntersection(ray, spheres, spheresNum);
  pairip planeIntersection = findClosestPlaneIntersection(ray, planes, planesNum);

  if (sphereIntersection.first != -1 && planeIntersection.first != -1)
  {
    if (distance(ray.a, sphereIntersection.second) < distance(ray.a, planeIntersection.second))
      return processPixelOnSphere(ray.a, sphereIntersection.second, sphereIntersection.first,
                                  recursionLevel, maxRecursionLevel, spheres, spheresNum, planes,
                                  planesNum, ambientCoefficient, light,
                                  background);
    else
      return processPixelOnPlane(ray.a, planeIntersection.second, planeIntersection.first,
                                 recursionLevel, maxRecursionLevel, spheres, spheresNum, planes,
                                 planesNum, ambientCoefficient, light,
                                 background);
  }

  if (sphereIntersection.first != -1)
    return processPixelOnSphere(ray.a, sphereIntersection.second, sphereIntersection.first,
                                recursionLevel, maxRecursionLevel, spheres, spheresNum, planes,
                                planesNum, ambientCoefficient, light,
                                background);

  if (planeIntersection.first != -1)
    return processPixelOnPlane(ray.a, planeIntersection.second, planeIntersection.first,
                               recursionLevel, maxRecursionLevel, spheres, spheresNum, planes,
                               planesNum, ambientCoefficient, light,
                               background);

  return processPixelOnBackground(background);
}
__global__ void computePixel(Sphere* spheres,
                             int spheresNum,
                             Plane* planes,
                             int planesNum,
                             RGB* bitmap,
                             int imageX, int imageY, int imageZ,
                             int antiAliasing,
                             int maxRecursionLevel,
                             float ambientCoefficient,
                             float observerX, float observerY, float observerZ,
                             float lX, float lY, float lZ,
                             uint8_t R, uint8_t G, uint8_t B)
{
  int thidX = (blockIdx.x * blockDim.x) + threadIdx.x;
  int thidY = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (thidX < 2 * imageY && thidY < 2 * imageZ)
  {
    Point const observer = {observerX, observerY, observerZ};
    Point const light = {lX, lY, lZ};
    RGB background = {R, G, B};
    Point point{static_cast<float>(imageX),
                static_cast<float>(thidX - imageY) / antiAliasing,
                static_cast<float>(thidY - imageZ) / antiAliasing};

    Segment ray{observer, point};
    int idx = thidX * imageZ * 2 + thidY;
    bitmap[idx] = processPixel(ray, 0, spheres, spheresNum, planes, planesNum, maxRecursionLevel,
                               ambientCoefficient, light, background);
  }
}
}
