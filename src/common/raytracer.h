#ifndef COMMON_RAYTRACER_H
#define COMMON_RAYTRACER_H

#include <thread>
#include <vector>

#include "common/structures_operators.h"

class RayTracer
{
public:
  RayTracer()
    : threadNumber(std::thread::hardware_concurrency())
  {
    std::cerr << threadNumber << " threads available\n";
  }
  // for antiAliasing = 4, 16 pixels are generated for each one from final scene
  static int const antiAliasing = 2;
  static int const maxRecursionLevel = 1;

  // We assume threadNumber < imageY
  int const threadNumber;

  Point const observer = {0, 0, 0};
  Point const light = {1000, 2000, 2500};

  // image is a rectangle with verticles (256, -+imageY/antiAliasing, -+imageZ/antiAliasing)
  static int const imageX = 512;
  Point const imageCenter = {imageX, 0, 0};
  static int const imageY = 384 * antiAliasing;
  static int const imageZ = 512 * antiAliasing;

  RGB bitmap[imageY * 2][imageZ * 2];

  float const diffuseCoefficient = 0.9;
  float const ambientCoefficient = 0.1;

  RGB processPixel(Segment const& ray, int recursionLevel);
  RGB processPixelOnBackground();
  RGB processPixelOnSphere(Point const& rayBeg, Point const& pointOnSphere, size_t sphereIndex,
                           int recursionLevel);
  RGB processPixelOnPlane(Point const& rayBeg, Point const& pointOnPlane, size_t planeIndex,
                          int recursionLevel);
  std::pair<int, Point> findClosestSphereIntersection(Segment const& seg);
  std::pair<int, Point> findClosestPlaneIntersection(Segment const& seg);
  void processPixelsThreads(int threadId);


  void processPixels();
  void printBitmap();

  std::vector<Sphere> spheres;
  std::vector<Plane> planes;
};

#endif // COMMON_RAYTRACER_H
