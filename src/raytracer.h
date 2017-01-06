#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "structures.h"


#include <array>
#include <vector>


class RayTracer
{
  // for antiAliasing = 4, 16 pixels are generated for each one from final scene
  static int const antiAliasing = 2;

  Point const observer = {0, 0, 0};
  Point const light = {2000, 2500, 2500};
  RGB background = {100, 100, 200};
  // image is a rectangle with verticles (256, -+imageY/antiAliasing, -+imageZ/antiAliasing)
  static int const imageX = 512;
  Point const imageCenter = {imageX, 0, 0};
  static int const imageY = 384 * antiAliasing;
  static int const imageZ = 512 * antiAliasing;

  std::array<std::array<RGB, imageZ * 2>, imageY * 2> bitmap;

  double const diffuseCoefficient = 0.9;
  double const ambientCoefficient = 0.1;


  void processPixel(std::vector<Sphere> const& spheres, Point const& point);
  void processPixelOnBackground(std::vector<Sphere> const& spheres, Point const& point);


public:
  void processPixels(std::vector<Sphere> const& spheres);
  void printBitmap();
};

#endif // RAYTRACER_H
