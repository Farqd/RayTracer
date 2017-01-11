#include <iomanip>
#include <iostream>
#include <vector>

#include "common/structures.h"
#include "cuda/raytracercuda.h"

int main()
{
  std::cout << std::fixed;
  std::cout << std::setprecision(3);
  int i = 0;

  RayTracerCuda tracer;

  std::vector<Sphere> spheres;

  spheres.push_back({{2500, 200, -600}, 600, {200, 0, 0}});

  spheres.push_back({{2300, 500, 800}, 400, {0, 200, 0}});

  tracer.processPixelsCuda(spheres);
  tracer.printBitmap();
}
