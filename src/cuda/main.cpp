#include <iomanip>
#include <iostream>
#include <vector>

#include "common/structures.h"
#include "cuda/raytracercuda.h"

int main()
{
  std::cout << std::fixed;
  std::cout << std::setprecision(3);

  RayTracerCuda tracer;
  tracer.spheres.push_back({{2500, 200, -600}, 600, {200, 0, 0}});
  tracer.spheres.push_back({{2300, 500, 800}, 400, {0, 200, 0}});

  tracer.processPixelsCuda();
  tracer.printBitmap();
}
