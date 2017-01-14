#include <iomanip>
#include <iostream>

#include "RayTracer.h"
#include "common/RayTracerConfig.h"
#include "common/Structures.h"

int main(int argc, char* argv[])
{
  RayTracerConfig config;
  if (argc > 1)
  {
    std::cerr << "Reading config from file " << argv[1] << std::endl;
    config = RayTracerConfig::fromFile(argv[1]);
  }
  else
  {
    std::cerr << "Using default config" << std::endl;
  }
  std::cerr << config;

  RayTracer tracer(config);

  // red
  tracer.spheres.push_back({{2500, -200, -600}, 600, {200, 0, 0}, 0.3});

  // green
  tracer.spheres.push_back({{2000, 0, 800}, 400, {0, 200, 0}, 0.1});


  // Plane has one face!

  // front
  tracer.planes.push_back({{6000, 0, 0}, {-1, 0, 0}, 6000, {178, 170, 30}, 0.05});

  // back
  tracer.planes.push_back({{-2000, 0, 0}, {1, 0, 0}, 2000, {245, 222, 179}});

  // top
  tracer.planes.push_back({{0, 3000, 0}, {0, -1, 0}, 3000, {255, 105, 180}, 0.05});

  // bottom
  tracer.planes.push_back({{0, -800, 0}, {0, 1, 0}, 800, {100, 100, 200}, 0.05});

  // left
  tracer.planes.push_back({{0, 0, -2500}, {0, 0, 1}, 2500, {32, 178, 170}, 0.05});

  // right
  tracer.planes.push_back({{0, 0, 3500}, {0, 0, -1}, 3500, {32, 178, 170}, 0.05});


  tracer.processPixels();
  tracer.printBitmap(std::cout);
}
