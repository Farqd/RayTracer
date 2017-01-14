#include <iomanip>
#include <iostream>

#include "common/RayTracerConfig.h"
#include "common/Structures.h"
#include "cuda/RayTracerCuda.h"

int main()
{
  RayTracerConfig config;
  config.antiAliasing = 2;
  config.maxRecursionLevel = 1;
  config.diffuseCoefficient = 0.9;
  config.ambientCoefficient = 0.1;
  config.imageX = 512;
  config.imageY = 384 * config.antiAliasing;
  config.imageZ = 512 * config.antiAliasing;
  config.imageCenter = {static_cast<float>(config.imageX), 0, 0};
  config.observer = {0, 0, 0};
  config.light = {1000, 2000, 2500};

  RayTracerCuda tracer(config);


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

  tracer.processPixelsCuda();
  tracer.printBitmap(std::cout);
}
