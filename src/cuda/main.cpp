#include <iomanip>
#include <iostream>

#include "common/RayTracerConfig.h"
#include "common/Structures.h"
#include "cuda/RayTracerCuda.h"

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
    config = RayTracerConfig::defaultConfig();
  }
  std::cerr << config;

  RayTracerCuda tracer(config);
  tracer.processPixelsCuda();
  tracer.printBitmap(std::cout);
}
