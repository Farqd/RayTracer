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
    config = RayTracerConfig::defaultConfig();
  }
  std::cerr << config;

  RayTracer tracer(config);
  tracer.processPixels();
  tracer.printBitmap(std::cout);
}
