#include <iomanip>
#include <iostream>

#include "common/RayTracerConfig.h"
#include "common/Utils.h"
#include "common/Structures.h"
#include "cpuTriangles/RayTracerTriangles.h"

int main(int argc, char* argv[])
{
  RayTracerConfig config;
  if (argc > 1)
  {
    std::cerr << "Reading config from file " << argv[1] << std::endl;
    config = RayTracerConfig::fromPlyFile(argv[1]);
    config.antiAliasing = 2;
    config.maxRecursionLevel = 1;
    config.ambientCoefficient = 0.1;
    config.imageX = 512;
    config.imageY = 768;
    //config.imageY = 500;
    config.imageZ = 1024;
    //config.imageZ = 500;
    config.observer = {0, 0, 0};
    config.light = {200, 1000, 200};
    config.scaleTriangles();
    config.background = {255, 255, 255};
  }
  else
  {
    std::cerr << "Using default config" << std::endl;
    config = RayTracerConfig::defaultConfig();

    Triangle tr;
    tr.x = {2500, -200, -600};
    tr.y = {2500, 600, 0};
    tr.z = {2500, 200, 300};
    tr.colorX = {0, 0, 200};
    tr.colorY = {200, 0, 0};
    tr.colorZ = {0, 200, 0};
    tr.reflectionCoefficient = 0.1;
    tr.normal = getNormalVector(tr);

    config.triangles.push_back(tr);


    for (int i = 0; i < 20; i++)
    {
      float x = rand() % 5000 + 1000.0;
      float y = rand() % 3000 - 800.0;
      float z = rand() % 5000 - 2000.0;

      float xx = rand() % 400 + 10.0;
      float yy = rand() % 400 + 10.0;
      float zz = rand() % 400 + 10.0;

      Triangle triangle;
      triangle.x = {x, y, z};
      triangle.y = {x, y + yy, z + zz};
      triangle.z = {x + xx, y + yy, z};

      triangle.colorX = {uint8_t(rand() % 200), uint8_t(rand() % 200), uint8_t(rand() % 200)};
      triangle.colorY = {uint8_t(rand() % 200), uint8_t(rand() % 200), uint8_t(rand() % 200)};
      triangle.colorZ = {uint8_t(rand() % 200), uint8_t(rand() % 200), uint8_t(rand() % 200)};
      triangle.reflectionCoefficient = 0.1;
      triangle.normal = getNormalVector(triangle);

      config.triangles.push_back(triangle);
    }
  }
  // std::cerr << config;

  RayTracerTriangles tracer(config);
  tracer.processPixels();
  tracer.printBitmap(std::cout);
}
