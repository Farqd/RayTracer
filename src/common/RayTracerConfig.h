#ifndef COMMON_RAYTRACERCONFIG_H
#define COMMON_RAYTRACERCONFIG_H

#include <exception>
#include <fstream>
#include <iostream>

#include "common/Structures.h"
#include "common/StructuresOperators.h"

struct RayTracerConfig
{
  int antiAliasing = 2;
  int maxRecursionLevel = 1;
  float diffuseCoefficient = 0.9;
  float ambientCoefficient = 0.1;

  int imageX = 512;
  int imageY = 384 * antiAliasing;
  int imageZ = 512 * antiAliasing;
  Point observer = {0, 0, 0};
  Point light = {1000, 2000, 2500};

  static RayTracerConfig fromFile(std::string const& path)
  {
    std::ifstream file(path);
    if (!file.is_open())
      throw std::invalid_argument("Unable to open file: " + path);

    RayTracerConfig config;
    std::string token;
    while (file >> token)
    {
      if (token == "aa")
        file >> config.antiAliasing;
      else if (token == "diffuse")
        file >> config.diffuseCoefficient;
      else if (token == "ambient")
        file >> config.ambientCoefficient;
      else if (token == "maxRecursion")
        file >> config.maxRecursionLevel;
      else if (token == "depth")
        file >> config.imageX;
      else if (token == "height")
        file >> config.imageY;
      else if (token == "width")
        file >> config.imageZ;
      else if (token == "observer")
        file >> config.observer.x >> config.observer.y >> config.observer.z;
      else if (token == "light")
        file >> config.light.x >> config.light.y >> config.light.z;
      else
        throw std::invalid_argument("Unknown token '" + token + "'");
      if (!file.good())
        throw std::invalid_argument("Invalid config file format.");
    }
    config.imageY *= config.antiAliasing;
    config.imageZ *= config.antiAliasing;
    return config;
  }

  friend std::ostream& operator<<(std::ostream& out, RayTracerConfig const& config)
  {
    return out << "antiAliasing: " << config.antiAliasing
               << "\nmaxRecursionLevel: " << config.maxRecursionLevel
               << "\ndiffuseCoefficient: " << config.diffuseCoefficient
               << "\nambientCoefficient: " << config.ambientCoefficient
               << "\nimageX: " << config.imageX
               << "\nimageY: " << config.imageY / config.antiAliasing
               << "\nimageZ: " << config.imageZ / config.antiAliasing
               << "\nobserver: " << config.observer << "\nlight: " << config.light << std::endl;
  }
};

#endif // COMMON_RAYTRACERCONFIG_H
