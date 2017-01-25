#include "common/RayTracerConfig.h"

#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>

#include "common/StructuresOperators.h"
#include "common/Utils.h"

namespace
{
std::istream& operator>>(std::istream& in, RGB& rgb)
{
  short r, g, b;
  in >> r >> g >> b;
  rgb.r = r;
  rgb.g = g;
  rgb.b = b;
  return in;
}

Sphere parseSphere(std::ifstream& file)
{
  Sphere sphere{};
  std::string token;
  while (file >> token)
  {
    if (token[0] == '#') // comment, skip entire line
      std::getline(file, token);
    else if (token == "endSphere")
      break;
    else if (token == "center")
      file >> sphere.center.x >> sphere.center.y >> sphere.center.z;
    else if (token == "radius")
      file >> sphere.radius;
    else if (token == "color")
      file >> sphere.color;
    else if (token == "reflection")
      file >> sphere.reflectionCoefficient;
    else
      throw std::invalid_argument("Unknown token '" + token + "'");
    if (!file.good())
      throw std::invalid_argument("Invalid config file format.");
  }
  return sphere;
}

Plane parsePlane(std::ifstream& file)
{
  Plane plane{};
  std::string token;
  while (file >> token)
  {
    if (token[0] == '#') // comment, skip entire line
      std::getline(file, token);
    else if (token == "endPlane")
      break;
    else if (token == "point")
      file >> plane.P.x >> plane.P.y >> plane.P.z;
    else if (token == "normalVector")
      file >> plane.normal.x >> plane.normal.y >> plane.normal.z;
    else if (token == "coef")
      file >> plane.d;
    else if (token == "color")
      file >> plane.color;
    else if (token == "reflection")
      file >> plane.reflectionCoefficient;
    else
      throw std::invalid_argument("Unknown token '" + token + "'");
    if (!file.good())
      throw std::invalid_argument("Invalid config file format.");
  }
  return plane;
}

} // anonymous namespace

RayTracerConfig RayTracerConfig::fromFile(std::string const& path)
{
  std::ifstream file(path);
  if (!file.is_open())
    throw std::invalid_argument("Unable to open file: " + path);

  RayTracerConfig config;
  std::string token;
  while (file >> token)
  {
    if (token[0] == '#') // comment, skip entire line
      std::getline(file, token);
    else if (token == "aa")
      file >> config.antiAliasing;
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
    else if (token == "sphere")
      config.spheres.push_back(parseSphere(file));
    else if (token == "plane")
      config.planes.push_back(parsePlane(file));
    else
      throw std::invalid_argument("Unknown token '" + token + "'");
    if (!file.good())
      throw std::invalid_argument("Invalid config file format.");
  }
  return config;
}

std::ostream& operator<<(std::ostream& out, RayTracerConfig const& config)
{
  out << "antiAliasing: " << config.antiAliasing
      << "\nmaxRecursionLevel: " << config.maxRecursionLevel
      << "\nambientCoefficient: " << config.ambientCoefficient << "\nimageX: " << config.imageX
      << "\nimageY: " << config.imageY << "\nimageZ: " << config.imageZ
      << "\nobserver: " << config.observer << "\nlight: " << config.light;
  out << "\nspheres:\n";
  for (Sphere const& sphere : config.spheres)
    out << sphere << '\n';
  out << "planes:\n";
  for (Plane const& plane : config.planes)
    out << plane << '\n';
  out << "planes:\n";
  for (Triangle const& triangle : config.triangles)
    out << triangle << '\n';
  return out;
}

RayTracerConfig RayTracerConfig::defaultConfig()
{
  RayTracerConfig config;

  config.antiAliasing = 2;
  config.maxRecursionLevel = 1;
  config.ambientCoefficient = 0.1;
  config.imageX = 512;
  config.imageY = 384 * 2;
  config.imageZ = 512 * 2;
  config.observer = {0, 0, 0};
  config.light = {1000, 2000, 2500};

  // red
  config.spheres.push_back({{2500, -200, -600}, 600, {200, 0, 0}, 0.3});
  // green
  config.spheres.push_back({{2000, 0, 800}, 400, {0, 200, 0}, 0.1});

  // Plane has one face!
  // front
  config.planes.push_back({{6000, 0, 0}, {-1, 0, 0}, 6000, {178, 170, 30}, 0.05});
  // back
  config.planes.push_back({{-2000, 0, 0}, {1, 0, 0}, 2000, {245, 222, 179}});
  // top
  config.planes.push_back({{0, 3000, 0}, {0, -1, 0}, 3000, {255, 105, 180}, 0.05});
  // bottom
  config.planes.push_back({{0, -800, 0}, {0, 1, 0}, 800, {100, 100, 200}, 0.05});
  // left
  config.planes.push_back({{0, 0, -2500}, {0, 0, 1}, 2500, {32, 178, 170}, 0.05});
  // right
  config.planes.push_back({{0, 0, 3500}, {0, 0, -1}, 3500, {32, 178, 170}, 0.05});

  return config;
}

RayTracerConfig RayTracerConfig::fromPlyFile(std::string const& path)
{
  std::ifstream file(path);
  if (!file.is_open())
    throw std::invalid_argument("Unable to open file: " + path);

  RayTracerConfig config;
  std::string token;
  file >> token;
  if (token != "ply")
    throw std::invalid_argument("Not a .ply file: " + path);
  file >> token;
  if (token != "format")
    throw std::invalid_argument("Missing 'format' specifier.");
  std::getline(file, token);

  int vertexCount = 0;
  int faceCount = 0;

  std::vector<Vector> vertices;
  std::vector<RGB> colors;
  bool isRGB = false;
  while (file >> token)
  {
    if (token == "comment")
      std::getline(file, token);
    if (token == "element")
    {
      file >> token;
      file >> vertexCount;
      if (token == "vertex")
      {
        // assume format x y z, ignore other
        file >> token;
        while (token == "property")
        {
          file >> token;
          if (token == "uchar")
            isRGB = true;

          std::getline(file, token);
          file >> token;
        }
      }
      file >> token;
      file >> faceCount;
      if (token == "face")
      {
        // assume property list uchar int vertex_indices
        file >> token;
        while (token == "property")
        {
          std::getline(file, token);
          file >> token;
        }
      }
      else
        throw std::invalid_argument("Unknown element '" + token
                                    + "', only 'vertex' and 'face' are supported");
    }
    if (token == "end_header")
    {
      for (int i = 0; i < vertexCount; ++i)
      {
        Vector v;
        file >> v.x >> v.y >> v.z;
        vertices.push_back(v);

        if (isRGB)
        {
          int r, g, b;
          file >> r >> g >> b;
          colors.push_back({uint8_t(r), uint8_t(g), uint8_t(b)});
        }
      }

      for (int i = 0; i < faceCount; ++i)
      {
        int count;
        file >> count;
        if (count != 3) // only triangles supported
        {
          std::cerr << "Warning: only triangles are supported" << std::endl;
          std::getline(file, token);
          continue;
        }
        int v1, v2, v3;
        file >> v1 >> v2 >> v3;
        // RGB color{uint8_t(rand() % 200), uint8_t(rand() % 200), uint8_t(rand() % 200)};
        if (colors.empty())
          config.triangles.emplace_back(Triangle{
              vertices[v1], vertices[v2], vertices[v3], {40, 200, 40}, {50, 50, 200}, {0, 0, 0}});
        else
          config.triangles.emplace_back(Triangle{vertices[v1], vertices[v2], vertices[v3],
                                                 colors[v1], colors[v2], colors[v3]});
      }
    }
    // else
    //  throw std::invalid_argument("Unknown token '" + token + "'");
    if (!file.good())
      throw std::invalid_argument("Invalid config file format.");
  }
  return config;
}

void swapVertex(Vector& a)
{
  // teapot
  // std::swap(a.y, a.z);
  std::swap(a.x, a.z);
}

void RayTracerConfig::scaleTriangles()
{
  for (Triangle& t : triangles)
  {
    swapVertex(t.x);
    swapVertex(t.y);
    swapVertex(t.z);

    t.x.x = -t.x.x;
    t.y.x = -t.y.x;
    t.z.x = -t.z.x;
  }

  float expectedSize = 3500.f;
  float expectedDist = 2000.f;
  float expectedY = 0.f;
  float expectedZ = 0.f;

  float minX = std::numeric_limits<float>::max();
  float minY = std::numeric_limits<float>::max();
  float minZ = std::numeric_limits<float>::max();

  float maxX = std::numeric_limits<float>::min();
  float maxY = std::numeric_limits<float>::min();
  float maxZ = std::numeric_limits<float>::min();

  for (Triangle& t : triangles)
  {
    Point minP = getMinPoint(t);
    minX = std::min(minX, minP.x);
    minY = std::min(minY, minP.y);
    minZ = std::min(minZ, minP.z);

    Point maxP = getMaxPoint(t);

    maxX = std::max(maxX, maxP.x);
    maxY = std::max(maxY, maxP.y);
    maxZ = std::max(maxZ, maxP.z);
  }

  float maxDiff = std::max({maxX - minX, maxY - minY, maxZ - minZ});
  float coef = expectedSize / maxDiff;

  float diffX = expectedDist - minX * coef;
  float diffY = expectedY - coef * (maxY + minY) / 2;
  float diffZ = expectedZ - coef * (maxZ + minZ) / 2;

  for (Triangle& t : triangles)
  {
    t.x *= coef;
    t.y *= coef;
    t.z *= coef;

    t.x.x += diffX;
    t.y.x += diffX;
    t.z.x += diffX;

    t.x.y += diffY;
    t.y.y += diffY;
    t.z.y += diffY;

    t.x.z += diffZ;
    t.y.z += diffZ;
    t.z.z += diffZ;
  }
}
