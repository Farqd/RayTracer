#include "common/RayTracerBase.h"

#include "common/StructuresOperators.h"
#include <iomanip>

void RayTracerBase::printBitmap(std::ostream& out)
{
  // see https://en.wikipedia.org/wiki/Netpbm_format for format details
  out << std::fixed << std::setprecision(3);

  out << "P3" << std::endl;
  out << config.imageZ * 2 / config.antiAliasing << " " << config.imageY * 2 / config.antiAliasing
      << std::endl
      << 255 << std::endl;
  for (int i = config.imageY * 2 - 1; i >= 0; i -= config.antiAliasing)
  {
    for (int j = 0; j < config.imageZ * 2; j += config.antiAliasing)
    {
      int r = 0;
      int g = 0;
      int b = 0;

      for (int ii = 0; ii < config.antiAliasing; ii++)
        for (int jj = 0; jj < config.antiAliasing; jj++)
        {
          r += bitmap(i - ii, j + jj).r;
          g += bitmap(i - ii, j + jj).g;
          b += bitmap(i - ii, j + jj).b;
        }

      RGB color = {0, 0, 0};
      color.r = r / (config.antiAliasing * config.antiAliasing);
      color.g = g / (config.antiAliasing * config.antiAliasing);
      color.b = b / (config.antiAliasing * config.antiAliasing);
      out << color << " ";
    }
    out << std::endl;
  }
}
