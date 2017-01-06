#include "raytracer.h"
#include "structures.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <vector>

// See http://www.ccs.neu.edu/home/fell/CSU540/programs/RayTracingFormulas.htm


int main()
{
  std::cout << std::fixed;
  std::cout << std::setprecision(3);

  RayTracer tracer;

  /*Segment seg{{0, 0, 0}, {100, 100,0}};
  Sphere sp{{100, 100, 0}, 20};
  auto const& res = intersection(seg, sp);
  if(res.first)
    std::cout<<res.second.first<<" "<<res.second.second<<std::endl;
  */
  std::vector<Sphere> spheres;

  spheres.push_back({{2500, 200, -600}, 600, {200, 0, 0}});

  spheres.push_back({{2300, 500, 800}, 400, {0, 200, 0}});

  tracer.processPixels(spheres);
  tracer.printBitmap();
}
