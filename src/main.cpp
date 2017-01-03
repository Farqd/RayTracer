#include "structures.h"
#include "raytracer.h"

#include <iostream>
#include <vector>
#include <array>
#include <iomanip>
#include <algorithm>

// See http://www.ccs.neu.edu/home/fell/CSU540/programs/RayTracingFormulas.htm


int main()
{
	std::cout<<std::fixed;
	std::cout<<std::setprecision(3);
	
	RayTracer tracer;
	/*Segment seg{{0, 0, 0}, {100, 100,0}};
	Sphere sp{{100, 100, 0}, 20};
	auto const& res = intersection(seg, sp);
	if(res.first)
		std::cout<<res.second.first<<" "<<res.second.second<<std::endl;
	*/
	std::vector<Sphere> spheres;

	spheres.push_back({{1500, 200, -400}, 400, {200, 0, 0}});

	spheres.push_back({{1300, 400, 400}, 200, {0, 200, 0}});

	tracer.processPixels(spheres);
	tracer.printBitmap();


}