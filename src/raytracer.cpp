#include "raytracer.h"

#include "structures.h"

#include <iostream>
#include <vector>
#include <array>
#include <iomanip>
#include <algorithm>


namespace {

	double vectorlen(Vector const& vec)
	{
		return std::sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
	}

	double dotProduct(Vector const&a, Vector const& b)
	{
		return a.x*b.x + a.y*b.y + a.z*b.z;
	}

	bool pointInShadow(Point const& point, Point const& light, Sphere const& sphere)
	{
		Segment seg = {point, light};
		return intersection(seg, sphere).first;
	}

	void normalize(Vector& vec)
	{
		double len = vectorlen(vec);
		vec.x = vec.x / len;
		vec.y = vec.y / len;
		vec.z = vec.z / len;
	}

}

void RayTracer::processPixelOnBackground(std::vector<Sphere> const& spheres, Point const& pixel)
{
	if(pixel.y - observer.y >= 0)
	{
		bitmap[pixel.y + imageY][pixel.z + imageZ] = {30, 30, 30};
		return;
	}

	Point pointOnFloor;
	pointOnFloor.y = -400;
	double times = - 400 / (pixel.y - observer.y);
	
	pointOnFloor.x = (pixel.x - observer.x) * times;
	pointOnFloor.z = (pixel.z - observer.z) * times;

	Segment seg = {pointOnFloor, light};

	bool isInShadow = false;
	for(auto const& sphere : spheres)
	{
		if(intersection(seg, sphere).first)
		{
			isInShadow = true; 
			break;
		}
	}

	if(isInShadow)
		bitmap[pixel.y + imageY][pixel.z + imageZ] = { uint8_t(background.r/2), uint8_t(background.g/2), uint8_t(background.b/2)};
	else
		bitmap[pixel.y + imageY][pixel.z + imageZ] = background;

}


void RayTracer::processPixel(std::vector<Sphere> const& spheres, Point const& point)
{
	Segment seg{observer, point };
	std::vector<std::pair<std::pair<Point, double>, size_t>> distanceIndex;
	for(size_t i = 0; i<spheres.size(); i++)
	{
		Sphere const& sphere = spheres[i];
		auto const& res = intersection(seg, sphere);
		if(res.first)
			distanceIndex.push_back({ {res.second}, i});
	}
	
	if(!distanceIndex.empty())
	{
		std::sort(distanceIndex.begin(), distanceIndex.end(),
			[](std::pair<std::pair<Point, double>, int> const& a, std::pair<std::pair<Point, double>, int> const& b)
			{ return a.first.second < b.first.second; } );

		Point const& pointOnSphere =  distanceIndex[0].first.first;
		Point const& center = spheres[distanceIndex[0].second].center;
		double radius = spheres[distanceIndex[0].second].radius;
		RGB rgb = spheres[distanceIndex[0].second].color;

		bool isInShadow = false;
		for(size_t i=0; i<spheres.size(); i++)
		{
			if(i != distanceIndex[0].second && pointInShadow(pointOnSphere, light, spheres[i]))
			{
				isInShadow = true; 
				break;
			}
		}

		if(isInShadow)
		{
			bitmap[point.y + imageY][point.z + imageZ] = rgb *ambientCoefficient;
		}
		else
		{
			Point normalVector = {(pointOnSphere.x - center.x)/radius, (pointOnSphere.y - center.y)/radius, (pointOnSphere.z - center.z)/radius};
			Point unitVec = {light.x - pointOnSphere.x, light.y - pointOnSphere.y, light.z - pointOnSphere.z};
			normalize(unitVec);
			double dot = dotProduct(normalVector, unitVec);

			bitmap[point.y + imageY][point.z + imageZ] = rgb* (std::max(0.0, diffuseCoefficient * dot) + ambientCoefficient);
		}
	}
	else
		processPixelOnBackground(spheres, point);
}

void RayTracer::processPixels(std::vector<Sphere> const& spheres)
{
	for(int y = -imageY; y < imageY; ++y)
		for(int z = -imageZ; z < imageZ; ++z)
			processPixel(spheres, {imageX, static_cast<double>(y)/antiAliasing, static_cast<double>(z)/antiAliasing});

}
void RayTracer::printBitmap()
{
	// see https://en.wikipedia.org/wiki/Netpbm_format for format details

	std::cout << "P3" << std::endl;
	std::cout << imageZ*2/antiAliasing << " " << imageY*2/antiAliasing << std::endl << 255 << std::endl;
	for(int i = imageY*2 - 1; i >= 0; i -= antiAliasing)
	{
		for(int j = 0; j < imageZ*2; j += antiAliasing)
		{
			int r = 0;
			int g = 0;
			int b = 0;

			for(int ii = 0; ii < antiAliasing; ii++)
				for(int jj=0; jj < antiAliasing; jj++)
				{
					r += bitmap[i-ii][j+jj].r;
					g += bitmap[i-ii][j+jj].g;
					b += bitmap[i-ii][j+jj].b;
				}
				
			RGB color = {0,0,0};
			color.r = r/(antiAliasing*antiAliasing);
			color.g = g/(antiAliasing*antiAliasing);
			color.b = b/(antiAliasing*antiAliasing);
			std::cout << color << " ";
		}
		std::cout << std::endl;
	}
}
