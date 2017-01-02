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

}

/*
If the pixel is background color, use the Ray-Sphere Intersection formulas with
            P0 = pixel = (x0, y0, 0)
            P1 = Light = (Lx, Ly, Lz)
Intersect this ray with each sphere in your scene.
           
If there is any intersection, the pixel is in shadow.
            Use half (or less) the R, G, B of your background color.
If there is no intersection, use the full R, G, B of the background color.

*/
void RayTracer::processPixelOnBackground(std::vector<Sphere> const& spheres, Point const& point)
{
	Segment seg = {point, light};
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
		bitmap[point.y + image_y][point.z + image_z] = { uint8_t(background.r/2), uint8_t(background.g/2), uint8_t(background.b/2)};
	else
		bitmap[point.y + image_y][point.z + image_z] = background;

}


void RayTracer::processPixel(std::vector<Sphere> const& spheres, Point const& point)
{
	// TODO FIX AND REFACTOR
	
	//std::cout << point << std::endl;
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

		Segment seg = {pointOnSphere, light};
		bool isInShadow = false;
		for(size_t i=0; i<spheres.size(); i++)
		{
			auto const& sphere = spheres[i];
			if(i != distanceIndex[0].second)
			if(intersection(seg, sphere).first)
			{
				isInShadow = true; 
				break;
			}
		}

		if(isInShadow)
		{
			bitmap[point.y + image_y][point.z + image_z] = rgb *ambientCoefficient;
		}
		else
		{

			Point normalVector = {(pointOnSphere.x - center.x)/radius, (pointOnSphere.y - center.y)/radius, (pointOnSphere.z - center.z)/radius};
			Point unitVec = {light.x - pointOnSphere.x, light.y - pointOnSphere.y, light.z - pointOnSphere.z};
			double len = vectorlen(unitVec);
			unitVec.x = unitVec.x / len;
			unitVec.y = unitVec.y / len;
			unitVec.z = unitVec.z / len;
			double dot = dotProduct(normalVector, unitVec);

			bitmap[point.y + image_y][point.z + image_z] = rgb* (diffuseCoefficient * dot + ambientCoefficient);
		}
	}
	else
		processPixelOnBackground(spheres, point);
}

void RayTracer::processPixels(std::vector<Sphere> const& spheres)
{
	for(int y = -image_y; y < image_y; ++y)
		for(int z = -image_z; z < image_z; ++z)
			processPixel(spheres, {image_x, static_cast<double>(y), static_cast<double>(z)});

}
void RayTracer::printBitmap()
{
	// see https://en.wikipedia.org/wiki/Netpbm_format for format details

	std::cout<<"P3"<<std::endl;
	std::cout<<512<<" "<<512<<std::endl<<255<<std::endl;
	for(auto const& row : bitmap)
	{
		for(auto const& pixel : row)
			std::cout << pixel << " ";
		std::cout << std::endl;
	}
}