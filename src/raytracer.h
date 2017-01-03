#ifndef UTILS_H
#define UTILS_H

#include "structures.h"


#include <vector>
#include <array>


class RayTracer
{
	Point const observer = {0, 0, 0};
	Point const light = {1000, 2000, 2000};
	RGB background = {100, 100, 200};
	// image is a rectangle with verticles (256, -+image_y, -+image_z)

	static int const image_x = 256;
	Point const imageCenter = {image_x, 0, 0}; 
	static int const image_y = 256;
	static int const image_z = 256;

	std::array<std::array<RGB, image_z*2>, image_y*2> bitmap;

	double const diffuseCoefficient = 0.9;
	double const ambientCoefficient = 0.1;
	void processPixel(std::vector<Sphere> const& spheres, Point const& point);
	void processPixelOnBackground(std::vector<Sphere> const& spheres, Point const& point);


public:
	void processPixels(std::vector<Sphere> const& spheres);
	void printBitmap();

};

#endif