#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <utility>
#include <cmath>
#include <limits>
#include <cstdint>
#include <iostream>

struct RGB
{
   uint8_t r;
   uint8_t g;
   uint8_t b;
};

inline RGB operator*(RGB rgb, double const& times)
{
  rgb.r *= times;
  rgb.g *= times;
  rgb.b *= times;
  
  return rgb;
}

std::ostream& operator <<( std::ostream& outs, RGB const& rgb);

struct Point
{
	double x; 
	double y; 
	double z;
};

std::ostream& operator <<( std::ostream& outs, Point const& point);

struct Sphere
{
	Point center;
	double radius;
	RGB color;
};
std::ostream& operator <<( std::ostream& outs, Sphere const& sphere);

struct Segment
{
	Point a;
	Point b;
};
std::ostream& operator <<( std::ostream& outs, Segment const& segment);


template <typename T>
bool isCloseToZero(T x)
{
    return std::abs(x) < std::numeric_limits<T>::epsilon();
}

// Returns true if intersection exists
// If two points of intersection exist closest is returned
std::pair<bool, std::pair<Point, double> > intersection(Segment segment, Sphere sphere);

using Vector = Point;


#endif