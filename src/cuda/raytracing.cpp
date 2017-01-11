//#include "raytracer.h"

//#include "structures.h"

#include <iostream>
#include <vector>
#include <array>
#include <iomanip>
#include <algorithm>

#include "cuda.h"
#include <cstdio>
#include <cstdlib>

//#include "structures.h"

#include <cmath>
#include <limits>
#include <utility>


#include <vector>
#include <array>

#include <utility>
#include <cmath>
#include <limits>
#include <cstdint>
#include <iostream>

struct RGB
{
   unsigned char r;
   unsigned char g;
   unsigned char b;
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


class RayTracer
{

    // for antiAliasing = 4, 16 pixels are generated for each one from final scene
    static int const antiAliasing = 2;

	Point const observer = {0, 0, 0};
	Point const light = {2000, 2500, 2500};
	RGB background = {100, 100, 200};
	// image is a rectangle with verticles (256, -+imageY/antiAliasing, -+imageZ/antiAliasing)
	static int const imageX = 512;
	Point const imageCenter = {imageX, 0, 0}; 
	static int const imageY = 384 * antiAliasing;
	static int const imageZ = 512 * antiAliasing;

	std::array<std::array<RGB, imageZ*2>, imageY*2> bitmap;

	double const diffuseCoefficient = 0.9;
	double const ambientCoefficient = 0.1;


	void processPixel(std::vector<Sphere> const& spheres, Point const& point);
	void processPixelOnBackground(std::vector<Sphere> const& spheres, Point const& point);


public:
	void processPixels(std::vector<Sphere> const& spheres);
	void printBitmap();

};



std::pair<bool, std::pair<Point, double> > intersection(Segment segment, Sphere sphere)
{
	double x0 = segment.a.x;
	double y0 = segment.a.y;
	double z0 = segment.a.z;

	double x1 = segment.b.x;
	double y1 = segment.b.y;
	double z1 = segment.b.z;

	double dx = x1 - x0;
	double dy = y1 - y0;
	double dz = z1 - z0;

	double cx = sphere.center.x;
	double cy = sphere.center.y;
	double cz = sphere.center.z;
	
	double a = dx*dx + dy*dy + dz*dz;
	double b = 2*dx*(x0-cx) +  2*dy*(y0-cy) +  2*dz*(z0-cz);
	double c = cx*cx + cy*cy + cz*cz + x0*x0 + y0*y0 + z0*z0 -2*(cx*x0 + cy*y0 + cz*z0) - sphere.radius * sphere.radius;

	double discriminant = b*b - 4*a*c;
	if(!isCloseToZero(discriminant) && discriminant < 0)
		return {false, {} };

	double t = (-b - std::sqrt(discriminant)) / (2*a);
	if(t < 0)
		return {false, {} };
	return {true, {{x0 + t*dx, y0 + t*dy, z0 + t*dz}, t }};

}


std::ostream& operator <<( std::ostream& outs, RGB const& rgb)
{
  outs << static_cast<int16_t>(rgb.r) << " " << static_cast<int16_t>(rgb.g) << " " << static_cast<int16_t>(rgb.b);
  return outs;
}

std::ostream& operator <<( std::ostream& outs, Point const& point)
{  
  outs << "{ " << point.x << " " << point.y << " " << point.z << "} ";
  return outs;
}

std::ostream& operator <<( std::ostream& outs, Segment const& segment)
{  
  outs << "A: " << segment.a << " B: " << segment.b;
  return outs;
}

std::ostream& operator <<( std::ostream& outs, Sphere const& sphere)
{  
  outs << "Center: " << sphere.center << " R: " << sphere.radius;
  return outs;
}
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

void RayTracer::processPixels(std::vector<Sphere> const& spheres)
{
//int i;
//std::cout<<"DDDD";
//std::cin>>i;
	cuInit(0);
    
    
    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot acquire device 0\n"); 
        exit(1);
    }

   
    CUcontext cuContext;
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        printf("cannot create context\n");
        exit(1);
    }

    
    CUmodule cuModule = (CUmodule)0;
    res = cuModuleLoad(&cuModule, "raytracing.ptx");
    if (res != CUDA_SUCCESS) {
        printf("cannot load module: %d\n", res);  
        exit(1); 
    }

    
    CUfunction processPixel;
    res = cuModuleGetFunction(&processPixel, cuModule, "processPixel");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }

	/*int* A = (int*) malloc(sizeof(int)*((r+1023)/1024));
	res = cuMemHostRegister(A, sizeof(int)*((r+1023)/1024), 0);
    if (res != CUDA_SUCCESS){
        exit(1);
    }*/

	int blocks_per_grid_x = (2*imageY + 31)/32;
	int blocks_per_grid_y = (2*imageZ + 31)/32;
    int threads_per_block_x = 32;
	int threads_per_block_y = 32;

	int spheresNum = spheres.size();

	RGB* bitmapTab = (RGB*) malloc(sizeof(RGB)*2*imageY*2*imageZ);
	res = cuMemHostRegister(bitmapTab, sizeof(RGB)*2*imageY*2*imageZ, 0);
    if (res != CUDA_SUCCESS){
        exit(1);
    }
//std::cout<<"AAA";
//std::cin>>i;
	for (int i=0; i<2*imageY*2*imageZ; ++i)
	{
		bitmapTab[i] = bitmap[i/(2*imageZ)][i%(2*imageZ)];
	}

	CUdeviceptr bitmapDev;
    res = cuMemAlloc(&bitmapDev, sizeof(RGB)*(2*imageY*2*imageZ));
    if (res != CUDA_SUCCESS){
		printf("cannot acquireB kernel handle\n");
        exit(1);
    }

	res = cuMemcpyHtoD(bitmapDev, bitmapTab, sizeof(RGB)*(2*imageY*2*imageZ));
    if (res != CUDA_SUCCESS){
		printf("cannot acquire kernel handle\n");        
		exit(1);
    }	

	Sphere* spheresTab = (Sphere*) malloc(sizeof(Sphere)*spheresNum);
	res = cuMemHostRegister(spheresTab, sizeof(Sphere)*spheresNum, 0);
    if (res != CUDA_SUCCESS){
        exit(1);
    }

	for(int i=0; i<spheresNum; ++i)
	{
		spheresTab[i]=spheres[i];
	}

	CUdeviceptr spheresDev;
    res = cuMemAlloc(&spheresDev, sizeof(Sphere)*(spheresNum));
    if (res != CUDA_SUCCESS){
		printf("cannot acquireB kernel handle\n");
        exit(1);
    }
//std::cout<<"BBB";
//std::cin>>i;
	res = cuMemcpyHtoD(spheresDev, spheresTab, sizeof(Sphere)*(spheresNum));
    if (res != CUDA_SUCCESS){
		printf("cannot acquire kernel handle\n");        
		exit(1);
    }	
	int iX = (int) imageX;
	int iY = (int) imageY;
	int iZ = (int) imageZ;
	int aA = (int) antiAliasing;
	double dC = (double) diffuseCoefficient;
	double aC = (double) ambientCoefficient;
	double oX = (double) observer.x;
	double oY = (double) observer.y;
	double oZ = (double) observer.z;
	double lX = (double) light.x;
	double lY = (double) light.y;
	double lZ = (double) light.z;
	unsigned char R = (unsigned char) background.r;
	unsigned char G = (unsigned char) background.g;
	unsigned char B = (unsigned char) background.b;

	void* args[] = {&spheresDev, &spheresNum, &bitmapDev, &iX, &iY, &iZ, &aA, &dC, &aC, &oX, &oY, &oZ, &lX, &lY, &lZ, &R, &G, &B};

    res = cuLaunchKernel(processPixel, blocks_per_grid_x, blocks_per_grid_y, 1, threads_per_block_x, threads_per_block_y, 1, 0, 0, args, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }
//std::cout<<"CCC";
//std::cin>>i;
	res = cuMemcpyDtoH(bitmapTab, bitmapDev, sizeof(RGB)*(2*imageY*2*imageZ));
    if (res != CUDA_SUCCESS){
		printf("cannot acquire kernel handle\n");        
		exit(1);
    }
//	std::cout<<"CDDDCC";
//std::cin>>i;
	for (int i=0; i<2*imageY*2*imageZ; ++i)
	{
		bitmap[i/(2*imageZ)][i%(2*imageZ)] = bitmapTab[i];
	}

	/*for(int y = -imageY; y < imageY; ++y)
		for(int z = -imageZ; z < imageZ; ++z)
			processPixel(spheres, {imageX, static_cast<double>(y)/antiAliasing, static_cast<double>(z)/antiAliasing});
	*/
//std::cout<<"CCCsad";
//std::cin>>i;
	free(bitmapTab);
	free(spheresTab);
//std::cout<<"asdCCC";
//std::cin>>i;
	cuCtxDestroy(cuContext);

}


int main()
{
	std::cout<<std::fixed;
	std::cout<<std::setprecision(3);
	int i=0;
	//std::cout<<"ASDA";
	//std::cin>>i;
	RayTracer tracer;
//std::cout<<"ASdADADAD";
//std::cin>>i;
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
