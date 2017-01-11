#include<cstdio>

struct Point
{
	double x; 
	double y; 
	double z;
};
struct pairPd
{
	Point first;
	double second;
};
struct pairbp
{
	bool first;
	pairPd second;
};
struct RGB
{
   unsigned char r;
   unsigned char g;
   unsigned char b;
};



struct Sphere
{
	Point center;
	double radius;
	RGB color;
};

struct Segment
{
	Point a;
	Point b;
};

extern "C" {
__device__
bool isCloseToZero(double x)
{
    return abs(x) < 0.00000001;
}
__device__
inline RGB operator*(RGB rgb, double const& times)
{
  rgb.r *= times;
  rgb.g *= times;
  rgb.b *= times;
  
  return rgb;
}
__device__
pairbp intersection(Segment segment, Sphere sphere)
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

	double t = (-b - sqrt(discriminant)) / (2*a);
	if(t < 0)
		return {false, {} };
	return {true, {{x0 + t*dx, y0 + t*dy, z0 + t*dz}, t }};

}



__device__
double vectorlen(Point const& vec)
{
	return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}
__device__
double dotProduct(Point const&a, Point const& b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__
bool pointInShadow(Point const& point, Point const& light, Sphere const& sphere)
{
	Segment seg = {point, light};
	return intersection(seg, sphere).first;
}
__device__
void normalize(Point& vec)
{
	double len = vectorlen(vec);
	vec.x = vec.x / len;
	vec.y = vec.y / len;
	vec.z = vec.z / len;
}

__device__
void processPixelOnBackground(RGB* bitmap, Sphere* spheres, Point const& pixel, int spheresNum, int imageY, int imageZ, Point const& observer, Point const& light, RGB const& background)
{
	
	int idx = ((blockIdx.x * blockDim.x) + threadIdx.x)*imageZ*2 + (blockIdx.y * blockDim.y) + threadIdx.y;
	//int idx = (pixel.y + imageY)*imageZ*2 +  pixel.z + imageZ; 

	if(pixel.y - observer.y >= 0)
	{
		//bitmap[pixel.y + imageY][pixel.z + imageZ] = {30, 30, 30};

		bitmap[idx].r = 30;
		bitmap[idx].g = 30;
		bitmap[idx].b = 30;
		return;
	}

	Point pointOnFloor;
	pointOnFloor.y = -400;
	double times = - 400 / (pixel.y - observer.y);
	
	pointOnFloor.x = (pixel.x - observer.x) * times;
	pointOnFloor.z = (pixel.z - observer.z) * times;

	Segment seg = {pointOnFloor, light};

	bool isInShadow = false;
	//for(auto const& sphere : spheres)
	for (int i=0; i<spheresNum; ++i)	
	{
		Sphere sphere = spheres[i];
		if(intersection(seg, sphere).first)
		{
			isInShadow = true; 
			break;
		}
	}

	if(isInShadow)
	{
		//bitmap[pixel.y + imageY][pixel.z + imageZ] = { uint8_t(background.r/2), uint8_t(background.g/2), uint8_t(background.b/2)};
		
		bitmap[idx].r = background.r/2;
		bitmap[idx].g = background.g/2;
		bitmap[idx].b = background.b/2;
	}

	else
	{
		bitmap[idx] = background;
	}

}

__global__
void processPixel(Sphere* spheres, int spheresNum, RGB* bitmap, int imageX, int imageY, int imageZ, int antiAliasing, double diffuseCoefficient, double ambientCoefficient, double observerX, double observerY, double observerZ, double lX, double lY, double lZ, unsigned char R, unsigned char G, unsigned char B) {
	    

	Point const observer = {observerX, observerY, observerZ};
	Point const light = {lX, lY, lZ};
	RGB background = {R, G, B};

	//int idx = (pixel.y + imageY)*imageZ*2 +  pixel.z + imageZ; 
	int thidX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int thidY = (blockIdx.y * blockDim.y) + threadIdx.y;
    
	
    if (thidX<2*imageY && thidY<2*imageZ)
    {

	Point point{imageX, ((double)(thidX-imageY))/antiAliasing, ((double)(thidY-imageZ))/antiAliasing};
		
	//Point point{imageX, (thidX-imageY), (thidY-imageZ)};
    Segment seg{observer, point };
	//std::vector<std::pair<std::pair<Point, double>, size_t>> distanceIndex;
	Point dIff;
	double dIfs;
	size_t dIs;
	
	bool intersected = false;
	//for(size_t i = 0; i<spheres.size(); i++)
	
	for (int i=0; i<spheresNum; ++i)	
	{
		Sphere const& sphere = spheres[i];
		pairbp const& res = intersection(seg, sphere);
		if(res.first)
			{
				if (!intersected || res.second.second < dIfs) {dIff = res.second.first; dIfs = res.second.second; dIs = i;}
				intersected = true;
				
				//distanceIndex.push_back({ {res.second}, i});
			}
	}
	
	//if(!distanceIndex.empty())
	if (intersected)	
	{
		//std::sort(distanceIndex.begin(), distanceIndex.end(),
		//	[](std::pair<std::pair<Point, double>, int> const& a, std::pair<std::pair<Point, double>, int> const& b)
		//	{ return a.first.second < b.first.second; } );

		Point const& pointOnSphere =  dIff; //distanceIndex[0].first.first;
		Point const& center = spheres[dIs].center; //spheres[distanceIndex[0].second].center;
		double radius = spheres[dIs].radius; //spheres[distanceIndex[0].second].radius;
		RGB rgb = spheres[dIs].color; //spheres[distanceIndex[0].second].color;

		bool isInShadow = false;
		//for(size_t i=0; i<spheres.size(); i++)
		for (int i=0; i<spheresNum; ++i)		
		{
			//if(i != distanceIndex[0].second && pointInShadow(pointOnSphere, light, spheres[i]))
			if (i!=dIs && pointInShadow(pointOnSphere, light, spheres[i]))			
			{
				isInShadow = true; 
				break;
			}
		}
		int idx = thidX*imageZ*2 + thidY;
		if(isInShadow)
		{
			bitmap[idx] = rgb *ambientCoefficient;
		}
		else
		{
			Point normalVector = {(pointOnSphere.x - center.x)/radius, (pointOnSphere.y - center.y)/radius, (pointOnSphere.z - center.z)/radius};
			Point unitVec = {light.x - pointOnSphere.x, light.y - pointOnSphere.y, light.z - pointOnSphere.z};
			normalize(unitVec);
			double dot = dotProduct(normalVector, unitVec);

			bitmap[idx] = rgb* (max(0.0, diffuseCoefficient * dot) + ambientCoefficient);
		}
	}
	else
		processPixelOnBackground(bitmap, spheres, point, spheresNum, imageY, imageZ, observer, light, background);    
    }
}
}

