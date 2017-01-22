#include "Utils.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "common/Structures.h"

float dotProduct(Vector const& a, Vector const& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vector crossProduct(Vector const& a, Vector const& b)
{
  Vector vec;
  vec.x = a.y * b.z - b.y * a.z;
  vec.y = a.z * b.x - a.x * b.z;
  vec.z = a.x * b.y - a.y * b.x;
  return vec;
}

float vectorLen(Vector const& vec)
{
  return std::sqrt(dotProduct(vec, vec));
}

Vector normalize(Vector const& vec)
{
  return vec / vectorLen(vec);
}

float distance(Point const& a, Point const& b)
{
  return vectorLen(b - a);
}

std::pair<bool, Point> intersection(Segment const& segment, Sphere const& sphere)
{
  float x0 = segment.a.x;
  float y0 = segment.a.y;
  float z0 = segment.a.z;

  float x1 = segment.b.x;
  float y1 = segment.b.y;
  float z1 = segment.b.z;

  float dx = x1 - x0;
  float dy = y1 - y0;
  float dz = z1 - z0;

  float cx = sphere.center.x;
  float cy = sphere.center.y;
  float cz = sphere.center.z;

  float a = dx * dx + dy * dy + dz * dz;
  float b = 2 * dx * (x0 - cx) + 2 * dy * (y0 - cy) + 2 * dz * (z0 - cz);
  float c = cx * cx + cy * cy + cz * cz + x0 * x0 + y0 * y0 + z0 * z0
            - 2 * (cx * x0 + cy * y0 + cz * z0) - sphere.radius * sphere.radius;

  float discriminant = b * b - 4 * a * c;
  if (!isCloseToZero(discriminant) && discriminant < 0)
    return {false, {}};

  float t = (-b - std::sqrt(discriminant)) / (2 * a);
  if (t < 0)
    return {false, {}};

  return {true, {x0 + t * dx, y0 + t * dy, z0 + t * dz}};
}

Segment reflection(Segment const& segment, Sphere const& sphere)
{
  Point normalVector = normalize((segment.b - sphere.center) / sphere.radius);

  Vector ri = normalize(segment.b - segment.a);
  float dot = dotProduct(ri, normalVector);
  ri -= normalVector * (2 * dot);

  return {segment.b, segment.b + ri};
}

std::pair<bool, Point> intersection(Segment const& segment, Plane const& plane)
{
  Vector V = segment.b - segment.a;
  float x = dotProduct(V, plane.normal);
  if (x == 0)
    return {false, {}};

  float t = -(dotProduct(segment.a, plane.normal) + plane.d) / x;
  if (t < 0 || isCloseToZero(t))
    return {false, {}};

  return {true, segment.a + V * t};
}

Segment reflection(Segment const& segment, Plane const& plane)
{
  Vector ri = segment.b - segment.a;
  ri -= plane.normal * (2 * dotProduct(ri, plane.normal));
  return {segment.b, segment.b + ri};
}

Segment reflection(Segment const& segment, Triangle const& triangle)
{
  Vector ri = segment.b - segment.a;
	Point v0 = triangle.x;
	Point v1 = triangle.y;
	Point v2 = triangle.z;

    
  Vector v0v1 = v1 - v0; 
  Vector v0v2 = v2 - v0; 
    
  Vector N = normalize(crossProduct(v0v1, v0v2)); 
 
	

 	ri -= N * (2 * dotProduct(ri, N));
  return {segment.b, segment.b + ri};
}

std::pair<bool, Point> intersection(Segment const& segment, Triangle const& triangle)
{
  float t;
  float u;
  float v;

  Point orig = segment.a;
  Point dir = segment.b;
  Point v0 = triangle.x;
  Point v1 = triangle.y;
  Point v2 = triangle.z;


  Vector v0v1 = v1 - v0;
  Vector v0v2 = v2 - v0;

  Vector N = crossProduct(v0v1, v0v2);
  float denom = dotProduct(N, N);


  float NdotRayDirection = dotProduct(N, dir);
  if (isCloseToZero(NdotRayDirection))
    return {false, {}};


  float d = dotProduct(N, v0);


  t = (dotProduct(N, orig) + d) / NdotRayDirection;

  if (t < 0)
    return {false, {}};


  Point P = orig + dir * t;


  Vector C;


  Vector edge0 = v1 - v0;
  Vector vp0 = P - v0;
  C = crossProduct(edge0, vp0);
  if (dotProduct(N, C) < 0)
    return {false, {}};


  Vector edge1 = v2 - v1;
  Vector vp1 = P - v1;
  C = crossProduct(edge1, vp1);
  if ((u = dotProduct(N, C)) < 0)
    return {false, {}};


  Vector edge2 = v0 - v2;
  Vector vp2 = P - v2;
  C = crossProduct(edge2, vp2);
  if ((v = dotProduct(N, C)) < 0)
    return {false, {}};

  u /= denom;
  v /= denom;

  return {true, P};
}

// We assume point is on triangle
RGB colorOfPoint(Point const& point, Triangle const& triangle)
{
  float u;
  float v;

  Point v0 = triangle.x;
  Point v1 = triangle.y;
  Point v2 = triangle.z;


  Vector v0v1 = v1 - v0;
  Vector v0v2 = v2 - v0;

  Vector N = crossProduct(v0v1, v0v2);
  float denom = dotProduct(N, N);

  Vector C;

  Vector edge1 = v2 - v1;
  Vector vp1 = point - v1;
  C = crossProduct(edge1, vp1);
  u = dotProduct(N, C);


  Vector edge2 = v0 - v2;
  Vector vp2 = point - v2;
  C = crossProduct(edge2, vp2);
  v = dotProduct(N, C);

  u /= denom;
  v /= denom;

  return triangle.colorX * u + triangle.colorY * v + triangle.colorZ * (1 - u - v);
}

bool intersectionWithRectangle(Segment const& segment, Point const& p0, Point const& p1, Point const& p2, Point const& p3)
{
	Triangle triangle = {p0, p1, p2, {}, {}, {}};
	if (intersection(segment, triangle).first) return true;
	triangle.y = p3;
	if (intersection(segment, triangle).first) return true;
	return false;
}

bool intersection(Segment const& segment, BoundingBox const& box)
{
	Point const& vMin = box.vMin;
	Point const& vMax = box.vMax;

	Point p0 = vMin;
	Point p1 = {vMin.x, vMin.y, vMax.z};
	Point p2 = {vMin.x, vMax.y, vMax.z};
	Point p3 = {vMin.x, vMax.y, vMin.z};
	if (intersectionWithRectangle(segment, p0, p1, p2, p3)) return true;
	//p0 = vMin;
	//p1 = {vMin.x, vMin.y, vMax.z};
	p2 = {vMax.x, vMin.y, vMax.z};
	p3 = {vMax.x, vMin.y, vMin.z};
	if (intersectionWithRectangle(segment, p0, p1, p2, p3)) return true;
	//p0 = vMin;
	p1 = {vMin.x, vMax.y, vMin.z};
	p2 = {vMax.x, vMax.y, vMin.z};
	//p3 = {vMax.x, vMin.y, vMin.z};
	if (intersectionWithRectangle(segment, p0, p1, p2, p3)) return true;

	p0 = vMax;
	p1 = {vMax.x, vMax.y, vMin.z};
	p2 = {vMax.x, vMin.y, vMin.z};
	p3 = {vMax.x, vMin.y, vMax.z};
	if (intersectionWithRectangle(segment, p0, p1, p2, p3)) return true;

	//p0 = vMax;
	//p1 = {vMax.x, vMax.y, vMin.z};
	p2 = {vMin.x, vMax.y, vMin.z};
	p3 = {vMin.x, vMax.y, vMax.z};
	if (intersectionWithRectangle(segment, p0, p1, p2, p3)) return true;
	//p0 = vMax;
	p1 = {vMax.x, vMin.y, vMax.z};
	p2 = {vMin.x, vMin.y, vMax.z};
	//p3 = {vMin.x, vMax.y, vMax.z};
	if (intersectionWithRectangle(segment, p0, p1, p2, p3)) return true;

  return false;
}
