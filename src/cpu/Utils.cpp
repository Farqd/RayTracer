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

bool intersects(Segment const& segment, Triangle const& triangle)
{
	Vector const& V1 = triangle.x;
	Vector const& V2 = triangle.y;
	Vector const& V3 = triangle.z;
	
	Vector const& O = segment.a;
	Vector const& D = segment.b;

  Vector e1, e2;  
  Vector P, Q, T;
  float det, inv_det, u, v;
  float t;

  e1 = V2 - V1;
  e2 = V3 - V1;

  P = crossProduct(D, e2);
  
  det = dotProduct(e1, P);

  if(isCloseToZero(det)) return false;
  inv_det = 1.f / det;
  T = O - V1;
  u = dotProduct(T, P) * inv_det;
  if(u < 0.f || u > 1.f) return false;

  Q = crossProduct(T, e1);

  v = dotProduct(D, Q) * inv_det;
 
  if(v < 0.f || u + v  > 1.f) return false;

  t = dotProduct(e2, Q) * inv_det;

  if(t > 0 && !isCloseToZero(t)) {
    return true;
  }

  
  return false;
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
	if (intersects(segment, triangle)) return true;
	triangle.y = p3;
	if (intersects(segment, triangle)) return true;
	return false;
}

bool intersection(Segment const& segment, BoundingBox const& box)
{
	float dirfracX;
	float dirfracY;
	float dirfracZ;
	dirfracX = 1.0f / segment.b.x;
	dirfracY = 1.0f / segment.b.y;
	dirfracZ = 1.0f / segment.b.z;
	
	float t1 = (box.vMin.x - segment.a.x)*dirfracX;
	float t2 = (box.vMax.x - segment.a.x)*dirfracX;
	float t3 = (box.vMin.y - segment.a.y)*dirfracY;
	float t4 = (box.vMax.y - segment.a.y)*dirfracY;
	float t5 = (box.vMin.z - segment.a.z)*dirfracZ;
	float t6 = (box.vMax.z - segment.a.z)*dirfracZ;

	float tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
	float tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));

	
	if (tmax < 0)
	{
		  return false;
	}


	if (tmin > tmax)
	{
		  return false;
	}
	return true;

}
