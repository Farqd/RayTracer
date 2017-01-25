#include <cfloat>
#include <cstdint>

#include <host_defines.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#include "common/Structures.h"
#include "cudaTriangles/CuUtils.h"
#include "cudaTriangles/KdTreeStructures.h"


struct FindTriangleResult
{
  bool exists = false;
  int triangle;
  float dist;
  Point point;
};

struct Stack
{
  size_t size = 0;
  int data[30];

  __device__ void push(int const& t)
  {
    data[size++] = t;
  }

  __device__ int pop()
  {
    return data[--size];
  }
};

extern "C" {

__device__ FindTriangleResult findTriangleLeafNode(
    int leafIdx, Segment const& ray, int excludedTriangle, KdTreeData const& treeData)
{
  FindTriangleResult res{};

  res.dist = FLT_MAX;

  LeafNode const leafNode = treeData.leafNodes[-leafIdx - 1];

  for (int i = leafNode.firstTriangle; i < leafNode.firstTriangle + leafNode.triangleCount; ++i)
  {
    if (i == excludedTriangle)
      continue;
    IntersectionResult const intersec = intersection(ray, treeData.triangles[i]);

    float dist = distance(ray.a, intersec.intersectionPoint);
    if (intersec.intersects && dist < res.dist)
    {
      res.dist = dist;
      res.exists = true;
      res.point = intersec.intersectionPoint;
      res.triangle = i;
    }
  }

  return res;
}

__device__ FindTriangleResult findTriangleSplitNode(
    int nodeIdx, Segment const& ray, int excludedTriangle, KdTreeData const& treeData)
{
  FindTriangleResult res{};
  res.dist = FLT_MAX;

  SplitNode currentNode;

  Stack stack;
  stack.push(nodeIdx);

  while (true)
  {
    if (stack.size == 0)
      return res;
    currentNode = treeData.splitNodes[stack.pop() - 1];

    int idxR = currentNode.rightChild;
    if (idxR < 0)
    {
      FindTriangleResult resR = findTriangleLeafNode(idxR, ray, excludedTriangle, treeData);
      if (resR.exists && resR.dist < res.dist)
        res = resR;
    }
    else if (idxR > 0)
    {
      SplitNode const& rightSplit = treeData.splitNodes[idxR - 1];
      if (intersectsBoundingBox(ray, rightSplit.bb))
        stack.push(idxR);
    }

    int idxL = currentNode.leftChild;
    if (idxL < 0)
    {
      FindTriangleResult resL = findTriangleLeafNode(idxL, ray, excludedTriangle, treeData);
      if (resL.exists && resL.dist < res.dist)
        res = resL;
    }
    else if (idxL > 0)
    {
      SplitNode const& leftSplit = treeData.splitNodes[idxL - 1];
      if (intersectsBoundingBox(ray, leftSplit.bb))
        stack.push(idxL);
    }
  }
}

__device__ RGB processPixelOnBackground(BaseConfig const& config)
{
  return config.background;
}

__device__ bool pointInShadow(Point const& pointOnTriangle, int excludedTriangle, KdTreeData const& treeData, BaseConfig const& config)
{
  Segment ray = {pointOnTriangle, config.light};
  FindTriangleResult res;
  if (treeData.treeRoot < 0)
    res = findTriangleLeafNode(treeData.treeRoot, ray, excludedTriangle, treeData);
  else // if (treeData.treeRoot > 0)
    res = findTriangleSplitNode(treeData.treeRoot, ray, excludedTriangle, treeData);

  return res.exists && distance(pointOnTriangle, res.point) < distance(pointOnTriangle, config.light);
}

__device__  RGB calculateColorInLight(Point const& pointOnTriangle, Triangle const& triangle, RGB color, BaseConfig const& config)
{
  Vector v0v1 = triangle.y - triangle.x;
  Vector v0v2 = triangle.z - triangle.x;

  Vector N = normalize(crossProduct(v0v1, v0v2));

  Vector lightVec = config.light - pointOnTriangle;
  float dot = fabsf(dotProduct(N, normalize(lightVec)));
  return color
         * (fmaxf(0.0f, (1 - config.ambientCoefficient) * dot) + config.ambientCoefficient);
}

__device__ RGB
processPixel(Segment const& ray, KdTreeData const& treeData, BaseConfig const& config, int recursionLevel = 0)
{
  FindTriangleResult triangleIntersec;
  if (treeData.treeRoot < 0)
    triangleIntersec = findTriangleLeafNode(treeData.treeRoot, ray, -1, treeData);
  else // if (treeData.treeRoot > 0)
    triangleIntersec = findTriangleSplitNode(treeData.treeRoot, ray, -1, treeData);

  if (!triangleIntersec.exists)
    return processPixelOnBackground(config);

  bool const isInShadow = pointInShadow(triangleIntersec.point, triangleIntersec.triangle, treeData, config);

  Triangle triangle = treeData.triangles[triangleIntersec.triangle];
  RGB color = colorOfPoint(triangleIntersec.point, triangle);

  RGB resultCol;
  if (isInShadow)
    resultCol = color * config.ambientCoefficient;
  else
    resultCol = calculateColorInLight(triangleIntersec.point, triangle, color, config);

  return resultCol;

  //float reflectionCoefficient = 0.1;
  //
  //if (recursionLevel >= config.maxRecursionLevel || isCloseToZero(reflectionCoefficient))
  //  return resultCol;
  //
  //Segment refl = reflection({ray.a, triangleIntersec.point}, triangle);
  //RGB reflectedColor = processPixel(refl, treeData, config, recursionLevel + 1);
  //
  //return calculateColorFromReflection(resultCol, reflectedColor, reflectionCoefficient);
}

__global__ void computePixel(RGB* bitmap,
                             BaseConfig config,
                             int treeRoot,
                             int trianglesNum,
                             Triangle* triangles,
                             int leafNodesNum,
                             LeafNode* leafNodes,
                             int splitNodesNum,
                             SplitNode* splitNodes)
{
  int thidX = (blockIdx.x * blockDim.x) + threadIdx.x;
  int thidY = (blockIdx.y * blockDim.y) + threadIdx.y;

  KdTreeData treeData;
  treeData.treeRoot = treeRoot;
  treeData.triangles = triangles;
  treeData.trianglesNum = trianglesNum;
  treeData.leafNodes = leafNodes;
  treeData.leafNodesNum = leafNodesNum;
  treeData.splitNodes = splitNodes;
  treeData.splitNodesNum = splitNodesNum;

  if (thidX < 2 * config.imageY && thidY < 2 * config.imageZ)
  {
    Point pixel{static_cast<float>(config.imageX),
                static_cast<float>(thidX - config.imageY) / config.antiAliasing,
                static_cast<float>(thidY - config.imageZ) / config.antiAliasing};

    Segment ray{config.observer, pixel};
    int idx = thidX * config.imageZ * 2 + thidY;
    bitmap[idx] = processPixel(ray, treeData, config);
  }
}

}
