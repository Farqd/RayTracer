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
  int data[22];

  __device__ void push(int const& t)
  {
    data[size++] = t;
  }

  __device__ int pop()
  {
    return data[--size];
  }
};

struct Fstack
{
  size_t size = 0;
  float data[22];

  __device__ void push(float const& t)
  {
    data[size++] = t;
  }

  __device__ float pop()
  {
    return data[--size];
  }
};


extern "C" {

__device__ FindTriangleResult findTriangleLeafNode(
    int leafIdx, Segment const& ray, int excludedTriangle, KdTreeData const& treeData, int depth, float t_enter, float t_exit)
{
  FindTriangleResult res{};

  float currDist = FLT_MAX;

  LeafNode const leafNode = treeData.leafNodes[-leafIdx - 1];

  for (int i = leafNode.firstTriangle; i < leafNode.firstTriangle + leafNode.triangleCount; ++i)
  {
    if (i == excludedTriangle)
      continue;
    IntersecRes const intersec = intersectionT(ray, treeData.triangles[i]);

    if (intersec.exists && intersec.t < currDist && intersec.t >= t_enter && intersec.t <= t_exit)
    {
      currDist = intersec.t;
      res.exists = true;
      res.point = intersec.point;
      res.triangle = i;
    }
  }

  return res;
}

__device__ FindTriangleResult search(Segment const& ray, int excludedTriangle, KdTreeData const& treeData)
{

  float tMin = FLT_MAX;
  float tMax = FLT_MIN;
  Vector V = normalize(ray.b - ray.a);
  float dirfracX = 1.0f / V.x;
  float dirfracY = 1.0f / V.y;
  float dirfracZ = 1.0f / V.z;

  float t1 = (treeData.p0 - ray.a.x) * dirfracX;
  float t2 = (treeData.p1 - ray.a.x) * dirfracX;
  float t3 = (treeData.p2 - ray.a.y) * dirfracY;
  float t4 = (treeData.p3 - ray.a.y) * dirfracY;
  float t5 = (treeData.p4 - ray.a.z) * dirfracZ;
  float t6 = (treeData.p5 - ray.a.z) * dirfracZ;

  tMin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
  tMax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));
  if (tMin > tMax || 0 > tMax) return FindTriangleResult{};
  bool foundLeaf = true;
  float t_enter = tMin;
  float t_exit = tMax;
  int i=0;
  while (foundLeaf && i<2500)
  {  ++i;
    foundLeaf = false;   
    int currNode = treeData.treeRoot;
    int depth = 0;
    while (currNode>0)
    {
      SplitNode node = treeData.splitNodes[currNode-1];
      int axis = depth%3;
      
      Vector normal{0, 0, 0};
      float vAxis;

      switch (axis)
      {
        case 0:
          normal.x = 1;
          vAxis = V.x;
          break;
        case 1:
          normal.y = 1;
          vAxis = V.y;
          break;
        case 2:
          normal.z = 1;
          vAxis = V.z;
          break;    
      }

      float x = dotProduct(V, normal);
      if (isCloseToZero(x))
  {
    float axisVal;
    switch(axis)
    {
      case 0:
        axisVal = ray.a.x;
        break;
      case 1:
        axisVal = ray.a.y;
        break;    
      case 2:
        axisVal = ray.a.z;
        break;
    }
    if (axisVal<=node.plane)
    {
      
        currNode = node.leftChild;
        depth = depth + 1;
        continue;
      
    }
    else
    {
      currNode = node.rightChild;
      depth = depth + 1;
      continue;
     
    }
  }    
  float t = -(dotProduct(ray.a, normal) - node.plane) / x;

  if (t<0)
  {
    if (vAxis<0)
    {
       currNode = node.leftChild;
        depth = depth + 1;
        continue;
    }
    else
    {
     currNode = node.rightChild;
      depth = depth + 1;
      continue;
    }
  }

  if (t < t_enter)
  {
    if (vAxis<0)
    {
      currNode = node.leftChild;
        depth = depth + 1;
        continue; 

   }
    else
    {
       currNode = node.rightChild;
      depth = depth + 1;
      continue;
    }

  }
  if (t > t_exit)
  {
    if (vAxis>0)
    {
        currNode = node.leftChild;
        depth = depth + 1;
        continue; 
    }
    else
    {
     currNode = node.rightChild;
      depth = depth + 1;
      continue;
    }
  }
  if (vAxis < 0)
  {
    currNode = node.rightChild;
      depth = depth + 1;
      t_exit = t;
      continue;
  }

  else
  {
     currNode = node.leftChild;
        depth = depth + 1;
        t_exit = t;
        continue; 
  }


    }  
    if (currNode < 0)
    {
      
      FindTriangleResult result = findTriangleLeafNode(currNode, ray, excludedTriangle, treeData, depth, t_enter, t_exit);
      if (result.exists) return result;
      if (isCloseToZero(t_exit - tMax)) return FindTriangleResult{};
      t_enter = t_exit;
      t_exit = tMax;
      foundLeaf = true;
    }
  }
  return FindTriangleResult{};
}

__device__ FindTriangleResult findTriangle(
int nodeIdx, Segment const& ray, int excludedTriangle, KdTreeData const& treeData, int dth, float f1, float f2)
{
  Vector V = normalize(ray.b - ray.a);
  SplitNode node;
  int depth = dth;
  float t_enter = f1;
  float t_exit = f2;
  Stack stack;
  Stack stack2;
  Fstack stack3;
  Fstack stack4;
  int id = nodeIdx;
  
  while (true)
  {
    if (id<0)
    {
      FindTriangleResult res = findTriangleLeafNode(id, ray, excludedTriangle, treeData, depth, t_enter, t_exit);
      if (res.exists) return res;
      if (stack.size == 0) return FindTriangleResult{};
      id = stack.pop();
      depth = stack2.pop();
      t_enter = stack3.pop();
      t_exit = stack4.pop();
      continue; 
    }
    else if (id>0)
    {
       node = treeData.splitNodes[id-1];
       int axis = depth%3;
  
  Vector normal{0, 0, 0};
  float vAxis;

  switch (axis)
  {
    case 0:
      normal.x = 1;
      vAxis = V.x;
      break;
    case 1:
      normal.y = 1;
      vAxis = V.y;
      break;
    case 2:
      normal.z = 1;
      vAxis = V.z;
      break;    
  }

  float x = dotProduct(V, normal);
  if (isCloseToZero(x))
  {
    float axisVal;
    switch(axis)
    {
      case 0:
        axisVal = ray.a.x;
        break;
      case 1:
        axisVal = ray.a.y;
        break;    
      case 2:
        axisVal = ray.a.z;
        break;
    }
    if (axisVal<=node.plane)
    {
      id = node.leftChild;
      depth = depth+1;
      continue;
    }
    else
    {
      id = node.rightChild;
      depth = depth+1;
      continue;
    }
  }    
  float t = -(dotProduct(ray.a, normal) - node.plane) / x;

  if (t<0)
  {
    if (vAxis<0)
    {
      id = node.leftChild;
      depth = depth+1;
      continue;
    }
    else
    {
      id = node.rightChild;
      depth = depth+1;
      continue;
    }
  }

  if (t < t_enter)
  {
    if (vAxis<0)
    {
      id = node.leftChild;
      depth = depth+1;
      continue;
    }
    else
    {
      id = node.rightChild;
      depth = depth+1;
      continue;
    }

  }
  if (t > t_exit)
  {
    if (vAxis>0)
    {
      id = node.leftChild;
      depth = depth+1;
      continue;
    }
    else
    {
      id = node.rightChild;
      depth = depth+1;
      continue;
    }
  }
  if (vAxis < 0)
  {
    stack.push(node.leftChild);
    stack2.push(depth+1);
    stack3.push(t);
    stack4.push(t_exit);

    id = node.rightChild;
    depth = depth+1;
    t_exit = t;
    continue;
  }

  else
  {
    stack.push(node.rightChild);
    stack2.push(depth+1);
    stack3.push(t);
    stack4.push(t_exit);

    id = node.leftChild;
    depth = depth+1;
    t_exit = t;
    continue;
  }

    } 
  }

}


__device__ FindTriangleResult findTriangleSplitNode(
    int nodeIdx, Segment const& ray, int excludedTriangle, KdTreeData const& treeData, int depth, float t_enter, float t_exit)
{

  SplitNode node = treeData.splitNodes[nodeIdx - 1];
  int axis = depth%3;
  Vector V = normalize(ray.b - ray.a);
  Vector normal{0, 0, 0};
  float vAxis;

  switch (axis)
  {
    case 0:
      normal.x = 1;
      vAxis = V.x;
      break;
    case 1:
      normal.y = 1;
      vAxis = V.y;
      break;
    case 2:
      normal.z = 1;
      vAxis = V.z;
      break;    
  }

  float x = dotProduct(V, normal);
  if (isCloseToZero(x))
  {
    float axisVal;
    switch(axis)
    {
      case 0:
        axisVal = ray.a.x;
        break;
      case 1:
        axisVal = ray.a.y;
        break;    
      case 2:
        axisVal = ray.a.z;
        break;
    }
    if (axisVal<=node.plane)
    {
      if (node.leftChild<0) return findTriangleLeafNode(node.leftChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      if (node.leftChild>0) return findTriangleSplitNode(node.leftChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      return FindTriangleResult{};
    }
    else
    {
      if (node.rightChild<0) return findTriangleLeafNode(node.rightChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      if (node.rightChild>0) return findTriangleSplitNode(node.rightChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      return FindTriangleResult{};
    }
  }    
  float t = -(dotProduct(ray.a, normal) - node.plane) / x;

  if (t<0)
  {
    if (vAxis<0)
    {
      if (node.leftChild<0) return findTriangleLeafNode(node.leftChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      if (node.leftChild>0) return findTriangleSplitNode(node.leftChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      return FindTriangleResult{};
    }
    else
    {
      if (node.rightChild<0) return findTriangleLeafNode(node.rightChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      if (node.rightChild>0) return findTriangleSplitNode(node.rightChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      return FindTriangleResult{};
    }
  }

  if (t < t_enter)
  {
    if (vAxis<0)
    {
      if (node.leftChild<0) return findTriangleLeafNode(node.leftChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      if (node.leftChild>0) return findTriangleSplitNode(node.leftChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      return FindTriangleResult{};
    }
    else
    {
      if (node.rightChild<0) return findTriangleLeafNode(node.rightChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      if (node.rightChild>0) return findTriangleSplitNode(node.rightChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      return FindTriangleResult{};
    }

  }
  if (t > t_exit)
  {
    if (vAxis>0)
    {
      if (node.leftChild<0) return findTriangleLeafNode(node.leftChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      if (node.leftChild>0) return findTriangleSplitNode(node.leftChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      return FindTriangleResult{};
    }
    else
    {
      if (node.rightChild<0) return findTriangleLeafNode(node.rightChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      if (node.rightChild>0) return findTriangleSplitNode(node.rightChild, ray, excludedTriangle, treeData, depth+1, t_enter, t_exit);
      return FindTriangleResult{};
    }
  }
  if (vAxis < 0)
  {
    FindTriangleResult resR{};
    if (node.rightChild<0) resR = findTriangleLeafNode(node.rightChild, ray, excludedTriangle, treeData, depth+1, t_enter, t);
    if (node.rightChild>0) resR = findTriangleSplitNode(node.rightChild, ray, excludedTriangle, treeData, depth+1, t_enter, t);
    if (resR.exists) return resR;

    if (node.leftChild<0) return findTriangleLeafNode(node.leftChild, ray, excludedTriangle, treeData, depth+1, t, t_exit);
    if (node.leftChild>0) return findTriangleSplitNode(node.leftChild, ray, excludedTriangle, treeData, depth+1, t, t_exit);

    return FindTriangleResult{};
  }

  else
  {
    FindTriangleResult resL{};
    if (node.leftChild<0) resL = findTriangleLeafNode(node.leftChild, ray, excludedTriangle, treeData, depth+1, t_enter, t);
    if (node.leftChild>0) resL = findTriangleSplitNode(node.leftChild, ray, excludedTriangle, treeData, depth+1, t_enter, t);
    if (resL.exists) return resL;

    if (node.rightChild<0) return findTriangleLeafNode(node.rightChild, ray, excludedTriangle, treeData, depth+1, t, t_exit);
    if (node.rightChild>0) return findTriangleSplitNode(node.rightChild, ray, excludedTriangle, treeData, depth+1, t, t_exit);

    return FindTriangleResult{};
  }
  

}

/*__device__ FindTriangleResult findTriangleSplitNode(
    int nodeIdx, Segment const& ray, int excludedTriangle, KdTreeData const& treeData, float t_enter, float t_exit)
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
      if (intersectsBoundingBoxN(ray, rightSplit.bb))
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
      if (intersectsBoundingBoxN(ray, leftSplit.bb))
        stack.push(idxL);
    }
  }
}
*/
__device__ RGB processPixelOnBackground(BaseConfig const& config)
{
  return config.background;
}

__device__ bool pointInShadow(Point const& pointOnTriangle, int excludedTriangle, KdTreeData const& treeData, BaseConfig const& config)
{
  Vector dir = normalize(config.light - pointOnTriangle);
  Segment ray = {pointOnTriangle, pointOnTriangle+dir};
  FindTriangleResult res;
//  res = search(ray, excludedTriangle, treeData);

  if (treeData.treeRoot < 0)
    res = findTriangleLeafNode(treeData.treeRoot, ray, excludedTriangle, treeData, 0, 0.0, FLT_MAX);
  else // if (treeData.treeRoot > 0)
    res = findTriangle(treeData.treeRoot, ray, excludedTriangle, treeData, 0, 0.0, FLT_MAX);

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
processPixel(Segment const& seg, KdTreeData const& treeData, BaseConfig const& config, int recursionLevel = 0)
{
  Vector dir = normalize(seg.b - seg.a);
  Segment ray = {seg.a, seg.a+dir};
  FindTriangleResult triangleIntersec;
//  triangleIntersec = search(ray, -1, treeData);
  if (treeData.treeRoot < 0)
    triangleIntersec = findTriangleLeafNode(treeData.treeRoot, ray, -1, treeData, 0, 0.0, FLT_MAX);
  else // if (treeData.treeRoot > 0)
    triangleIntersec = findTriangle(treeData.treeRoot, ray, -1, treeData, 0, 0.0, FLT_MAX);
  
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

  float reflectionCoefficient = 0.1;

  if (recursionLevel >= config.maxRecursionLevel || isCloseToZero(reflectionCoefficient))
    return resultCol;

  Segment refl = reflection({ray.a, triangleIntersec.point}, triangle);
  RGB reflectedColor = processPixel(refl, treeData, config, recursionLevel + 1);

  return calculateColorFromReflection(resultCol, reflectedColor, reflectionCoefficient);
}

__global__ void computePixel(RGB* bitmap,
                             BaseConfig config,
                             int treeRoot,
                             int trianglesNum,
                             Triangle* triangles,
                             int leafNodesNum,
                             LeafNode* leafNodes,
                             int splitNodesNum,
                             SplitNode* splitNodes, float p0, float p1, float p2, float p3, float p4, float p5)
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
  treeData.p0 = p0;
  treeData.p1 = p1;
  treeData.p2 = p2;
  treeData.p3 = p3;
  treeData.p4 = p4;
  treeData.p5 = p5;

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
