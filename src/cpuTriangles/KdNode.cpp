#include "cpuTriangles/KdNode.h"
#include "common/Structures.h"
#include "common/Utils.h"

#include <algorithm>
#include <vector>

KdNode* KdNode::build(std::vector<Triangle>& triangles, std::vector<float> &ranges, int depth)
{
  if (triangles.size() == 0)
    return nullptr;


  KdNode* node = new KdNode();

  if (triangles.size() < 10 || depth > 20)
  {
    node->triangles = triangles;
    return node;
  }

  std::vector<Triangle> leftTrs;
  std::vector<Triangle> rightTrs;

  int axis = depth % 3;

  float midValue = (ranges[axis*2] + ranges[axis*2+1]) / 2;
  for (auto const& triangle : triangles)
  {
    switch (axis)
    {
      case 0:
        if(getMinPoint(triangle).x <= midValue)
            leftTrs.push_back(triangle);
        if(getMaxPoint(triangle).x > midValue)
            rightTrs.push_back(triangle);
        break;
      case 1:
        if(getMinPoint(triangle).y <= midValue)
            leftTrs.push_back(triangle);
        if(getMaxPoint(triangle).y > midValue)
            rightTrs.push_back(triangle);
        break;
      case 2:
        if(getMinPoint(triangle).z <= midValue)
            leftTrs.push_back(triangle);
        if(getMaxPoint(triangle).z > midValue)
            rightTrs.push_back(triangle);
        break;
    }
  }

  node->plane = midValue;

  float tmp = ranges[axis*2];
  ranges[axis*2] = midValue;
  node->right = KdNode::build(rightTrs, ranges, depth+1);
  ranges[axis*2] = tmp;
  tmp = ranges[axis*2+1];
  ranges[axis*2+1] = midValue;
  node->left = KdNode::build(leftTrs, ranges, depth+1);
  ranges[axis*2+1] = tmp;

  return node;
}

FindResult KdNode::findInTriangles(Segment seg, Triangle const& excludedTriangle, float t_enter, float t_exit)
{
  //std::cerr<<"A"<<std::endl;
  FindResult res{};

  float currDist = std::numeric_limits<float>::max();

  for (auto const& triangle : triangles)
  {
    if (triangle == excludedTriangle)
      continue;
    IntersecRes const& intersec = intersectionT(seg, triangle);

    //if(intersec.exists && intersec.t < currDist && (intersec.t < t_enter || intersec.t > t_exit))
    //  std::cerr << seg <<" " <<  t_enter <<" " << intersec.t <<" "<<t_exit<<std::endl;
    if (intersec.exists && intersec.t < currDist && intersec.t >= t_enter && intersec.t <= t_exit)
    {
      currDist = intersec.t;
      res.exists = true;
      res.point = intersec.point;
      res.triangle = triangle;
    }
  }

  return res;
}

FindResult KdNode::findRecursive(Segment seg, Triangle const& excludedTriangle, int depth, float t_enter, float t_exit)
{
  int axis = depth%3;
  Vector V = normalize(seg.b - seg.a);
  Vector normal{0, 0, 0};
  float vAxis;

  switch(axis)
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
        axisVal = seg.a.x;
        break;
      case 1:
        axisVal = seg.a.y;
        break;
      case 2:
        axisVal = seg.a.z;
        break;
    }
  //std::cerr << " chuj"<<std::endl;
   if(axisVal <= plane)
      return left ? left->find(seg, excludedTriangle, depth+1, t_enter, t_exit) : FindResult{};
   else
      return right ? right->find(seg, excludedTriangle, depth+1, t_enter, t_exit) : FindResult{};
  }

  float t = -(dotProduct(seg.a, normal) - plane) / x;
  //std::cerr << t <<" " << plane << std::endl;
  
  if(t < 0)
  {
    // check only one side
    
    if(vAxis < 0)
    {
      return left ? left->find(seg, excludedTriangle, depth+1, t_enter, t_exit) : FindResult{};
    }
    else
    {
      return right ? right->find(seg, excludedTriangle, depth+1, t_enter, t_exit) : FindResult{};
    }
  }
  
  if(t < t_enter)
  {
    if(vAxis < 0)
      return left ? left->find(seg, excludedTriangle, depth+1, t_enter, t_exit) : FindResult{};
    else
      return right ? right->find(seg, excludedTriangle, depth+1, t_enter, t_exit) : FindResult{};
  }

  // for sure t < t_exit
  if(vAxis < 0)
  {
    FindResult const& resR = right ? right->find(seg, excludedTriangle, depth+1, t_enter, t) : FindResult{};
    if(resR.exists)
      return resR;
    return left ? left->find(seg, excludedTriangle, depth+1, t, t_exit) : FindResult{};
    
  }
  else
  {
    FindResult const& resL = left ? left->find(seg, excludedTriangle, depth+1, t_enter, t) : FindResult{};
    if(resL.exists)
      return resL;
    return right ? right->find(seg, excludedTriangle, depth+1, t, t_exit) : FindResult{};
  }
}

FindResult KdNode::find(Segment seg, Triangle const& excludedTriangle, int depth, float t_enter, float t_exit)
{
  if (triangles.empty())
    return findRecursive(seg, excludedTriangle, depth, t_enter, t_exit);
  else
    return findInTriangles(seg, excludedTriangle, t_enter, t_exit);
}

KdNode::~KdNode()
{
  delete left;
  delete right;
}
