#include "cpuTriangles/KdNode.h"
#include "common/Structures.h"
#include "common/Utils.h"

#include <algorithm>
#include <vector>

KdNode* KdNode::build(std::vector<Triangle>& triangles, int depth)
{
  if (triangles.size() == 0)
    return nullptr;


  KdNode* node = new KdNode();

  node->bb.vMin.x = std::numeric_limits<float>::max();
  node->bb.vMin.y = std::numeric_limits<float>::max();
  node->bb.vMin.z = std::numeric_limits<float>::max();

  node->bb.vMax.x = std::numeric_limits<float>::min();
  node->bb.vMax.y = std::numeric_limits<float>::min();
  node->bb.vMax.z = std::numeric_limits<float>::min();


  for (auto const& triangle : triangles)
  {
    Point const& minP = getMinPoint(triangle);
    Point const& maxP = getMaxPoint(triangle);

    node->bb.vMin.x = std::min(node->bb.vMin.x, minP.x);
    node->bb.vMin.y = std::min(node->bb.vMin.y, minP.y);
    node->bb.vMin.z = std::min(node->bb.vMin.z, minP.z);

    node->bb.vMax.x = std::max(node->bb.vMax.x, maxP.x);
    node->bb.vMax.y = std::max(node->bb.vMax.y, maxP.y);
    node->bb.vMax.z = std::max(node->bb.vMax.z, maxP.z);
  }

  if (triangles.size() < 10)
  {
    node->triangles = triangles;
    return node;
  }

  std::vector<Triangle> leftTrs;
  std::vector<Triangle> rightTrs;

  int axis = depth % 3;

  float midValue;
  switch (axis)
  {
    case 0:
      midValue = (node->bb.vMax.x + node->bb.vMin.x) / 2;
      break;
    case 1:
      midValue = (node->bb.vMax.y + node->bb.vMin.y) / 2;
      break;
    case 2:
      midValue = (node->bb.vMax.z + node->bb.vMin.z) / 2;
      break;
  }

  for (auto const& triangle : triangles)
  {
    switch (axis)
    {
      case 0:
        getMinPoint(triangle).x < midValue ? leftTrs.push_back(triangle)
                                           : rightTrs.push_back(triangle);
        break;
      case 1:
        getMinPoint(triangle).y < midValue ? leftTrs.push_back(triangle)
                                           : rightTrs.push_back(triangle);
        break;
      case 2:
        getMinPoint(triangle).z < midValue ? leftTrs.push_back(triangle)
                                           : rightTrs.push_back(triangle);
        break;
    }
  }

  if (leftTrs.empty() || rightTrs.empty())
  {
    switch (axis)
    {
      case 0:
        std::sort(triangles.begin(), triangles.end(), [](auto const& t1, auto const& t2) {
          return getMinPoint(t1).x < getMinPoint(t2).x;
        });
        break;
      case 1:
        std::sort(triangles.begin(), triangles.end(), [](auto const& t1, auto const& t2) {
          return getMinPoint(t1).y < getMinPoint(t2).y;
        });
        break;
      case 2:
        std::sort(triangles.begin(), triangles.end(), [](auto const& t1, auto const& t2) {
          return getMinPoint(t1).z < getMinPoint(t2).z;
        });
        break;
    }
    auto mid = triangles.begin() + triangles.size() / 2;
    leftTrs.clear();
    rightTrs.clear();
    leftTrs.insert(leftTrs.end(), triangles.begin(), mid);
    rightTrs.insert(rightTrs.end(), mid, triangles.end());
  }
  node->left = KdNode::build(leftTrs, depth + 1);
  node->right = KdNode::build(rightTrs, depth + 1);

  return node;
}

FindResult KdNode::findInTriangles(Segment seg, Triangle const& excludedTriangle)
{
  FindResult res{};

  float currDist = std::numeric_limits<float>::max();

  for (auto const& triangle : triangles)
  {
    if (triangle == excludedTriangle)
      continue;
    std::pair<bool, Point> const& intersec = intersection(seg, triangle);

    if (intersec.first && distance(seg.a, intersec.second) < currDist)
    {
      currDist = distance(seg.a, intersec.second);
      res.exists = true;
      res.point = intersec.second;
      res.triangle = triangle;
    }
  }

  return res;
}

FindResult KdNode::findRecursive(Segment seg, Triangle const& excludedTriangle)
{
  FindResult const& resL = left ? left->find(seg, excludedTriangle) : FindResult{};
  FindResult const& resR = right ? right->find(seg, excludedTriangle) : FindResult{};

  FindResult res{};
  if (!resL.exists && !resR.exists)
    return res;

  if (resL.exists)
    res = resL;

  if (resR.exists)
  {
    if (!res.exists || (distance(seg.a, resR.point) < distance(seg.a, res.point)))
      res = resR;
  }

  return res;
}

FindResult KdNode::find(Segment seg, Triangle const& excludedTriangle)
{
  FindResult res{};

  if (!intersection(seg, bb))
    return res;

  if (triangles.empty())
    return findRecursive(seg, excludedTriangle);
  else
    return findInTriangles(seg, excludedTriangle);
}

KdNode::~KdNode()
{
  delete left;
  delete right;
}
