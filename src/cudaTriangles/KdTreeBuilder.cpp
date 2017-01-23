#include "cudaTriangles/KdTreeBuilder.h"

#include "common/Utils.h"
#include <algorithm>
#include <cassert>
#include <vector>

static BoundingBox getBoundBoxForTriangles(std::vector<Triangle> const& triangles)
{
  BoundingBox bb;

  bb.vMin.x = std::numeric_limits<float>::max();
  bb.vMin.y = std::numeric_limits<float>::max();
  bb.vMin.z = std::numeric_limits<float>::max();

  bb.vMax.x = std::numeric_limits<float>::min();
  bb.vMax.y = std::numeric_limits<float>::min();
  bb.vMax.z = std::numeric_limits<float>::min();


  for (auto const& triangle : triangles)
  {
    Point const& minP = getMinPoint(triangle);
    Point const& maxP = getMaxPoint(triangle);

    bb.vMin.x = std::min(bb.vMin.x, minP.x);
    bb.vMin.y = std::min(bb.vMin.y, minP.y);
    bb.vMin.z = std::min(bb.vMin.z, minP.z);

    bb.vMax.x = std::max(bb.vMax.x, maxP.x);
    bb.vMax.y = std::max(bb.vMax.y, maxP.y);
    bb.vMax.z = std::max(bb.vMax.z, maxP.z);
  }

  return bb;
}

static float getSplitValue(BoundingBox const& bb, int const axis)
{
  switch (axis)
  {
    case 0:
      return (bb.vMax.x + bb.vMin.x) / 2;
    case 1:
      return (bb.vMax.y + bb.vMin.y) / 2;
    case 2:
      return (bb.vMax.z + bb.vMin.z) / 2;
  }
  assert(false);
  return 0;
}

static bool goesLeft(Triangle const& triangle, int const axis, float const splitValue)
{
  switch (axis)
  {
    case 0:
      return getMinPoint(triangle).x < splitValue;
    case 1:
      return getMinPoint(triangle).y < splitValue;
    case 2:
      return getMinPoint(triangle).z < splitValue;
  }
  assert(false);
  return false;
}

int KdTreeBuilder::build(std::vector<Triangle> const& triangles, int depth)
{
  if (triangles.size() == 0)
    return 0; // valid index is either negative or positive, see SplitNode

  if (triangles.size() < trianglesInLeafBound)
    return addLeaf(triangles);

  int const axis = depth % 3;
  BoundingBox const bb = getBoundBoxForTriangles(triangles);

  float const splitValue = getSplitValue(bb, axis);

  std::vector<Triangle> leftTrs;
  std::vector<Triangle> rightTrs;

  for (auto const& triangle : triangles)
    goesLeft(triangle, axis, splitValue) ? leftTrs.push_back(triangle)
                                         : rightTrs.push_back(triangle);

  if (leftTrs.empty() || rightTrs.empty())
    return addLeaf(triangles);

  splitNodes.emplace_back();
  int const nodeIdx = static_cast<int>(splitNodes.size()) - 1;

  splitNodes[nodeIdx].bb = bb;
  int left = build(leftTrs, depth + 1);
  int right = build(rightTrs, depth + 1);
  splitNodes[nodeIdx].leftChild = left;
  splitNodes[nodeIdx].rightChild = right;

  return nodeIdx + 1;
}

int KdTreeBuilder::addLeaf(std::vector<Triangle> const& triangles)
{
  LeafNode leaf;
  leaf.firstTriangle = static_cast<int>(treeTriangles.size());
  leaf.triangleCount = static_cast<int>(triangles.size());
  treeTriangles.insert(treeTriangles.end(), triangles.begin(), triangles.end());
  leafNodes.push_back(leaf);
  return -static_cast<int>(leafNodes.size()); // leaf nodes have negative indices
}
