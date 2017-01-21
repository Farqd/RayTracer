#include "cpuTriangles/KdNode.h"
#include "common/Structures.h"

#include <algorithm>
#include <vector>

KdNode* KdNode::build(std::vector<Triangle>& triangles, int depth /* = 0*/)
{
  return nullptr;
}

KdNode::~KdNode()
{
  delete left;
  delete right;
}