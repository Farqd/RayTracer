#ifndef KDNODE_H
#define KDNODE_H


#include "common/Structures.h"

#include <vector>

struct FindResult
{
  bool exists = false;
  Triangle triangle;
  Point point;
};

class KdNode
{
private:
  FindResult findRecursive(Segment seg, Triangle const& excludedTriangle);
  FindResult findInTriangles(Segment seg, Triangle const& excludedTriangle);

public:
  BoundingBox bb;
  KdNode* left = nullptr;
  KdNode* right = nullptr;

  // Consider changing to Triangle*
  std::vector<Triangle> triangles;

  FindResult find(Segment seg, Triangle const& excludedTriangle);

  static KdNode* build(std::vector<Triangle> const& triangles, int depth = 0);

  ~KdNode();
};

#endif // KDNODE_H
