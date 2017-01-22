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
  FindResult findInTriangles(Segment seg);
  FindResult findRecursive(Segment seg);

public:
  BoundingBox bb;
  KdNode* left = nullptr;
  KdNode* right = nullptr;

  // Consider changing to Triangle*
  std::vector<Triangle> triangles;

  static KdNode* build(std::vector<Triangle> const& triangles, int depth = 0);

  FindResult find(Segment seg);

  ~KdNode();
};

#endif // KDNODE_H
