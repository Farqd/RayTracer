#ifndef KDNODE_H
#define KDNODE_H


#include "common/Structures.h"

#include <vector>

struct FindResult
{
  bool exists;
  Triangle triangle;
  Point point;
};

class KdNode
{
public:
  BoundingBox bb;
  KdNode* left = nullptr;
  KdNode* right = nullptr;

  // Consider changing to Triangle*
  std::vector<Triangle> triangles;

  static KdNode* build(std::vector<Triangle>& triangles, int depth = 0);

  // TODO create funciton that finds intersection with a ray (returns triangle/point)
  FindResult find(Segment seg);

  ~KdNode();
};

#endif // KDNODE_H
