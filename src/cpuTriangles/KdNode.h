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
  FindResult findRecursive(Segment seg, Triangle const& excludedTriangle, int depth, float t_enter, float t_exit);
  FindResult findInTriangles(Segment seg, Triangle const& excludedTriangle, float t_enter, float t_exit);

public:
  float plane;
  KdNode* left = nullptr;
  KdNode* right = nullptr;

  // Consider changing to Triangle*
  std::vector<Triangle> triangles;

  FindResult find(Segment seg, Triangle const& excludedTriangle, int depth, float t_enter, float t_exit);

  static KdNode* build(std::vector<Triangle>& triangles, std::vector<float>& ranges, int depth = 0);

  ~KdNode();
};

#endif // KDNODE_H
