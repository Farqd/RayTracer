#include "common/Structures.h"

#include <vector>

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

  ~KdNode();
};