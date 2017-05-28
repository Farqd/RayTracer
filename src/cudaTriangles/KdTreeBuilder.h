#ifndef CUDA_TRIANGLES_KDTREEBUILDER_H
#define CUDA_TRIANGLES_KDTREEBUILDER_H

#include "cudaTriangles/KdTreeStructures.h"
#include "common/Utils.h"
#include <vector>

struct KdTreeBuilder
{
  KdTreeBuilder(size_t const leafTrianglesLimit = 16)
    : trianglesInLeafBound(leafTrianglesLimit)
  {
  }

  std::vector<SplitNode> splitNodes;
  std::vector<LeafNode> leafNodes;
  std::vector<Triangle> treeTriangles;

  int build(std::vector<Triangle>& triangles)
  {
    float ranges[6];
    if (triangles.size() > 0)
    {
      ranges[0] = triangles[0].x.x;
      ranges[1] = triangles[0].x.x;
      ranges[2] = triangles[0].x.y;
      ranges[3] = triangles[0].x.y;
      ranges[4] = triangles[0].x.z;
      ranges[5] = triangles[0].x.z;
    }
    for (Triangle const& triangle : triangles)
    {
      Point pMin = getMinPoint(triangle);
      Point pMax = getMaxPoint(triangle);
      
      ranges[0] = std::min(ranges[0], pMin.x);
      ranges[1] = std::max(ranges[1], pMax.x);
      ranges[2] = std::min(ranges[2], pMin.y);
      ranges[3] = std::max(ranges[3], pMax.y);
      ranges[4] = std::min(ranges[4], pMin.z);
      ranges[5] = std::max(ranges[5], pMax.z);
    }
    return build(triangles, ranges, 0);
  }

private:
  size_t const trianglesInLeafBound;

  int build(std::vector<Triangle>& triangles, float* ranges, int depth);

  int addLeaf(std::vector<Triangle> const& triangles);
};


#endif // CUDA_TRIANGLES_KDTREEBUILDER_H
