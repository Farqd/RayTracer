#ifndef CUDA_TRIANGLES_KDTREEBUILDER_H
#define CUDA_TRIANGLES_KDTREEBUILDER_H

#include "cudaTriangles/KdTreeStructures.h"

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
    return build(triangles, 0);
  }

private:
  size_t const trianglesInLeafBound;

  int build(std::vector<Triangle>& triangles, int depth);

  int addLeaf(std::vector<Triangle> const& triangles);
};


#endif // CUDA_TRIANGLES_KDTREEBUILDER_H
