#ifndef CUDA_TRIANGLES_KDTREESTRUCTURES_H
#define CUDA_TRIANGLES_KDTREESTRUCTURES_H

#include "common/Structures.h"

#include <vector_types.h>

// a leaf node in the kd-tree
struct LeafNode
{
  int triangleCount;
  int firstTriangle;
};

// an internal node in the kd-tree
struct SplitNode
{
  // > 0 - SplitNode, < 0 - LeafNode, points to position + 1
  int leftChild;
  int rightChild;
  BoundingBox bb;
};

struct KdTreeData
{
  int treeRoot;

  int trianglesNum;
  Triangle* triangles;

  int leafNodesNum;
  LeafNode* leafNodes;

  int splitNodesNum;
  SplitNode* splitNodes;
};

#endif // CUDA_TRIANGLES_KDTREESTRUCTURES_H
