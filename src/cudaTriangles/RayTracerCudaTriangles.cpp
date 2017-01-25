#include "cudaTriangles/RayTracerCudaTriangles.h"

#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <exception>

#include "cudaTriangles/KdTreeBuilder.h"
#include "cudaTriangles/KdTreeStructures.h"

#define CU_CHECK(ans) cuAssert((ans), __FILE__, __LINE__);

inline void cuAssert(CUresult code, const char* file, int line, bool abort = true)
{
  if (code != CUDA_SUCCESS)
  {
    const char* error_string;
    cuGetErrorString(code, &error_string);
    std::cerr << file << ":" << line << " - CUDA error (" << code << "): " << error_string
              << std::endl;
    if (abort)
      exit(code);
  }
}

template <typename T>
CUdeviceptr toDeviceCopy(T* data, size_t count)
{
  CUdeviceptr dataDev{};
  if (count > 0)
  {
    CU_CHECK(cuMemHostRegister(data, sizeof(T) * count, 0));
    CU_CHECK(cuMemAlloc(&dataDev, sizeof(T) * (count)));
    CU_CHECK(cuMemcpyHtoD(dataDev, data, sizeof(T) * (count)));
  }
  return dataDev;
}

void RayTracerCudaTriangles::processPixelsCuda()
{
  if (config.triangles.size() == 0)
    return;

  cuInit(0);

  CUdevice cuDevice;
  CU_CHECK(cuDeviceGet(&cuDevice, 1));

  CUcontext cuContext;
  CU_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));

  CUmodule cuModule = (CUmodule) 0;
  CU_CHECK(cuModuleLoad(&cuModule, "raytracingtriangles.ptx"));

  CUfunction computePixel;
  CU_CHECK(cuModuleGetFunction(&computePixel, cuModule, "computePixel"));

  cudaError code = cudaDeviceSetLimit(cudaLimitStackSize, 2048 * 2);
  if (code != cudaSuccess)
  {
    std::cerr << "Setting stack limit failed " << code << " " << std::endl;
    exit(code);
  }

  KdTreeBuilder treeBuilder(20);

  int root = treeBuilder.build(config.triangles);
  assert(root != 0);

  RGB* bitmapTab = bitmap.data();
  CUdeviceptr bitmapDev;
  CU_CHECK(cuMemAlloc(&bitmapDev, sizeof(RGB) * (bitmap.size())));

  int trianglesNum = treeBuilder.treeTriangles.size();
  Triangle* trianglesTab = treeBuilder.treeTriangles.data();
  CUdeviceptr trianglesDev = toDeviceCopy(trianglesTab, trianglesNum);

  int leafNodesNum = treeBuilder.leafNodes.size();
  LeafNode* leafNodesTab = treeBuilder.leafNodes.data();
  CUdeviceptr leafNodesDev = toDeviceCopy(leafNodesTab, leafNodesNum);

  int splitNodesNum = treeBuilder.splitNodes.size();
  SplitNode* splitNodesTab = treeBuilder.splitNodes.data();
  CUdeviceptr splitNodesDev = toDeviceCopy(splitNodesTab, splitNodesNum);

  BaseConfig baseConfig = config;
  baseConfig.imageY = bitmap.rows / 2;
  baseConfig.imageZ = bitmap.cols / 2;

  void* args[] = {&bitmapDev,    &baseConfig,   &root,          &trianglesNum, &trianglesDev,
                  &leafNodesNum, &leafNodesDev, &splitNodesNum, &splitNodesDev};

  int threadsX = 32;
  int threadsY = 16;
  int blocksX = (bitmap.rows + threadsX - 1) / threadsX;
  int blocksY = (bitmap.cols + threadsY - 1) / threadsY;

  CU_CHECK(cuLaunchKernel(computePixel, blocksX, blocksY, 1, threadsX, threadsY, 1, 0, 0, args, 0));

  CU_CHECK(cuMemHostRegister(bitmapTab, sizeof(RGB) * bitmap.size(), 0));
  CU_CHECK(cuMemcpyDtoH(bitmapTab, bitmapDev, sizeof(RGB) * bitmap.size()));
  CU_CHECK(cuMemHostUnregister(bitmapTab));

  if (trianglesNum != 0)
    CU_CHECK(cuMemHostUnregister(trianglesTab));
  if (splitNodesNum != 0)
    CU_CHECK(cuMemHostUnregister(splitNodesTab));
  if (leafNodesNum != 0)
    CU_CHECK(cuMemHostUnregister(leafNodesTab));

  cuCtxDestroy(cuContext);
}
