#include "cuda/raytracercuda.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#define CU_CHECK(ans)                                                                              \
  {                                                                                                \
    cuAssert((ans), __FILE__, __LINE__);                                                           \
  }
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

void RayTracerCuda::processPixelsCuda()
{
  cuInit(0);

  CUdevice cuDevice;
  CU_CHECK(cuDeviceGet(&cuDevice, 0));

  CUcontext cuContext;
  CU_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));

  CUmodule cuModule = (CUmodule) 0;
  CU_CHECK(cuModuleLoad(&cuModule, "raytracing.ptx"));

  CUfunction processPixel;
  CU_CHECK(cuModuleGetFunction(&processPixel, cuModule, "processPixel"));

  CU_CHECK(cuMemHostRegister(bitmap, sizeof(RGB) * 2 * imageY * 2 * imageZ, 0));

  CUdeviceptr bitmapDev;
  CU_CHECK(cuMemAlloc(&bitmapDev, sizeof(RGB) * (2 * imageY * 2 * imageZ)));
  CU_CHECK(cuMemcpyHtoD(bitmapDev, bitmap, sizeof(RGB) * (2 * imageY * 2 * imageZ)));

  int spheresNum = spheres.size();
  Sphere* spheresTab = spheres.data();
  CU_CHECK(cuMemHostRegister(spheresTab, sizeof(Sphere) * spheresNum, 0));

  CUdeviceptr spheresDev;
  CU_CHECK(cuMemAlloc(&spheresDev, sizeof(Sphere) * (spheresNum)));
  CU_CHECK(cuMemcpyHtoD(spheresDev, spheresTab, sizeof(Sphere) * (spheresNum)));

  int iX = imageX;
  int iY = imageY;
  int iZ = imageZ;
  int aA = antiAliasing;
  double dC = diffuseCoefficient;
  double aC = ambientCoefficient;
  double oX = observer.x;
  double oY = observer.y;
  double oZ = observer.z;
  double lX = light.x;
  double lY = light.y;
  double lZ = light.z;
  uint8_t R = background.r;
  uint8_t G = background.g;
  uint8_t B = background.b;

  void* args[] = {&spheresDev, &spheresNum, &bitmapDev, &iX, &iY, &iZ, &aA, &dC, &aC,
                  &oX,         &oY,         &oZ,        &lX, &lY, &lZ, &R,  &G,  &B};

  int blocks_per_grid_x = (2 * imageY + 31) / 32;
  int blocks_per_grid_y = (2 * imageZ + 31) / 32;
  int threads_per_block_x = 32;
  int threads_per_block_y = 32;

  CU_CHECK(cuLaunchKernel(processPixel, blocks_per_grid_x, blocks_per_grid_y, 1,
                          threads_per_block_x, threads_per_block_y, 1, 0, 0, args, 0));

  CU_CHECK(cuMemcpyDtoH(bitmap, bitmapDev, sizeof(RGB) * (2 * imageY * 2 * imageZ)));

  CU_CHECK(cuMemHostUnregister(bitmap));
  CU_CHECK(cuMemHostUnregister(spheresTab));

  cuCtxDestroy(cuContext);
}
