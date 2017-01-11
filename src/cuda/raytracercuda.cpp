#include "cuda/raytracercuda.h"

#include "cuda.h"

void RayTracerCuda::processPixelsCuda(std::vector<Sphere> const& spheres)
{
  cuInit(0);

  CUdevice cuDevice;
  CUresult res = cuDeviceGet(&cuDevice, 0);
  if (res != CUDA_SUCCESS)
  {
    printf("cannot acquire device 0\n");
    exit(1);
  }

  CUcontext cuContext;
  res = cuCtxCreate(&cuContext, 0, cuDevice);
  if (res != CUDA_SUCCESS)
  {
    printf("cannot create context\n");
    exit(1);
  }

  CUmodule cuModule = (CUmodule) 0;
  res = cuModuleLoad(&cuModule, "raytracing.ptx");
  if (res != CUDA_SUCCESS)
  {
    printf("cannot load module: %d\n", res);
    exit(1);
  }

  CUfunction processPixel;
  res = cuModuleGetFunction(&processPixel, cuModule, "processPixel");
  if (res != CUDA_SUCCESS)
  {
    printf("cannot acquire kernel handle\n");
    exit(1);
  }

  int blocks_per_grid_x = (2 * imageY + 31) / 32;
  int blocks_per_grid_y = (2 * imageZ + 31) / 32;
  int threads_per_block_x = 32;
  int threads_per_block_y = 32;

  int spheresNum = spheres.size();

  RGB* bitmapTab = (RGB*) malloc(sizeof(RGB) * 2 * imageY * 2 * imageZ);
  res = cuMemHostRegister(bitmapTab, sizeof(RGB) * 2 * imageY * 2 * imageZ, 0);
  if (res != CUDA_SUCCESS)
  {
    exit(1);
  }

  for (int i = 0; i < 2 * imageY * 2 * imageZ; ++i)
  {
    bitmapTab[i] = bitmap[i / (2 * imageZ)][i % (2 * imageZ)];
  }

  CUdeviceptr bitmapDev;
  res = cuMemAlloc(&bitmapDev, sizeof(RGB) * (2 * imageY * 2 * imageZ));
  if (res != CUDA_SUCCESS)
  {
    printf("cannot acquireB kernel handle\n");
    exit(1);
  }

  res = cuMemcpyHtoD(bitmapDev, bitmapTab, sizeof(RGB) * (2 * imageY * 2 * imageZ));
  if (res != CUDA_SUCCESS)
  {
    printf("cannot acquire kernel handle\n");
    exit(1);
  }

  Sphere* spheresTab = (Sphere*) malloc(sizeof(Sphere) * spheresNum);
  res = cuMemHostRegister(spheresTab, sizeof(Sphere) * spheresNum, 0);
  if (res != CUDA_SUCCESS)
  {
    exit(1);
  }

  for (int i = 0; i < spheresNum; ++i)
  {
    spheresTab[i] = spheres[i];
  }

  CUdeviceptr spheresDev;
  res = cuMemAlloc(&spheresDev, sizeof(Sphere) * (spheresNum));
  if (res != CUDA_SUCCESS)
  {
    printf("cannot acquireB kernel handle\n");
    exit(1);
  }

  res = cuMemcpyHtoD(spheresDev, spheresTab, sizeof(Sphere) * (spheresNum));
  if (res != CUDA_SUCCESS)
  {
    printf("cannot acquire kernel handle\n");
    exit(1);
  }
  int iX = (int) imageX;
  int iY = (int) imageY;
  int iZ = (int) imageZ;
  int aA = (int) antiAliasing;
  double dC = (double) diffuseCoefficient;
  double aC = (double) ambientCoefficient;
  double oX = (double) observer.x;
  double oY = (double) observer.y;
  double oZ = (double) observer.z;
  double lX = (double) light.x;
  double lY = (double) light.y;
  double lZ = (double) light.z;
  unsigned char R = (unsigned char) background.r;
  unsigned char G = (unsigned char) background.g;
  unsigned char B = (unsigned char) background.b;

  void* args[] = {&spheresDev, &spheresNum, &bitmapDev, &iX, &iY, &iZ, &aA, &dC, &aC,
                  &oX,         &oY,         &oZ,        &lX, &lY, &lZ, &R,  &G,  &B};

  res = cuLaunchKernel(processPixel, blocks_per_grid_x, blocks_per_grid_y, 1, threads_per_block_x,
                       threads_per_block_y, 1, 0, 0, args, 0);
  if (res != CUDA_SUCCESS)
  {
    printf("cannot run kernel\n");
    exit(1);
  }

  res = cuMemcpyDtoH(bitmapTab, bitmapDev, sizeof(RGB) * (2 * imageY * 2 * imageZ));
  if (res != CUDA_SUCCESS)
  {
    printf("cannot acquire kernel handle\n");
    exit(1);
  }

  for (int i = 0; i < 2 * imageY * 2 * imageZ; ++i)
  {
    bitmap[i / (2 * imageZ)][i % (2 * imageZ)] = bitmapTab[i];
  }

  free(bitmapTab);
  free(spheresTab);

  cuCtxDestroy(cuContext);
}
