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

  CUfunction computePixel;
  CU_CHECK(cuModuleGetFunction(&computePixel, cuModule, "computePixel"));

  CU_CHECK(cuMemHostRegister(bitmap, sizeof(RGB) * 2 * imageY * 2 * imageZ, 0));

  CUdeviceptr bitmapDev;
  CU_CHECK(cuMemAlloc(&bitmapDev, sizeof(RGB) * (2 * imageY * 2 * imageZ)));
  CU_CHECK(cuMemcpyHtoD(bitmapDev, bitmap, sizeof(RGB) * (2 * imageY * 2 * imageZ)));

	int planesNum = planes.size();
	Plane* planesTab = planes.data();
	CUdeviceptr planesDev;
	if (planesNum!=0)
	{
  	CU_CHECK(cuMemHostRegister(planesTab, sizeof(Plane) * planesNum, 0));
	

  
		CU_CHECK(cuMemAlloc(&planesDev, sizeof(Plane) * (planesNum)));
	 	CU_CHECK(cuMemcpyHtoD(planesDev, planesTab, sizeof(Plane) * (planesNum)));
	}

  int spheresNum = spheres.size();
  Sphere* spheresTab = spheres.data();
	CUdeviceptr spheresDev;
	if (spheresNum!=0)
	{
		CU_CHECK(cuMemHostRegister(spheresTab, sizeof(Sphere) * spheresNum, 0));

		
		CU_CHECK(cuMemAlloc(&spheresDev, sizeof(Sphere) * (spheresNum)));
		CU_CHECK(cuMemcpyHtoD(spheresDev, spheresTab, sizeof(Sphere) * (spheresNum)));
	}

  int iX = imageX;
  int iY = imageY;
  int iZ = imageZ;
  int aA = antiAliasing;
	int mRL = maxRecursionLevel;
  float dC = diffuseCoefficient;
  float aC = ambientCoefficient;
  float oX = observer.x;
  float oY = observer.y;
  float oZ = observer.z;
  float lX = light.x;
  float lY = light.y;
  float lZ = light.z;
  uint8_t R = 0;
  uint8_t G = 0;
  uint8_t B = 0;

  void* args[] = {&spheresDev, &spheresNum,
									&planesDev, &planesNum,
									&bitmapDev,
									&iX, &iY, &iZ,
									&aA, &mRL,
									&dC, &aC,
                  &oX, &oY, &oZ,
									&lX, &lY, &lZ,
									&R,  &G,  &B};
	int threadsNum = 16;
  int blocks_per_grid_x = (2 * imageY + threadsNum-1) / threadsNum;
  int blocks_per_grid_y = (2 * imageZ + threadsNum-1) / threadsNum;
  int threads_per_block_x = threadsNum;
  int threads_per_block_y = threadsNum;

  CU_CHECK(cuLaunchKernel(computePixel, blocks_per_grid_x, blocks_per_grid_y, 1,
                          threads_per_block_x, threads_per_block_y, 1, 0, 0, args, 0));

  CU_CHECK(cuMemcpyDtoH(bitmap, bitmapDev, sizeof(RGB) * (2 * imageY * 2 * imageZ)));

  CU_CHECK(cuMemHostUnregister(bitmap));
	if (spheresNum!=0){
  CU_CHECK(cuMemHostUnregister(spheresTab));}
	if (planesNum!=0){
	CU_CHECK(cuMemHostUnregister(planesTab));}

  cuCtxDestroy(cuContext);
}
