cmake_minimum_required(VERSION 3.1)
project(RayTracer)

# Packages

find_package(CUDA)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# CUDA prerequisites
if (NOT ${CUDA_FOUND})
  message("WARNING: CUDA CMake package not found!"
      " If you want to run CUDA targets, rerun cmake with CUDA_TOOLKIT_ROOT_DIR specified.")
else ()
  if ((CUDA_HOST_COMPILER STREQUAL CMAKE_C_COMPILER)
      AND (CMAKE_C_COMPILER_VERSION VERSION_GREATER 5))
    set(CUDA_HOST_COMPILER /usr/bin/gcc-5)
    message("WARNING: CUDA does not support gcc version later than 5."
        " Trying with ${CUDA_HOST_COMPILER}.")
    if (NOT EXISTS ${CUDA_HOST_COMPILER})
      message(FATAL_ERROR
          "CUDA compatible compiler not found!"
          " Run cmake again with -DCUDA_HOST_COMPILER=<path/to/gcc>")
    endif ()
  endif ()
  message("INFO: ${CUDA_HOST_COMPILER} will be used to compile CUDA files.")
endif ()

# Common

set(BASEPATH "${CMAKE_SOURCE_DIR}/src")
include_directories("${BASEPATH}")

set(CMAKE_CXX_FLAGS "-Wall -Wextra") # forget previous flags to avoid unwanted errors

set(CMAKE_CXX_STANDARD 14)
set(RELEASE_CXX_FLAGS -Werror -O2)
set(DEBUG_CXX_FLAGS -ggdb)

set(COMMON_SOURCE_FILES
    src/common/structures_operators.cpp
    src/common/utils.cpp
    src/common/raytracer.cpp)
set(COMMON_HEADER_FILES
    src/common/structures.h
    src/common/structures_operators.h
    src/common/raytracer.h
    src/common/utils.h)

# CPU executable

set(SOURCE_FILES
    src/main.cpp
    ${COMMON_SOURCE_FILES})
set(HEADER_FILES
    ${COMMON_HEADER_FILES})

add_executable(RayTracer ${SOURCE_FILES} ${HEADER_FILES})

target_compile_options(RayTracer PUBLIC "$<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>")
target_compile_options(RayTracer PUBLIC "$<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>")

target_link_libraries(RayTracer Threads::Threads)

# CUDA executable
if (CUDA_FOUND)
  set(CUDA_SOURCE_FILES
      src/cuda/main.cpp
      src/cuda/raytracercuda.cpp
      ${COMMON_SOURCE_FILES})

  set(CUDA_HEADER_FILES
      src/cuda/raytracercuda.h
      ${COMMON_HEADER_FILES})

  set(CUDA_NVCC_FLAGS
      ${CUDA_NVCC_FLAGS}
      --compiler-options
      -O2 -std=c++11
      -gencode arch=compute_20,code=sm_20
      -Wno-deprecated-gpu-targets)

  set(CUDA_CU_FILES
      src/cuda/raytracing.cu)

  set(CUDA_PTX_FILES
      ${CMAKE_BINARY_DIR}/raytracing.ptx)

  cuda_compile_ptx(
      CUDA_GENERATED_PTX_FILES
      ${CUDA_CU_FILES})

  add_custom_command(
      OUTPUT ${CMAKE_BINARY_DIR}/raytracing.ptx
      COMMAND ${CMAKE_COMMAND} -E
      rename ${CUDA_GENERATED_PTX_FILES} ${CMAKE_BINARY_DIR}/raytracing.ptx
      DEPENDS ${CUDA_GENERATED_PTX_FILES})

  cuda_add_executable(
      RayTracerCuda
      ${CUDA_SOURCE_FILES} ${CUDA_PTX_FILES} ${CUDA_HEADER_FILES})

  target_link_libraries(RayTracerCuda "${CUDA_LIBRARIES}" cuda)
endif ()
