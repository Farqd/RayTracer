cmake_minimum_required(VERSION 3.1)
project(RayTracer)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
      src/main.cpp
      src/raytracer.cpp
      src/structures.cpp
      src/utils.cpp)
set(HEADER_FILES
      src/raytracer.h
      src/structures.h
      src/utils.h)

add_executable(RayTracer ${SOURCE_FILES} ${HEADER_FILES})

set(RELEASE_CXX_FLAGS -Wall -Werror -Wextra -O2)
set(DEBUG_CXX_FLAGS -Wall -Wextra -ggdb)


target_compile_options(RayTracer PUBLIC "$<$<CONFIG:Release>:${RELEASE_CXX_FLAGS}>")
target_compile_options(RayTracer PUBLIC "$<$<CONFIG:Debug>:${DEBUG_CXX_FLAGS}>")


