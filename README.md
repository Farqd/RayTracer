# RayTracer
CUDA Ray Tracer

# Building

Available targets for `make`:

* `RayTracer` - multithreaded CPU implementation
* `RayTracerCuda` - CUDA implementation (requires CUDA libraries installed)

**Debug**
```sh
mkdir Debug
cd Debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

**Release**
```sh
mkdir Release
cd Release
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

**CMake flags**

* `CMAKE_BUILD_TYPE=[Debug|Release]` - target build type
* `CUDA_HOST_COMPILER=path/to/gcc` - path to CUDA-compatible compiler, by default points to `CMAKE_C_COMPILER`
* `CUDA_TOOLKIT_ROOT_DIR=path/to/cuda` - path to CUDA installation directory

# Running

Enter the build directory:
```sh
# CPU
./RayTracer > image.ppm
# CUDA, requires raytracing.ptx to run
./RayTracerCuda > image.ppm
```

# RayTracer config file

Each program can read config from file, for example:
```sh
cd Release
./RayTracer ../resources/RayTracerConfig.example > image.ppm
```

See format of the config file in `resources/RayTracerConfig.example`.
If no file is specified, default config is used (same as in the example config).

# Contributing
All source files should comply with `.clang-format` configuration.
Run `clang-format` on each file before commit or add git hook to run it automatically:
```sh
cp -i resources/pre_commit .git/hooks/
```
and follow displayed instructions.
