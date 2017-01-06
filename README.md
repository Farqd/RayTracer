# RayTracer
CUDA Ray Tracer

# Building

**Debug**
```
mkdir Debug
cd Debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

**Release**
```
mkdir Release
cd Release
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

# Contributing
All source files should comply with `.clang-format` configuration.
Run `clang-format` on each file before commit or add git hook to run it automatically:
```
cp -i resources/pre_commit .git/hooks/
```
and follow displayed instructions.
