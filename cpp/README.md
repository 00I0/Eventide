# Eventide C++ Core

This directory contains the C++17 simulation engine used by the Python package.

Recent updates: added support for alternating case and introduced `synthetic.py`.

## Contents

- `include/`: public headers (`Simulator`, `Sampler`, `Scenario`, `Criterion`, `Collector`, etc.)
- `src/`: implementations
- `tests/`: unit test and benchmark targets

## Build Standalone

From repository root:

```bash
cmake -S cpp -B build-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp -j
```

## Dependencies

Fetched automatically by CMake:

- `exprtk` (expressions)
- `googletest` (tests)
- `boost::math` (test support)

## Test and Benchmark

```bash
ctest --test-dir build-cpp --output-on-failure
```

The benchmark executable is built as `benchmark` in `build-cpp` (location depends on generator/configuration). Run it directly after build.

## Notes

- Compiler flags are configured in `../cmake/CompilerSettings.cmake`.
- Top-level project build (`../CMakeLists.txt`) also builds Python bindings; use standalone `cpp` build when you only want core C++ work.
