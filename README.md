# Eventide

Eventide is a branching-process simulation project with:

- A C++17 simulation core (`cpp/`)
- A `pybind11` extension module (`bindings/`)
- Python analysis and plotting workflows (`python/`)

Recent updates: added support for alternating case and introduced `synthetic.py`.

The top-level CMake build compiles the C++ core, builds the Python extension (`_eventide`), and installs the Python package in editable mode.

## Repository Layout

- `cpp/`: Core simulator library, criteria, samplers, collectors, tests, benchmark
- `bindings/`: `pybind11` module exposing C++ API to Python
- `python/`: Python package (`eventide`) and research/plot scripts
- `cmake/`: Shared CMake compiler settings
- `*.csv`, `img/`, `optimization_logs/`: input data and generated outputs

## Prerequisites

- CMake `>= 3.15`
- C++17 compiler
- Python 3 with development headers (`Interpreter` + `Development` components)
- `pip`
- Network access during first configure/build (CMake `FetchContent` downloads `exprtk`, `pybind11`, `googletest`, `boost::math`)

## Build (Top Level)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This build:

- Produces the `_eventide` extension in `python/eventide/`
- Runs `pip install -e python/eventide` through the `python_install` target

## Run Tests

```bash
ctest --test-dir build --output-on-failure
```

## Where To Start

- C++ internals: see [`cpp/README.md`](cpp/README.md)
- Python workflows and scripts: see [`python/README.md`](python/README.md)
