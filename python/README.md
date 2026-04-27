# Eventide Python

This directory contains:

- The Python package wrapping the C++ extension (`eventide/`)
- Analysis and plotting scripts for epidemic case studies

## Structure

- `eventide/`: installable package (`setup.py`, wrappers, collectors, criteria, samplers)
- `plots/`: plotting helpers and figure builders
- Top-level `*.py` scripts: scenario-specific runs/analysis
- `robustness_outputs/`, `optimization_logs/`: generated outputs
- `map/`: GIS assets and notebook

## Environment Setup

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pandas lmfit scikit-learn seaborn
```

Install the local package:

```bash
python -m pip install -e python/eventide
```

## Build the C++ Extension for Python

The extension module `_eventide` is built by top-level CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

After build, `_eventide` is placed under `python/eventide/` and the package is installed editable by the `python_install` target.

## Running Scripts

From repository root, with your virtual environment active:

```bash
python python/plot_final.py
```

Other common entry scripts are in `python/` (for example `plot_john_snow_cholera.py`, `plot_FMD.py`, `plot_TDK.py`, `robustness_appendix_analysis.py`).

## Notes

- Several scripts expect repository-root relative paths to CSV files.
- Some scripts are computationally heavy; start by reducing trajectory counts when iterating.
