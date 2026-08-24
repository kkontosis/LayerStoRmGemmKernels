# Build & Install — SM120 Quantized GEMM Kernels

## Prerequisites

- NVIDIA GPU: SM120 (RTX 5090/5080)
- CUDA Toolkit: 13.x (or 12.8+ for SM120 target)
- Linux with `apt` package manager

### System packages

```bash
sudo apt-get install python3-dev
```

## Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install ninja pybind11 pytest
```

## CUTLASS

CUTLASS 4.4.2+ is included as a git submodule:

```bash
git submodule update --init --recursive
```

## Build

```bash
pip install -e . --no-build-isolation
```

Smoke test:

```bash
python -c "import sm120_gemm_kernels; print('OK')"
```

## Test

```bash
pytest tests/test_kernels.py -v
python tests/test_reference.py -v
```

## Troubleshooting

### CUTLASS version

CUTLASS **4.4.2+** is required for NVFP4 GEMM on SM120. The pip package `nvidia-cutlass` (4.2.0) has a bug. Use the bundled submodule or set `CUTLASS_PATH` to a 4.4.2+ checkout.

### CUDA version mismatch

`setup.py` bypasses PyTorch's strict CUDA version check. System CUDA 13.1 with PyTorch cu130 is ABI-compatible.

### Missing Python.h

```bash
sudo apt-get install python3-dev
```
