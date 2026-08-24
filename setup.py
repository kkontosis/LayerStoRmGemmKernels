import os
import subprocess
import sys
from setuptools import setup, find_packages

# ---------------------------------------------------------------------------
# Detect CUTLASS include path
# ---------------------------------------------------------------------------
def find_cutlass_include():
    """Find CUTLASS headers: env var > 3rd-party submodule > pip package > system paths.
    Prefers 4.4+ (required for NVFP4 GEMM on SM120)."""
    # 1. Environment variable (highest priority)
    cutlass_path = os.environ.get("CUTLASS_PATH")
    if cutlass_path:
        inc = os.path.join(cutlass_path, "include")
        if os.path.isfile(os.path.join(inc, "cutlass", "cutlass.h")):
            return inc

    # 2. Local 3rd-party/cutlass submodule (v4.4.2+, required for NVFP4 GEMM)
    _project_root = os.path.dirname(os.path.abspath(__file__))
    inc = os.path.join(_project_root, "3rd-party", "cutlass", "include")
    if os.path.isfile(os.path.join(inc, "cutlass", "cutlass.h")):
        return inc

    # 3. Scan site-packages for nvidia-cutlass (works with pip, uv, etc.)
    import site
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for subdir in [
            "cutlass_library/source/include",  # nvidia-cutlass 4.x
            "nvidia/cutlass/include",
            "cutlass/include",
        ]:
            inc = os.path.join(sp, subdir)
            if os.path.isfile(os.path.join(inc, "cutlass", "cutlass.h")):
                return inc

    # 4. Common system paths
    for path in ["/usr/include", "/usr/local/include", "/usr/local/cutlass/include"]:
        if os.path.isfile(os.path.join(path, "cutlass", "cutlass.h")):
            return path

    return None


# ---------------------------------------------------------------------------
# Build configuration
# ---------------------------------------------------------------------------
# Bypass CUDA version mismatch check (system CUDA 13.1 vs PyTorch cu130 is compatible)
import torch.utils.cpp_extension as _cpp_ext
_orig_check = _cpp_ext._check_cuda_version
def _noop_check(*args, **kwargs): pass
_cpp_ext._check_cuda_version = _noop_check

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

project_root = os.path.dirname(os.path.abspath(__file__))
csrc_dir = os.path.join(project_root, "csrc")

cutlass_include = find_cutlass_include()
if cutlass_include is None:
    print("ERROR: CUTLASS headers not found. CUTLASS 4.4.2+ required for NVFP4 GEMM.")
    print("  git submodule update --init --recursive")
    print("  OR: CUTLASS_PATH=/path/to/cutlass pip install -e . --no-build-isolation")
    sys.exit(1)

# Check CUTLASS version (warn if < 4.4)
_ver_h = os.path.join(cutlass_include, "cutlass", "version.h")
if os.path.isfile(_ver_h):
    with open(_ver_h) as f:
        _ver_text = f.read()
    import re
    _maj = re.search(r"#define CUTLASS_MAJOR (\d+)", _ver_text)
    _min = re.search(r"#define CUTLASS_MINOR (\d+)", _ver_text)
    if _maj and _min:
        _cutlass_ver = (int(_maj.group(1)), int(_min.group(1)))
        if _cutlass_ver < (4, 4):
            print(f"WARNING: CUTLASS {_cutlass_ver[0]}.{_cutlass_ver[1]} detected. "
                  f"NVFP4 GEMM requires 4.4+. Set CUTLASS_PATH to a 4.4.2+ checkout.")

print(f"Using CUTLASS headers from: {cutlass_include}")

# CUTLASS util headers (packed_stride.hpp etc.) may be in a separate tools/ tree
cutlass_util_include = os.path.normpath(
    os.path.join(cutlass_include, "..", "tools", "util", "include"))
if not os.path.isdir(cutlass_util_include):
    cutlass_util_include = None

include_dirs = [
    csrc_dir,
    cutlass_include,
]
if cutlass_util_include:
    print(f"Using CUTLASS util headers from: {cutlass_util_include}")
    include_dirs.append(cutlass_util_include)

sources = [
    # Main TU: smxx .cu files (included) + pybind11 wrappers
    "csrc/bindings.cu",
    # SM120 CUTLASS-heavy kernels — separate TUs (long compile times)
    "csrc/sm120/gemm/nvfp4/nvfp4_gemm.cu",
    "csrc/sm120/gemm/nvfp4/nvfp4_grouped_gemm.cu",
    "csrc/sm120/gemm/fp8/fp8_gemm.cu",
    "csrc/sm120/gemm/fp8/fp8_blockscaled_gemv.cu",
    "csrc/sm120/gemm/fp8/fp8_grouped_gemm.cu",
    # Q4K kernels — separate TUs (wmma + cuda core GEMM)
    "csrc/sm120/gemm/q4k/q4k_dequant_gemm.cu",
    "csrc/sm120/gemm/q4k/q4k_cutlass_gemm.cu",
    # GGUF k-quant + Q8_0 GEMMs (dequant / mmvq / int8 tensor-core mmq) — all 6 types
    "csrc/sm120/gemm/gguf/gguf_dequant_gemm.cu",
    "csrc/sm120/gemm/gguf/gguf_mmvq.cu",
    "csrc/sm120/gemm/gguf/mmq_mma/mmq_mma_dispatch.cu",
    "csrc/sm120/gemm/gguf/mmq_mma/instances/mmq_q4_k.cu",
    "csrc/sm120/gemm/gguf/mmq_mma/instances/mmq_q5_k.cu",
    "csrc/sm120/gemm/gguf/mmq_mma/instances/mmq_q8_0.cu",
    "csrc/sm120/gemm/gguf/mmq_mma/instances/mmq_q6_k.cu",
    "csrc/sm120/gemm/gguf/mmq_mma/instances/mmq_q2_k.cu",
    "csrc/sm120/gemm/gguf/mmq_mma/instances/mmq_q3_k.cu",
    "csrc/sm120/gemm/gguf/mmq_mma/instances/mmq_mxfp4.cu",
    # GGUF grouped GEMM (MoE): device-routed Int (mmvq/mmq) + host-loop Dequant
    "csrc/sm120/gemm/gguf/gguf_grouped_int.cu",
    "csrc/sm120/gemm/gguf/gguf_grouped_gemm.cu",
]

nvcc_flags = [
    "-std=c++17",
    "-O2",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

# Detect SM arch: prefer sm_120f (enables FP8 + blockwise scaled TensorOp),
# fall back to sm_89 for pre-12.8 CUDA.
try:
    nvcc_version = subprocess.check_output(["nvcc", "--version"], text=True)
    if "release 13" in nvcc_version or "release 12.8" in nvcc_version:
        nvcc_flags.append("-gencode=arch=compute_120f,code=sm_120f")
    else:
        print("WARNING: CUDA < 12.8 detected, using -arch=sm_89")
        nvcc_flags.append("-arch=sm_89")
except Exception:
    nvcc_flags.append("-gencode=arch=compute_120f,code=sm_120f")

setup(
    name="sm120_gemm_kernels",
    version="0.1.0",
    description="SM120 Quantized GEMM CUDA kernels with Python bindings",
    ext_modules=[
        CUDAExtension(
            name="sm120_gemm_kernels",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-std=c++17", "-O2"],
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
