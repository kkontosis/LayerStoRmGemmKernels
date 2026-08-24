# Development Guide — LayerStoRmGemmKernels

This document defines the standards for this project. All conventions mirror the sibling projects **LayerStoRmKernels** (SM120 SnapMLA attention kernels) and **LayerStoRmExpertKernels** (SM120 Expert/MoE kernels) so that all three libraries share the same structure, style, and integration patterns.

**Read this document at the start of every session.**

## 1. Project Identity

| Property | This project |
|----------|-------------|
| Domain | Quantized GEMM (single + grouped) and quantization helpers |
| Kernel types | NVFP4 GEMM, FP8 GEMM, Q4K GEMM, grouped GEMM, BF16→FP4/FP8 quant, scale reformatting |
| Python module | `sm120_gemm_kernels` |
| Namespace | `layerstorm::compute` (kernels), `layerstorm::formats` (Q4K block structs) |
| Target GPU | SM120 (RTX 5090/5080) |
| Quantization | NVFP4 E2M1, FP8 E4M3/E5M2, Q4_K (GGUF) |
| Consumers | LayerStoRmKernels (attention), LayerStoRmExpertKernels (expert), LayerStoRm3 (engine) |

## 2. Directory Structure

```
csrc/
  sm120/gemm/                               # SM120 GEMM kernels (CUTLASS, compiled as separate TUs)
    grouped_gemm.h                          # Nvfp4GroupedGemmParams + Fp8GroupedGemmParams
    nvfp4/
      nvfp4_gemm.h                          # Nvfp4GemmParams + GemmOutputDtype + SF query funcs
      nvfp4_gemm.cu                         # CUTLASS 3.x BlockScaledTensorOp, M-dispatch
      nvfp4_grouped_gemm.cu                 # CUTLASS GroupProblemShape + PtrArray grouped GEMM
    fp8/
      fp8_gemm.h                            # Fp8GemmParams + launch
      fp8_gemm.cu                           # CUTLASS 3.x OpClassTensorOp + blockwise scales
      fp8_grouped_gemm.cu                   # Per-expert dispatch loop (calls fp8_gemm)
    q4k/
      q4k_dequant_gemm.h                    # Q4KDequantGemmParams + launch
      q4k_dequant_gemm.cu                   # Hand-written CUDA core GEMM/GEMV (M-dispatch)
      q4k_cutlass_gemm.h                    # Q4KCutlassGemmParams + launch
      q4k_cutlass_gemm.cu                   # wmma tensor-core GEMM (cp.async + BF16 MMA)
  smxx/                                     # Arch-generic kernels
    utils.h                                 # CHECK_CUDA, FLASH_ASSERT macros
    quant/
      bf16_to_nvfp4.h                       # Bf16ToNvfp4Params + launch
      bf16_to_nvfp4.cu                      # BF16→FP4 quantization + Sm1xx scale layout
      reformat_scales.h                     # ReformatScalesParams + launch
      reformat_scales.cu                    # Row-major float32→Sm1xx UE4M3 reformatter
      dynamic_fp8_quant.h                   # DynamicFp8QuantParams + launch
      dynamic_fp8_quant.cu                  # BF16→FP8 E4M3 per-block quantization
  formats/
    q4k_format.h                            # Q4K block struct + dequant helpers
  bindings.cu                               # Main TU: #includes smxx .cu + headers + bindings_python
  bindings_python.cu                        # pybind11 wrappers for all kernel launch functions
```

## 3. Naming Conventions

### Namespaces

```cpp
layerstorm::compute     // All kernel launch functions, param structs, enums
layerstorm::formats     // Q4K block_q4_K struct + format helpers
```

### Param Structs

Each kernel defines its own param struct in the header:

```cpp
namespace layerstorm::compute {
struct Nvfp4GemmParams {
    int M, N, K;
    const void* A;           // [M, K/2] packed FP4
    const void* B;           // [N, K/2] packed FP4
    void* D;                 // [M, N] output
    const void* scale_A;     // UE4M3 scales
    const void* scale_B;     // UE4M3 scales
    float alpha;
    GemmOutputDtype output_dtype;
};
}
```

### Launch Functions

Same pattern: param struct + workspace + stream.

```cpp
namespace layerstorm::compute {
void launch_nvfp4_gemm(const Nvfp4GemmParams& params, void* workspace, void* stream);
size_t query_nvfp4_gemm_workspace_size(int M, int N, int K, GemmOutputDtype);
}
```

## 4. Build System

### setup.py Structure

Mirrors ExpertKernels exactly. Key points:
- CUTLASS from `3rd-party/cutlass` submodule (v4.4.2+) preferred, env var fallback
- Version check warns if < 4.4 (NVFP4 requires 4.4+)
- CUDA version bypass for PyTorch cu130 mismatch
- `-gencode=arch=compute_120f,code=sm_120f` (the `f` flag is required)
- Separate TUs for CUTLASS-heavy kernels; smxx .cu files #included into bindings.cu

### Compiler Flags

```
-std=c++17 -O2 --expt-relaxed-constexpr --expt-extended-lambda
-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
-U__CUDA_NO_BFLOAT16_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__
-gencode=arch=compute_120f,code=sm_120f
```

## 5. Test Conventions

### Reference Tests (`tests/test_reference.py`)
Pure-PyTorch implementations that establish error budgets. No GPU needed.

### GPU Kernel Tests (`tests/test_kernels.py`)
Per-kernel validation against reference. CUDA events for timing.

### Benchmarks (`benchmarks/benchmark_speed.py`)
50 warmup + 500 timed iterations, CUDA events, JSON output.

## 6. Code Style

- C++17, `#pragma once` for all headers
- `__nv_bfloat16`, `__nv_fp8_e4m3` for CUDA numeric types
- Raw pointers in param structs
- `const __restrict__` on read-only kernel parameters
- `__forceinline__ __device__` for small device helpers
- Apache-2.0 attribution when adapting reference code
