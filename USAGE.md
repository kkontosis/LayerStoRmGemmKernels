# USAGE.md — SM120 Quantized GEMM Kernel Library

## What This Is

CUDA kernel library for quantized GEMM on SM120 GPUs (RTX 5090/5080). Provides NVFP4, FP8, and Q4_K GEMM variants (single + grouped) plus quantization helpers. Shared dependency for both attention (LayerStoRmKernels) and expert (LayerStoRmExpertKernels) pipelines.

## Kernel Inventory

| Kernel | File | Precision | Purpose |
|--------|------|-----------|---------|
| **nvfp4_gemm** | `sm120/gemm/nvfp4/nvfp4_gemm.cu` | NVFP4, BF16/FP16 out | Single NVFP4 GEMM (M-dispatch) |
| **nvfp4_grouped_gemm** | `sm120/gemm/nvfp4/nvfp4_grouped_gemm.cu` | NVFP4, BF16/FP16 out | Multi-expert grouped NVFP4 GEMM |
| **fp8_gemm** | `sm120/gemm/fp8/fp8_gemm.cu` | FP8 E4M3, blockwise | Single FP8 GEMM (M>=128) |
| **fp8_grouped_gemm** | `sm120/gemm/fp8/fp8_grouped_gemm.cu` | FP8 E4M3, blockwise | Multi-expert FP8 dispatch loop |
| **q4k_dequant_gemm** | `sm120/gemm/q4k/q4k_dequant_gemm.cu` | Q4_K, BF16 out | CUDA core GEMM/GEMV |
| **q4k_cutlass_gemm** | `sm120/gemm/q4k/q4k_cutlass_gemm.cu` | Q4_K, BF16 out | wmma tensor-core GEMM |
| **bf16_to_nvfp4** | `smxx/quant/bf16_to_nvfp4.cu` | BF16→FP4 | Activation quantization |
| **reformat_scales** | `smxx/quant/reformat_scales.cu` | float32→UE4M3 | Sm1xx scale reformatter |
| **dynamic_fp8_quant** | `smxx/quant/dynamic_fp8_quant.cu` | BF16→FP8 | Per-block FP8 quantization |

## C++ API

All functions in `namespace layerstorm::compute`.

```cpp
#include "sm120/gemm/nvfp4/nvfp4_gemm.h"
#include "sm120/gemm/fp8/fp8_gemm.h"
#include "sm120/gemm/grouped_gemm.h"
#include "sm120/gemm/q4k/q4k_dequant_gemm.h"
#include "sm120/gemm/q4k/q4k_cutlass_gemm.h"
#include "smxx/quant/dynamic_fp8_quant.h"
#include "smxx/quant/bf16_to_nvfp4.h"
#include "smxx/quant/reformat_scales.h"

using namespace layerstorm::compute;

// NVFP4 single GEMM
Nvfp4GemmParams p{.M=M, .N=N, .K=K, .A=a, .B=b, .D=d, .scale_A=sa, .scale_B=sb, .alpha=1.0f, .output_dtype=GemmOutputDtype::kBFloat16};
launch_nvfp4_gemm(p, workspace, stream);

// FP8 single GEMM
Fp8GemmParams fp{.M=M, .N=N, .K=K, .A=a, .B=b, .D=d, .scale_A=sa, .scale_B=sb, .output_dtype=GemmOutputDtype::kBFloat16};
launch_fp8_gemm(fp, workspace, stream);

// Q4K dequant GEMM
Q4KDequantGemmParams qp{.M=M, .N=N, .K=K, .A=a, .B_q4k=b, .C=c};
launch_q4k_dequant_gemm(qp, stream);

// Dynamic FP8 quantization
DynamicFp8QuantParams dp{.num_tokens=M, .hidden_size=K, .input=in, .output=out, .scales=scales};
launch_dynamic_fp8_quant(dp, stream);
```

## Python API

```python
import sm120_gemm_kernels as GK

# NVFP4 GEMM (low-level: pre-quantized FP4 inputs)
D = GK.nvfp4_gemm(A, B, scale_A, scale_B, M, N, K, alpha=1.0, output_dtype="bf16")

# NVFP4 Grouped GEMM
GK.nvfp4_grouped_gemm(A, B, D, sA, sB, alphas, offsets, sf_offsets, sizes, N, K, "bf16")

# FP8 GEMM
D = GK.fp8_gemm(A, B, scale_A, scale_B, M, N, K, "bf16")

# FP8 Grouped GEMM
GK.fp8_grouped_gemm(A, B, D, sA, sB, offsets, sizes, N, K, "bf16")

# Q4K dequant GEMM
D = GK.q4k_dequant_gemm(x_bf16, weight_q4k_packed)

# Q4K tensor-core GEMM
D = GK.q4k_cutlass_gemm(x_bf16, weight_q4k_packed)

# Dynamic FP8 quantization
output_fp8, scales = GK.dynamic_fp8_quant(input_bf16)

# NVFP4 high-level: BF16 input → quantize → GEMM (attention weight path)
D = GK.nvfp4_gemm_from_bf16(x_bf16, weight_packed, weight_scales, weight_scale_global)
wt_scales = GK.nvfp4_gemm_preprocess(weight_scales, N, K)  # reformat scales once

# Workspace queries
ws = GK.query_nvfp4_gemm_workspace_size(M, N, K, "bf16")
ws = GK.query_fp8_gemm_workspace_size(M, N, K, "bf16")
sf_a = GK.query_sf_buffer_size_a(M, N, K)
sf_b = GK.query_sf_buffer_size_b(M, N, K)
```

## Build

```bash
pip install -e . --no-build-isolation
python -c "import torch; import sm120_gemm_kernels; print('OK')"
```

Requires: CUDA 12.8+ (SM120), CUTLASS 4.4.2+ (in `3rd-party/cutlass`), PyTorch 2.x.
Note: `import torch` must precede `import sm120_gemm_kernels` (standard for PyTorch C++ extensions).

Tested with: CUDA 13.1, PyTorch 2.9.1+cu130, CUTLASS 4.4.2, RTX 5090 (SM 12.0).

## License

Apache-2.0 (see `LICENSE.md`). Portions are derived from third-party projects
(llama.cpp/ggml, SGLang, vLLM, NVIDIA CUTLASS, FlashMLA) — see
`THIRD_PARTY_NOTICES.md` for per-upstream attributions and license texts.
