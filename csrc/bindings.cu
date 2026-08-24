// SM120 Quantized GEMM Kernels — single compilation unit.
//
// Arch-generic kernel sources (csrc/smxx/) are #included here for single-TU
// compilation. SM120-specific CUTLASS-heavy kernels + Q4K kernels are compiled
// as separate TUs listed in setup.py.

// Arch-generic quantization helper implementations
#include "smxx/quant/bf16_to_nvfp4.cu"
#include "smxx/quant/bf16_to_nvfp4_grouped.cu"
#include "smxx/quant/silu_mul_to_nvfp4_grouped.cu"
#include "smxx/quant/reformat_scales.cu"
#include "smxx/quant/dynamic_fp8_quant.cu"
#include "smxx/quant/weight_fp8_quant.cu"
#include "smxx/quant/nvfp4_dequant_bf16.cu"

// SM120 kernel headers (type visibility for Python wrappers)
#include "sm120/gemm/nvfp4/nvfp4_gemm.h"
#include "sm120/gemm/grouped_gemm.h"
#include "sm120/gemm/fp8/fp8_gemm.h"
#include "sm120/gemm/q4k/q4k_dequant_gemm.h"
#include "sm120/gemm/q4k/q4k_cutlass_gemm.h"
#include "sm120/gemm/gguf/gguf_grouped_gemm.h"   // GGUF grouped GEMM (MoE)
#include "sm120/gemm/gguf/gguf_mmvq.h"           // GGUF single mmvq (decode GEMV)

// Python bindings (pybind11 module definition)
#include "bindings_python.cu"
