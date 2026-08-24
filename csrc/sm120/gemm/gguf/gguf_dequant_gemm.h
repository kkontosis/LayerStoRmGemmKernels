#pragma once

//==============================================================================
// GGUF dequant-to-float GEMM — strategy 3 of 3 (the "dequant + GEMM" fallback,
// mirroring llama.cpp's dequantize-to-float path). Format-agnostic: one launcher
// dispatches over all supported GGUF weight types.
//
// Computes  C[M,N] = A[M,K] @ dequant(B)^T  with A,C in BF16 and FP32 accumulate.
// B is a row-major [N, K/QK * block_bytes] packed GGUF weight (one row = one
// output channel), matching the existing q4k_dequant_gemm convention.
//
// M <= 8 : GEMV (1 warp per output channel) — decode path.
// M  > 8 : tiled GEMM (32x32) — prefill correctness path. Performance-critical
//          prefill should prefer the mmq integer path; this is the fallback.
//==============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace layerstorm::compute {

// Supported GGUF weight quantization types (weight side only).
enum class GgufType {
    Q2_K = 0,
    Q3_K = 1,
    Q4_K = 2,
    Q5_K = 3,
    Q6_K = 4,
    Q8_0 = 5,
    MXFP4 = 6,  // OCP MX e2m1 + E8M0 block scale (17 B / 32 values)
};

struct GgufDequantGemmParams {
    int M, N, K;
    const __nv_bfloat16* A;   // [M, K] activations, row-major
    const void*          B;    // [N, K/QK * block_bytes] packed GGUF weight
    __nv_bfloat16*       C;    // [M, N] output, row-major
    GgufType             type;
};

// Number of weight values per packed block for a given type (32 for Q8_0, 256
// for the k-quants). Useful for host-side size validation.
int gguf_block_values(GgufType type);

// Packed bytes per block for a given type (Q2_K=84, Q3_K=110, Q4_K=144,
// Q5_K=176, Q6_K=210, Q8_0=34).
int gguf_block_bytes(GgufType type);

void launch_gguf_dequant_gemm(const GgufDequantGemmParams& params, cudaStream_t stream);

}  // namespace layerstorm::compute
