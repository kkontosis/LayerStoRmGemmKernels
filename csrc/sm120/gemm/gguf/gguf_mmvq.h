#pragma once

//==============================================================================
// GGUF integer mat-vec (mmvq) — strategy "int" of the GGUF GEMM, mirroring
// llama.cpp's mul_mat_vec_q: the activation is quantized to Q8_1 (8-bit) and
// each output is an integer dp4a dot-product against the still-packed GGUF
// weight, with per-sub-block scale/min applied. Decode (small-M) path.
//
// Computes  C[M,N] = A[M,K] @ dequant(B)^T  with A,C in BF16. Carries the
// standard ~8-bit activation-quant error vs the lossless-activation dequant
// path; intended for decode throughput.
//==============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "sm120/gemm/gguf/gguf_dequant_gemm.h"  // GgufType, gguf_block_bytes/values

namespace layerstorm::compute {

struct GgufMmvqParams {
    int M, N, K;
    const __nv_bfloat16* A;   // [M, K] activations, row-major
    const void*          B;    // [N, (K/QK)*block_bytes] packed GGUF weight
    __nv_bfloat16*       C;    // [M, N] output, row-major
    GgufType             type;
};

// Workspace (caller-allocated) for the quantized activation: M * (K/32) Q8_1
// blocks of 36 bytes each. Shared by mmvq and mmq.
size_t gguf_mmvq_workspace_bytes(int M, int K);

// Quantize a BF16 activation [M,K] into Q8_1 blocks (M*(K/32) block_q8_1).
// Exposed so the mmq path reuses the same activation quantizer.
void gguf_quantize_q8_1_cuda(const __nv_bfloat16* x, void* q8_1_out,
                             int M, int K, cudaStream_t stream);

void launch_gguf_mmvq(const GgufMmvqParams& params, void* q8_1_workspace, cudaStream_t stream);

}  // namespace layerstorm::compute
