#pragma once

//==============================================================================
// Q8_0 (GGUF) Quantized Weight Format
//
// Block layout (34 bytes, 32 values, 8.5 bits/value):
//   [0:2]   ggml_half d   — block scale
//   [2:34]  int8_t qs[32] — 32 signed 8-bit quants
//
// Dequant: y[l] = d * qs[l]
//
// Unlike the k-quants, Q8_0 uses a flat 32-value block (no super-block, no
// sub-block scales/mins). Binary-compatible with ggml block_q8_0.
//
// Ported from llama.cpp ggml (MIT, "The ggml authors"):
//   - ggml/src/ggml-common.h          (block_q8_0)
//   - ggml/src/ggml-cuda/convert.cu   (dequantize_block_q8_0_f16)
// Source revision: commit 1191758c (tag b9780).
//==============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace layerstorm::formats {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int QK8_0 = 32;            // values per block
static constexpr int BLOCK_Q8_0_SIZE = 34;  // bytes per block

// ---------------------------------------------------------------------------
// Block struct
// ---------------------------------------------------------------------------

struct block_q8_0 {
    half    d;          // block scale
    int8_t  qs[QK8_0];  // 32 signed quants
};

static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

// ---------------------------------------------------------------------------
// dequant_q8_0_block — dequantize one block (32 values) to float.
//
// Port of ggml dequantize_block_q8_0 math (y = d * qs).
// ---------------------------------------------------------------------------

__device__ __forceinline__
void dequant_q8_0_block(const block_q8_0& block, float* __restrict__ out) {
    const float d = __half2float(block.d);
    #pragma unroll
    for (int l = 0; l < QK8_0; ++l) {
        out[l] = d * static_cast<float>(block.qs[l]);
    }
}

}  // namespace layerstorm::formats
