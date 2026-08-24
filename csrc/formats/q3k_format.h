#pragma once

//==============================================================================
// Q3_K (GGUF) Quantized Weight Format
//
// Super-block layout (110 bytes, 256 values, 3.4375 bits/value):
//   [0:32]    uint8_t hmask[32]  — high (3rd) bit of each quant
//   [32:96]   uint8_t qs[64]     — low 2 bits of each quant, 4 per byte
//   [96:108]  uint8_t scales[12] — 16 sub-block scales, packed 6-bit
//   [108:110] ggml_half d        — super-block scale
//
// 16 sub-blocks of 16 values. Each 3-bit quant = (qs 2 bits) with the high bit
// from hmask; reconstructed value is centered ((hmask bit) ? 0 : -4) then scaled
// by dl = d * (us - 32), where us is the 6-bit sub-block scale.
//
// Binary-compatible with ggml block_q3_K.
//
// Ported from llama.cpp ggml (MIT, "The ggml authors"):
//   - ggml/src/ggml-common.h          (block_q3_K)
//   - ggml/src/ggml-cuda/convert.cu   (dequantize_block_q3_K)
// Source revision: commit 1191758c (tag b9780).
//==============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "formats/kquant_common.h"  // QK_K

namespace layerstorm::formats {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int BLOCK_Q3K_SIZE = 110;  // bytes per super-block

// ---------------------------------------------------------------------------
// Block struct
// ---------------------------------------------------------------------------

struct block_q3_K {
    uint8_t hmask[QK_K / 8];  // 32: high bit of each quant
    uint8_t qs[QK_K / 4];     // 64: low 2 bits, 4 per byte
    uint8_t scales[12];       // 12: 6-bit sub-block scales
    half    d;                // super-block scale
};

static_assert(sizeof(block_q3_K) == 110, "block_q3_K must be 110 bytes");

// ---------------------------------------------------------------------------
// dequant_q3k_block — dequantize one super-block (256 values) to float.
//
// Port of ggml dequantize_block_q3_K (convert.cu). The reference kernel uses
// 64 threads with a non-trivial index decomposition; we replay it
// single-threaded (t = threadIdx.x in [0,64)) so the ordering and the 6-bit
// scale unpacking match ggml exactly.
// ---------------------------------------------------------------------------

__device__ __forceinline__
void dequant_q3k_block(const block_q3_K& block, float* __restrict__ out) {
    const float d_all = __half2float(block.d);
    const uint8_t* q_base  = block.qs;
    const uint8_t* hm      = block.hmask;
    const uint8_t* scales  = block.scales;

    #pragma unroll
    for (int t = 0; t < 64; ++t) {
        const int r   = t / 4;            // 0..15
        const int tid = r / 2;            // 0..7
        const int is0 = r % 2;            // 0..1
        const int l0  = 16 * is0 + 4 * (t % 4);
        const int n   = tid / 4;          // 0..1
        const int j   = tid - 4 * n;      // 0..3

        const uint8_t m = 1 << (4 * n + j);
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;

        const int us =
            is <  4 ? (scales[is - 0] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4) :
            is <  8 ? (scales[is - 0] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4) :
            is < 12 ? (scales[is - 8] >>  4) | (((scales[is + 0] >> 4) & 3) << 4) :
                      (scales[is - 8] >>  4) | (((scales[is - 4] >> 6) & 3) << 4);
        const float dl = d_all * (us - 32);

        float* y = out + 128 * n + 32 * j;
        const uint8_t* q = q_base + 32 * n;

        #pragma unroll
        for (int l = l0; l < l0 + 4; ++l) {
            y[l] = dl * (static_cast<int8_t>((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
        }
    }
}

}  // namespace layerstorm::formats
