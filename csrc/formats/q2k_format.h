#pragma once

//==============================================================================
// Q2_K (GGUF) Quantized Weight Format
//
// Super-block layout (84 bytes, 256 values, 2.625 bits/value):
//   [0:16]   uint8_t scales[16] — 16 sub-block (scale,min) pairs, 4 bits each
//                                  (low nibble = scale, high nibble = min)
//   [16:80]  uint8_t qs[64]     — 256 2-bit quants, 4 per byte
//   [80:84]  half2   dm         — d (scale for quantized scales),
//                                  dmin (scale for quantized mins)
//
// 16 sub-blocks of 16 values. Dequant:
//   y[i] = d * (scales[j] & 0xF) * q2[i] - dmin * (scales[j] >> 4)
//
// Binary-compatible with ggml block_q2_K.
//
// Ported from llama.cpp ggml (MIT, "The ggml authors"):
//   - ggml/src/ggml-common.h          (block_q2_K)
//   - ggml/src/ggml-cuda/convert.cu   (dequantize_block_q2_K)
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

static constexpr int BLOCK_Q2K_SIZE = 84;  // bytes per super-block

// ---------------------------------------------------------------------------
// Block struct
// ---------------------------------------------------------------------------

struct block_q2_K {
    uint8_t scales[QK_K / 16];  // 16: 4-bit scales and mins
    uint8_t qs[QK_K / 4];       // 64: 2-bit quants, 4 per byte
    half2   dm;                 // d (low) + dmin (high)
};

static_assert(sizeof(block_q2_K) == 84, "block_q2_K must be 84 bytes");

// ---------------------------------------------------------------------------
// dequant_q2k_block — dequantize one super-block (256 values) to float.
//
// Port of ggml dequantize_block_q2_K (convert.cu). The reference kernel uses
// 64 threads (n = tid/32 in {0,1}, l = tid%32), each producing 4 outputs at
// +0/+32/+64/+96 within a 128-value half; we replay it single-threaded so the
// ordering matches ggml exactly.
// ---------------------------------------------------------------------------

__device__ __forceinline__
void dequant_q2k_block(const block_q2_K& block, float* __restrict__ out) {
    const float dall = __half2float(__low2half(block.dm));
    const float dmin = __half2float(__high2half(block.dm));

    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            const int is = 8 * n + l / 16;
            const uint8_t q = block.qs[32 * n + l];
            float* y = out + 128 * n;

            y[l +  0] = dall * (block.scales[is + 0] & 0xF) * ((q >> 0) & 3) - dmin * (block.scales[is + 0] >> 4);
            y[l + 32] = dall * (block.scales[is + 2] & 0xF) * ((q >> 2) & 3) - dmin * (block.scales[is + 2] >> 4);
            y[l + 64] = dall * (block.scales[is + 4] & 0xF) * ((q >> 4) & 3) - dmin * (block.scales[is + 4] >> 4);
            y[l + 96] = dall * (block.scales[is + 6] & 0xF) * ((q >> 6) & 3) - dmin * (block.scales[is + 6] >> 4);
        }
    }
}

}  // namespace layerstorm::formats
