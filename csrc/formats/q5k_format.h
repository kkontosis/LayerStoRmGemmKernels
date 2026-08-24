#pragma once

//==============================================================================
// Q5_K (GGUF) Quantized Weight Format
//
// Super-block layout (176 bytes, 256 values, 5.5 bits/value):
//   [0:4]     half2   dm         — d (super-block scale), dmin (super-block min)
//   [4:16]    uint8_t scales[12] — 8 sub-block (scale,min) pairs, packed 6-bit
//   [16:48]   uint8_t qh[32]     — 5th (high) bit of each quant
//   [48:176]  uint8_t qs[128]    — low 4 bits of each quant, 2 per byte
//
// 8 sub-blocks of 32 values. Each 5-bit quant = (qs nibble) + (qh bit ? 16 : 0).
// Dequant: y[i] = d * sc[j] * q5[i] - dmin * m[j], j = i/32
//
// Binary-compatible with ggml block_q5_K.
//
// Ported from llama.cpp ggml (MIT, "The ggml authors"):
//   - ggml/src/ggml-common.h          (block_q5_K)
//   - ggml/src/ggml-cuda/convert.cu   (dequantize_block_q5_K)
// Source revision: commit 1191758c (tag b9780).
//==============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "formats/kquant_common.h"  // QK_K, K_SCALE_SIZE, get_scale_min_k4

namespace layerstorm::formats {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int BLOCK_Q5K_SIZE = 176;  // bytes per super-block

// ---------------------------------------------------------------------------
// Block struct
// ---------------------------------------------------------------------------

struct block_q5_K {
    half2   dm;                     // super-block scale (low) + min (high)
    uint8_t scales[K_SCALE_SIZE];   // 12: packed 6-bit scales and mins
    uint8_t qh[QK_K / 8];           // 32: high (5th) bit of each quant
    uint8_t qs[QK_K / 2];           // 128: low 4 bits, 2 per byte
};

static_assert(sizeof(block_q5_K) == 176, "block_q5_K must be 176 bytes");

// ---------------------------------------------------------------------------
// dequant_q5k_block — dequantize one super-block (256 values) to float.
//
// Port of ggml dequantize_block_q5_K (convert.cu). The reference kernel uses
// 64 threads, each producing 4 outputs (positions +0/+1/+32/+33); we replay
// that decomposition single-threaded so the ordering matches ggml exactly.
// ---------------------------------------------------------------------------

__device__ __forceinline__
void dequant_q5k_block(const block_q5_K& block, float* __restrict__ out) {
    const float dall = __half2float(__low2half(block.dm));
    const float dmin = __half2float(__high2half(block.dm));

    #pragma unroll
    for (int t = 0; t < 64; ++t) {
        const int il = t / 16;   // 0..3
        const int ir = t % 16;   // 0..15
        const int is = 2 * il;   // 0,2,4,6

        float* y = out + 64 * il + 2 * ir;
        const uint8_t* ql = block.qs + 32 * il + 2 * ir;
        const uint8_t* qh = block.qh + 2 * ir;

        uint8_t sc, m;
        get_scale_min_k4(is + 0, block.scales, sc, m);
        const float d1 = dall * sc; const float m1 = dmin * m;
        get_scale_min_k4(is + 1, block.scales, sc, m);
        const float d2 = dall * sc; const float m2 = dmin * m;

        uint8_t hm = 1 << (2 * il);
        y[ 0] = d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1;
        y[ 1] = d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1;
        hm <<= 1;
        y[32] = d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2;
        y[33] = d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2;
    }
}

}  // namespace layerstorm::formats
