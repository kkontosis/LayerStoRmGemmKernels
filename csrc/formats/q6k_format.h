#pragma once

//==============================================================================
// Q6_K (GGUF) Quantized Weight Format
//
// Super-block layout (210 bytes, 256 values, 6.5625 bits/value):
//   [0:128]   uint8_t ql[128]    — lower 4 bits of each quant
//   [128:192] uint8_t qh[64]     — upper 2 bits of each quant
//   [192:208] int8_t  scales[16] — 16 sub-block scales, 8-bit signed
//   [208:210] ggml_half d        — super-block scale
//
// 16 sub-blocks of 16 values. Each 6-bit quant is reconstructed from 4 low bits
// (ql) + 2 high bits (qh), centered by subtracting 32.
// Dequant: y[i] = d * scales[i/16] * (q6[i] - 32)
//
// Binary-compatible with ggml block_q6_K.
//
// Ported from llama.cpp ggml (MIT, "The ggml authors"):
//   - ggml/src/ggml-common.h          (block_q6_K)
//   - ggml/src/ggml-cuda/convert.cu   (dequantize_block_q6_K)
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

static constexpr int BLOCK_Q6K_SIZE = 210;  // bytes per super-block

// ---------------------------------------------------------------------------
// Block struct
// ---------------------------------------------------------------------------

struct block_q6_K {
    uint8_t ql[QK_K / 2];       // 128: lower 4 bits
    uint8_t qh[QK_K / 4];       // 64:  upper 2 bits
    int8_t  scales[QK_K / 16];  // 16:  8-bit sub-block scales
    half    d;                  // super-block scale
};

static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 bytes");

// ---------------------------------------------------------------------------
// dequant_q6k_block — dequantize one super-block (256 values) to float.
//
// Port of ggml dequantize_block_q6_K (convert.cu). The reference kernel uses
// 64 threads, each producing 4 outputs (positions +0/+32/+64/+96 within a
// 128-value half); we replay that decomposition single-threaded so the output
// ordering matches ggml exactly.
// ---------------------------------------------------------------------------

__device__ __forceinline__
void dequant_q6k_block(const block_q6_K& block, float* __restrict__ out) {
    const float d = __half2float(block.d);

    #pragma unroll
    for (int t = 0; t < 64; ++t) {
        const int ip = t / 32;           // 0 or 1 (which 128-value half)
        const int il = t - 32 * ip;      // 0..31
        const int is = 8 * ip + il / 16; // scale base index

        float* y = out + 128 * ip + il;
        const uint8_t* ql = block.ql + 64 * ip + il;
        const uint8_t  qh = block.qh[32 * ip + il];
        const int8_t*  sc = block.scales + is;

        y[ 0] = d * sc[0] * (static_cast<int8_t>((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
        y[32] = d * sc[2] * (static_cast<int8_t>((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
        y[64] = d * sc[4] * (static_cast<int8_t>((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
        y[96] = d * sc[6] * (static_cast<int8_t>((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
    }
}

}  // namespace layerstorm::formats
