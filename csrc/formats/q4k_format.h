#pragma once

//==============================================================================
// Q4_K (GGUF) Quantized Weight Format
//
// Super-block layout (144 bytes, 256 values, 4.5 bits/value):
//   [0:4]    half2 dm       — d (low half) = super-block scale,
//                             dmin (high half) = super-block min scale
//   [4:16]   uint8_t[12]    — 8 sub-block scales + 8 mins, packed 6-bit
//   [16:144] uint8_t[128]   — 256 4-bit quants, 2 per byte
//
// Dequant: y[i] = d * sc[j] * q[i] - dmin * m[j], j = i/32
//
// 8 sub-blocks of 32 values each. Scales/mins are 6-bit (0-63).
// Quants are 4-bit unsigned (0-15), packed low nibble first.
//
// Binary-compatible with ggml block_q4_K (ggml-common.h).
//==============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "formats/kquant_common.h"  // QK_K, get_scale_min_k4

namespace layerstorm::formats {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// QK_K (256) is defined in kquant_common.h
static constexpr int BLOCK_Q4K_SIZE = 144;    // bytes per super-block
static constexpr int Q4K_N_SUB_BLOCKS = 8;    // sub-blocks per super-block
static constexpr int Q4K_SUB_BLOCK_SIZE = 32;  // values per sub-block

// ---------------------------------------------------------------------------
// Block struct
// ---------------------------------------------------------------------------

struct block_q4_K {
    half2 dm;              // super-block scale (low) + min scale (high)
    uint8_t scales[12];    // packed 6-bit sub-block scales and mins
    uint8_t qs[128];       // 256 4-bit quants, 2 per byte
};

static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");

// get_scale_min_k4 is provided by kquant_common.h (shared with Q5_K).

// ---------------------------------------------------------------------------
// extract_q4k_scales — extract all 8 scales and 8 mins as half
// ---------------------------------------------------------------------------

__device__ __forceinline__
void extract_q4k_scales(const block_q4_K& block, half* __restrict__ sc,
                        half* __restrict__ mn) {
    #pragma unroll
    for (int j = 0; j < Q4K_N_SUB_BLOCKS; ++j) {
        uint8_t d_val, m_val;
        get_scale_min_k4(j, block.scales, d_val, m_val);
        sc[j] = __int2half_rn(d_val);
        mn[j] = __int2half_rn(m_val);
    }
}

// ---------------------------------------------------------------------------
// dequant_q4k_block — dequantize one super-block (256 values) to half
//
// Port of ggml dequantize_block_q4_K (dequantize.cuh:166-196), single-thread
// version. Writes 256 half values to out[0..255].
//
// For GEMM kernels, prefer using get_scale_min_k4 directly with per-tile
// dequant to reduce register pressure.
// ---------------------------------------------------------------------------

__device__ __forceinline__
void dequant_q4k_block(const block_q4_K& block, half* __restrict__ out) {
    const half dall = __low2half(block.dm);
    const half dmin = __high2half(block.dm);

    half sc[Q4K_N_SUB_BLOCKS];
    half mn[Q4K_N_SUB_BLOCKS];
    extract_q4k_scales(block, sc, mn);

    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const int is = pair * 2;
        const half d1 = __hmul(dall, sc[is]);
        const half m1 = __hmul(dmin, mn[is]);
        const half d2 = __hmul(dall, sc[is + 1]);
        const half m2 = __hmul(dmin, mn[is + 1]);

        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            const uint8_t qbyte = block.qs[pair * 32 + l];
            out[pair * 64 + l]      = __hsub(__hmul(d1, __int2half_rn(qbyte & 0xF)), m1);
            out[pair * 64 + l + 32] = __hsub(__hmul(d2, __int2half_rn(qbyte >> 4)),  m2);
        }
    }
}

// ---------------------------------------------------------------------------
// dequant_q4k_block (float overload) — dequantize one super-block to float.
//
// Float-output variant matching the other GGUF format headers (q2k/q3k/q5k/
// q6k/q8_0), for use by the gguf_mul_mat dispatcher and q_absorb where FP32
// accumulation against a BF16 activation is wanted.
// Port of ggml dequantize_block_q4_K (convert.cu).
// ---------------------------------------------------------------------------

__device__ __forceinline__
void dequant_q4k_block(const block_q4_K& block, float* __restrict__ out) {
    const float dall = __half2float(__low2half(block.dm));
    const float dmin = __half2float(__high2half(block.dm));

    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const int is = pair * 2;
        uint8_t sc, m;
        get_scale_min_k4(is + 0, block.scales, sc, m);
        const float d1 = dall * sc; const float m1 = dmin * m;
        get_scale_min_k4(is + 1, block.scales, sc, m);
        const float d2 = dall * sc; const float m2 = dmin * m;

        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            const uint8_t qbyte = block.qs[pair * 32 + l];
            out[pair * 64 + l]      = d1 * (qbyte & 0xF) - m1;
            out[pair * 64 + l + 32] = d2 * (qbyte >> 4)  - m2;
        }
    }
}

}  // namespace layerstorm::formats
