#pragma once

//==============================================================================
// MXFP4 (GGUF, OCP Microscaling) Quantized Weight Format
//
// Block layout (17 bytes, 32 values, 4.25 bits/value):
//   [0:1]   uint8_t e      — E8M0 block scale (power of two, bias 127)
//   [1:17]  uint8_t qs[16] — 32 e2m1 (FP4) codes, two per byte
//
// Nibble → value ordering (NOT interleaved): the LOW nibble of qs[j] is
// element j (j in [0,16)), the HIGH nibble of qs[j] is element j+16.
//
// Dequant: y[j]     = kvalues_mxfp4[qs[j] & 0xF] * d
//          y[j+16]  = kvalues_mxfp4[qs[j] >> 4 ] * d
// where kvalues_mxfp4 holds the DOUBLED e2m1 values as integers
// {0,±1,±2,±3,±4,±6,±8,±12} and d = e8m0_to_fp32_half(e) = 2^(e-127) / 2
// (the halving un-doubles the table). Values are exact integers times a
// power-of-two scale — dp4a-exact in the integer paths (A = d, B = 0).
//
// Ported from llama.cpp ggml (MIT, "The ggml authors"):
//   - ggml/src/ggml-common.h   (block_mxfp4, kvalues_fp4/kvalues_mxfp4)
//   - ggml/src/ggml-quants.c   (dequantize_row_mxfp4)
//   - ggml/src/ggml-impl.h     (ggml_e8m0_to_fp32_half)
// OCP MX spec: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
//==============================================================================

#include <cuda_runtime.h>
#include <cstdint>

namespace layerstorm::formats {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int QK_MXFP4 = 32;           // values per block
static constexpr int BLOCK_MXFP4_SIZE = 17;   // bytes per block

// ---------------------------------------------------------------------------
// Block struct (binary-compatible with ggml block_mxfp4)
// ---------------------------------------------------------------------------

struct block_mxfp4 {
    uint8_t e;                 // E8M0 block scale
    uint8_t qs[QK_MXFP4 / 2];  // 32 e2m1 codes, two per byte
};

static_assert(sizeof(block_mxfp4) == BLOCK_MXFP4_SIZE,
              "block_mxfp4 must be 17 bytes");

// ---------------------------------------------------------------------------
// e2m1 value table (doubled, as signed integers). Index = 4-bit code.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int8_t mxfp4_kvalue(int code) {
    // {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12} — port of ggml kvalues_fp4.
    constexpr int8_t kValues[16] = {0, 1, 2,  3,  4,  6,  8,  12,
                                    0, -1, -2, -3, -4, -6, -8, -12};
    return kValues[code & 0xF];
}

// ---------------------------------------------------------------------------
// 8-nibble → 8-int8 table lookup via __byte_perm. Port of llama.cpp
// get_int_from_table_16 (ggml/src/ggml-cuda/vecdotq.cuh, MIT, "The ggml
// authors"): __byte_perm selects with 3-bit indices, so look up the low and
// high table halves separately and select on the 4th bit. Returns
// {x = values of the 4 LOW nibbles, y = values of the 4 HIGH nibbles} of the
// packed int `q4` (byte i → x-lane i / y-lane i).
// ---------------------------------------------------------------------------

__device__ __forceinline__ int2 mxfp4_table_lookup8(uint32_t q4) {
    // kvalues_fp4 as four 32-bit words (little-endian lanes).
    constexpr uint32_t kTab0 = 0x03020100u;  // {0, 1, 2, 3}
    constexpr uint32_t kTab1 = 0x0C080604u;  // {4, 6, 8, 12}
    constexpr uint32_t kTab2 = 0xFDFEFF00u;  // {0, -1, -2, -3}
    constexpr uint32_t kTab3 = 0xF4F8FAFCu;  // {-4, -6, -8, -12}
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low  = __byte_perm(kTab0, kTab1, q4 >> shift);
        const uint32_t high = __byte_perm(kTab2, kTab3, q4 >> shift);
        tmp[i] = __byte_perm(low, high, sel >> shift);
    }
    return make_int2(static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x6420)),
                     static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x7531)));
}

// ---------------------------------------------------------------------------
// E8M0 scale → fp32, pre-divided by 2 (un-doubles the value table).
// Port of ggml_e8m0_to_fp32_half: x < 2 → denormal bit patterns
// (0x00200000 << x = 2^(x-128)); x >= 2 → normalized exponent (x-1).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;
    if (x < 2) {
        bits = 0x00200000u << x;
    } else {
        bits = static_cast<uint32_t>(x - 1) << 23;
    }
    return __uint_as_float(bits);
}

// ---------------------------------------------------------------------------
// dequant_mxfp4_block — dequantize one block (32 values) to float.
// Port of ggml dequantize_row_mxfp4 (single-block body).
// ---------------------------------------------------------------------------

__device__ __forceinline__
void dequant_mxfp4_block(const block_mxfp4& block, float* __restrict__ out) {
    const float d = e8m0_to_fp32_half(block.e);
    #pragma unroll
    for (int j = 0; j < QK_MXFP4 / 2; ++j) {
        out[j] = static_cast<float>(mxfp4_kvalue(block.qs[j] & 0xF)) * d;
        out[j + QK_MXFP4 / 2] =
            static_cast<float>(mxfp4_kvalue(block.qs[j] >> 4)) * d;
    }
}

}  // namespace layerstorm::formats
