#pragma once

//==============================================================================
// Single-element GGUF dequant: dequant_<type>_one(block, p) returns the one
// dequantized weight at position p within a packed block, without materializing
// the whole block. Used where the access pattern touches scattered single
// elements (e.g. q_absorb's W_UK contraction), so dequanting a full 256-value
// super-block per access would be wasteful.
//
// Each function is the single-position specialization of the corresponding
// dequant_<type>_block in the format headers (same ggml-faithful math and
// output ordering), so per-element and whole-block dequant agree exactly.
//==============================================================================

#include "formats/kquant_common.h"
#include "formats/mxfp4_format.h"
#include "formats/q2k_format.h"
#include "formats/q3k_format.h"
#include "formats/q4k_format.h"
#include "formats/q5k_format.h"
#include "formats/q6k_format.h"
#include "formats/q8_0_format.h"

namespace layerstorm::formats {

__device__ __forceinline__
float dequant_q8_0_one(const block_q8_0& b, int p) {
    return __half2float(b.d) * static_cast<float>(b.qs[p]);
}

__device__ __forceinline__
float dequant_mxfp4_one(const block_mxfp4& b, int p) {
    // p < 16: LOW nibble of qs[p]; p >= 16: HIGH nibble of qs[p-16].
    const uint8_t byte = b.qs[p & 15];
    const int code = (p < QK_MXFP4 / 2) ? (byte & 0xF) : (byte >> 4);
    return static_cast<float>(mxfp4_kvalue(code)) * e8m0_to_fp32_half(b.e);
}

__device__ __forceinline__
float dequant_q4k_one(const block_q4_K& b, int p) {
    const float dall = __half2float(__low2half(b.dm));
    const float dmin = __half2float(__high2half(b.dm));
    const int pair = p / 64, within = p % 64, half = within / 32, l = within % 32;
    const int sub = 2 * pair + half;
    uint8_t sc, m; get_scale_min_k4(sub, b.scales, sc, m);
    const uint8_t qb = b.qs[pair * 32 + l];
    const int nib = half ? (qb >> 4) : (qb & 0xF);
    return dall * sc * nib - dmin * m;
}

__device__ __forceinline__
float dequant_q5k_one(const block_q5_K& b, int p) {
    const float dall = __half2float(__low2half(b.dm));
    const float dmin = __half2float(__high2half(b.dm));
    const int il = p / 64, within = p % 64, half = within / 32, w = within % 32;
    const int sub = 2 * il + half;
    uint8_t sc, m; get_scale_min_k4(sub, b.scales, sc, m);
    const uint8_t ql = b.qs[32 * il + w];
    const uint8_t qh = b.qh[w];
    const uint8_t hm = half ? (uint8_t)(1 << (2 * il + 1)) : (uint8_t)(1 << (2 * il));
    const int nib = half ? (ql >> 4) : (ql & 0xF);
    const int val = nib + ((qh & hm) ? 16 : 0);
    return dall * sc * val - dmin * m;
}

__device__ __forceinline__
float dequant_q6k_one(const block_q6_K& b, int p) {
    const float d = __half2float(b.d);
    const int ip = p / 128, within = p % 128, quad = within / 32, il = within % 32;
    const int sub = 8 * ip + il / 16 + 2 * quad;
    const int qoff = (quad == 1 || quad == 3) ? 32 : 0;
    const int shift_lo = (quad >= 2) ? 4 : 0;
    const int shift_qh = 2 * quad;
    const uint8_t ql = b.ql[64 * ip + qoff + il];
    const uint8_t qh = b.qh[32 * ip + il];
    const int q6 = ((ql >> shift_lo) & 0xF) | (((qh >> shift_qh) & 3) << 4);
    return d * b.scales[sub] * (q6 - 32);
}

__device__ __forceinline__
float dequant_q2k_one(const block_q2_K& b, int p) {
    const float dall = __half2float(__low2half(b.dm));
    const float dmin = __half2float(__high2half(b.dm));
    const int ni = p / 128, within = p % 128, quad = within / 32, l = within % 32;
    const int sub = 8 * ni + l / 16 + 2 * quad;
    const uint8_t scb = b.scales[sub];
    const int q = (b.qs[32 * ni + l] >> (2 * quad)) & 3;
    return dall * (scb & 0xF) * q - dmin * (scb >> 4);
}

__device__ __forceinline__
float dequant_q3k_one(const block_q3_K& b, int p) {
    const float d = __half2float(b.d);
    const int ni = p / 128, within = p % 128, j = within / 32;
    const int w32 = within % 32, sub_half = w32 / 16, l0 = w32 % 16;
    const int is = 8 * ni + 2 * j + sub_half;
    const uint8_t* sc = b.scales;
    const int us =
        is <  4 ? (sc[is - 0] & 0xF) | (((sc[is + 8] >> 0) & 3) << 4) :
        is <  8 ? (sc[is - 0] & 0xF) | (((sc[is + 4] >> 2) & 3) << 4) :
        is < 12 ? (sc[is - 8] >>  4) | (((sc[is + 0] >> 4) & 3) << 4) :
                  (sc[is - 8] >>  4) | (((sc[is - 4] >> 6) & 3) << 4);
    const float dl = d * (us - 32);
    const int idx = sub_half * 16 + l0;
    const int q2 = (b.qs[32 * ni + idx] >> (2 * j)) & 3;
    const uint8_t m = 1 << (4 * ni + j);
    return dl * (q2 - ((b.hmask[idx] & m) ? 0 : 4));
}

}  // namespace layerstorm::formats
