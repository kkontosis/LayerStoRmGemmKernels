#pragma once

//==============================================================================
// Per-type integer policies for the GGUF integer GEMM paths (mmvq, mmq).
//
// Each policy maps a packed GGUF weight block to the integer form used by the
// dp4a / int8-MMA paths: for a 4-value group g (output positions [4g, 4g+4)),
// group() returns 4 signed-int8 "centered" weight quants packed into an int,
// plus the per-sub-block (A, B) such that  w = A*Q + B  (B folded into Q for
// the center-quant types q3/q6/q8). The ordering matches dequant_<type>_block
// exactly, so the integer paths stay numerically consistent with dequant.
//
// Weight loads are vectorized (one/two 32-bit loads per 4-value group instead
// of per-byte loads), exploiting the per-type block alignment:
//   * Q2_K/Q4_K/Q5_K blocks start with a half2 `dm` and have sizes that are
//     multiples of 4 (84/144/176), so every block — and every 4-value qs/qh
//     slice used below — is 4-byte aligned.
//   * Q8_0/Q3_K/Q6_K carry a bare half scale (block sizes 34/110/210), so only
//     2-byte alignment is guaranteed; their 32-bit loads go through a
//     2×uint16_t combine (SASS: LD.16 pairs, still 2-4× fewer transactions
//     than per-byte loads).
// The 32-bit nibble/2-bit extraction idiom ((w >> s) & 0x03030303 etc.)
// follows llama.cpp ggml-cuda vecdotq.cuh get_int_b2/get_int_b4 usage (MIT,
// "The ggml authors", commit 1191758c / tag b9780). Byte-lane arithmetic uses
// __vadd4/__vsub4 (per-byte modular add/sub == exact signed int8 math).
//==============================================================================

#include "formats/kquant_common.h"
#include "formats/mxfp4_format.h"
#include "formats/q2k_format.h"
#include "formats/q3k_format.h"
#include "formats/q4k_format.h"
#include "formats/q5k_format.h"
#include "formats/q6k_format.h"
#include "formats/q8_0_format.h"

#include <cuda_fp16.h>

namespace layerstorm::compute {
namespace gguf_int {

namespace fmt = layerstorm::formats;

__device__ __forceinline__ int pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// 32-bit little-endian load from a 2-byte-aligned address (all GGUF blocks are
// at least 2-byte aligned: every type carries a half scale field).
__device__ __forceinline__ int load_i32_a2(const void* p) {
    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(p);
    return static_cast<int>(static_cast<uint32_t>(p16[0]) |
                            (static_cast<uint32_t>(p16[1]) << 16));
}

// 32-bit load from a 4-byte-aligned address (types whose blocks start with a
// half2 and whose block size is a multiple of 4: Q2_K/Q4_K/Q5_K).
__device__ __forceinline__ int load_i32_a4(const void* p) {
    return *reinterpret_cast<const int*>(p);
}

struct Q8_0 {
    static constexpr int VALS = 32;
    static constexpr int BYTES = fmt::BLOCK_Q8_0_SIZE;
    static constexpr bool kPipeline = true;
    __device__ static void scales(const void* b, float& dw, float& dmw) {
        dw = __half2float(reinterpret_cast<const fmt::block_q8_0*>(b)->d); dmw = 0.0f;
    }
    __device__ static void group(const void* b, int g, float dw, float /*dmw*/,
                                 int& Q, float& A, float& B) {
        // qs at byte offset 2 of a 34-byte block: 2-byte aligned.
        Q = load_i32_a2(reinterpret_cast<const fmt::block_q8_0*>(b)->qs + 4 * g);
        A = dw; B = 0.0f;
    }
};

struct Mxfp4 {
    static constexpr int VALS = 32;
    static constexpr int BYTES = fmt::BLOCK_MXFP4_SIZE;
    static constexpr bool kPipeline = true;
    __device__ static void scales(const void* b, float& dw, float& dmw) {
        dw = fmt::e8m0_to_fp32_half(
            reinterpret_cast<const fmt::block_mxfp4*>(b)->e);
        dmw = 0.0f;
    }
    __device__ static void group(const void* b, int g, float dw, float /*dmw*/,
                                 int& Q, float& A, float& B) {
        // 17-byte block → only 1-byte alignment guaranteed; assemble the
        // 4-byte nibble word per-byte, then one __byte_perm table lookup
        // (llama.cpp get_int_from_table_16 trick — see mxfp4_format.h).
        // Output positions [4g, 4g+4): g < 4 → LOW nibbles of qs[4g..4g+3]
        // (lookup .x), g >= 4 → HIGH nibbles of qs[4g-16..4g-13] (lookup .y).
        // Table values are exact signed integers {0,±1,±2,±3,±4,±6,±8,±12}
        // → dp4a-exact with A = d, B = 0.
        auto p = reinterpret_cast<const fmt::block_mxfp4*>(b);
        const int base = (4 * g) & 15;
        const uint32_t q4 =
            static_cast<uint32_t>(p->qs[base + 0]) |
            (static_cast<uint32_t>(p->qs[base + 1]) << 8) |
            (static_cast<uint32_t>(p->qs[base + 2]) << 16) |
            (static_cast<uint32_t>(p->qs[base + 3]) << 24);
        const int2 v = fmt::mxfp4_table_lookup8(q4);
        Q = ((4 * g) < (fmt::QK_MXFP4 / 2)) ? v.x : v.y;
        A = dw;
        B = 0.0f;
    }
};

struct Q4K {
    static constexpr int VALS = 256;
    static constexpr int BYTES = fmt::BLOCK_Q4K_SIZE;
    static constexpr bool kPipeline = true;
    __device__ static void scales(const void* b, float& dw, float& dmw) {
        auto p = reinterpret_cast<const fmt::block_q4_K*>(b);
        dw = __half2float(__low2half(p->dm)); dmw = __half2float(__high2half(p->dm));
    }
    __device__ static void group(const void* b, int g, float dw, float dmw,
                                 int& Q, float& A, float& B) {
        auto p = reinterpret_cast<const fmt::block_q4_K*>(b);
        const int p0 = 4 * g, pair = p0 / 64, within = p0 % 64, half = within / 32, l = within % 32;
        uint8_t sc, m; fmt::get_scale_min_k4(2 * pair + half, p->scales, sc, m);
        A = dw * sc; B = -dmw * m;
        // qs at offset 16 of a 144-byte block, l a multiple of 4: 4-byte aligned.
        const int w = load_i32_a4(&p->qs[pair * 32 + l]);
        Q = (half ? (w >> 4) : w) & 0x0F0F0F0F;
    }
};

struct Q5K {
    static constexpr int VALS = 256;
    static constexpr int BYTES = fmt::BLOCK_Q5K_SIZE;
    // 256-VALS K-quant: eligible for the superblock-unrolled pipelined GEMV
    // (raises memory-level parallelism at M_e=1 decode). See the pipelined
    // grouped kernel in gguf_grouped_int.cu.
    static constexpr bool kPipeline = true;
    __device__ static void scales(const void* b, float& dw, float& dmw) {
        auto p = reinterpret_cast<const fmt::block_q5_K*>(b);
        dw = __half2float(__low2half(p->dm)); dmw = __half2float(__high2half(p->dm));
    }
    __device__ static void group(const void* b, int g, float dw, float dmw,
                                 int& Q, float& A, float& B) {
        auto p = reinterpret_cast<const fmt::block_q5_K*>(b);
        const int p0 = 4 * g, il = p0 / 64, within = p0 % 64, half = within / 32, w = within % 32;
        uint8_t sc, m; fmt::get_scale_min_k4(2 * il + half, p->scales, sc, m);
        A = dw * sc; B = -dmw * m;
        // qs at offset 48, qh at offset 16 of a 176-byte block, w a multiple
        // of 4: both 4-byte aligned.
        const int wl = load_i32_a4(&p->qs[32 * il + w]);
        const int wh = load_i32_a4(&p->qh[w]);
        const int nib = (half ? (wl >> 4) : wl) & 0x0F0F0F0F;
        const int hbit = 2 * il + half;                       // 5th-bit position
        Q = nib | (((wh >> hbit) & 0x01010101) << 4);
    }
};

struct Q6K {
    static constexpr int VALS = 256;
    static constexpr int BYTES = fmt::BLOCK_Q6K_SIZE;
    static constexpr bool kPipeline = true;   // see Q5K::kPipeline
    __device__ static void scales(const void* b, float& dw, float& dmw) {
        dw = __half2float(reinterpret_cast<const fmt::block_q6_K*>(b)->d); dmw = 0.0f;
    }
    __device__ static void group(const void* b, int g, float dw, float /*dmw*/,
                                 int& Q, float& A, float& B) {
        auto p = reinterpret_cast<const fmt::block_q6_K*>(b);
        const int p0 = 4 * g, ip = p0 / 128, within = p0 % 128, quad = within / 32, il0 = within % 32;
        A = dw * (float)p->scales[8 * ip + il0 / 16 + 2 * quad]; B = 0.0f;
        const int qoff = (quad == 1 || quad == 3) ? 32 : 0;
        const int shift_lo = (quad >= 2) ? 4 : 0;
        const int shift_qh = 2 * quad;
        // 210-byte block: only 2-byte alignment guaranteed (offsets even).
        const int wl = load_i32_a2(&p->ql[64 * ip + qoff + il0]);
        const int wh = load_i32_a2(&p->qh[32 * ip + il0]);
        const int nib  = (wl >> shift_lo) & 0x0F0F0F0F;
        const int hi2  = (wh >> shift_qh) & 0x03030303;
        Q = __vsub4(nib | (hi2 << 4), 0x20202020);   // per-byte q - 32
    }
};

struct Q2K {
    static constexpr int VALS = 256;
    static constexpr int BYTES = fmt::BLOCK_Q2K_SIZE;
    static constexpr bool kPipeline = true;
    __device__ static void scales(const void* b, float& dw, float& dmw) {
        auto p = reinterpret_cast<const fmt::block_q2_K*>(b);
        dw = __half2float(__low2half(p->dm)); dmw = __half2float(__high2half(p->dm));
    }
    __device__ static void group(const void* b, int g, float dw, float dmw,
                                 int& Q, float& A, float& B) {
        auto p = reinterpret_cast<const fmt::block_q2_K*>(b);
        const int p0 = 4 * g, ni = p0 / 128, within = p0 % 128, quad = within / 32, l0 = within % 32;
        const uint8_t scb = p->scales[8 * ni + l0 / 16 + 2 * quad];
        A = dw * (scb & 0xF); B = -dmw * (scb >> 4);
        // qs at offset 16 of an 84-byte block, l0 a multiple of 4: 4-byte aligned.
        const int w = load_i32_a4(&p->qs[32 * ni + l0]);
        Q = (w >> (2 * quad)) & 0x03030303;
    }
};

struct Q3K {
    static constexpr int VALS = 256;
    static constexpr int BYTES = fmt::BLOCK_Q3K_SIZE;
    static constexpr bool kPipeline = true;
    __device__ static void scales(const void* b, float& dw, float& dmw) {
        dw = __half2float(reinterpret_cast<const fmt::block_q3_K*>(b)->d); dmw = 0.0f;
    }
    __device__ static void group(const void* b, int g, float dw, float /*dmw*/,
                                 int& Q, float& A, float& B) {
        auto p = reinterpret_cast<const fmt::block_q3_K*>(b);
        const int p0 = 4 * g, ni = p0 / 128, within = p0 % 128, j = within / 32;
        const int w32 = within % 32, sub_half = w32 / 16, l0 = w32 % 16;
        const int is = 8 * ni + 2 * j + sub_half;
        const uint8_t* sc = p->scales;
        const int us =
            is <  4 ? (sc[is - 0] & 0xF) | (((sc[is + 8] >> 0) & 3) << 4) :
            is <  8 ? (sc[is - 0] & 0xF) | (((sc[is + 4] >> 2) & 3) << 4) :
            is < 12 ? (sc[is - 8] >>  4) | (((sc[is + 0] >> 4) & 3) << 4) :
                      (sc[is - 8] >>  4) | (((sc[is - 4] >> 6) & 3) << 4);
        A = dw * (us - 32); B = 0.0f;
        const int bit = 4 * ni + j;                  // hmask bit position (0..7)
        const int shift = 2 * j;
        // 110-byte block: only 2-byte alignment guaranteed (offsets even).
        const int wq = load_i32_a2(&p->qs[32 * ni + sub_half * 16 + l0]);
        const int wh = load_i32_a2(&p->hmask[sub_half * 16 + l0]);
        const int q2   = (wq >> shift) & 0x03030303;
        const int hset = (wh >> bit) & 0x01010101;   // 1 where high bit set
        // per-byte q2 - (hset ? 0 : 4) == (q2 - 4) + 4*hset
        Q = __vadd4(__vsub4(q2, 0x04040404), hset << 2);
    }
};

}  // namespace gguf_int
}  // namespace layerstorm::compute
