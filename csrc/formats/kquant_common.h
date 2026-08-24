#pragma once

//==============================================================================
// GGUF k-quant shared constants and helpers
//
// Definitions common to the k-quant super-block formats (Q2_K..Q6_K). Factored
// out so multiple format headers can be included in the same translation unit
// (e.g. the gguf_mul_mat dispatcher) without redefining shared symbols.
//
// Ported from llama.cpp ggml (MIT, "The ggml authors"):
//   - ggml/src/ggml-common.h  (QK_K, K_SCALE_SIZE)
//   - ggml/src/ggml-cuda/convert.cu  (get_scale_min_k4)
// Source revision: commit 1191758c (tag b9780).
//==============================================================================

#include <cuda_runtime.h>
#include <cstdint>

namespace layerstorm::formats {

// values per super-block (all k-quants pack 256 weights per super-block)
static constexpr int QK_K = 256;

// bytes of packed 6-bit (scale,min) data in Q4_K/Q5_K super-blocks
static constexpr int K_SCALE_SIZE = 12;

// ---------------------------------------------------------------------------
// get_scale_min_k4 — extract one 6-bit (scale, min) pair for sub-block j.
//
// Used by Q4_K and Q5_K (8 sub-blocks of 32 values, scales/mins packed 6-bit
// into 12 bytes). j in [0,7]; returns d (scale) and m (min), each 0-63.
//
// Port of ggml get_scale_min_k4 (ggml-cuda/convert.cu, "static inline __device__").
// ---------------------------------------------------------------------------

__device__ __forceinline__
void get_scale_min_k4(int j, const uint8_t* __restrict__ q,
                      uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

}  // namespace layerstorm::formats
