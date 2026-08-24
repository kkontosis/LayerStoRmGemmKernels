#pragma once

//==============================================================================
// Q8_1 activation block — the quantized-activation format consumed by the
// integer mat-vec (mmvq) and mat-mat (mmq) GGUF paths, mirroring llama.cpp.
//
// Per 32-element block: ds.x = d (scale = amax/127), ds.y = s (sum of the
// original activations, Σx), qs[i] = round(x[i]/d) as signed int8.
//
// Binary-compatible with ggml block_q8_1 (ggml-common.h).
//
// Ported from llama.cpp ggml (MIT, "The ggml authors"):
//   - ggml/src/ggml-common.h         (block_q8_1)
//   - ggml/src/ggml-cuda/quantize.cu (quantize_q8_1: d=amax/127, ds=(d, Σx))
// Source revision: commit 1191758c (tag b9780).
//==============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace layerstorm::formats {

static constexpr int QK8_1 = 32;             // activations per block
static constexpr int BLOCK_Q8_1_SIZE = 36;   // bytes per block

struct block_q8_1 {
    half2   ds;          // ds.x = d (scale), ds.y = s (Σ of original values)
    int8_t  qs[QK8_1];   // 32 signed int8 quants
};

static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 bytes");

}  // namespace layerstorm::formats
