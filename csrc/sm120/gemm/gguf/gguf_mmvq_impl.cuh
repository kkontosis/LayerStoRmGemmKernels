#pragma once

//==============================================================================
// Shared device-side mmvq accumulation + reduction (TD-GGUF-MMVQ-DECODE-
// EFFICIENCY redesign), used by both the single-weight kernel (gguf_mmvq.cu)
// and the device-routed grouped kernel (gguf_grouped_int.cu).
//
// Structure: NW warps per CTA cooperate on ONE output channel n — the
// flattened 4-value-group index space [0, K/4) of the weight row is strided
// across all NW*32 lanes (every lane does dp4a work every iteration, for
// Q8_0 as well as the k-quants), per-thread partials are reduced warp-locally
// (shfl) then across warps via shared memory in fixed order. Up to MT=8
// activation rows are accumulated in registers in a single pass over the
// weight blocks. Reduction order is fixed per shape -> bit-run-deterministic.
// No host sync / no attribute setting: CUDA-graph-capture safe.
//
// The multi-warp k-split + shared-memory cross-warp reduction + register row
// tile structure is adapted from llama.cpp ggml-cuda mul_mat_vec_q
// (MIT, "The ggml authors", commit 1191758c / tag b9780):
//   ggml/src/ggml-cuda/mmvq.cu (nwarps/blocks_per_iter thread mapping,
//   tmp_shared cross-warp reduce, rows_per_cuda_block register tiling).
// The per-type 4-value-group integer dot math (P::scales / P::group) is this
// project's own policy layer (gguf_int_policy.h).
//==============================================================================

#include "sm120/gemm/gguf/gguf_int_policy.h"
#include "formats/q8_1_format.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace layerstorm::compute {
namespace gguf_mmvq_impl {

constexpr int kWarps = 4;   // NW: warps per CTA (all on one output channel)
constexpr int kMTile = 8;   // MT: activation rows per register pass

// Strides the flattened 4-value-group index space [0, total_groups) of ONE
// weight row across all NW*32 threads and accumulates up to MT row partials
// in registers. `xrow0` points at the Q8_1 blocks of the first activation
// row; consecutive rows are x_bpr blocks apart. `mt` = live rows this pass
// (<= MT, uniform across the CTA).
template <class P, int NW, int MT>
__device__ __forceinline__ void accumulate(
        const uint8_t* __restrict__ Wrow,
        const layerstorm::formats::block_q8_1* __restrict__ xrow0,
        int x_bpr, int total_groups, int mt, int tid, float (&acc)[MT]) {
    constexpr int GPS = P::VALS / 4;   // 4-value groups per super-block
    for (int gg = tid; gg < total_groups; gg += NW * 32) {
        const int sb = gg / GPS;
        const int g  = gg - sb * GPS;
        const void* wblk = Wrow + static_cast<size_t>(sb) * P::BYTES;
        float dw, dmw; P::scales(wblk, dw, dmw);
        int Q; float A, B; P::group(wblk, g, dw, dmw, Q, A, B);
        const int b32 = gg >> 3;          // q8_1 block index (gg*4 / 32)
        const int ofs = (gg & 7) * 4;     // byte offset within qs[32]
        #pragma unroll
        for (int j = 0; j < MT; ++j) {
            if (j < mt) {
                const layerstorm::formats::block_q8_1* xb =
                    &xrow0[static_cast<size_t>(j) * x_bpr + b32];
                const float dx = __half2float(__low2half(xb->ds));
                const int qx = *reinterpret_cast<const int*>(&xb->qs[ofs]);
                const int dotQ  = __dp4a(Q, qx, 0);
                const int sumqx = __dp4a(qx, 0x01010101, 0);
                acc[j] += dx * (A * dotQ + B * sumqx);
            }
        }
    }
}

// Cross-CTA reduction of the per-thread partials: warp shfl reduce, one smem
// slot per (warp, row), then warp 0 folds the NW partials in fixed order and
// lane j (< mt) writes output row j at out0 + j*out_stride. Deterministic
// (fixed order per shape). Callers must __syncthreads() before reusing `red`
// for another pass.
template <int NW, int MT>
__device__ __forceinline__ void reduce_store(
        float (&acc)[MT], float (&red)[NW][MT], int mt, int lane, int warp,
        __nv_bfloat16* __restrict__ out0, int out_stride) {
    #pragma unroll
    for (int j = 0; j < MT; ++j) {
        if (j < mt) {
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                acc[j] += __shfl_xor_sync(0xffffffff, acc[j], o);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < MT; ++j) red[warp][j] = acc[j];
    }
    __syncthreads();
    if (warp == 0 && lane < mt) {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < NW; ++w) v += red[w][lane];
        out0[static_cast<size_t>(lane) * out_stride] = __float2bfloat16_rn(v);
    }
}

}  // namespace gguf_mmvq_impl
}  // namespace layerstorm::compute
