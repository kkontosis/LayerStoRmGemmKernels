#pragma once

//==============================================================================
// Device-side ROUTED grouped GGUF integer GEMM (mmvq + mmq) for MoE expert FFN.
//
// This is the CUDA-graph-capturable replacement for the host per-expert loop in
// gguf_grouped_gemm.cu (TD-GG5-GROUPED-HOST-SYNC). Instead of copying
// expert_offsets + B_ptrs to host and looping, the experts are routed ON DEVICE:
//
//   * Q8_1 activation quant runs ONCE over all total_tokens rows (no per-expert
//     loop), into a shared workspace.
//   * mmvq (decode, small per-expert M): grid is (N, num_experts); each CTA reads
//     its expert's [off, off+M_e) token slice from expert_offsets on device and
//     its weight from B_ptrs[e]. Empty experts (M_e == 0) early-return.
//   * mmq (prefill, large per-expert M): a tiny prep kernel reads expert_offsets
//     on device and builds a flattened tile -> (expert, m_start) map (the
//     mul_mat_id "ids" routing pattern); the grouped mmq kernel grid is
//     (n_tiles, T_max) and each CTA looks up its expert + row origin from the map
//     and its weight from B_ptrs[e]. Tiles past the live count are marked -1 and
//     early-return.
//
// The host makes ONE up-front mmvq-vs-mmq choice by the overall token count
// (total_tokens); there is NO per-expert loop and NO host<-device sync anywhere
// on the int path, so the whole path captures into a cudaGraph.
//
// Routing pattern adapted from llama.cpp / ggml mul_mat_id (MIT, "The ggml
// authors", commit 1191758c / tag b9780): ggml/src/ggml-cuda/{mmid.cu,mmq.cu,
// mmvq.cu,quantize.cu} — the per-row/per-tile expert "ids" map. The per-tile
// GGUF dot-product math is our own SM120 CuTe int8 mmq / dp4a mmvq kernels; only
// the on-device expert routing is adapted from ggml.
//==============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "formats/q8_1_format.h"                 // block_q8_1
#include "sm120/gemm/gguf/gguf_dequant_gemm.h"   // GgufType

namespace layerstorm::compute {

// Arguments for the grouped (device-routed) mmvq kernel. The activation has
// already been quantized to Q8_1 over all total_tokens rows (xq).
struct GgufGroupedMmvqArgs {
    int N;                                     // output channels
    int K;                                     // input cols
    const layerstorm::formats::block_q8_1* xq; // [total_tokens, K/32] quantized activations
    const int32_t*  expert_offsets;            // [num_experts + 1] cumulative token counts
    const void* const* B_ptrs;                 // [num_experts] packed GGUF weight blocks
    __nv_bfloat16*  D_base;                     // [total_tokens, N] output, row-major
};

// Arguments for the grouped (device-routed) mmq kernels. Adds the tile->expert
// routing map built by the prep kernel.
struct GgufGroupedMmqArgs {
    int N;                                     // output channels
    int K;                                     // input cols
    const layerstorm::formats::block_q8_1* xq; // [total_tokens, K/32] quantized activations
    const int32_t*  expert_offsets;            // [num_experts + 1] cumulative token counts
    const void* const* B_ptrs;                 // [num_experts] packed GGUF weight blocks
    __nv_bfloat16*  D_base;                     // [total_tokens, N] output, row-major
    const int*      tile_expert;               // [T_max] expert per flattened m-tile (-1 = dead)
    const int*      tile_mstart;               // [T_max] global row origin per m-tile
};

// Per-type grouped mmq launchers (defined in mmq_mma/instances/mmq_*.cu). They
// pick the SmallMTileCfg (64x128) iff use_small, matching the TILE_M the prep
// kernel used to build the tile map; T_max is the map length (grid.y).
void gguf_mmq_mma_grouped_launch_q4k(const GgufGroupedMmqArgs& args, bool use_small,
                                     int T_max, cudaStream_t stream);
void gguf_mmq_mma_grouped_launch_q5k(const GgufGroupedMmqArgs& args, bool use_small,
                                     int T_max, cudaStream_t stream);
void gguf_mmq_mma_grouped_launch_q8_0(const GgufGroupedMmqArgs& args, bool use_small,
                                      int T_max, cudaStream_t stream);
void gguf_mmq_mma_grouped_launch_mxfp4(const GgufGroupedMmqArgs& args, bool use_small,
                                      int T_max, cudaStream_t stream);
void gguf_mmq_mma_grouped_launch_q6k(const GgufGroupedMmqArgs& args, bool use_small,
                                     int T_max, cudaStream_t stream);
void gguf_mmq_mma_grouped_launch_q2k(const GgufGroupedMmqArgs& args, bool use_small,
                                     int T_max, cudaStream_t stream);
void gguf_mmq_mma_grouped_launch_q3k(const GgufGroupedMmqArgs& args, bool use_small,
                                     int T_max, cudaStream_t stream);

// Tile-map TILE_M for the chosen config. The 32-aligned and 16-subblock kernels
// share the same TILE_M (Cfg::kTILE_M), so the map is config-, not type-,
// specific. 128 = DefaultTileCfg, 64 = SmallMTileCfg.
inline int gguf_grouped_mmq_tile_m(bool use_small) { return use_small ? 64 : 128; }

// Worst-case flattened m-tile count for a given tile height: each expert
// contributes ceil(M_e / TILE_M) <= M_e/TILE_M + 1 tiles, so the sum is bounded
// by ceil(total_tokens / TILE_M) + num_experts.
inline int gguf_grouped_mmq_tmax(int total_tokens, int num_experts, int tile_m) {
    if (tile_m <= 0) tile_m = 64;
    return (total_tokens + tile_m - 1) / tile_m + (num_experts > 0 ? num_experts : 0);
}

// Force selector for the grouped int path (testing / benchmarking).
enum class GgufGroupedIntForce {
    Auto = 0,   // host picks mmvq vs mmq by overall M
    Mmvq = 1,   // force decode mmvq
    Mmq  = 2,   // force prefill mmq
};

// Device-routed grouped GGUF integer GEMM. Quantizes A_base -> Q8_1 once, then
// routes experts on device (mmvq or mmq). NO host<-device sync; CUDA-graph
// capturable. `workspace` holds the Q8_1 scratch + (for mmq) the tile map; size
// it with gguf_grouped_gemm_workspace_bytes(total_tokens, K, num_experts).
//
//   type           : weight quant family (shared across experts)
//   num_experts    : active expert groups
//   N, K           : GEMM dims (shared across experts)
//   total_tokens   : sum of per-expert token counts (== expert_offsets[num_experts]);
//                    a host-known capture-time constant (the permuted row count).
//   A_base/D_base  : [total_tokens, K] / [total_tokens, N] BF16 (row-major)
//   expert_offsets : [num_experts + 1] device cumulative token counts
//   B_ptrs         : [num_experts] device array of packed GGUF weight pointers
void launch_gguf_grouped_int(
    GgufType type, int num_experts, int N, int K, int total_tokens,
    const __nv_bfloat16* A_base, __nv_bfloat16* D_base,
    const int32_t* expert_offsets, const void* const* B_ptrs,
    void* workspace, size_t workspace_bytes,
    GgufGroupedIntForce force, cudaStream_t stream);

}  // namespace layerstorm::compute
