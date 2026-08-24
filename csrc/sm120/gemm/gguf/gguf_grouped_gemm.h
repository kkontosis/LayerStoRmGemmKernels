#pragma once

//==============================================================================
// GGUF grouped GEMM for MoE expert FFN on SM120.
//
// One projection (gate, OR up, OR down) over a batch of routed experts. Mirrors
// the FP8 grouped path (fp8_grouped_gemm.cu): a per-expert dispatch loop that
// copies expert_offsets (+ the scattered per-expert weight pointers) to host
// once, then slices the contiguous permuted activation / output by token offset
// and dispatches the SINGLE-expert GGUF launcher per expert.
//
// All experts in one call share ONE GgufType (GG-6: the stacked ffn_*_exps
// tensor carries a single ggml_type per projection) and ONE strategy.
//
// Strategy / per-expert M routing (mirrors the non-grouped device path):
//   Dequant            -> launch_gguf_dequant_gemm  (no q8_1 workspace)
//   Int  & M_e <= 8    -> launch_gguf_mmvq          (decode GEMV)
//   Int  & M_e  > 8    -> launch_gguf_mmq_mma       (prefill int8 tensor-core)
//
// Layout (all device pointers):
//   A_base : [total_tokens, K] BF16 permuted activations (row-major)
//   D_base : [total_tokens, N] BF16 output (row-major)
//   B_ptrs[e] : expert e's packed GGUF weight, [N, (K/QK)*block_bytes] row-major
//   expert_offsets : [num_experts + 1] cumulative permuted-token counts
//==============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>

#include "sm120/gemm/gguf/gguf_dequant_gemm.h"  // GgufType, gguf_block_bytes/values
#include "sm120/gemm/gguf/gguf_grouped_int.h"   // GgufGroupedIntForce, device router

namespace layerstorm::compute {

/// Compute strategy for the GGUF grouped GEMM (mirrors GgufStrategy in the engine).
enum class GgufGroupedStrategy {
    Dequant = 0,  ///< dequantize weight to BF16 then GEMM (lossless activation)
    Int     = 1,  ///< quantize activation to Q8_1; integer dp4a / int8 tensor-core
};

struct GgufGroupedGemmKernelParams {
    GgufType type;                  ///< weight quant family (shared across experts)
    GgufGroupedStrategy strategy;   ///< Dequant or Int
    int num_experts;                ///< active expert groups this call
    int N;                          ///< output rows (gate/up: I; down: H)
    int K;                          ///< input cols  (gate/up: H; down: I)

    const __nv_bfloat16* A_base;    ///< [total_tokens, K] BF16 activations
    __nv_bfloat16*       D_base;    ///< [total_tokens, N] BF16 output

    const int32_t* expert_offsets;  ///< [num_experts + 1] cumulative token counts (device)
    const void**   B_ptrs;          ///< [num_experts] packed GGUF weight blocks (device array)
};

/// Workspace bytes for the GGUF grouped GEMM (Int strategy).
///
/// The device-routed Int path quantizes ALL activations to Q8_1 ONCE (a
/// total_tokens*(K/32)-block scratch) and, for the mmq sub-path, appends a
/// flattened tile->expert routing map sized for the worst-case 64-tall tile
/// (ceil(total_tokens/64)+num_experts tiles, 2 ints each). The Dequant strategy
/// needs no workspace (pass 0 bytes / null).
size_t gguf_grouped_gemm_workspace_bytes(int total_tokens, int K, int num_experts);

/// Launch the GGUF grouped GEMM. Throws std::runtime_error on a bad config.
///
/// Int strategy: device-routed (mmvq/mmq), NO host<-device sync — CUDA-graph
/// capturable (TD-GG5). Dequant strategy: the existing host per-expert dispatch
/// loop (one D2H sync; the correctness reference). `total_tokens` is the permuted
/// row count (== expert_offsets[num_experts]); a host-known capture-time
/// constant. `workspace` may be null when strategy == Dequant.
void launch_gguf_grouped_gemm(
    const GgufGroupedGemmKernelParams& params, int total_tokens,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream);

/// Force a specific Int sub-path (mmvq or mmq) for the device-routed grouped int
/// GEMM. Testing / benchmarking hook; the public launcher uses Auto.
void launch_gguf_grouped_gemm_int_forced(
    const GgufGroupedGemmKernelParams& params, int total_tokens,
    void* workspace, size_t workspace_bytes,
    GgufGroupedIntForce force, cudaStream_t stream);

/// Original host per-expert dispatch loop (copies expert_offsets+B_ptrs to host,
/// one D2H sync, loops the single-expert launchers). Retained as the golden
/// reference for both strategies and as the Dequant implementation. NOT
/// CUDA-graph capturable (host sync). `total_tokens` only used to validate the
/// workspace; per-expert M is read on host.
void launch_gguf_grouped_gemm_hostloop(
    const GgufGroupedGemmKernelParams& params,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream);

}  // namespace layerstorm::compute
