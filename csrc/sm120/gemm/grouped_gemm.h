// Grouped GEMM for MoE expert FFN execution on SM120.
// Batches multiple expert GEMMs into a single CUTLASS grouped kernel launch.
//
// Two quantization variants (separate .cu files):
//   NVFP4: OpClassBlockScaledTensorOp, UE4M3 scales, FP4 E2M1 data
//   FP8:   OpClassTensorOp, Sm120BlockwiseScaleConfig, FP8 E4M3 data
//
// Adapted from sglang jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh (Apache-2.0)
// and CUTLASS examples 79a/87a (BSD-3-Clause).

#pragma once

#include "sm120/gemm/nvfp4/nvfp4_gemm.h"  // GemmOutputDtype

#include <cstddef>
#include <cstdint>

namespace layerstorm::compute {

// ── NVFP4 Grouped GEMM ────────────────────────────────────────────────────

/// Parameters for NVFP4 grouped GEMM (gate_up or down projection).
///
/// All pointers are device memory. expert_offsets and problem_sizes are
/// computed by the caller (typically from moe_permute output).
///
/// Memory layout:
///   A_base: [total_tokens, K/2] contiguous permuted FP4 activations
///   B_base: [num_all_experts, N, K/2] expert weights (col-major per expert)
///   D_base: [total_tokens, N] contiguous output
///   scale_A_base: [sum_of_sf_sizes, K/group_size] padded+swizzled UE4M3
///   scale_B_base: [num_all_experts, N, K/group_size] UE4M3
///   alphas: [num_all_experts] float32 per-expert output scale
struct Nvfp4GroupedGemmParams {
    int num_experts;          ///< number of active expert groups this call
    int N;                    ///< output dimension (same for all experts)
    int K;                    ///< input dimension (in elements, not bytes)

    const void* A_base;       ///< [total_tokens, K/2] FP4 activations
    const void* B_base;       ///< [num_all_experts, N, K/2] FP4 weights
    void*       D_base;       ///< [total_tokens, N] output

    const void* scale_A_base; ///< UE4M3 activation scales (padded per sf_offsets)
    const void* scale_B_base; ///< UE4M3 weight scales
    const float* alphas;      ///< [num_all_experts] per-expert output scale

    const int32_t* expert_offsets; ///< [num_experts + 1] cumulative token counts
    const int32_t* sf_offsets;     ///< [num_experts + 1] scale factor offsets (128-byte TMA aligned)
    const int32_t* problem_sizes;  ///< [num_experts, 3] packed {M_i, N, K} per expert

    GemmOutputDtype output_dtype;

    /// Optional per-expert pointer arrays (for scattered ExpertCache slots).
    /// When non-null, used instead of computing B/scale_B from base + stride.
    const void** B_ptrs = nullptr;       ///< [num_experts] pre-computed per-expert B pointers
    const void** scale_B_ptrs = nullptr; ///< [num_experts] pre-computed per-expert scale_B pointers
};

/// Query workspace bytes for NVFP4 grouped GEMM.
/// Workspace holds per-expert pointer arrays, stride arrays, layout arrays,
/// and CUTLASS internal workspace. Allocate once for max_experts.
size_t query_nvfp4_grouped_gemm_workspace_size(
    int max_experts, int N, int K,
    GemmOutputDtype output_dtype);

/// Launch NVFP4 grouped GEMM on SM120.
/// Uses CUTLASS 3.x GroupProblemShape + KernelPtrArrayTmaWarpSpecializedPingpong.
/// Throws std::runtime_error on CUTLASS failure.
void launch_nvfp4_grouped_gemm(
    const Nvfp4GroupedGemmParams& params,
    void* workspace, size_t workspace_bytes,
    void* stream /*cudaStream_t*/);

// ── FP8 Grouped GEMM ──────────────────────────────────────────────────────

/// Parameters for FP8 grouped GEMM.
///
/// Memory layout:
///   A_base: [total_tokens, K] FP8 E4M3 activations
///   B_base: [num_all_experts, N, K] FP8 E4M3 weights (col-major per expert)
///   D_base: [total_tokens, N] output
///   scale_A_base: float32 [total_tokens, ceil(K/128)] per-block activation scales
///   scale_B_base: float32 [num_all_experts, ceil(K/128), ceil(N/128)] weight scales
struct Fp8GroupedGemmParams {
    int num_experts;
    int N;
    int K;

    const void* A_base;
    const void* B_base;
    void*       D_base;

    const void* scale_A_base;
    const void* scale_B_base;

    const int32_t* expert_offsets;
    const int32_t* problem_sizes;

    GemmOutputDtype output_dtype;

    /// Optional per-expert pointer arrays (for scattered ExpertCache slots).
    /// When non-null, used instead of computing B/scale_B from base + stride.
    const void** B_ptrs = nullptr;       ///< [num_experts] pre-computed per-expert B pointers
    const void** scale_B_ptrs = nullptr; ///< [num_experts] pre-computed per-expert scale_B pointers
};

/// Query workspace bytes for FP8 grouped GEMM.
size_t query_fp8_grouped_gemm_workspace_size(
    int max_experts, int N, int K,
    GemmOutputDtype output_dtype);

/// Launch FP8 grouped GEMM on SM120.
/// Uses CUTLASS 3.x GroupProblemShape with blockwise scales.
/// Falls back to per-expert dispatch loop if grouped mode unavailable.
/// Throws std::runtime_error on CUTLASS failure.
void launch_fp8_grouped_gemm(
    const Fp8GroupedGemmParams& params,
    void* workspace, size_t workspace_bytes,
    void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
