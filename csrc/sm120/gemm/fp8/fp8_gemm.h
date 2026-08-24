// FP8 E4M3 blockwise-scaled GEMM wrapper for SM120.
// Adapted from sglang fp8_blockwise_gemm_kernel.cu (Apache-2.0)
// and CUTLASS example 87a (BSD-3-Clause).
//
// Uses CUTLASS OpClassTensorOp with Sm120BlockwiseScaleConfig.
// Both activations and weights must be pre-quantized to FP8 E4M3.
// Per-block float32 scales: one scale per 128-element block.

#pragma once

#include "sm120/gemm/nvfp4/nvfp4_gemm.h"  // GemmOutputDtype

#include <cstddef>
#include <cstdint>

namespace layerstorm::compute {

/// Parameters for an FP8 blockwise-scaled GEMM launch.
///
/// Alignment requirements: K % 16 == 0, N % 16 == 0.
/// Scale tensors:
///   scale_A: float32 [M, ceil(K/128)] column-major (M-major)
///   scale_B: float32 [ceil(K/128), ceil(N/128)] column-major (K-major)
struct Fp8GemmParams {
    int M;                   ///< rows of A / rows of D (tokens)
    int N;                   ///< cols of B / cols of D (output dim)
    int K;                   ///< cols of A / cols of B (input dim)

    const void* A;           ///< [M, K] FP8 E4M3, row-major
    const void* B;           ///< [N, K] FP8 E4M3, col-major
    void*       D;           ///< [M, N] output (BF16/FP16), row-major

    const void* scale_A;     ///< float32 per-block activation scales
    const void* scale_B;     ///< float32 per-block weight scales

    GemmOutputDtype output_dtype;
};

/// Query workspace bytes required for an FP8 blockwise GEMM.
/// Thread-safe, no CUDA calls. Returns 0 if no workspace needed.
size_t query_fp8_gemm_workspace_size(int M, int N, int K,
                                      GemmOutputDtype output_dtype);

/// Launch an FP8 blockwise-scaled GEMM on SM120.
///
///   workspace: device memory >= query_fp8_gemm_workspace_size() bytes,
///              or nullptr if workspace size is 0
///   stream:    cudaStream_t (as void*), or nullptr for default stream
///
/// Throws std::runtime_error on CUTLASS failure or alignment violation.
void launch_fp8_gemm(const Fp8GemmParams& params,
                      void* workspace,
                      void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
