#pragma once
// FP8 E4M3 blockwise-scaled GEMV (M == 1) for SM120.
//
// Hand-written CUDA-core dequant-GEMV for the decode path. At M=1 the CUTLASS
// warp-specialized pingpong GEMM under-occupies (~12 CTAs) and is bandwidth-
// starved (~370 GB/s of the 5090's 1.79 TB/s). This kernel bypasses CUTLASS:
// it parallelizes over N (one warp per output row), streams the full FP8
// weight coalesced, dequantizes per 128-wide K-block with scale_B, multiplies
// by the (shared-memory-staged) dequantized activation, accumulates in float,
// and writes BF16/FP16.
//
// Numerics match launch_fp8_gemm exactly (alpha == 1.0, per-128-block scales):
//   D[0,n] = sum_k ( A_fp8[0,k] * scale_A[k_blk] ) * ( B_fp8[n,k] * scale_B[k_blk,n_blk] )
//
// Scale layouts (same contract as Fp8GemmParams):
//   scale_A: M-major [ceil(K/128), M]; at M=1 it is ceil(K/128) values,
//            one per K-block: scale_A[k_blk].
//   scale_B: K-major [ceil(K/128), ceil(N/128)]: scale_B[k_blk + n_blk*ceil(K/128)].
//   A:  [1, K] FP8 E4M3, row-major.
//   B:  [N, K] FP8 E4M3, row-major (N rows of K contiguous).
//   D:  [1, N] output (BF16/FP16), row-major.

#include "sm120/gemm/fp8/fp8_gemm.h"  // Fp8GemmParams, GemmOutputDtype

#include <cuda_runtime.h>

namespace layerstorm::compute {

/// Launch the M==1 FP8 blockwise-scaled GEMV on SM120.
/// Requires params.M == 1, K % 16 == 0. No workspace needed.
/// Throws std::runtime_error on unsupported output dtype.
void launch_fp8_blockscaled_gemv(const Fp8GemmParams& params,
                                 void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
