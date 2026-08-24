// NVFP4 (FP4 E2M1) GEMM wrapper for SM120.
// Adapted from vLLM nvfp4_scaled_mm_sm120_kernels.cu (Apache-2.0)
// and CUTLASS example 79a (BSD-3-Clause).
//
// Uses CUTLASS OpClassBlockScaledTensorOp with UE4M3 scale factors.
// Weights must be pre-packed as nv_float4_t<float_e2m1_t> pairs.
// Activations must be pre-quantized to FP4 before calling.

#pragma once

#include <cstddef>
#include <cstdint>

namespace layerstorm::compute {

/// Output data type for quantized GEMMs.
enum class GemmOutputDtype {
    kBFloat16,
    kFloat16,
};

/// Parameters for an NVFP4 GEMM launch.
///
/// Alignment requirements: K % 32 == 0, N % 32 == 0.
/// Scale tensors must be padded and swizzled per CUTLASS Sm1xxBlkScaledConfig:
///   scale_A: [round_up(M, 128), round_up(K/16, 4)] UE4M3
///   scale_B: [round_up(N, 128), round_up(K/16, 4)] UE4M3
struct Nvfp4GemmParams {
    int M;                   ///< rows of A / rows of D (tokens)
    int N;                   ///< cols of B / cols of D (output dim)
    int K;                   ///< cols of A / cols of B (input dim, in elements not bytes)

    const void* A;           ///< [M, K/2] packed FP4 E2M1 pairs, row-major
    const void* B;           ///< [N, K/2] packed FP4 E2M1 pairs, col-major
    void*       D;           ///< [M, N] output, row-major

    const void* scale_A;     ///< UE4M3 activation scales (padded+swizzled)
    const void* scale_B;     ///< UE4M3 weight scales (padded+swizzled)

    float alpha;             ///< global output scale factor

    GemmOutputDtype output_dtype;
};

/// Query workspace bytes required for an NVFP4 GEMM.
/// Thread-safe, no CUDA calls. Returns 0 if no workspace needed.
size_t query_nvfp4_gemm_workspace_size(int M, int N, int K,
                                        GemmOutputDtype output_dtype);

/// Launch an NVFP4 GEMM on SM120.
///
///   workspace: device memory >= query_nvfp4_gemm_workspace_size() bytes,
///              or nullptr if workspace size is 0
///   stream:    cudaStream_t (as void*), or nullptr for default stream
///
/// Throws std::runtime_error on CUTLASS failure or alignment violation.
void launch_nvfp4_gemm(const Nvfp4GemmParams& params,
                        void* workspace,
                        void* stream /*cudaStream_t*/);

/// Query the Sm1xx-interleaved scale buffer size (in bytes) for SFA or SFB.
/// Used by the BF16→NVFP4 quantization pipeline to allocate scale buffers.
size_t query_sf_buffer_size_a(int M, int N, int K);
size_t query_sf_buffer_size_b(int M, int N, int K);

}  // namespace layerstorm::compute
