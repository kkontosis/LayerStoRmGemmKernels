// Dynamic FP8 E4M3 quantization kernel: BF16 → FP8 with per-block float32 scales.
//
// Adapted from vLLM csrc/quantization/fp8/scaled_fp8_quant_kernel.cu (Apache-2.0).
// Block size = 128 elements (matching Fp8GemmParams scale_A layout).
// Architecture-agnostic (no SM120-specific ops).

#pragma once

#include <cstddef>

namespace layerstorm::compute {

/// Parameters for dynamic FP8 quantization of BF16 activations.
///
/// Quantizes [num_tokens, hidden_size] BF16 → FP8 E4M3 with per-block
/// float32 scales.  Block size = 128 elements along the hidden dimension.
///
/// Scale layout — TWO modes, equal only at M == 1:
///   m_major_scales = false (default): [M, ceil(K/128)] ROW-major,
///     scales[m*nb + k_blk]. Historical layout; standalone consumers.
///   m_major_scales = true: M-MAJOR (column-major), scales[m + k_blk*M] —
///     the Fp8GemmParams::scale_A contract (CUTLASS SFA). REQUIRED when the
///     scales feed launch_fp8_gemm with M > 1; the row-major layout reads
///     scrambled per-token scales there (found via LayerStoRm3 prefill).
struct DynamicFp8QuantParams {
    int num_tokens;     ///< M: number of tokens (rows)
    int hidden_size;    ///< K: elements per token (columns)
    const void* input;  ///< [M, K] BF16, row-major
    void* output;       ///< [M, K] FP8 E4M3, row-major
    void* scales;       ///< [M, ceil(K/128)] float32, per-block scales
    bool m_major_scales = false;  ///< see scale-layout modes above
};

/// Launch dynamic per-block FP8 quantization.
///
///   stream: cudaStream_t (as void*), or nullptr for default stream.
///
/// Throws std::runtime_error on invalid dimensions.
void launch_dynamic_fp8_quant(const DynamicFp8QuantParams& params,
                               void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
