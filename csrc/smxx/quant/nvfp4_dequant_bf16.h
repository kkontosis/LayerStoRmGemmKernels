#pragma once
// NVFP4 → BF16 weight dequantization.
//
// Dequantizes [N, K/2] packed FP4 E2M1 weights with per-group FP8 E4M3
// scales and a global F32 scalar to [N, K] BF16 row-major.
// Used by Engine::quantize_attention_weights() as the first step of the
// two-step NVFP4→BF16→FP8 online weight conversion for o_proj.
//
// Architecture-agnostic (no SM120-specific ops).

#include <cstddef>

namespace layerstorm::compute {

/// Parameters for NVFP4 → BF16 weight dequantization.
///
/// input:   [N, K/2] packed FP4 E2M1 pairs, row-major (K/2 stride-1)
/// scales:  [N, K/16] raw FP8 E4M3 group scales, row-major
/// output:  [N, K] BF16, row-major (K stride-1)
///
/// Dequant formula per element (n, k):
///   nibble = (k even) ? input[n*K/2 + k/2] & 0xF : input[n*K/2 + k/2] >> 4
///   value  = fp4_e2m1_to_float(nibble) * e4m3_to_float(scales[n*K/16 + k/16])
///            * weight_scale_2
struct Nvfp4DequantBf16Params {
    int N;                    ///< rows of weight matrix (output dim)
    int K;                    ///< cols of weight matrix (input dim, in elements)
    const void* input;        ///< [N, K/2] packed FP4 E2M1 pairs
    const void* scales;       ///< [N, K/16] FP8 E4M3 group scales
    float weight_scale_2;     ///< global dequant scalar
    void* output;             ///< [N, K] BF16, row-major
};

/// Launch NVFP4 → BF16 weight dequantization.
///
///   stream: cudaStream_t (as void*), or nullptr for default stream.
///
/// Throws std::runtime_error on invalid dimensions or null pointers.
void launch_nvfp4_dequant_bf16(const Nvfp4DequantBf16Params& params,
                                void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
