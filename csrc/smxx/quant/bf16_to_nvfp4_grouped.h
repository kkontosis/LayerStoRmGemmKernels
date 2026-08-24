#pragma once
// Grouped BF16 activation quantization to NVFP4 format.
//
// Per-expert quantization: binary searches expert_offsets to determine each
// row's expert, then writes Sm1xx interleaved scales at the corresponding
// sf_offsets position with per-expert padded_rows.
//
// Packed FP4 output is contiguous (indexed by token_offset * K/2, same as
// GEMM A_base). Scale output has 128-row-aligned per-expert sections
// (indexed by sf_offsets[e] * groups_per_row, matching NVFP4 grouped GEMM).
//
// All pointers are void* for inclusion from non-CUDA .cpp translation units.

#include <cstdint>

namespace layerstorm::compute {

struct Bf16ToNvfp4GroupedParams {
    const void* input;           ///< [total_tokens, K] BF16
    void* output_packed;          ///< [total_tokens, K/2] packed FP4 E2M1 pairs
    void* output_scales;          ///< Sm1xx interleaved UE4M3 activation scales
    const void* expert_offsets;   ///< [num_experts + 1] INT32 device cumulative tokens
    const void* sf_offsets;       ///< [num_experts + 1] INT32 device scale offsets
    int total_tokens;
    int num_experts;
    int K;                        ///< must be divisible by 16
    /// Optional [num_experts] FLOAT32 device per-tensor calibrated activation
    /// global scales (TRT-LLM input_scale). When set, group scales are derived
    /// as amax/(6*is) and values quantize as x/(scale_repr*is); the consuming
    /// GEMM's alpha must then include the SAME is (alpha = weight_scale_2*is).
    /// nullptr, or any entry <= 0, behaves as 1.0 (pure dynamic scaling).
    const void* input_scales = nullptr;
};

void launch_bf16_to_nvfp4_grouped(const Bf16ToNvfp4GroupedParams& params,
                                   void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
