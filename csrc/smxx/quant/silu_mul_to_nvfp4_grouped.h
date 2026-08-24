#pragma once
// Fused SiLU-mul + grouped BF16→NVFP4 activation quantization.
//
// Reads the gate and up grouped-GEMM outputs separately, computes the SwiGLU
// activation r[t,i] = silu(gate[t,i]) * up[t,i] (silu(x) = x*sigmoid(x), fp32),
// then quantizes r to NVFP4 BYTE-IDENTICALLY to bf16_to_nvfp4_grouped — same
// per-16-group amax/scale math, same per-expert input_scales (TRT-LLM) scheme,
// same 2^-9 UE4M3 clamp, same FP4 E2M1 packing, same Sm1xx-interleaved scale
// write. The output is indistinguishable from the unfused 4-op sequence
// (interleave gate+up → SwiGLU → bf16_to_nvfp4_grouped), so the down GEMM
// contract is unchanged.
//
// All pointers are void* for inclusion from non-CUDA .cpp translation units.

#include <cstdint>

namespace layerstorm::compute {

struct SiluMulToNvfp4GroupedParams {
    const void* gate;             ///< [total_tokens, K] BF16 (gate GEMM output)
    const void* up;               ///< [total_tokens, K] BF16 (up GEMM output)
    void* output_packed;          ///< [total_tokens, K/2] packed FP4 E2M1 pairs
    void* output_scales;          ///< Sm1xx interleaved UE4M3 activation scales
    const void* expert_offsets;   ///< [num_experts + 1] INT32 device cumulative tokens
    const void* sf_offsets;       ///< [num_experts + 1] INT32 device scale offsets
    int total_tokens;
    int num_experts;
    int K;                        ///< intermediate size I; must be divisible by 16
    /// Optional [num_experts] FLOAT32 device per-tensor calibrated activation
    /// global scales (TRT-LLM input_scale). Identical semantics to
    /// bf16_to_nvfp4_grouped: group scale = amax/(6*is), values quantize as
    /// x/(scale_repr*is); consuming GEMM alpha must include the SAME is.
    /// nullptr, or any entry <= 0, behaves as 1.0 (pure dynamic scaling).
    const void* input_scales = nullptr;
};

void launch_silu_mul_to_nvfp4_grouped(const SiluMulToNvfp4GroupedParams& params,
                                       void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
