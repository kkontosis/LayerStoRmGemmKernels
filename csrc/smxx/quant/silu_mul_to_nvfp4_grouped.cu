#ifndef SILU_MUL_TO_NVFP4_GROUPED_CU_INCLUDED
#define SILU_MUL_TO_NVFP4_GROUPED_CU_INCLUDED
#include "smxx/quant/silu_mul_to_nvfp4_grouped.h"

// Reuse the EXACT quant device helpers from bf16_to_nvfp4_grouped so the FP4
// packing, UE4M3 scale conversion, Sm1xx scale-offset layout, the input_scale
// math, and the 2^-9 clamp stay bit-identical to the unfused path. Only the
// input read differs (two BF16 tensors + SwiGLU instead of one).
// NOTE: include the .cuh (internal-linkage helpers only), NOT the .cu — the
// .cu's __global__ kernel + launch_* have external linkage and would duplicate
// at link time in LayerStoRm3's separate-TU CMake build (both .cu are listed).
#include "smxx/quant/bf16_to_nvfp4_grouped_detail.cuh"

#include <cuda_bf16.h>
#include <cfloat>

namespace layerstorm::compute {

// SwiGLU on a single element, matching LayerStoRmExpertKernels fused_swiglu.cu:
//   result = silu(gate) * up, silu(x) = x * sigmoid(x), computed in fp32.
// The reference SwiGLU uses silu(x) = x * (0.5*tanh(0.5*x) + 0.5) on its
// vectorized path; 0.5*tanh(0.5*x)+0.5 == sigmoid(x) exactly as a real
// identity, and the fused kernel here uses the same tanh form so the fp32
// activation matches the unfused SwiGLU kernel bit-for-bit before quant.
__device__ __forceinline__
float silu_mul_f32(float gate, float up) {
    float silu_g = gate * (0.5f * tanhf(0.5f * gate) + 0.5f);
    return silu_g * up;
}

// ── Main kernel ──────────────────────────────────────────────────────────────
// One block per token row (identical grid to bf16_to_nvfp4_grouped_kernel).
// Reads gate[row,:] and up[row,:] (both [total_tokens, K] BF16), forms the
// SwiGLU activation r per element, then quantizes r EXACTLY as
// bf16_to_nvfp4_grouped_kernel quantizes its `input`.

__global__ void __launch_bounds__(256)
silu_mul_to_nvfp4_grouped_kernel(
        const __nv_bfloat16* __restrict__ gate,
        const __nv_bfloat16* __restrict__ up,
        uint8_t* __restrict__ output_packed,
        uint8_t* __restrict__ output_scales,
        const int32_t* __restrict__ expert_offsets,
        const int32_t* __restrict__ sf_offsets,
        const float* __restrict__ input_scales,
        int total_tokens, int num_experts, int K) {

    const int row = blockIdx.x;
    if (row >= total_tokens) return;

    const int groups_per_row = K / GROUPED_GROUP_SIZE;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Find which expert this row belongs to (same binary search as unfused).
    int expert_id = find_expert(expert_offsets, num_experts, row);
    int expert_start = expert_offsets[expert_id];
    int M_e = expert_offsets[expert_id + 1] - expert_start;
    int local_row = row - expert_start;

    int padded_rows_e = ((M_e + 127) / 128) * 128;
    int padded_groups = ((groups_per_row + 3) / 4) * 4;

    int sf_base = sf_offsets[expert_id] * groups_per_row;

    // Per-expert calibrated activation global scale (TRT-LLM input_scale).
    // <= 0 guards missing experts (gathered input_scale 0.0) → behaves as 1.0.
    float is = 1.0f;
    if (input_scales) {
        float v = input_scales[expert_id];
        if (v > 0.0f) is = v;
    }

    const __nv_bfloat16* gate_in = gate + row * K;
    const __nv_bfloat16* up_in   = up + row * K;
    uint8_t* row_packed = output_packed + row * (K / 2);

    for (int g = tid; g < groups_per_row; g += num_threads) {
        int base = g * GROUPED_GROUP_SIZE;

        // SwiGLU per element into fp32, then amax — exactly the values the
        // unfused path's bf16_to_nvfp4_grouped would see in `input` (the
        // SwiGLU kernel rounds silu(gate)*up to BF16 and stores it; we
        // reproduce that BF16 round-trip so the quant input is bit-identical).
        float vals[GROUPED_GROUP_SIZE];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < GROUPED_GROUP_SIZE; i++) {
            float gf = __bfloat162float(gate_in[base + i]);
            float uf = __bfloat162float(up_in[base + i]);
            float r = silu_mul_f32(gf, uf);
            // The unfused SwiGLU kernel writes BF16; the quantizer then reads
            // that BF16 back to fp32. Round-trip through BF16 here so the
            // fused output is byte-identical to the unfused sequence.
            vals[i] = __bfloat162float(__float2bfloat16(r));
            amax = fmaxf(amax, fabsf(vals[i]));
        }

        float scale = amax / (GROUPED_FP4_MAX_ABS * is);
        scale = fmaxf(scale, 0.001953125f);  // 2^-9

        uint8_t scale_bits = grouped_float_to_ue4m3(scale);
        float scale_repr = grouped_ue4m3_to_float(scale_bits);
        scale_repr = fmaxf(scale_repr, 1e-12f);

        int sf_offset = grouped_sm1xx_sf_offset(local_row, g, padded_rows_e, padded_groups);
        output_scales[sf_base + sf_offset] = scale_bits;

        float inv_scale = 1.0f / (scale_repr * is);
        #pragma unroll
        for (int i = 0; i < GROUPED_GROUP_SIZE; i += 2) {
            float v0 = vals[i] * inv_scale;
            float v1 = vals[i + 1] * inv_scale;
            uint8_t idx0 = grouped_quantize_to_fp4(v0);
            uint8_t idx1 = grouped_quantize_to_fp4(v1);
            row_packed[(base + i) / 2] = (idx0 & 0x0F) | ((idx1 & 0x0F) << 4);
        }
    }
}

void launch_silu_mul_to_nvfp4_grouped(const SiluMulToNvfp4GroupedParams& params,
                                       void* stream) {
    if (params.K % GROUPED_GROUP_SIZE != 0) return;
    if (params.total_tokens <= 0) return;

    int grid = params.total_tokens;
    silu_mul_to_nvfp4_grouped_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(params.gate),
        static_cast<const __nv_bfloat16*>(params.up),
        static_cast<uint8_t*>(params.output_packed),
        static_cast<uint8_t*>(params.output_scales),
        static_cast<const int32_t*>(params.expert_offsets),
        static_cast<const int32_t*>(params.sf_offsets),
        static_cast<const float*>(params.input_scales),
        params.total_tokens, params.num_experts, params.K);
}

}  // namespace layerstorm::compute

#endif  // SILU_MUL_TO_NVFP4_GROUPED_CU_INCLUDED
