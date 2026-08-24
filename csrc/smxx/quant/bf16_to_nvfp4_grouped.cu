#ifndef BF16_TO_NVFP4_GROUPED_CU_INCLUDED
#define BF16_TO_NVFP4_GROUPED_CU_INCLUDED
#include "smxx/quant/bf16_to_nvfp4_grouped.h"
// Shared internal-linkage device helpers + constants (also used by the fused
// silu_mul_to_nvfp4_grouped kernel; see the header for why it is a .cuh and
// not this .cu — separate-TU ODR safety).
#include "smxx/quant/bf16_to_nvfp4_grouped_detail.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cfloat>

namespace layerstorm::compute {

// ── Main kernel ──────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(256)
bf16_to_nvfp4_grouped_kernel(
        const __nv_bfloat16* __restrict__ input,
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

    // Find which expert this row belongs to.
    int expert_id = find_expert(expert_offsets, num_experts, row);
    int expert_start = expert_offsets[expert_id];
    int M_e = expert_offsets[expert_id + 1] - expert_start;
    int local_row = row - expert_start;

    int padded_rows_e = ((M_e + 127) / 128) * 128;
    int padded_groups = ((groups_per_row + 3) / 4) * 4;

    int sf_base = sf_offsets[expert_id] * groups_per_row;

    // Per-expert calibrated activation global scale (TRT-LLM input_scale).
    // <= 0 guards missing experts whose B_ptrs point at a zero buffer: their
    // gathered input_scale reads 0.0 and must not produce inf/NaN here.
    float is = 1.0f;
    if (input_scales) {
        float v = input_scales[expert_id];
        if (v > 0.0f) is = v;
    }

    const __nv_bfloat16* row_in = input + row * K;
    uint8_t* row_packed = output_packed + row * (K / 2);

    for (int g = tid; g < groups_per_row; g += num_threads) {
        int base = g * GROUPED_GROUP_SIZE;

        float vals[GROUPED_GROUP_SIZE];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < GROUPED_GROUP_SIZE; i++) {
            vals[i] = __bfloat162float(row_in[base + i]);
            amax = fmaxf(amax, fabsf(vals[i]));
        }

        // With a calibrated input_scale, the stored group scale is amax/(6*is):
        // for low-magnitude tensors (is << 1) this lifts the scale out of the
        // UE4M3 underflow region, so tiny groups quantize finely instead of
        // clamping. The GEMM alpha must multiply the same is back.
        float scale = amax / (GROUPED_FP4_MAX_ABS * is);
        // Clamp to the UE4M3 minimum positive subnormal (2^-9 ≈ 1.95e-3):
        // anything smaller rounds to 0 in grouped_float_to_ue4m3 and the
        // whole group dequants to ZERO downstream. Low-magnitude activations
        // (e.g. MLA kv_b_v latents, group amax ~1e-3) hit this constantly —
        // found via LayerStoRm3 TD-GOLDEN (o_proj output direction broken).
        // With the clamp, tiny groups quantize coarsely instead of vanishing.
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

void launch_bf16_to_nvfp4_grouped(const Bf16ToNvfp4GroupedParams& params,
                                   void* stream) {
    if (params.K % GROUPED_GROUP_SIZE != 0) return;
    if (params.total_tokens <= 0) return;

    int grid = params.total_tokens;
    bf16_to_nvfp4_grouped_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(params.input),
        static_cast<uint8_t*>(params.output_packed),
        static_cast<uint8_t*>(params.output_scales),
        static_cast<const int32_t*>(params.expert_offsets),
        static_cast<const int32_t*>(params.sf_offsets),
        static_cast<const float*>(params.input_scales),
        params.total_tokens, params.num_experts, params.K);
}

}  // namespace layerstorm::compute

#endif  // BF16_TO_NVFP4_GROUPED_CU_INCLUDED
