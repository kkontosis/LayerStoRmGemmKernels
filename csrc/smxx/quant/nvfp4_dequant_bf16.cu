// NVFP4 → BF16 weight dequantization kernel.
//
// One CUDA block per row (N dimension).  Each thread processes multiple
// elements (K dimension) in a strided loop.

#ifndef NVFP4_DEQUANT_BF16_CU_INCLUDED
#define NVFP4_DEQUANT_BF16_CU_INCLUDED
#include "smxx/quant/nvfp4_dequant_bf16.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace layerstorm::compute {

namespace {

// FP4 E2M1 dequantization LUT (16 entries: 8 positive + 8 negative).
// idx 0-7:  +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0
// idx 8-15: -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
__device__ __constant__ float kFp4E2M1Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

// Convert FP8 E4M3 byte to float32 (reinterpret as __nv_fp8_e4m3).
__device__ __forceinline__
float e4m3_to_float(uint8_t bits) {
    __nv_fp8_e4m3 fp8 = *reinterpret_cast<const __nv_fp8_e4m3*>(&bits);
    return float(fp8);
}

static constexpr int kThreadsPerBlock = 256;
static constexpr int kGroupSize = 16;  // one FP8 scale per 16 FP4 elements

__global__ void __launch_bounds__(kThreadsPerBlock)
nvfp4_dequant_bf16_kernel(
    __nv_bfloat16* __restrict__ output,         // [N, K] row-major
    const uint8_t* __restrict__ input_packed,    // [N, K/2] packed FP4 pairs
    const uint8_t* __restrict__ input_scales,    // [N, K/16] FP8 E4M3
    float weight_scale_2,
    int N, int K) {

    const int row = blockIdx.x;
    if (row >= N) return;

    const int half_K = K / 2;
    const int groups_per_row = K / kGroupSize;

    const uint8_t* __restrict__ row_packed = input_packed + static_cast<int64_t>(row) * half_K;
    const uint8_t* __restrict__ row_scales = input_scales + static_cast<int64_t>(row) * groups_per_row;
    __nv_bfloat16* __restrict__ row_output = output + static_cast<int64_t>(row) * K;

    for (int k = threadIdx.x; k < K; k += kThreadsPerBlock) {
        // Read packed FP4 byte and extract nibble.
        uint8_t packed_byte = __ldg(&row_packed[k / 2]);
        uint8_t nibble = (k & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);

        // FP4 E2M1 → float via LUT.
        float fp4_val = kFp4E2M1Lut[nibble];

        // Per-group FP8 E4M3 scale → float.
        float group_scale = e4m3_to_float(__ldg(&row_scales[k / kGroupSize]));

        // Dequantized value.
        float val = fp4_val * group_scale * weight_scale_2;

        row_output[k] = __float2bfloat16(val);
    }
}

}  // namespace

void launch_nvfp4_dequant_bf16(const Nvfp4DequantBf16Params& params,
                                void* stream) {
    if (params.N <= 0 || params.K <= 0) return;
    if (!params.input || !params.scales || !params.output) {
        throw std::runtime_error(
            "launch_nvfp4_dequant_bf16: null pointer in params");
    }
    if (params.K % 16 != 0) {
        throw std::runtime_error(
            "launch_nvfp4_dequant_bf16: K must be divisible by 16 (got " +
            std::to_string(params.K) + ")");
    }

    dim3 grid(params.N);
    dim3 block(kThreadsPerBlock);

    nvfp4_dequant_bf16_kernel<<<grid, block, 0,
                                 static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16*>(params.output),
        static_cast<const uint8_t*>(params.input),
        static_cast<const uint8_t*>(params.scales),
        params.weight_scale_2,
        params.N, params.K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_nvfp4_dequant_bf16 failed: ") +
            cudaGetErrorString(err));
    }
}

}  // namespace layerstorm::compute

#endif  // NVFP4_DEQUANT_BF16_CU_INCLUDED
