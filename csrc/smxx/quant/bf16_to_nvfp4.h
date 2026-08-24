#pragma once
// BF16 activation quantization to NVFP4 format.
//
// Per group of 16 elements: compute amax, derive UE4M3 scale, quantize
// to nearest FP4 E2M1, pack pairs into uint8.
// Scales are written directly in Sm1xx interleaved layout.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace layerstorm::compute {

struct Bf16ToNvfp4Params {
    const __nv_bfloat16* __restrict__ input;   // [M, K] BF16 activation
    uint8_t* __restrict__ output_packed;        // [M, K/2] packed FP4 pairs
    uint8_t* __restrict__ output_scales;        // Sm1xx interleaved UE4M3 scales
    int M;
    int N;   // needed for Sm1xx layout computation
    int K;   // must be divisible by 16
};

void launch_bf16_to_nvfp4(const Bf16ToNvfp4Params& params, cudaStream_t stream);

}  // namespace layerstorm::compute
