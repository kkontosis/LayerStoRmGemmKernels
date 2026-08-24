#pragma once
// Reformat per-group scales from simple row-major to Sm1xx interleaved layout.
//
// Source: [rows, K/16] row-major float32 scales (from Python reference format)
// Dest:   flat buffer in Sm1xx interleaved format (for CUTLASS BlockScaledTensorOp)

#include <cuda_runtime.h>
#include <cstdint>

namespace layerstorm::compute {

struct ReformatScalesParams {
    const float* __restrict__ src_scales;  // [rows, groups_per_row] float32
    uint8_t* __restrict__ dst_scales;      // Sm1xx interleaved UE4M3 buffer
    int rows;             // N (weight) or M (activation)
    int groups_per_row;   // K / 16
    int M, N, K;          // full problem dims (for Sm1xx layout computation)
    bool is_scale_a;      // true = SFA layout, false = SFB layout
};

void launch_reformat_scales(const ReformatScalesParams& params, cudaStream_t stream);

}  // namespace layerstorm::compute
