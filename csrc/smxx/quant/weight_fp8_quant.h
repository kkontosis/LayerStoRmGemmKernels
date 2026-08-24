// Offline BF16 → FP8 E4M3 weight quantization with tile-level scales.
//
// Quantizes a [N, K] row-major BF16 weight matrix to [N, K] row-major FP8 E4M3
// with [K_blocks, N_blocks] K-major float32 scales (one per 128x128 tile) —
// the scale_B layout launch_fp8_gemm consumes (fp8_gemm.h).
// Output layout is row-major (same as input) — despite the CUTLASS type alias
// LayoutBTag = ColumnMajor, the TN kernel reads row-major B correctly via
// make_cute_packed_stride.
//
// Architecture-agnostic (no SM120-specific ops).

#pragma once

#include <cstddef>

namespace layerstorm::compute {

/// Parameters for offline BF16 → FP8 weight quantization.
///
/// Quantizes [N, K] BF16 row-major → [N, K] FP8 E4M3 row-major with
/// per-tile float32 scales.  Tile size = 128x128.
///
/// Scale layout: [ceil(K/128), ceil(N/128)] K-major (column-major), i.e.
/// scales[k_blk + n_blk*ceil(K/128)] — exactly the scale_B contract of
/// launch_fp8_gemm (fp8_gemm.h).
///
/// NOT in-place safe: output must NOT alias input (the kernel re-reads
/// input during the write pass; aliasing corrupts data race-dependently).
struct WeightFp8QuantParams {
    int N;              ///< rows of weight matrix (output dim)
    int K;              ///< cols of weight matrix (input dim)
    const void* input;  ///< [N, K] BF16, row-major (K stride-1)
    void* output;       ///< [N, K] FP8 E4M3, row-major (K stride-1; no alias)
    void* scales;       ///< [ceil(K/128), ceil(N/128)] float32, K-major
};

/// Launch offline BF16 → FP8 weight quantization with tile-level scales.
///
///   stream: cudaStream_t (as void*), or nullptr for default stream.
///
/// Throws std::runtime_error on invalid dimensions or null pointers.
void launch_weight_fp8_quant(const WeightFp8QuantParams& params,
                              void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute
