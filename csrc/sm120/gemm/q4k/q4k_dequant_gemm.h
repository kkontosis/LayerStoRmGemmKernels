#pragma once
// Q4_K dequant-GEMM: C[M,N] = A[M,K] @ dequant(B_q4k[N, K/256 blocks]).T
//
// Hand-written tiled GEMM on CUDA cores (no tensor cores / no CUTLASS).
// A is BF16 activation (row-major), B is Q4_K quantized weight stored as
// contiguous block_q4_K structs (row-major per-N row), C is BF16 output.
//
// Algorithm per thread block (TILE_M=64, TILE_N=64, 256 threads):
//   For each K step of 256 (one Q4_K super-block):
//     1. Cooperative load A tile [64, 256] BF16 → shared memory
//     2. Cooperative load + dequant B tile [64 blocks] → [64, 256] half in smem
//     3. Each thread computes 4x4 output sub-tile with FP32 FMA accumulation
//   Write output as BF16.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace layerstorm::compute {

struct Q4KDequantGemmParams {
    int M;                                    // rows of A / rows of C (tokens)
    int N;                                    // output features (rows of B_q4k)
    int K;                                    // inner dimension (must be % 256 == 0)

    const __nv_bfloat16* __restrict__ A;      // [M, K] BF16, row-major
    const void* __restrict__ B_q4k;           // [N * (K/256)] block_q4_K structs (144 bytes each)
    __nv_bfloat16* __restrict__ C;            // [M, N] BF16 output, row-major
};

void launch_q4k_dequant_gemm(const Q4KDequantGemmParams& params, cudaStream_t stream);

}  // namespace layerstorm::compute
