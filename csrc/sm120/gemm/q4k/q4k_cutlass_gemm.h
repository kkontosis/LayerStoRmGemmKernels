#pragma once
// Q4_K tensor-core GEMM: C[M,N] = A[M,K] @ dequant(B_q4k[N, K/256 blocks]).T
//
// Shared-memory dequant + BF16 wmma tensor-core MMA on SM120.
// A is BF16 activation (row-major), B is Q4_K quantized weight stored as
// contiguous block_q4_K structs, C is BF16 output.
//
// For M <= 8 (decode): delegates to Phase 2 GEMV (memory-bound, tensor cores
// don't help). For M > 8 (prefill): tensor-core kernel with cooperative Q4_K
// dequant to BF16 shared memory, then wmma 16x16x16 BF16 MMA.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace layerstorm::compute {

struct Q4KCutlassGemmParams {
    int M;                                    // rows of A / rows of C (tokens)
    int N;                                    // output features (rows of B_q4k)
    int K;                                    // inner dimension (must be % 256 == 0)

    const __nv_bfloat16* __restrict__ A;      // [M, K] BF16, row-major
    const void* __restrict__ B_q4k;           // [N * (K/256)] block_q4_K structs (144 bytes each)
    __nv_bfloat16* __restrict__ C;            // [M, N] BF16 output, row-major
};

void launch_q4k_cutlass_gemm(const Q4KCutlassGemmParams& params, cudaStream_t stream);

}  // namespace layerstorm::compute
