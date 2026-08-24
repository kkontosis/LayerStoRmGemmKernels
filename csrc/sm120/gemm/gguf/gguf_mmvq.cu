// GGUF integer mat-vec (mmvq), strategy "int".
//
// Pipeline: quantize the BF16 activation to Q8_1 (per-32 block d + int8 qs),
// then for each output channel a CTA accumulates, over the weight's packed
// blocks, the integer dp4a dot-product of the (centered) weight quants against
// the activation quants, scaled per sub-block.
//
// Each weight value is written as w = A_j*Q + B_j with Q a signed int8
// "centered" quant and (A_j, B_j) per-sub-block (B folded into Q for the
// center-quant types q3_K/q6_K/q8_0). Then, summing over a sub-block,
//   Σ w*x  ≈  d_x * ( A_j*Σ(Q*qx) + B_j*Σ qx )
// using x ≈ d_x*qx. The per-type "policy" produces (Q packed ×4, A, B) for each
// 4-value group, matching the dequant_*_block output ordering exactly so this
// path is numerically consistent with the dequant GEMM.
//
// Kernel structure (TD-GGUF-MMVQ-DECODE-EFFICIENCY redesign): NW=4 warps per
// CTA cooperate on ONE output channel with the K dimension k-split across all
// lanes, MT=8 activation rows per register pass, warp-shfl + shared-memory
// cross-warp reduction — see gguf_mmvq_impl.cuh (structure adapted from
// llama.cpp ggml-cuda mul_mat_vec_q; attribution there).

#include "sm120/gemm/gguf/gguf_mmvq.h"
#include "sm120/gemm/gguf/gguf_mmvq_impl.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

namespace layerstorm::compute {

namespace {

namespace fmt = layerstorm::formats;
namespace impl = gguf_mmvq_impl;
using fmt::block_q8_1;

// ===========================================================================
// Activation quantizer: BF16 x[M,K] -> block_q8_1[M, K/32]
// ===========================================================================

__global__ void quantize_bf16_to_q8_1(const __nv_bfloat16* __restrict__ x,
                                      block_q8_1* __restrict__ y, int M, int K) {
    const int kb = blockIdx.x;        // 0..K/32
    const int m = blockIdx.y;         // 0..M
    const int lane = threadIdx.x;     // 0..31
    const int k = kb * 32 + lane;

    const float xi = (k < K) ? __bfloat162float(x[m * K + k]) : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        sum += __shfl_xor_sync(0xffffffff, sum, o);
    }
    const float d = amax / 127.0f;
    const int q = (amax == 0.0f) ? 0 : __float2int_rn(xi / d);

    block_q8_1* blk = &y[m * (K / 32) + kb];
    blk->qs[lane] = static_cast<int8_t>(q);
    if (lane == 0) blk->ds = make_half2(__float2half(d), __float2half(sum));
}

// ===========================================================================
// mmvq kernel: NW warps per output channel, k-split, MT-row register tile.
// ===========================================================================

template <class P, int NW = impl::kWarps, int MT = impl::kMTile>
__global__ void __launch_bounds__(NW * 32)
gguf_mmvq_kernel(GgufMmvqParams params, const block_q8_1* __restrict__ xq) {
    const int n = blockIdx.x;
    if (n >= params.N) return;                 // uniform: whole CTA exits
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int K = params.K, N = params.N, M = params.M;
    const int nblk = K / P::VALS;
    const int x_bpr = K / 32;
    const int total_groups = K / 4;
    const uint8_t* Wrow = reinterpret_cast<const uint8_t*>(params.B)
                        + static_cast<size_t>(n) * nblk * P::BYTES;

    __shared__ float red[NW][MT];

    for (int m0 = 0; m0 < M; m0 += MT) {
        const int mt = min(M - m0, MT);
        float acc[MT];
        #pragma unroll
        for (int j = 0; j < MT; ++j) acc[j] = 0.0f;

        impl::accumulate<P, NW, MT>(
            Wrow, xq + static_cast<size_t>(m0) * x_bpr, x_bpr,
            total_groups, mt, tid, acc);

        impl::reduce_store<NW, MT>(
            acc, red, mt, lane, warp,
            params.C + static_cast<size_t>(m0) * N + n, N);
        if (m0 + MT < M) __syncthreads();      // red[] reused next pass
    }
}

template <class P>
void launch_mmvq(const GgufMmvqParams& params, const block_q8_1* xq, cudaStream_t stream) {
    const dim3 grid(params.N), block(impl::kWarps * 32);
    if (params.M == 1) {
        // Decode fast path: single-row register tile (no dead guarded rows).
        gguf_mmvq_kernel<P, impl::kWarps, 1><<<grid, block, 0, stream>>>(params, xq);
    } else {
        gguf_mmvq_kernel<P><<<grid, block, 0, stream>>>(params, xq);
    }
}

}  // namespace

// ===========================================================================
// Host entry points
// ===========================================================================

size_t gguf_mmvq_workspace_bytes(int M, int K) {
    return static_cast<size_t>(M) * (K / 32) * sizeof(block_q8_1);
}

void gguf_quantize_q8_1_cuda(const __nv_bfloat16* x, void* q8_1_out,
                             int M, int K, cudaStream_t stream) {
    quantize_bf16_to_q8_1<<<dim3(K / 32, M), dim3(32), 0, stream>>>(
        x, reinterpret_cast<block_q8_1*>(q8_1_out), M, K);
}

void launch_gguf_mmvq(const GgufMmvqParams& params, void* q8_1_workspace, cudaStream_t stream) {
    const int qk = gguf_block_values(params.type);
    if (params.K % qk != 0) {
        throw std::invalid_argument(
            "gguf_mmvq: K must be divisible by " + std::to_string(qk) +
            ", got K=" + std::to_string(params.K));
    }
    block_q8_1* xq = reinterpret_cast<block_q8_1*>(q8_1_workspace);

    // 1. Quantize activation to Q8_1.
    gguf_quantize_q8_1_cuda(params.A, xq, params.M, params.K, stream);

    // 2. Integer mat-vec.
    switch (params.type) {
        case GgufType::Q2_K: launch_mmvq<gguf_int::Q2K>(params, xq, stream);  break;
        case GgufType::Q3_K: launch_mmvq<gguf_int::Q3K>(params, xq, stream);  break;
        case GgufType::Q4_K: launch_mmvq<gguf_int::Q4K>(params, xq, stream);  break;
        case GgufType::Q5_K: launch_mmvq<gguf_int::Q5K>(params, xq, stream);  break;
        case GgufType::Q6_K: launch_mmvq<gguf_int::Q6K>(params, xq, stream);  break;
        case GgufType::Q8_0: launch_mmvq<gguf_int::Q8_0>(params, xq, stream); break;
        case GgufType::MXFP4: launch_mmvq<gguf_int::Mxfp4>(params, xq, stream); break;
    }
}

}  // namespace layerstorm::compute
