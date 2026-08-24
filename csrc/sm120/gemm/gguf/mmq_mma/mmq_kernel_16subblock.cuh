#pragma once

//==============================================================================
// FAST GGUF mmq kernel body for the 16-SUBBLOCK types (Q2_K/Q3_K/Q6_K).
//
// These types have 16-element weight sub-blocks, so the MMA chunk is k=16
// (SM80_16x8x16_S32S8S8S32_TN) — one MMA per 16-K chunk, with the affine
// rescale applied once per 16-chunk (exactly one A/B/sumqx per chunk). The
// Q8_1 activation block is 32 elements and spans two consecutive 16-chunks
// (same d_x), so d_x is indexed per 32-block (c/2).
//
// This mirrors mmq_kernel_32aligned.cuh structurally (single-buffered, optional
// cp.async activation streaming) but at k=16 granularity. It reuses the common
// scaffolding: the activation loaders (via the as_block32 adaptor), the per-16
// sumqx helper, the per-16 weight loader, and the per-16 affine rescale. See
// mmq_mma_common.cuh for the ggml technique attribution.
//
// Cost vs the 32-aligned path: 2x the MMAs (k=16 atom) and 2x the rescales for
// the same VALS, so it is expected to be slower than Q4_K/Q5_K/Q8_0 — but it is
// the correct path for the 16-subblock types and still beats the v2 k=16 tile
// and the dp4a path.
//==============================================================================

#include "sm120/gemm/gguf/mmq_mma/mmq_mma_common.cuh"
#include "sm120/gemm/gguf/gguf_mmvq.h"          // GgufMmvqParams
#include "sm120/gemm/gguf/gguf_grouped_int.h"   // GgufGroupedMmqArgs (device routing)

namespace layerstorm::compute {
namespace mmq_mma {

// Per-tile compute body for the 16-subblock mmq kernel. Shared by the
// single-expert and device-routed grouped globals (see the 32-aligned header
// for the parameter contract: M = global row limit, m_start = global row
// origin, Wbase = expert weight base, C = output base).
template <class P, bool USE_CPASYNC, class Cfg>
__device__ __forceinline__ void gguf_mmq_mma_16subblock_body(
        char* smem, const block_q8_1* __restrict__ xq, const uint8_t* __restrict__ Wbase,
        __nv_bfloat16* __restrict__ C, int M, int N, int K, int n_start, int m_start,
        int tid) {
    constexpr int VALS = P::VALS;
    constexpr int NC   = VALS / 16;     // 16-K chunks per super-block
    constexpr int BM = Cfg::kTILE_M, BN = Cfg::kTILE_N;
    constexpr int NT = Cfg::kBLOCK_SIZE, FC = Cfg::kFRAG_C;
    using S = SmemSizes16<VALS, BM, BN>;

    SmemBlock16<VALS, BM, BN> sm(smem);
    // Adaptor view for the (VALS-keyed) activation loaders, which only touch the
    // shared qx/dx prefix of the smem layout.
    SmemBlock<VALS, BM, BN> sm32 = as_block32<VALS, BM, BN>(sm);

    const int nblk = K / VALS;

    // Persistent float accumulator + the partition_C (m,n) mapping. The k=16 and
    // k=32 atoms share CLayout (SM80_16x8_Row), so this mapping is identical.
    typename Cfg::TiledMma16 tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);
    auto cIdent = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
    auto tCcAcc = thr_mma.partition_C(cIdent);     // -> (m,n) per frag elem
    const int frag_c = size(tCcAcc);

    float Acc[FC];
    #pragma unroll
    for (int i = 0; i < FC; ++i) Acc[i] = 0.0f;

    // smem operand tensors (row-major, k-major within row).
    auto sA_t = make_tensor(make_smem_ptr(sm.sQx),
                            make_layout(make_shape(Int<BM>{}, Int<VALS>{}),
                                        make_stride(Int<VALS>{}, _1{})));
    auto sB_t = make_tensor(make_smem_ptr(sm.sQw),
                            make_layout(make_shape(Int<BN>{}, Int<VALS>{}),
                                        make_stride(Int<VALS>{}, _1{})));

    for (int kb = 0; kb < nblk; ++kb) {
        if (USE_CPASYNC) {
            // Issue the activation qx cp.async, do the independent weight dequant
            // while it streams, then wait before reading sQx.
            load_activation_tile_cpasync<VALS, BM, BN, NT>(xq, sm32, kb, m_start, M, K, tid);
            cp_async_fence();
            load_weight_tile16<P, BM, BN, NT>(Wbase, sm, kb, n_start, N, nblk, tid);
            cp_async_wait<0>();
            __syncthreads();
        } else {
            load_activation_tile<VALS, BM, BN, NT>(xq, sm32, kb, m_start, M, K, tid);
            load_weight_tile16<P, BM, BN, NT>(Wbase, sm, kb, n_start, N, nblk, tid);
            __syncthreads();
        }
        compute_sumqx_tile16<VALS, BM, BN, NT>(sm, tid);
        __syncthreads();

        // One k=16 MMA per 16-chunk; rescale int32 partial into Acc.
        #pragma unroll 4
        for (int c = 0; c < NC; ++c) {
            auto sAchunk = local_tile(sA_t, make_shape(Int<BM>{}, Int<K_MMA16>{}),
                                      make_coord(0, c));
            auto sBchunk = local_tile(sB_t, make_shape(Int<BN>{}, Int<K_MMA16>{}),
                                      make_coord(0, c));

            auto tCrA = thr_mma.partition_fragment_A(sAchunk);
            auto tCrB = thr_mma.partition_fragment_B(sBchunk);

            auto rP = partition_fragment_C(tiled_mma, Shape<Int<BM>, Int<BN>>{});
            clear(rP);

            auto tCsA = thr_mma.partition_A(sAchunk);
            auto tCsB = thr_mma.partition_B(sBchunk);
            cute::copy(tCsA(_, _, 0), tCrA(_, _, 0));
            cute::copy(tCsB(_, _, 0), tCrB(_, _, 0));
            cute::gemm(tiled_mma, tCrA(_, _, 0), tCrB(_, _, 0), rP);

            rescale_chunk_into_acc16<VALS, BM, BN, FC>(sm, tCcAcc, rP, c, frag_c, Acc);
        }
        __syncthreads();
    }

    // Epilogue: write BF16 via the partition_C (m,n) mapping.
    #pragma unroll
    for (int i = 0; i < FC; ++i) {
        if (i >= frag_c) break;
        const int m = (int)get<0>(tCcAcc(i));
        const int n = (int)get<1>(tCcAcc(i));
        const int mg = m_start + m;
        const int ng = n_start + n;
        if (mg < M && ng < N)
            C[(size_t)mg * N + ng] = __float2bfloat16_rn(Acc[i]);
    }
}

// Single-expert global: row origin from blockIdx.y, weight/output from params.
template <class P, bool USE_CPASYNC = false, class Cfg = DefaultTileCfg>
__global__ void __launch_bounds__(Cfg::kBLOCK_SIZE)
gguf_mmq_mma_16subblock(GgufMmvqParams params, const block_q8_1* __restrict__ xq) {
    extern __shared__ char smem[];
    const int n_start = blockIdx.x * Cfg::kTILE_N;
    const int m_start = blockIdx.y * Cfg::kTILE_M;
    gguf_mmq_mma_16subblock_body<P, USE_CPASYNC, Cfg>(
        smem, xq, reinterpret_cast<const uint8_t*>(params.B), params.C,
        params.M, params.N, params.K, n_start, m_start, threadIdx.x);
}

// Device-routed grouped global: see the 32-aligned header for the routing.
template <class P, bool USE_CPASYNC = false, class Cfg = DefaultTileCfg>
__global__ void __launch_bounds__(Cfg::kBLOCK_SIZE)
gguf_mmq_mma_16subblock_grouped(GgufGroupedMmqArgs args) {
    const int t = blockIdx.y;
    const int e = args.tile_expert[t];
    if (e < 0) return;                              // dead (padding) tile
    const int Mend    = args.expert_offsets[e + 1]; // global row limit
    const int m_start = args.tile_mstart[t];        // global row origin
    const uint8_t* Wbase = reinterpret_cast<const uint8_t*>(args.B_ptrs[e]);
    const int n_start = blockIdx.x * Cfg::kTILE_N;
    extern __shared__ char smem[];
    gguf_mmq_mma_16subblock_body<P, USE_CPASYNC, Cfg>(
        smem, args.xq, Wbase, args.D_base, Mend, args.N, args.K,
        n_start, m_start, threadIdx.x);
}

}  // namespace mmq_mma
}  // namespace layerstorm::compute
