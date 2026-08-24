#pragma once

//==============================================================================
// FAST GGUF mmq kernel body for the 32-subblock-aligned types (Q4_K/Q5_K/Q8_0).
//
// One k=32 MMA (SM80_16x8x32_S32S8S8S32_TN) per 32-K chunk over a 128x128 / 256
// thread tile, with the affine rescale applied once per chunk (exactly one d_x /
// A / B / sumqx per chunk for these types). Templated on the integer policy P.
//
// Single-buffered, cute::copy operand loads (step-1 correctness baseline). The
// cp.async pipeline / ldmatrix variants are layered on top in
// mmq_mma_common.cuh as the optimizations land. See mmq_mma_common.cuh for the
// ggml technique attribution.
//==============================================================================

#include "sm120/gemm/gguf/mmq_mma/mmq_mma_common.cuh"
#include "sm120/gemm/gguf/gguf_mmvq.h"          // GgufMmvqParams
#include "sm120/gemm/gguf/gguf_grouped_int.h"   // GgufGroupedMmqArgs (device routing)

namespace layerstorm::compute {
namespace mmq_mma {

// Per-tile compute body for the 32-aligned mmq kernel. Shared by the
// single-expert global (m_start = blockIdx.y*BM, M = params.M, Wbase = params.B,
// C = params.C) and the device-routed grouped global (m_start / M / Wbase / C
// resolved from the tile map + B_ptrs on device). The body itself is identical;
// only the row origin / row limit / weight base / output base differ.
//
//   M        : global row LIMIT (rows >= M are masked off)
//   m_start  : global row ORIGIN of this CTA's M-tile (into xq / C)
//   Wbase    : this expert's packed GGUF weight base ([N, nblk*BYTES])
//   C        : output base ([*, N] BF16, row-major); written at global (mg, ng)
template <class P, bool USE_CPASYNC, bool USE_LDMATRIX, class Cfg>
__device__ __forceinline__ void gguf_mmq_mma_32aligned_body(
        char* smem, const block_q8_1* __restrict__ xq, const uint8_t* __restrict__ Wbase,
        __nv_bfloat16* __restrict__ C, int M, int N, int K, int n_start, int m_start,
        int tid) {
    constexpr int VALS = P::VALS;
    constexpr int N32  = VALS / 32;     // 32-K chunks per super-block
    constexpr int BM = Cfg::kTILE_M, BN = Cfg::kTILE_N;
    constexpr int NT = Cfg::kBLOCK_SIZE, FC = Cfg::kFRAG_C;
    using S = SmemSizes32<VALS, BM, BN>;

    SmemBlock<VALS, BM, BN> sm(smem);

    const int nblk = K / VALS;

    // Persistent float accumulator + the partition_C (m,n) mapping.
    typename Cfg::TiledMma tiled_mma;
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

    // ldmatrix tiled-copies for the int8 operand smem->reg loads (step-3).
    auto ldsm_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, int8_t>{}, tiled_mma);
    auto ldsm_b = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, int8_t>{}, tiled_mma);
    auto thr_ldsm_a = ldsm_a.get_slice(tid);
    auto thr_ldsm_b = ldsm_b.get_slice(tid);

    for (int kb = 0; kb < nblk; ++kb) {
        if (USE_CPASYNC) {
            // Issue the activation qx cp.async, then do the independent weight
            // dequant while it streams, then wait before reading sQx.
            load_activation_tile_cpasync<VALS, BM, BN, NT>(xq, sm, kb, m_start, M, K, tid);
            cp_async_fence();
            load_weight_tile<P, BM, BN, NT>(Wbase, sm, kb, n_start, N, nblk, tid);
            cp_async_wait<0>();
            __syncthreads();
        } else {
            load_activation_tile<VALS, BM, BN, NT>(xq, sm, kb, m_start, M, K, tid);
            load_weight_tile<P, BM, BN, NT>(Wbase, sm, kb, n_start, N, nblk, tid);
            __syncthreads();
        }
        compute_sumqx_tile<VALS, BM, BN, NT>(sm, tid);
        __syncthreads();

        // One k=32 MMA per 32-chunk; rescale int32 partial into Acc.
        #pragma unroll 4
        for (int c = 0; c < N32; ++c) {
            auto sAchunk = local_tile(sA_t, make_shape(Int<BM>{}, Int<K_MMA>{}),
                                      make_coord(0, c));
            auto sBchunk = local_tile(sB_t, make_shape(Int<BN>{}, Int<K_MMA>{}),
                                      make_coord(0, c));

            auto tCrA = thr_mma.partition_fragment_A(sAchunk);
            auto tCrB = thr_mma.partition_fragment_B(sBchunk);

            auto rP = partition_fragment_C(tiled_mma, Shape<Int<BM>, Int<BN>>{});
            clear(rP);

            if (USE_LDMATRIX) {
                // ldmatrix smem->reg operand loads.
                auto tAsA = thr_ldsm_a.partition_S(sAchunk);
                auto tArA = thr_ldsm_a.retile_D(tCrA);
                auto tBsB = thr_ldsm_b.partition_S(sBchunk);
                auto tBrB = thr_ldsm_b.retile_D(tCrB);
                cute::copy(ldsm_a, tAsA, tArA);
                cute::copy(ldsm_b, tBsB, tBrB);
            } else {
                auto tCsA = thr_mma.partition_A(sAchunk);
                auto tCsB = thr_mma.partition_B(sBchunk);
                cute::copy(tCsA(_, _, 0), tCrA(_, _, 0));
                cute::copy(tCsB(_, _, 0), tCrB(_, _, 0));
            }
            cute::gemm(tiled_mma, tCrA(_, _, 0), tCrB(_, _, 0), rP);

            rescale_chunk_into_acc<VALS, BM, BN, FC>(sm, tCcAcc, rP, c, frag_c, Acc);
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
template <class P, bool USE_CPASYNC = false, bool USE_LDMATRIX = false,
          class Cfg = DefaultTileCfg>
__global__ void __launch_bounds__(Cfg::kBLOCK_SIZE)
gguf_mmq_mma_32aligned(GgufMmvqParams params, const block_q8_1* __restrict__ xq) {
    extern __shared__ char smem[];
    const int n_start = blockIdx.x * Cfg::kTILE_N;
    const int m_start = blockIdx.y * Cfg::kTILE_M;
    gguf_mmq_mma_32aligned_body<P, USE_CPASYNC, USE_LDMATRIX, Cfg>(
        smem, xq, reinterpret_cast<const uint8_t*>(params.B), params.C,
        params.M, params.N, params.K, n_start, m_start, threadIdx.x);
}

// Device-routed grouped global: row origin / row limit / weight / output base
// resolved on device from the tile->expert map (tile_expert/tile_mstart),
// expert_offsets and B_ptrs. Dead tiles (expert == -1) early-return.
template <class P, bool USE_CPASYNC = false, bool USE_LDMATRIX = false,
          class Cfg = DefaultTileCfg>
__global__ void __launch_bounds__(Cfg::kBLOCK_SIZE)
gguf_mmq_mma_32aligned_grouped(GgufGroupedMmqArgs args) {
    const int t = blockIdx.y;
    const int e = args.tile_expert[t];
    if (e < 0) return;                              // dead (padding) tile
    const int Mend    = args.expert_offsets[e + 1]; // global row limit
    const int m_start = args.tile_mstart[t];        // global row origin
    const uint8_t* Wbase = reinterpret_cast<const uint8_t*>(args.B_ptrs[e]);
    const int n_start = blockIdx.x * Cfg::kTILE_N;
    extern __shared__ char smem[];
    gguf_mmq_mma_32aligned_body<P, USE_CPASYNC, USE_LDMATRIX, Cfg>(
        smem, args.xq, Wbase, args.D_base, Mend, args.N, args.K,
        n_start, m_start, threadIdx.x);
}

}  // namespace mmq_mma
}  // namespace layerstorm::compute
