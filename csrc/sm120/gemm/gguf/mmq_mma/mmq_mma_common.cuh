#pragma once

//==============================================================================
// Shared scaffolding for the FAST GGUF mmq int8 tensor-core path (mmq_mma).
//
// This is the "v3" GGUF mat-mat: C[M,N] = A[M,K] @ dequant(W)^T, with the
// activation quantized to Q8_1 and the weight dequanted to centered int8 in
// shared memory. The compute uses the SM80 *m16n8k32* int8 MMA atom
// (SM80_16x8x32_S32S8S8S32_TN, int8 in / int32 out) instead of the v2 path's
// m16n8k16 atom. The win:
//
//   For the 32-aligned GGUF types (Q4_K/Q5_K/Q8_0) the weight sub-blocks are
//   32 elements AND the Q8_1 activation blocks are 32 elements, so each k=32
//   MMA chunk has EXACTLY ONE d_x, ONE A_j, ONE B_j. The int32 partial
//       P[m,n] = Σ_{32k} Q[n,k]*qx[m,k]
//   is rescaled once per 32-K chunk:
//       Acc[m,n] += d_x[m,c] * ( A_j[n,c]*P[m,n] + B_j[n,c]*sumqx[m,c] )
//   This HALVES the rescale frequency vs the v2 k=16 kernel and uses a 2x
//   bigger atom — the perf win.
//
// Technique reference (HOW, not copied): llama.cpp's mmq
//   ggml/src/ggml-cuda/{mmq.cuh,mma.cuh,quantize.cu}  (MIT, "The ggml authors",
//   commit b9780): the k=32 MMA, ldmatrix operand loads, the cp.async K-pipeline
//   and keeping the scale-rescale off the hot MMA path. Reimplemented in CuTe.
//
// This header is intentionally policy-generic: the activation loader, weight
// loader, smem-sizing and the affine-rescale epilogue all take `VALS` / the
// policy `P`, so later agents can reuse them for the other GGUF types (the
// 16-subblock types will pair them with a k=16 MMA variant; the 32-subblock
// kernel body lives in mmq_kernel_32aligned.cuh).
//==============================================================================

#include "formats/q8_1_format.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/copy_traits_sm75.hpp>

namespace layerstorm::compute {
namespace mmq_mma {

using namespace cute;

namespace fmt = layerstorm::formats;
using fmt::block_q8_1;

//------------------------------------------------------------------------------
// Tile configuration shared by the mmq_mma kernels.
//------------------------------------------------------------------------------
static constexpr int TILE_M     = 128;
static constexpr int TILE_N     = 128;
static constexpr int BLOCK_SIZE = 256;   // 8 warps
static constexpr int K_MMA      = 32;    // k of the SM80 m16n8k32 atom
static constexpr int K_MMA16    = 16;    // k of the SM80 m16n8k16 atom

// C-fragment elements per thread for the default 128x128 / 256-thread tile.
// (Per-config FRAG_C lives in MmqTileCfg::kFRAG_C; this global is the default
// for the back-compat helper template args.)
static constexpr int FRAG_C = (TILE_M * TILE_N) / BLOCK_SIZE;   // 64

// int8 tensor-core MMA: m16n8k32, int8 in / int32 accumulate, TN.
// 8 warps in a 4x2 layout over the 128x128 output tile; one K-step == one
// 32-K chunk (== one Q8_1 block == one Q4_K weight sub-block).
using MmaAtomS8    = MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>;
using MmaAtomS8K16 = MMA_Atom<SM80_16x8x16_S32S8S8S32_TN>;

//------------------------------------------------------------------------------
// Tile-configuration trait. Parameterizes the output tile (BM x BN), thread
// count (NT) and the matching TiledMMA warp layout, so the launch can pick a
// SMALLER tile at small M (more CTAs -> higher occupancy when the 128-tall tile
// would underfill the GPU). The default (128,128,256) reproduces the original
// constants byte-for-byte; the k=32 and k=16 TiledMMAs share the same warp
// layout / CLayout so the partition_C (m,n) epilogue mapping is identical.
//
//   WM x WN warps (WM*WN*32 == NT); the SM80 16x8 atom replicated over the warp
//   grid + a Tile<BM,BN,K> permutation covers the BM x BN tile. FRAG_C =
//   BM*BN/NT C-fragment elements per thread.
//------------------------------------------------------------------------------
template <int BM, int BN, int NT, int WM, int WN>
struct MmqTileCfg {
    static constexpr int kTILE_M     = BM;
    static constexpr int kTILE_N     = BN;
    static constexpr int kBLOCK_SIZE = NT;
    static constexpr int kFRAG_C     = (BM * BN) / NT;
    static_assert(WM * WN * 32 == NT, "warp grid must match thread count");

    using WarpLayout = Layout<Shape<Int<WM>, Int<WN>, _1>>;
    using TiledMma   = TiledMMA<MmaAtomS8,    WarpLayout, Tile<Int<BM>, Int<BN>, _32>>;
    using TiledMma16 = TiledMMA<MmaAtomS8K16, WarpLayout, Tile<Int<BM>, Int<BN>, _16>>;
};

// Default tile: 128x128, 256 threads, 4x2 warps (matches the original layout).
using DefaultTileCfg = MmqTileCfg<TILE_M, TILE_N, BLOCK_SIZE, 4, 2>;
// Small-M tile: 64x128, 256 threads, 2x4 warps. Halves the M-tile height so a
// 128-row M produces 2x the M-tile rows (2x CTAs), raising occupancy at small M.
// FRAG_C = 64*128/256 = 32.
using SmallMTileCfg  = MmqTileCfg<64, 128, 256, 2, 4>;

using TiledMmaS8    = DefaultTileCfg::TiledMma;     // back-compat aliases
using TiledMmaS8K16 = DefaultTileCfg::TiledMma16;

//------------------------------------------------------------------------------
// Launch heuristic: should this (M,N) use the SmallMTileCfg (64x128) instead of
// the DefaultTileCfg (128x128)?
//
// The default 128-tall M-tile underfills the GPU at small M (too few CTAs) and
// wastes compute when the last M-tile is only partly populated. Halving TILE_M
// to 64 doubles the M-tile count (2x CTAs). Measured on RTX 5090 across Q4_K/
// Q5_K/Q8_0/Q2_K/Q3_K/Q6_K at q_b (N=12288,K=1536) and o_proj (N=7168,K=8192),
// the small tile wins exactly when EITHER:
//   (a) the default grid's last M-tile has <=64 valid rows  (M%128 in (0,64]),
//       i.e. half the top M-tile is wasted compute  — OR —
//   (b) the default grid is heavily CTA-underfilled: default_CTAs*2 <= numSMs,
//       so the doubled CTA count better fills the SMs.
// This rule matched all 20 measured (shape x M) points with zero mispredictions
// (+11..+38% where it fires; never picks the slower config). Only host code.
//------------------------------------------------------------------------------
__host__ inline int mmq_mma_sm_count() {
    static int sm = -1;
    if (sm < 0) {
        int dev = 0; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
        if (sm <= 0) sm = 170;   // sane fallback
    }
    return sm;
}

__host__ inline bool mmq_mma_use_small_m(int M, int N) {
    const int n_tiles = (N + TILE_N - 1) / TILE_N;
    const int m_tiles = (M + TILE_M - 1) / TILE_M;
    const int default_ctas = n_tiles * m_tiles;
    const int top = M % TILE_M;                       // rows in the last M-tile
    const bool partial_top = (top != 0 && top <= 64); // <=64 valid rows wasted
    const bool underfilled = default_ctas * 2 <= mmq_mma_sm_count();
    return partial_top || underfilled;
}

//------------------------------------------------------------------------------
// Shared-memory byte layout for the 32-aligned kernel, given VALS.
//   N32 = VALS/32 = number of 32-K chunks per K super-block. For the 32-aligned
//   types each chunk carries exactly one (d_x, A, B, sumqx).
//------------------------------------------------------------------------------
template <int VALS, int BM = TILE_M, int BN = TILE_N>
struct SmemSizes32 {
    static constexpr int N32 = VALS / 32;          // 32-K chunks per super-block
    static constexpr int qx_bytes = BM * VALS;                 // int8 qx
    static constexpr int qw_bytes = BN * VALS;                 // int8 Qw
    static constexpr int dx_bytes = BM * N32 * (int)sizeof(float);  // d_x
    static constexpr int a_bytes  = BN * N32 * (int)sizeof(float);  // A_j
    static constexpr int b_bytes  = BN * N32 * (int)sizeof(float);  // B_j
    static constexpr int sx_bytes = BM * N32 * (int)sizeof(float);  // sumqx
    static constexpr int total =
        qx_bytes + qw_bytes + dx_bytes + a_bytes + b_bytes + sx_bytes;
};

// Pointers into one smem super-block region. Reusable by any kernel that uses
// the same {qx, qw, dx, A, B, sumqx} layout (single- or multi-buffered: each
// buffer is one SmemBlock laid out back-to-back).
template <int VALS, int BM = TILE_M, int BN = TILE_N>
struct SmemBlock {
    using S = SmemSizes32<VALS, BM, BN>;
    int8_t* sQx;   // [TILE_M][VALS]   int8 activation quants
    int8_t* sQw;   // [TILE_N][VALS]   int8 centered weight quants
    float*  sDx;   // [TILE_M][N32]    activation scale d_x
    float*  sA;    // [TILE_N][N32]    weight affine A
    float*  sB;    // [TILE_N][N32]    weight affine B
    float*  sSx;   // [TILE_M][N32]    Σ qx over each 32-chunk

    __device__ explicit SmemBlock(char* base) {
        sQx = reinterpret_cast<int8_t*>(base);
        sQw = reinterpret_cast<int8_t*>(base + S::qx_bytes);
        sDx = reinterpret_cast<float*>(base + S::qx_bytes + S::qw_bytes);
        sA  = reinterpret_cast<float*>(base + S::qx_bytes + S::qw_bytes + S::dx_bytes);
        sB  = reinterpret_cast<float*>(base + S::qx_bytes + S::qw_bytes + S::dx_bytes + S::a_bytes);
        sSx = reinterpret_cast<float*>(base + S::qx_bytes + S::qw_bytes + S::dx_bytes
                                            + S::a_bytes + S::b_bytes);
    }
};

//------------------------------------------------------------------------------
// Activation loader: stage one K super-block of the Q8_1 activation into smem.
// Loads int8 qx [TILE_M][VALS], per-32 scale d_x [TILE_M][N32], and the per-32
// sum sumqx [TILE_M][N32]. Policy-generic (only depends on VALS), so the
// k=16 kernel can reuse it verbatim.
//
//   xq      : Q8_1 workspace, row-major [M][K/32] blocks
//   kb      : K super-block index (0..K/VALS)
//   m_start : tile row origin
//------------------------------------------------------------------------------
template <int VALS, int BM = TILE_M, int BN = TILE_N, int NT = BLOCK_SIZE>
__device__ __forceinline__ void load_activation_tile(
        const block_q8_1* __restrict__ xq, const SmemBlock<VALS, BM, BN>& sm,
        int kb, int m_start, int M, int K, int tid) {
    constexpr int N32 = VALS / 32;
    const int x_bpr = K / 32;

    // int8 qx
    for (int idx = tid; idx < BM * VALS; idx += NT) {
        const int mr = idx / VALS, v = idx % VALS;
        const int mg = m_start + mr;
        int8_t val = 0;
        if (mg < M) {
            const block_q8_1* b = &xq[(size_t)mg * x_bpr + kb * N32 + v / 32];
            val = b->qs[v % 32];
        }
        sm.sQx[idx] = val;
    }
    // d_x  +  sumqx (one of each per 32-chunk).
    for (int idx = tid; idx < BM * N32; idx += NT) {
        const int mr = idx / N32, b = idx % N32;
        const int mg = m_start + mr;
        if (mg < M) {
            const block_q8_1& blk = xq[(size_t)mg * x_bpr + kb * N32 + b];
            sm.sDx[idx] = __half2float(__low2half(blk.ds));
            // ds.y already holds Σx (original activations); but sumqx must be the
            // integer Σ qx to match the dp4a oracle's B*Σqx term, so accumulate
            // the staged int8 quants directly below (after they are in smem).
        } else {
            sm.sDx[idx] = 0.0f;
        }
    }
}

//------------------------------------------------------------------------------
// cp.async activation loader: same staging as load_activation_tile, but the
// int8 qx body is streamed global->smem via cp.async 4-byte transfers so its
// global-load latency overlaps the (independent) weight-dequant compute. The
// scalar d_x stage is a normal store. Caller must cp_async_fence()+wait before
// reading sQx (e.g. before compute_sumqx_tile / the MMA loop).
//
// Smem-free (single-buffered): no extra buffers needed — the overlap comes
// from issuing the async copies, then doing weight dequant, then waiting.
// Policy-generic (depends only on VALS); reusable by the k=16 kernel.
//------------------------------------------------------------------------------
template <int VALS, int BM = TILE_M, int BN = TILE_N, int NT = BLOCK_SIZE>
__device__ __forceinline__ void load_activation_tile_cpasync(
        const block_q8_1* __restrict__ xq, const SmemBlock<VALS, BM, BN>& sm,
        int kb, int m_start, int M, int K, int tid) {
    constexpr int N32 = VALS / 32;
    constexpr int VALS4 = VALS / 4;   // 4-byte words of qx per row
    const int x_bpr = K / 32;

    // qx body via cp.async (4-byte words). Each Q8_1 block's qs[] is 32 int8 ==
    // 8 words; word w of row mr maps to block (kb*N32 + w/8), qs offset (w%8)*4.
    using CpAsync = SM80_CP_ASYNC_CACHEALWAYS<uint32_t>;
    for (int idx = tid; idx < BM * VALS4; idx += NT) {
        const int mr = idx / VALS4, w = idx % VALS4;   // word within row
        const int mg = m_start + mr;
        uint32_t* dst = reinterpret_cast<uint32_t*>(sm.sQx + mr * VALS) + w;
        if (mg < M) {
            const block_q8_1* b = &xq[(size_t)mg * x_bpr + kb * N32 + w / 8];
            const uint32_t* src = reinterpret_cast<const uint32_t*>(b->qs) + (w % 8);
            CpAsync::copy(*src, *dst);
        } else {
            *dst = 0u;
        }
    }
    // d_x: cheap scalar stage (no async).
    for (int idx = tid; idx < BM * N32; idx += NT) {
        const int mr = idx / N32, b = idx % N32;
        const int mg = m_start + mr;
        sm.sDx[idx] = (mg < M)
            ? __half2float(__low2half(xq[(size_t)mg * x_bpr + kb * N32 + b].ds))
            : 0.0f;
    }
}

//------------------------------------------------------------------------------
// Compute sumqx[TILE_M][N32] = Σ of the 32 staged qx per chunk per m-row.
// Must run after load_activation_tile + __syncthreads (reads sm.sQx).
//------------------------------------------------------------------------------
template <int VALS, int BM = TILE_M, int BN = TILE_N, int NT = BLOCK_SIZE>
__device__ __forceinline__ void compute_sumqx_tile(
        const SmemBlock<VALS, BM, BN>& sm, int tid) {
    constexpr int N32 = VALS / 32;
    for (int idx = tid; idx < BM * N32; idx += NT) {
        const int mr = idx / N32, c = idx % N32;
        const int base = mr * VALS + 32 * c;
        int s = 0;
        #pragma unroll
        for (int q = 0; q < 8; ++q) {            // 8 int4-vectors = 32 int8
            const int qx4 = *reinterpret_cast<const int*>(&sm.sQx[base + 4 * q]);
            s = __dp4a(qx4, 0x01010101, s);
        }
        sm.sSx[idx] = (float)s;
    }
}

//------------------------------------------------------------------------------
// Weight loader: dequant one K super-block of the packed GGUF weight into
// centered int8 [TILE_N][VALS] plus per-32-chunk affine (A,B). Uses the policy.
//
// For the 32-aligned types (P::VALS sub-blocks of 32) the policy's (A,B) are
// constant across the 8 four-groups of a 32-chunk, so we store one (A,B) per
// chunk (N32) rather than per 4-group (NG) — an 8x smem reduction.
//------------------------------------------------------------------------------
template <class P, int BM = TILE_M, int BN = TILE_N, int NT = BLOCK_SIZE>
__device__ __forceinline__ void load_weight_tile(
        const uint8_t* __restrict__ Wbase, const SmemBlock<P::VALS, BM, BN>& sm,
        int kb, int n_start, int N, int nblk, int tid) {
    constexpr int VALS = P::VALS;
    constexpr int NG   = VALS / 4;     // 4-value groups per super-block
    for (int idx = tid; idx < BN * NG; idx += NT) {
        const int nr = idx / NG, g = idx % NG;
        const int ng = n_start + nr;
        int Q = 0; float A = 0.0f, B = 0.0f;
        if (ng < N) {
            const void* wblk = Wbase + ((size_t)ng * nblk + kb) * P::BYTES;
            float dw, dmw; P::scales(wblk, dw, dmw);
            P::group(wblk, g, dw, dmw, Q, A, B);
        }
        *reinterpret_cast<int*>(&sm.sQw[nr * VALS + 4 * g]) = Q;
        // (A,B) constant within a 32-chunk == 8 four-groups; store once.
        if ((g & 7) == 0) {
            const int c = g >> 3;      // 32-chunk index
            sm.sA[nr * (VALS / 32) + c] = A;
            sm.sB[nr * (VALS / 32) + c] = B;
        }
    }
}

//------------------------------------------------------------------------------
// Affine-rescale epilogue helper: fold one chunk's int32 MMA partial into the
// persistent float accumulator using the partition_C (m,n) mapping.
//
//   Acc[i] += d_x[m,c] * ( A[n,c]*P[m,n] + B[n,c]*sumqx[m,c] )
//
// tCcAcc : partition_C(identity) -> (m,n) coord per fragment elem (caller-built)
// rP     : the int32 C-fragment from cute::gemm for THIS 32-chunk
// c      : the 32-chunk index within the super-block
//------------------------------------------------------------------------------
template <int VALS, int BM = TILE_M, int BN = TILE_N, int FC = FRAG_C,
          class CCoord, class FragP, class AccArr>
__device__ __forceinline__ void rescale_chunk_into_acc(
        const SmemBlock<VALS, BM, BN>& sm, const CCoord& tCcAcc, const FragP& rP,
        int c, int frag_c, AccArr& Acc) {
    constexpr int N32 = VALS / 32;
    #pragma unroll
    for (int i = 0; i < FC; ++i) {
        if (i >= frag_c) break;
        const int m = (int)get<0>(tCcAcc(i));
        const int n = (int)get<1>(tCcAcc(i));
        const float dx = sm.sDx[m * N32 + c];
        const float A  = sm.sA[n * N32 + c];
        const float B  = sm.sB[n * N32 + c];
        const float sx = sm.sSx[m * N32 + c];
        const int   Pi = rP(i);
        Acc[i] += dx * (A * (float)Pi + B * sx);
    }
}

//==============================================================================
// 16-SUBBLOCK SCAFFOLDING (Q2_K / Q3_K / Q6_K).
//
// These types have weight sub-blocks of 16 elements, so the MMA chunk is k=16
// (one chunk == one sub-block == exactly one (A,B)). The Q8_1 activation block
// is still 32 elements and spans TWO consecutive 16-chunks (same d_x), while
// sumqx is computed PER 16-chunk to match the per-chunk B*Σqx rescale term.
//
//   per 16-chunk c:  P[m,n] = Σ_{16k} Q[n,k]*qx[m,k]
//   Acc[m,n] += d_x[m, c/2] * ( A[n,c]*P[m,n] + B[n,c]*sumqx[m,c] )
//
// The qx/dx staging reuses load_activation_tile[_cpasync] verbatim (they depend
// only on VALS and the 32-block d_x layout). Only the chunk granularity for
// qw / A / B / sumqx and the MMA loop differ — handled by the helpers below.
//==============================================================================

//------------------------------------------------------------------------------
// Smem byte layout for the 16-subblock kernel, given VALS.
//   NC  = VALS/16 = number of 16-K chunks (== sub-blocks) per super-block.
//   N32 = VALS/32 = number of Q8_1 activation blocks (each spans 2 chunks).
// qx/qw are full int8 tiles; d_x is per-32-block; A/B/sumqx are per-16-chunk.
//------------------------------------------------------------------------------
template <int VALS, int BM = TILE_M, int BN = TILE_N>
struct SmemSizes16 {
    static constexpr int NC  = VALS / 16;          // 16-K chunks per super-block
    static constexpr int N32 = VALS / 32;          // Q8_1 blocks per super-block
    static constexpr int qx_bytes = BM * VALS;                  // int8 qx
    static constexpr int qw_bytes = BN * VALS;                  // int8 Qw
    static constexpr int dx_bytes = BM * N32 * (int)sizeof(float);  // d_x
    static constexpr int a_bytes  = BN * NC  * (int)sizeof(float);  // A_j
    static constexpr int b_bytes  = BN * NC  * (int)sizeof(float);  // B_j
    static constexpr int sx_bytes = BM * NC  * (int)sizeof(float);  // sumqx
    static constexpr int total =
        qx_bytes + qw_bytes + dx_bytes + a_bytes + b_bytes + sx_bytes;
};

// Pointers into one 16-subblock smem region.
template <int VALS, int BM = TILE_M, int BN = TILE_N>
struct SmemBlock16 {
    using S = SmemSizes16<VALS, BM, BN>;
    int8_t* sQx;   // [TILE_M][VALS]   int8 activation quants
    int8_t* sQw;   // [TILE_N][VALS]   int8 centered weight quants
    float*  sDx;   // [TILE_M][N32]    activation scale d_x (per 32-block)
    float*  sA;    // [TILE_N][NC]     weight affine A     (per 16-chunk)
    float*  sB;    // [TILE_N][NC]     weight affine B     (per 16-chunk)
    float*  sSx;   // [TILE_M][NC]     Σ qx over each 16-chunk

    __device__ explicit SmemBlock16(char* base) {
        sQx = reinterpret_cast<int8_t*>(base);
        sQw = reinterpret_cast<int8_t*>(base + S::qx_bytes);
        sDx = reinterpret_cast<float*>(base + S::qx_bytes + S::qw_bytes);
        sA  = reinterpret_cast<float*>(base + S::qx_bytes + S::qw_bytes + S::dx_bytes);
        sB  = reinterpret_cast<float*>(base + S::qx_bytes + S::qw_bytes + S::dx_bytes + S::a_bytes);
        sSx = reinterpret_cast<float*>(base + S::qx_bytes + S::qw_bytes + S::dx_bytes
                                            + S::a_bytes + S::b_bytes);
    }
};

// SmemBlock16 shares the qx/qw/dx prefix layout with SmemBlock, so the
// activation loaders (which only touch sQx/sDx and are keyed on VALS) work
// verbatim. This adaptor lets us call load_activation_tile[_cpasync] with a
// SmemBlock16 by reinterpreting its base pointer.
template <int VALS, int BM = TILE_M, int BN = TILE_N>
__device__ __forceinline__ SmemBlock<VALS, BM, BN> as_block32(
        const SmemBlock16<VALS, BM, BN>& s) {
    // The qx/qw/dx fields coincide byte-for-byte; the trailing A/B/sumqx differ
    // in count but the loaders never touch them, so reusing the prefix is safe.
    SmemBlock<VALS, BM, BN> b((char*)s.sQx);
    return b;
}

//------------------------------------------------------------------------------
// Per-16-chunk sumqx[TILE_M][NC] = Σ of the 16 staged qx per chunk per m-row.
// Must run after the activation staging + __syncthreads (reads sm.sQx).
//------------------------------------------------------------------------------
template <int VALS, int BM = TILE_M, int BN = TILE_N, int NT = BLOCK_SIZE>
__device__ __forceinline__ void compute_sumqx_tile16(
        const SmemBlock16<VALS, BM, BN>& sm, int tid) {
    constexpr int NC = VALS / 16;
    for (int idx = tid; idx < BM * NC; idx += NT) {
        const int mr = idx / NC, c = idx % NC;
        const int base = mr * VALS + 16 * c;
        int s = 0;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {            // 4 int4-vectors = 16 int8
            const int qx4 = *reinterpret_cast<const int*>(&sm.sQx[base + 4 * q]);
            s = __dp4a(qx4, 0x01010101, s);
        }
        sm.sSx[idx] = (float)s;
    }
}

//------------------------------------------------------------------------------
// Weight loader for the 16-subblock types: dequant one K super-block into
// centered int8 [TILE_N][VALS] plus per-16-chunk affine (A,B). The policy's
// (A,B) are constant across the 4 four-groups of a 16-chunk, so we store one
// (A,B) per chunk (NC) — a 4x smem reduction vs per-4-group.
//------------------------------------------------------------------------------
template <class P, int BM = TILE_M, int BN = TILE_N, int NT = BLOCK_SIZE>
__device__ __forceinline__ void load_weight_tile16(
        const uint8_t* __restrict__ Wbase, const SmemBlock16<P::VALS, BM, BN>& sm,
        int kb, int n_start, int N, int nblk, int tid) {
    constexpr int VALS = P::VALS;
    constexpr int NG   = VALS / 4;     // 4-value groups per super-block
    constexpr int NC   = VALS / 16;
    for (int idx = tid; idx < BN * NG; idx += NT) {
        const int nr = idx / NG, g = idx % NG;
        const int ng = n_start + nr;
        int Q = 0; float A = 0.0f, B = 0.0f;
        if (ng < N) {
            const void* wblk = Wbase + ((size_t)ng * nblk + kb) * P::BYTES;
            float dw, dmw; P::scales(wblk, dw, dmw);
            P::group(wblk, g, dw, dmw, Q, A, B);
        }
        *reinterpret_cast<int*>(&sm.sQw[nr * VALS + 4 * g]) = Q;
        // (A,B) constant within a 16-chunk == 4 four-groups; store once.
        if ((g & 3) == 0) {
            const int c = g >> 2;      // 16-chunk index
            sm.sA[nr * NC + c] = A;
            sm.sB[nr * NC + c] = B;
        }
    }
}

//------------------------------------------------------------------------------
// Affine-rescale epilogue for one 16-chunk: fold the int32 MMA partial into the
// float accumulator. Same partition_C (m,n) mapping as the k=32 path; only the
// scale indexing differs (A/B/sumqx per 16-chunk c, d_x per 32-block c/2).
//------------------------------------------------------------------------------
template <int VALS, int BM = TILE_M, int BN = TILE_N, int FC = FRAG_C,
          class CCoord, class FragP, class AccArr>
__device__ __forceinline__ void rescale_chunk_into_acc16(
        const SmemBlock16<VALS, BM, BN>& sm, const CCoord& tCcAcc, const FragP& rP,
        int c, int frag_c, AccArr& Acc) {
    constexpr int NC  = VALS / 16;
    constexpr int N32 = VALS / 32;
    const int c32 = c >> 1;            // containing 32-block (shared d_x)
    #pragma unroll
    for (int i = 0; i < FC; ++i) {
        if (i >= frag_c) break;
        const int m = (int)get<0>(tCcAcc(i));
        const int n = (int)get<1>(tCcAcc(i));
        const float dx = sm.sDx[m * N32 + c32];
        const float A  = sm.sA[n * NC + c];
        const float B  = sm.sB[n * NC + c];
        const float sx = sm.sSx[m * NC + c];
        const int   Pi = rP(i);
        Acc[i] += dx * (A * (float)Pi + B * sx);
    }
}

}  // namespace mmq_mma
}  // namespace layerstorm::compute
