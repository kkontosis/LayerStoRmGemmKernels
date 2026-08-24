// Device-routed grouped GGUF integer GEMM (mmvq + mmq) — the CUDA-graph
// capturable replacement for the host per-expert loop (TD-GG5-GROUPED-HOST-SYNC).
//
// Pipeline (NO host<-device sync anywhere):
//   1. Quantize A_base[total_tokens,K] -> Q8_1 ONCE (gguf_quantize_q8_1_cuda).
//   2a. mmvq (decode): grid (N, num_experts); each CTA reads its expert slice
//       [off, off+M_e) from expert_offsets on device and B_ptrs[e].
//   2b. mmq (prefill): a tiny prep kernel reads expert_offsets on device and
//       builds a flattened tile->(expert, m_start) map; the per-type grouped
//       mmq kernel grid is (n_tiles, T_max), each CTA routes off the map.
//
// On-device expert/tile routing pattern adapted from llama.cpp / ggml
// mul_mat_id (MIT, "The ggml authors", commit 1191758c / tag b9780):
//   ggml/src/ggml-cuda/{mmid.cu,mmq.cu,mmvq.cu,quantize.cu}. The per-tile GGUF
// dot-product math (dp4a mmvq / CuTe int8 mmq) is this project's own SM120
// kernels; only the routing (the "ids"/tile->expert map) is adapted from ggml.

#include "sm120/gemm/gguf/gguf_grouped_int.h"
#include "sm120/gemm/gguf/gguf_mmvq.h"          // gguf_quantize_q8_1_cuda, workspace
#include "sm120/gemm/gguf/gguf_int_policy.h"    // per-type integer policies
#include "sm120/gemm/gguf/gguf_mmvq_impl.cuh"   // k-split inner loop (A/B toggle)

#include "formats/q8_1_format.h"

#include <cstdlib>   // getenv (A/B toggle)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

namespace layerstorm::compute {

namespace {

namespace fmt = layerstorm::formats;
using fmt::block_q8_1;

// ── Tile-map prep kernel ───────────────────────────────────────────────────
// Reads expert_offsets on device and builds the flattened m-tile routing map:
//   tile_expert[t] = expert owning tile t   (-1 for padding tiles past the live
//                    count, so their CTAs early-return)
//   tile_mstart[t] = global row origin of tile t (off_e + i*tile_m)
// Single thread: num_experts is small (<=256) and this runs once per call; the
// serial scan keeps it trivially CUDA-graph capturable.
__global__ void build_mmq_tile_map(const int32_t* __restrict__ expert_offsets,
                                   int num_experts, int tile_m, int T_max,
                                   int* __restrict__ tile_expert,
                                   int* __restrict__ tile_mstart) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int t = 0;
    for (int e = 0; e < num_experts; ++e) {
        const int off = expert_offsets[e];
        const int M_e = expert_offsets[e + 1] - off;
        const int nt  = (M_e + tile_m - 1) / tile_m;
        for (int i = 0; i < nt && t < T_max; ++i) {
            tile_expert[t] = e;
            tile_mstart[t] = off + i * tile_m;
            ++t;
        }
    }
    for (; t < T_max; ++t) {        // pad the rest as dead tiles
        tile_expert[t] = -1;
        tile_mstart[t] = 0;
    }
}

// ── Grouped mmvq kernel (decode small-M) ───────────────────────────────────
// One warp per (output channel n, expert e). Mirrors the single-expert
// gguf_mmvq_kernel inner math, but the token slice [off, off+M_e) and the
// weight B_ptrs[e] are resolved on device from expert_offsets. Empty experts
// (M_e == 0) early-return.
//
// WPB warps per CTA (each warp owns one output channel n = blockIdx.x*WPB+warp);
// vs one warp per CTA this cuts the launched-CTA count WPB-fold and raises
// occupancy/packing for the (N x num_experts) decode grid (measured opt
// step). Warps are independent (no __syncthreads), so a
// warp whose n >= N simply returns.
//
// NOTE (TD-GGUF-MMVQ-DECODE-EFFICIENCY): the single-tensor gguf_mmvq_kernel was
// redesigned to a 4-warp k-split CTA (gguf_mmvq_impl.cuh) — a large win at the
// N-parallel attention/indexer shapes. The SAME k-split applied HERE REGRESSED
// the keeper52 routed-FFN decode (grid (N, num_experts), M_e≈1): measured MoE
// GPU-compute wall 58→129 ms/token, engine tok/s 5.79→4.63. At the routed FFN's
// N (2048/6144) × experts the grid-x already saturates the SMs, so the k-split
// only adds cross-warp __syncthreads + shared-mem reduction traffic for no extra
// parallelism. This variant therefore KEEPS the WPB=8 one-warp-per-channel
// layout; it still gains from the vectorized policy loads (shared, in
// gguf_int_policy.h).
template <class P, int WPB>
__global__ void __launch_bounds__(WPB * 32)
gguf_mmvq_grouped_kernel(GgufGroupedMmvqArgs args) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * WPB + warp;
    if (n >= args.N) return;
    const int e   = blockIdx.y;
    const int off = args.expert_offsets[e];
    const int M_e = args.expert_offsets[e + 1] - off;
    if (M_e <= 0) return;                              // empty expert
    // INV-MOE-OVERLAP (engine wave passes): a NULL expert weight pointer means
    // "skip this expert in THIS pass" (its rows are pre-zeroed by the caller).
    // CTA-uniform (e is uniform per CTA) -> barrier-safe early return. The old
    // zero-weight-buffer convention still works (writes zeros) but pays the
    // full latency-bound K-walk; NULL skips it.
    if (args.B_ptrs[e] == nullptr) return;

    const int K = args.K, N = args.N;
    const int nblk  = K / P::VALS;
    const int x_bpr = K / 32;
    const uint8_t* Wbase = reinterpret_cast<const uint8_t*>(args.B_ptrs[e])
                         + static_cast<size_t>(n) * nblk * P::BYTES;
    const block_q8_1* __restrict__ xq = args.xq;

    for (int m = 0; m < M_e; ++m) {
        const int gm = off + m;                       // global token row
        float acc = 0.0f;
        for (int sb = 0; sb < nblk; ++sb) {
            const void* wblk = Wbase + static_cast<size_t>(sb) * P::BYTES;
            float dw, dmw; P::scales(wblk, dw, dmw);
            for (int g = lane; g < P::VALS / 4; g += 32) {
                const int p0 = 4 * g;
                const int k  = sb * P::VALS + p0;
                const int b32 = k / 32, ofs = k % 32;
                const block_q8_1* xb = &xq[static_cast<size_t>(gm) * x_bpr + b32];
                const float dx = __half2float(__low2half(xb->ds));
                const int qx = *reinterpret_cast<const int*>(&xb->qs[ofs]);
                int Q; float A, B; P::group(wblk, g, dw, dmw, Q, A, B);
                const int dotQ  = __dp4a(Q, qx, 0);
                const int sumqx = __dp4a(qx, 0x01010101, 0);
                acc += dx * (A * dotQ + B * sumqx);
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
        if (lane == 0)
            args.D_base[static_cast<size_t>(gm) * N + n] = __float2bfloat16_rn(acc);
    }
}

// ── COMPACTED grouped mmvq (LS_GGUF_GROUPED_KSPLIT=5) ───────────────────────
// vLLM small-batch behaviour (fused_moe.py: EM = batch*topk*BLOCK_M caps the
// grid to the ACTIVE-expert count, not the global expert count). Adapted from
// vLLM (Apache-2.0, "the vLLM project"; ref/vllm/.../fused_moe/{fused_moe.py,
// moe_align_block_size.py} — the "only schedule blocks for experts that have
// tokens" idea; the dp4a GEMV math is this project's own).
//
// The default grouped launch is grid=(N/WPB, num_experts=256) but at decode only
// ~2-8 experts are resident → ~97% of the (N/WPB × 256) CTAs early-return. Here
// a device prep kernel scans expert_offsets ONCE and writes a DENSE list of the
// active (M_e>0) expert indices; the kernel launches grid=(N/WPB, cap) where
// cap = min(num_experts, total_tokens) is a HOST-known capture-time constant
// (total_tokens = permuted rows = an upper bound on active experts), and each
// CTA maps blockIdx.y → active_experts[blockIdx.y]. Grid rows past n_active hold
// -1 and early-return. Same weights read once; ~cap/num_experts fewer CTAs.
__global__ void build_active_expert_list(
        const int32_t* __restrict__ expert_offsets, int num_experts, int cap,
        int* __restrict__ active_experts /* [cap] */) {
    // One warp cooperatively stream-compacts the active (M_e>0) expert indices
    // into ascending order, then pads the tail with -1. Bit-identical output to
    // the former single-thread serial scan; ~32x fewer serial loads (the scan
    // was 10-14 us at decode, launched 3x/layer/GPU — see the engine handoff
    // spec/reports/DECODE_KERNEL_HUNT_HANDOFF.md 8b). Deterministic: within each
    // 32-expert chunk the ballot preserves lane (=expert) order, and chunks are
    // visited in ascending order, so writes land in the same order as before.
    const int lane = threadIdx.x & 31;
    int j = 0;  // running count of active experts written so far (warp-uniform)

    for (int base = 0; base < num_experts && j < cap; base += 32) {
        const int e = base + lane;
        int active = 0;
        if (e < num_experts)
            active = (expert_offsets[e + 1] - expert_offsets[e] > 0) ? 1 : 0;

        const unsigned mask = __ballot_sync(0xffffffffu, active);
        if (active) {
            // Compacted slot = j + (# active lanes strictly before me).
            const int prior = __popc(mask & ((1u << lane) - 1u));
            const int slot  = j + prior;
            if (slot < cap) active_experts[slot] = e;
        }
        j += __popc(mask);
    }

    // Pad remaining slots (warp-strided) with the early-return sentinel.
    for (int s = j + lane; s < cap; s += 32) active_experts[s] = -1;
}

template <class P, int WPB>
__global__ void __launch_bounds__(WPB * 32)
gguf_mmvq_grouped_compact_kernel(GgufGroupedMmvqArgs args,
                                 const int* __restrict__ active_experts) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * WPB + warp;
    if (n >= args.N) return;
    const int e = active_experts[blockIdx.y];         // dense active index
    if (e < 0) return;                                // padding row
    const int off = args.expert_offsets[e];
    const int M_e = args.expert_offsets[e + 1] - off;
    if (M_e <= 0) return;                             // defensive (shouldn't hit)
    // INV-MOE-OVERLAP (engine wave passes): a NULL expert weight pointer means
    // "skip this expert in THIS pass" (its rows are pre-zeroed by the caller).
    // CTA-uniform (e is uniform per CTA) -> barrier-safe early return. The old
    // zero-weight-buffer convention still works (writes zeros) but pays the
    // full latency-bound K-walk; NULL skips it.
    if (args.B_ptrs[e] == nullptr) return;

    const int K = args.K, N = args.N;
    const int nblk  = K / P::VALS;
    const int x_bpr = K / 32;
    const uint8_t* Wbase = reinterpret_cast<const uint8_t*>(args.B_ptrs[e])
                         + static_cast<size_t>(n) * nblk * P::BYTES;
    const block_q8_1* __restrict__ xq = args.xq;

    for (int m = 0; m < M_e; ++m) {
        const int gm = off + m;
        float acc = 0.0f;
        for (int sb = 0; sb < nblk; ++sb) {
            const void* wblk = Wbase + static_cast<size_t>(sb) * P::BYTES;
            float dw, dmw; P::scales(wblk, dw, dmw);
            for (int g = lane; g < P::VALS / 4; g += 32) {
                const int p0 = 4 * g;
                const int k  = sb * P::VALS + p0;
                const int b32 = k / 32, ofs = k % 32;
                const block_q8_1* xb = &xq[static_cast<size_t>(gm) * x_bpr + b32];
                const float dx = __half2float(__low2half(xb->ds));
                const int qx = *reinterpret_cast<const int*>(&xb->qs[ofs]);
                int Q; float A, B; P::group(wblk, g, dw, dmw, Q, A, B);
                const int dotQ  = __dp4a(Q, qx, 0);
                const int sumqx = __dp4a(qx, 0x01010101, 0);
                acc += dx * (A * dotQ + B * sumqx);
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
        if (lane == 0)
            args.D_base[static_cast<size_t>(gm) * N + n] = __float2bfloat16_rn(acc);
    }
}

// ── PIPELINED + ROW-TILED compact kernel (superblock-unrolled, MT rows) ──────
// Two changes over gguf_mmvq_grouped_compact_kernel, both pure scheduling (the
// per-row reduction order is BIT-IDENTICAL):
//   (1) The K-walk is UNROLLED over superblocks so UNROLL*GPL weight+activation
//       loads are in flight before their dp4a chains — memory-level parallelism
//       at M_e=1 decode, where the GEMV is memory-LATENCY-bound (~5% of DRAM
//       roof; the weight-read-floor claim was refuted — LayerStoRm3
//       spec/reports/DECODE_KERNEL_HUNT_HANDOFF.md §8c).
//   (2) MT-ROW REGISTER TILING: the expert's rows are processed in tiles of MT,
//       and each superblock's weight is DECODED ONCE and reused across the tile's
//       mt rows (llama.cpp mul_mat_vec_q `ncols_y`, MIT). The OLD kernel wrapped
//       the whole K-walk in `for m` → it re-read the 8.7 MB expert weight ONCE
//       PER ROW; at the M_e<=MT decode norm this now reads it ONCE. Since the
//       GEMV is memory-bound, that is up to an MT× DRAM-traffic cut for M_e>1.
//
// VALS-generic: each lane walks its groups exactly as the scalar kernel does
// (g = lane, lane+32, … < VALS/4; 256-VALS K-quants: 2 groups/lane; Q8_0
// VALS=32: 1 group on lanes 0-7). GROUPS_PER_LANE + MT are compile-time so all
// inner loops fully unroll. Deterministic; graph-capture-safe (no host sync).
template <class P, int WPB, int UNROLL, int MT>
__global__ void __launch_bounds__(WPB * 32)
gguf_mmvq_grouped_compact_pipe_kernel(GgufGroupedMmvqArgs args,
                                      const int* __restrict__ active_experts) {
    constexpr int NGROUP = P::VALS / 4;                       // groups/superblock
    constexpr int GPL    = (NGROUP + 31) / 32;                // groups per lane
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * WPB + warp;
    if (n >= args.N) return;
    const int e = active_experts[blockIdx.y];
    if (e < 0) return;
    const int off = args.expert_offsets[e];
    const int M_e = args.expert_offsets[e + 1] - off;
    if (M_e <= 0) return;
    // INV-MOE-OVERLAP (engine wave passes): a NULL expert weight pointer means
    // "skip this expert in THIS pass" (its rows are pre-zeroed by the caller).
    // CTA-uniform (e is uniform per CTA) -> barrier-safe early return. The old
    // zero-weight-buffer convention still works (writes zeros) but pays the
    // full latency-bound K-walk; NULL skips it.
    if (args.B_ptrs[e] == nullptr) return;

    const int K = args.K, N = args.N;
    const int nblk  = K / P::VALS;                 // superblocks along K
    const int x_bpr = K / 32;
    const uint8_t* Wbase = reinterpret_cast<const uint8_t*>(args.B_ptrs[e])
                         + static_cast<size_t>(n) * nblk * P::BYTES;
    const block_q8_1* __restrict__ xq = args.xq;

    // One superblock's contribution for a TILE of `mt` (<= MT) rows: the weight
    // is DECODED ONCE per group and reused across all mt rows (llama.cpp
    // mul_mat_vec_q `ncols_y` register-tiling, MIT — "The ggml authors", commit
    // 1191758c / tag b9780). At M_e<=MT this reads the 8.7 MB expert weight ONCE
    // instead of M_e times — the decisive win in this memory-bound regime, since
    // the row loop wrapping the whole K-walk previously re-read the weight per
    // row. Each row's per-superblock, per-group accumulation order is UNCHANGED,
    // so acc[j] is BIT-IDENTICAL to the single-row kernel.
    auto do_block_tiled = [&](int s, const block_q8_1* __restrict__ xrow0,
                              int mt, float* acc /* [MT] */) {
        const void* wblk = Wbase + static_cast<size_t>(s) * P::BYTES;
        float dw, dmw; P::scales(wblk, dw, dmw);
        #pragma unroll
        for (int gi = 0; gi < GPL; ++gi) {
            const int g = lane + 32 * gi;
            if (g >= NGROUP) continue;                 // idle lanes (e.g. Q8_0)
            const int p0 = 4 * g;
            const int k  = s * P::VALS + p0;
            const int b32 = k / 32, ofs = k % 32;
            int Q; float A, B; P::group(wblk, g, dw, dmw, Q, A, B);  // decode ONCE
            #pragma unroll
            for (int j = 0; j < MT; ++j) {
                if (j >= mt) break;
                const block_q8_1* xb = &xrow0[static_cast<size_t>(j) * x_bpr + b32];
                const float dx = __half2float(__low2half(xb->ds));
                const int qx = *reinterpret_cast<const int*>(&xb->qs[ofs]);
                const int dotQ  = __dp4a(Q, qx, 0);
                const int sumqx = __dp4a(qx, 0x01010101, 0);
                acc[j] += dx * (A * dotQ + B * sumqx);
            }
        }
    };

    // Process the expert's rows in tiles of MT (weight read once per tile).
    for (int m0 = 0; m0 < M_e; m0 += MT) {
        const int mt = min(M_e - m0, MT);
        const block_q8_1* __restrict__ xrow0 =
            xq + static_cast<size_t>(off + m0) * x_bpr;
        float acc[MT];
        #pragma unroll
        for (int j = 0; j < MT; ++j) acc[j] = 0.0f;

        int sb = 0;
        // Steady state: UNROLL superblocks at a time → UNROLL*GPL weight loads in
        // flight before the dp4a chains (MLP); each weight reused across mt rows.
        for (; sb + UNROLL <= nblk; sb += UNROLL) {
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) do_block_tiled(sb + u, xrow0, mt, acc);
        }
        for (; sb < nblk; ++sb) do_block_tiled(sb, xrow0, mt, acc);

        #pragma unroll
        for (int j = 0; j < MT; ++j) {
            float a = acc[j];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
            if (lane == 0 && j < mt)
                args.D_base[static_cast<size_t>(off + m0 + j) * N + n] =
                    __float2bfloat16_rn(a);
        }
    }
}

// ── cp.async SHARED-MEMORY-STAGED compact kernel (superblock double-buffer) ──
// Same compaction + MT-tiled math as gguf_mmvq_grouped_compact_pipe_kernel, but
// each warp's NEXT weight superblock is staged into SHARED MEMORY via cp.async
// (`cp.async.ca.shared.global`, non-blocking) while the CURRENT superblock is
// dp4a'd from smem — a true software-pipelined DOUBLE BUFFER. It gets the
// memory-level parallelism the k-split reached only by paying a cross-warp
// __syncthreads (a measured loss): here each warp keeps ONE channel and reduces
// via __shfl only, so there is NO reduction barrier — the loads-in-flight come
// from the 2-deep pipeline, not from cooperating warps. (Adapted from the
// project's own q4k_cutlass_gemm cp.async idiom + llama.cpp ggml-cuda staging
// pattern, MIT — "The ggml authors".) Per-row reduction order UNCHANGED →
// BIT-IDENTICAL to the non-staged kernel. Graph-capture-safe (no host sync).
//
// 2-deep is optimal: a smoke benchmark + engine A/B (see OPTIMIZATION_
// IMPROVEMENTS.md) showed depth 3/4 only add smem pressure without benefit —
// the K-walk's latency is already covered by a double buffer. So the depth is
// hard-coded 2 (no template knob).
//
// 256-VALS K-quants with 16-byte-aligned blocks ONLY (Q4_K 144, Q5_K 176):
// cp.async 16-byte copies need 16-aligned block bases, and SLOT == P::BYTES so
// every chunk is fully in-bounds (no tail guard). Q6_K/Q2_K/Q3_K/Q8_0 use the
// register-pipelined path.
template <class P, int WPB, int MT>
__global__ void __launch_bounds__(WPB * 32)
gguf_mmvq_grouped_compact_cpasync_kernel(GgufGroupedMmvqArgs args,
                                         const int* __restrict__ active_experts) {
    static_assert(P::VALS == 256, "cp.async staged path is for 256-VALS K-quants");
    static_assert(P::BYTES % 16 == 0,
        "cp.async 16-byte copy needs 16-byte-aligned block bases (BYTES%16==0); "
        "sub-16 types (Q6_K/Q2_K/Q3_K) must use the register-pipelined path");
    constexpr int NGROUP = P::VALS / 4;            // 64 groups/superblock
    constexpr int GPL    = (NGROUP + 31) / 32;     // 2 groups per lane
    constexpr int SLOT   = P::BYTES;               // 16-aligned by static_assert
    constexpr int NCHUNK = SLOT / 16;              // whole 16-byte cp.async chunks

    __shared__ uint8_t sW[2][WPB][SLOT];           // double buffer

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * WPB + warp;
    const bool active_n = (n < args.N);
    const int e = active_experts[blockIdx.y];
    if (e < 0) return;
    const int off = args.expert_offsets[e];
    const int M_e = args.expert_offsets[e + 1] - off;
    if (M_e <= 0) return;
    // INV-MOE-OVERLAP (engine wave passes): a NULL expert weight pointer means
    // "skip this expert in THIS pass" (its rows are pre-zeroed by the caller).
    // CTA-uniform (e is uniform per CTA) -> barrier-safe early return. The old
    // zero-weight-buffer convention still works (writes zeros) but pays the
    // full latency-bound K-walk; NULL skips it.
    if (args.B_ptrs[e] == nullptr) return;

    const int K = args.K, N = args.N;
    const int nblk  = K / P::VALS;
    const int x_bpr = K / 32;
    // Per-warp weight base for THIS output channel (guard OOB warps to n=0 so
    // their cp.async has a valid source; their results are never stored).
    const int n_safe = active_n ? n : 0;
    const uint8_t* Wbase = reinterpret_cast<const uint8_t*>(args.B_ptrs[e])
                         + static_cast<size_t>(n_safe) * nblk * P::BYTES;
    const block_q8_1* __restrict__ xq = args.xq;

    // Stage superblock `s` for THIS warp into sW[buf][warp] via cp.async.
    // SLOT == P::BYTES (16-aligned), so every 16-byte chunk is fully in-bounds
    // within the block — no tail over-copy, no size guard needed.
    auto stage = [&](int s, int buf) {
        const uint8_t* gsrc = Wbase + static_cast<size_t>(s) * P::BYTES;
        uint8_t* dst = &sW[buf][warp][0];
        if (lane < NCHUNK) {
            uint32_t saddr = __cvta_generic_to_shared(dst + lane * 16);
            const uint8_t* src = gsrc + lane * 16;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(saddr), "l"(src));
        }
    };

    // Process rows in tiles of MT; weight decoded once per group, reused across
    // the tile's mt rows (from smem). Double-buffered over superblocks.
    for (int m0 = 0; m0 < M_e; m0 += MT) {
        const int mt = min(M_e - m0, MT);
        const block_q8_1* __restrict__ xrow0 =
            xq + static_cast<size_t>(off + m0) * x_bpr;
        float acc[MT];
        #pragma unroll
        for (int j = 0; j < MT; ++j) acc[j] = 0.0f;

        // Prologue: kick off superblock 0.
        stage(0, 0);
        asm volatile("cp.async.commit_group;\n" :::);

        for (int sb = 0; sb < nblk; ++sb) {
            const int cur = sb & 1;
            // Issue the NEXT superblock's copy (if any) before consuming `sb`,
            // then wait until only 1 group remains pending (i.e. `sb` has landed).
            if (sb + 1 < nblk) {
                stage(sb + 1, (sb + 1) & 1);
                asm volatile("cp.async.commit_group;\n" :::);
                asm volatile("cp.async.wait_group 1;\n" :::);
            } else {
                asm volatile("cp.async.wait_group 0;\n" :::);
            }
            __syncwarp();

            // Consume superblock `sb` from smem (same math as do_block_tiled).
            const void* wblk = &sW[cur][warp][0];
            float dw, dmw; P::scales(wblk, dw, dmw);
            #pragma unroll
            for (int gi = 0; gi < GPL; ++gi) {
                const int g = lane + 32 * gi;
                if (g >= NGROUP) continue;
                const int p0 = 4 * g;
                const int k  = sb * P::VALS + p0;
                const int b32 = k / 32, ofs = k % 32;
                int Q; float A, B; P::group(wblk, g, dw, dmw, Q, A, B);
                #pragma unroll
                for (int j = 0; j < MT; ++j) {
                    if (j >= mt) break;
                    const block_q8_1* xb = &xrow0[static_cast<size_t>(j) * x_bpr + b32];
                    const float dx = __half2float(__low2half(xb->ds));
                    const int qx = *reinterpret_cast<const int*>(&xb->qs[ofs]);
                    const int dotQ  = __dp4a(Q, qx, 0);
                    const int sumqx = __dp4a(qx, 0x01010101, 0);
                    acc[j] += dx * (A * dotQ + B * sumqx);
                }
            }
        }

        #pragma unroll
        for (int j = 0; j < MT; ++j) {
            float a = acc[j];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
            if (lane == 0 && active_n && j < mt)
                args.D_base[static_cast<size_t>(off + m0 + j) * N + n] =
                    __float2bfloat16_rn(a);
        }
    }
}

// ── A/B diagnostic k-split grouped kernel (env LS_GGUF_GROUPED_KSPLIT=1) ─────
// NW=4 warps cooperate on ONE output channel with the K-dim split across all
// lanes (the single-tensor redesign layout). Kept behind a runtime toggle,
// DEFAULT OFF — this layout is a PROVEN e2e LOSS for the routed FFN and must
// NOT be enabled in production. A clean idle-box A/B (alternating WPB=8/k-split
// ×2, TD-GGUF-MMVQ-DECODE-EFFICIENCY §7b) measured keeper52 EP=4 at 7.31-7.37
// tok/s (WPB=8) vs 4.22-4.23 (k-split) — a reproducible ~43% regression, MoE
// GPU-compute wall 52 -> 150 ms/token. The k-split is FASTER per-launch on an
// empty GPU but launches 8x more CTAs with smem+__syncthreads, which serializes
// inside the captured finalize FFN graph. The toggle is retained only as a
// documented diagnostic / regression guard.
template <class P, int NW = gguf_mmvq_impl::kWarps, int MT = gguf_mmvq_impl::kMTile>
__global__ void __launch_bounds__(NW * 32)
gguf_mmvq_grouped_ksplit_kernel(GgufGroupedMmvqArgs args) {
    namespace impl = gguf_mmvq_impl;
    const int n = blockIdx.x;
    if (n >= args.N) return;
    const int e   = blockIdx.y;
    const int off = args.expert_offsets[e];
    const int M_e = args.expert_offsets[e + 1] - off;
    if (M_e <= 0) return;
    // INV-MOE-OVERLAP (engine wave passes): a NULL expert weight pointer means
    // "skip this expert in THIS pass" (its rows are pre-zeroed by the caller).
    // CTA-uniform (e is uniform per CTA) -> barrier-safe early return. The old
    // zero-weight-buffer convention still works (writes zeros) but pays the
    // full latency-bound K-walk; NULL skips it.
    if (args.B_ptrs[e] == nullptr) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int K = args.K, N = args.N;
    const int nblk = K / P::VALS, x_bpr = K / 32, total_groups = K / 4;
    const uint8_t* Wrow = reinterpret_cast<const uint8_t*>(args.B_ptrs[e])
                        + static_cast<size_t>(n) * nblk * P::BYTES;
    __shared__ float red[NW][MT];
    for (int m0 = 0; m0 < M_e; m0 += MT) {
        const int mt = min(M_e - m0, MT);
        float acc[MT];
        #pragma unroll
        for (int j = 0; j < MT; ++j) acc[j] = 0.0f;
        impl::accumulate<P, NW, MT>(
            Wrow, args.xq + static_cast<size_t>(off + m0) * x_bpr, x_bpr,
            total_groups, mt, tid, acc);
        impl::reduce_store<NW, MT>(
            acc, red, mt, lane, warp,
            args.D_base + static_cast<size_t>(off + m0) * N + n, N);
        if (m0 + MT < M_e) __syncthreads();
    }
}

// COMPACT k-split (LS_GGUF_GROUPED_KSPLIT=6): the NW=4 llama.cpp-structure
// k-split (4 warps cooperate on ONE output channel, K striped across 128 threads
// = 4× the memory-level parallelism of the 1-warp-per-channel kernels), but on
// the ACTIVE-expert grid (N, cap) instead of the old (N, num_experts=256). The
// original NW=4 k-split (KSPLIT=1) was measured an e2e LOSS — but on the STALE
// 256-wide grid, which drowned the win in ~256/cap× empty-CTA scheduling +
// barrier cost. This variant removes that confound: grid.y = cap ≈ active
// experts, so it is the honest test of whether llama.cpp's 4-warp-per-channel
// K-parallelism (the path to >7%-of-peak weight bandwidth) beats the
// barrier-free 1-warp WPB layout once the empty-CTA bloat is gone. blockIdx.y
// maps through active_experts[]; padding rows (-1) early-return.
template <class P, int NW, int MT>
__global__ void __launch_bounds__(NW * 32)
gguf_mmvq_grouped_ksplit_compact_kernel(GgufGroupedMmvqArgs args,
                                        const int* __restrict__ active_experts) {
    namespace impl = gguf_mmvq_impl;
    const int n = blockIdx.x;
    if (n >= args.N) return;
    const int e = active_experts[blockIdx.y];
    if (e < 0) return;                       // padding row
    const int off = args.expert_offsets[e];
    const int M_e = args.expert_offsets[e + 1] - off;
    if (M_e <= 0) return;
    // INV-MOE-OVERLAP (engine wave passes): a NULL expert weight pointer means
    // "skip this expert in THIS pass" (its rows are pre-zeroed by the caller).
    // CTA-uniform (e is uniform per CTA) -> barrier-safe early return. The old
    // zero-weight-buffer convention still works (writes zeros) but pays the
    // full latency-bound K-walk; NULL skips it.
    if (args.B_ptrs[e] == nullptr) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int K = args.K, N = args.N;
    const int nblk = K / P::VALS, x_bpr = K / 32, total_groups = K / 4;
    const uint8_t* Wrow = reinterpret_cast<const uint8_t*>(args.B_ptrs[e])
                        + static_cast<size_t>(n) * nblk * P::BYTES;
    __shared__ float red[NW][MT];
    for (int m0 = 0; m0 < M_e; m0 += MT) {
        const int mt = min(M_e - m0, MT);
        float acc[MT];
        #pragma unroll
        for (int j = 0; j < MT; ++j) acc[j] = 0.0f;
        impl::accumulate<P, NW, MT>(
            Wrow, args.xq + static_cast<size_t>(off + m0) * x_bpr, x_bpr,
            total_groups, mt, tid, acc);
        impl::reduce_store<NW, MT>(
            acc, red, mt, lane, warp,
            args.D_base + static_cast<size_t>(off + m0) * N + n, N);
        if (m0 + MT < M_e) __syncthreads();
    }
}

// Milder k-split (LS_GGUF_GROUPED_KSPLIT=2): 4 warps/CTA = 2 channels × 2 warps.
// Warps {0,1} cooperate (K split 2 ways) on channel n0 = blockIdx.x*2+0; warps
// {2,3} on n0+1. Grid (N/2, E) → only 2× the WPB=8 CTA count (vs 8× for NW=4),
// testing whether a gentler CTA-count increase keeps the per-channel k-parallel
// win without the in-graph serialization that sank NW=4. Independent channels →
// no cross-channel sync; each channel uses its own smem reduce slot.
template <class P, int MT = gguf_mmvq_impl::kMTile>
__global__ void __launch_bounds__(128)
gguf_mmvq_grouped_ksplit2_kernel(GgufGroupedMmvqArgs args) {
    namespace impl = gguf_mmvq_impl;
    const int tid = threadIdx.x;              // 0..127
    const int chan = tid >> 6;               // 0/1: which of the 2 channels
    const int sub  = (tid >> 5) & 1;         // 0/1: warp within the channel (NW=2)
    const int lane = tid & 31;
    const int e   = blockIdx.y;
    const int off = args.expert_offsets[e];
    const int M_e = args.expert_offsets[e + 1] - off;
    if (M_e <= 0) return;                     // CTA-uniform (e is per-CTA) — safe
    // INV-MOE-OVERLAP (engine wave passes): a NULL expert weight pointer means
    // "skip this expert in THIS pass" (its rows are pre-zeroed by the caller).
    // CTA-uniform (e is uniform per CTA) -> barrier-safe early return. The old
    // zero-weight-buffer convention still works (writes zeros) but pays the
    // full latency-bound K-walk; NULL skips it.
    if (args.B_ptrs[e] == nullptr) return;

    // NOTE: `reduce_store` has an internal __syncthreads(), so a per-CHANNEL
    // early-return on out-of-range n would deadlock for odd N (one channel exits,
    // the other waits at the barrier). Instead every thread runs the loop + syncs;
    // an out-of-range channel just skips the global-memory WRITE (guarded in the
    // store). Reads use a clamped row so they stay in-bounds.
    const int n = blockIdx.x * 2 + chan;
    const bool nvalid = (n < args.N);
    const int  n_rd   = nvalid ? n : (args.N - 1);   // clamp reads in-bounds

    const int K = args.K, N = args.N;
    const int nblk = K / P::VALS, x_bpr = K / 32, total_groups = K / 4;
    const uint8_t* Wrow = reinterpret_cast<const uint8_t*>(args.B_ptrs[e])
                        + static_cast<size_t>(n_rd) * nblk * P::BYTES;
    // 2 channels × NW=2 reduce slots each.
    __shared__ float red[2][2][MT];
    // Local tid within this channel's 2-warp group (0..63) so the shared impl
    // helpers (which key on tid = 32*warp + lane) see a clean 2-warp CTA.
    const int local_tid = sub * 32 + lane;

    for (int m0 = 0; m0 < M_e; m0 += MT) {
        const int mt = min(M_e - m0, MT);
        float acc[MT];
        #pragma unroll
        for (int j = 0; j < MT; ++j) acc[j] = 0.0f;
        impl::accumulate<P, /*NW=*/2, MT>(
            Wrow, args.xq + static_cast<size_t>(off + m0) * x_bpr, x_bpr,
            total_groups, mt, local_tid, acc);
        // Send out-of-range channels to a scratch sink so the internal
        // __syncthreads() still fires uniformly but no valid row is clobbered.
        __shared__ __nv_bfloat16 sink[MT];
        __nv_bfloat16* out = nvalid
            ? (args.D_base + static_cast<size_t>(off + m0) * N + n)
            : sink;
        impl::reduce_store</*NW=*/2, MT>(
            acc, red[chan], mt, lane, sub, out, nvalid ? N : 1);
        if (m0 + MT < M_e) __syncthreads();
    }
}

// Grouped GEMV layout selector (env LS_GGUF_GROUPED_KSPLIT, read ONCE — static,
// so the hot path stays branch-free and CUDA-graph-capture-stable). These are
// kept as PERMANENT, re-selectable alternatives, not throwaway experiments: the
// best layout is CONTEXTUAL to the surrounding decode pipeline, and the pipeline
// shifts as other kernels are optimized (a layout that loses today can win once
// its FFN-graph neighbours get cheaper). Re-A/B on the CURRENT binary before
// trusting any prior verdict. See TD-GGUF-MMVQ-DECODE-EFFICIENCY §7b + the decode
// kernel-hunt handoff.
// The routed FFN GEMV runs at only ~7% of peak weight bandwidth (~125 GB/s vs
// ~1790) — it is OCCUPANCY-bound (WPB=8 → grid (N/8, E) = 256 CTAs on 170 SMs),
// NOT bandwidth- or compute-bound. The k-split variants (1,2) raise the CTA
// count but ADD a smem + __syncthreads reduction barrier, which serializes in
// the captured FFN graph (measured MoE-wall regression PROPORTIONAL to the CTA
// multiplier: WPB8 52 → NW2 106 → NW4 150 ms/token). The WPB kernel itself is
// BARRIER-FREE (warps independent, only warp-local __shfl), so LOWERING WPB
// raises the CTA count the SAME way WITHOUT the barrier — the clean way to test
// whether occupancy alone (sans reduction cost) recovers the 7%-of-peak.
//   0 / unset : WPB=8, grid (N/8, E) = 256 CTAs  — default (best barrier'd-A/B)
//   1         : NW=4 k-split, grid (N, E) = 8x CTAs   — PROVEN LOSS (barrier)
//   2         : NW=2 k-split, grid (N/2, E) = 2x CTAs — PROVEN LOSS (barrier)
//   3         : WPB=4, grid (N/4, E) = 2x CTAs, BARRIER-FREE
//   4         : WPB=2, grid (N/2, E) = 4x CTAs, BARRIER-FREE
//   5         : COMPACT, grid (N/8, cap) — cap=min(E, total_tokens) ACTIVE experts
//               only (vLLM small-batch); ~E/cap fewer CTAs, no empty-expert waste
enum class GroupedLayout { kWpb8 = 0, kKsplit4 = 1, kKsplit2 = 2, kWpb4 = 3,
                           kWpb2 = 4, kCompact = 5 };

inline GroupedLayout grouped_layout() {
    static const GroupedLayout sel = [] {
        const char* v = std::getenv("LS_GGUF_GROUPED_KSPLIT");
        // DEFAULT (unset) = kCompact: the measured best (+6.7% median tok/s vs
        // WPB=8 on keeper52 EP=4, golden 12089, graph-capture-safe). The other
        // layouts stay explicitly selectable as documented alternatives (the
        // best layout is pipeline-contextual — re-A/B on the CURRENT binary).
        if (!v || !v[0]) return GroupedLayout::kCompact;
        if (v[0] == '0') return GroupedLayout::kWpb8;      // legacy default
        if (v[0] == '1') return GroupedLayout::kKsplit4;
        if (v[0] == '2') return GroupedLayout::kKsplit2;
        if (v[0] == '3') return GroupedLayout::kWpb4;
        if (v[0] == '4') return GroupedLayout::kWpb2;
        if (v[0] == '5') return GroupedLayout::kCompact;
        return GroupedLayout::kCompact;
    }();
    return sel;
}

// Superblock-unrolled pipelined COMPACT GEMV for 256-VALS K-quants (Q5_K/Q6_K).
// Read ONCE (static → branch-free hot path, capture-stable). Default ON: the
// pipelined kernel raises memory-level parallelism at M_e=1 decode where the
// GEMV is latency-bound (the kernel is bit-identical to the non-pipelined
// compact path, so this is a pure scheduling change). Set LS_GGUF_GROUPED_PIPE=0
// to A/B against the scalar compact kernel on the SAME binary (Trap #2/#3 —
// always verify e2e on an idle box, not the isolated bench).
inline bool pipeline_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("LS_GGUF_GROUPED_PIPE");
        return !(v && v[0] == '0');
    }();
    return on;
}

// cp.async shared-memory-staged variant (256-VALS K-quants). Read ONCE (static,
// capture-stable). DEFAULT ON — MEASURED a win e2e (keeper52 EP=4, 3-run idle-box
// A/B: MoE GPU-compute wall 40.2 → 38.7-39.2 ms/tok vs the register pipe, median
// 10.55-10.59 vs 10.38 tok/s, golden 12089 bit-identical). It gets the k-split's
// memory-level parallelism (loads in flight) WITHOUT the cross-warp reduction
// barrier that made k-split a loss (each warp keeps one channel, __shfl-only
// reduce). Set LS_GGUF_GROUPED_CPASYNC=0 to A/B against the register pipe on the
// SAME binary. Only for BYTES%16==0 K-quants (Q4_K/Q5_K); others use the pipe.
inline bool cpasync_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("LS_GGUF_GROUPED_CPASYNC");
        return !(v && v[0] == '0');
    }();
    return on;
}

// COMPACT k-split A/B (LS_GGUF_GROUPED_KSPLIT_COMPACT=1): route the compact path
// to the NW=4 llama.cpp-structure k-split on the ACTIVE-expert grid (N, cap).
// Read ONCE. DEFAULT OFF — this re-tests the k-split (proven a loss on the STALE
// 256-wide grid) now that the compact grid removes the empty-CTA confound. It is
// the honest measurement of whether 4-warp-per-channel K-parallelism (llama.cpp's
// route to >7%-of-peak bandwidth) beats the barrier-free WPB layout in-engine.
inline bool ksplit_compact_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("LS_GGUF_GROUPED_KSPLIT_COMPACT");
        return (v && v[0] == '1');
    }();
    return on;
}

// E=1 GEMV fast path (LS_GGUF_E1_KSPLIT, DEFAULT ON; =0 disables): dense-FFN
// and shared-expert grouped calls arrive with num_experts == 1, where the
// channel-packed WPB=8 grid degenerates to (N/8, 1) ≈ 256 CTAs — an SM
// under-fill that measured 96 µs vs a ~10 µs floor for the Q8_0 shared expert
// (LayerStoRm3 bench/gemv_chain, handoff §12; llama.cpp's mmvq geometry holds
// the DRAM floor at these shapes). Routes these through the NW=4 k-split
// kernel on the (N, 1) grid — the llama.cpp mul_mat_vec_q launch geometry
// (see the gguf_mmvq_impl.cuh attribution) on the grouped B_ptrs plumbing.
// GEMV regime only (total_tokens <= 8); prefill M keeps the tiled paths.
// DEFAULT flipped OFF→ON 2026-07-20 (perf saga, spec/reports/PERF_SAGA.md):
// under KEEPER52_REEF_ORCH the EP-combine is gated by the 5090 rank's OWN FFN
// chain (nsys: dev2 feeds each AllReduce with a 2-3 µs gap; the 5080 fold
// arrives ~300 µs early) — NOT the 5080s as in the §12d era — so the
// shared-expert shave now converts into wall. 5-seed in-process keeper eval:
// ON wins 4/5 seeds, median-of-medians 10.88 vs 9.48 tok/s, hit-adjusted MoE
// wall −4.3 ms/tok at equal hit. The §12c single-seed −2% was trajectory luck
// (Trap #10). FP reduction-order perturbation documented: non-TQ golden top1
// 0.4836 → 0.4750-class (token 12089 unchanged, correct range).
inline bool e1_ksplit_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("LS_GGUF_E1_KSPLIT");
        return !(v && v[0] == '0');
    }();
    return on;
}

// active_scratch: device int[cap] for the compact layout (else may be null).
// total_tokens: host-known upper bound on active experts (cap = min(E, tt)).
template <class P>
void launch_mmvq_grouped(const GgufGroupedMmvqArgs& args, int num_experts,
                         int total_tokens, int* active_scratch,
                         cudaStream_t stream) {
    if (grouped_layout() == GroupedLayout::kCompact && active_scratch) {
        // Cap = min(num_experts, total_tokens): the max distinct active experts
        // (each permuted row belongs to one expert). Host-known → graph-safe.
        const int cap = num_experts < total_tokens ? num_experts : total_tokens;
        if (cap > 0) {
            build_active_expert_list<<<1, 32, 0, stream>>>(
                args.expert_offsets, num_experts, cap, active_scratch);
            constexpr int WPB = 8;
            dim3 grid((args.N + WPB - 1) / WPB, cap);
            // E=1 GEMV fast path (dense/shared FFN): full-N k-split grid — see
            // e1_ksplit_enabled(). MT=1 decode variant avoids dead guarded rows.
            if (e1_ksplit_enabled() && num_experts == 1 && total_tokens <= 8) {
                constexpr int NW = 4;
                dim3 kgrid(args.N, cap);
                if (total_tokens == 1)
                    gguf_mmvq_grouped_ksplit_compact_kernel<P, NW, 1>
                        <<<kgrid, dim3(NW * 32), 0, stream>>>(args, active_scratch);
                else
                    gguf_mmvq_grouped_ksplit_compact_kernel<P, NW,
                                                            gguf_mmvq_impl::kMTile>
                        <<<kgrid, dim3(NW * 32), 0, stream>>>(args, active_scratch);
                return;
            }
            // A/B: NW=4 llama.cpp-structure k-split on the compact grid (N, cap).
            // 4 warps cooperate on ONE channel (4× K-parallelism) → the honest
            // re-test of the k-split now the empty-CTA bloat is gone.
            if (ksplit_compact_enabled()) {
                constexpr int NW = 4, MT = gguf_mmvq_impl::kMTile;
                dim3 kgrid(args.N, cap);
                gguf_mmvq_grouped_ksplit_compact_kernel<P, NW, MT>
                    <<<kgrid, dim3(NW * 32), 0, stream>>>(args, active_scratch);
            } else if constexpr (P::kPipeline) {
                // Superblock-unrolled + MT-row-tiled variant: raises MLP and
                // reads each expert weight ONCE per MT rows (vs once per row) —
                // the decisive win in the memory-bound M_e<=4 decode regime.
                // Bit-identical reduction order per row. LS_GGUF_GROUPED_PIPE=0
                // falls back to the scalar compact kernel.
                constexpr int UNROLL = 4;
                constexpr int MT     = 4;   // row tile (M_e chunked in 4s)
                // cp.async 16-byte staging REQUIRES 16-byte-aligned block bases.
                // Blocks are packed contiguously so the base alignment == the
                // block-size alignment: only types with BYTES % 16 == 0 (Q4_K
                // 144, Q5_K 176) qualify. Q6_K(210)/Q2_K(84)/Q3_K(110)/Q8_0(34)
                // are sub-16 aligned → cp.async would read misaligned garbage
                // (measured: cosine=nan). They keep the register-pipelined path.
                constexpr bool kCpAsyncOk = (P::VALS == 256) && (P::BYTES % 16 == 0);
                if constexpr (kCpAsyncOk) {
                    // cp.async smem 2-deep double buffer — MLP without a
                    // cross-warp barrier (the measured win; 2-deep is optimal).
                    if (cpasync_enabled())
                        gguf_mmvq_grouped_compact_cpasync_kernel<P, WPB, MT>
                            <<<grid, dim3(WPB * 32), 0, stream>>>(args, active_scratch);
                    else if (pipeline_enabled())
                        gguf_mmvq_grouped_compact_pipe_kernel<P, WPB, UNROLL, MT>
                            <<<grid, dim3(WPB * 32), 0, stream>>>(args, active_scratch);
                    else
                        gguf_mmvq_grouped_compact_kernel<P, WPB>
                            <<<grid, dim3(WPB * 32), 0, stream>>>(args, active_scratch);
                } else if (pipeline_enabled()) {
                    // Sub-16-aligned or VALS!=256 types: register pipe only.
                    gguf_mmvq_grouped_compact_pipe_kernel<P, WPB, UNROLL, MT>
                        <<<grid, dim3(WPB * 32), 0, stream>>>(args, active_scratch);
                } else {
                    gguf_mmvq_grouped_compact_kernel<P, WPB>
                        <<<grid, dim3(WPB * 32), 0, stream>>>(args, active_scratch);
                }
            } else {
                gguf_mmvq_grouped_compact_kernel<P, WPB>
                    <<<grid, dim3(WPB * 32), 0, stream>>>(args, active_scratch);
            }
        }
        return;
    }
    switch (grouped_layout()) {
        case GroupedLayout::kKsplit4: {
            // NW=4 warps cooperate on one channel; grid (N, E) = 8x WPB=8 CTAs.
            dim3 grid(args.N, num_experts);
            gguf_mmvq_grouped_ksplit_kernel<P, 4>
                <<<grid, dim3(4 * 32), 0, stream>>>(args);
            return;
        }
        case GroupedLayout::kKsplit2: {
            // NW=2 warps/channel, 2 channels/CTA → grid (N/2, E) = 2x WPB=8 CTAs.
            // (2 warps × 2 channels = 128 threads; each channel's K split 2 ways.)
            constexpr int CH = 2;  // channels per CTA
            dim3 grid((args.N + CH - 1) / CH, num_experts);
            gguf_mmvq_grouped_ksplit2_kernel<P><<<grid, dim3(CH * 2 * 32), 0, stream>>>(args);
            return;
        }
        case GroupedLayout::kWpb4: {
            // Barrier-free: 4 warps (channels) per CTA → grid (N/4, E) = 2x CTAs.
            constexpr int WPB = 4;
            dim3 grid((args.N + WPB - 1) / WPB, num_experts);
            gguf_mmvq_grouped_kernel<P, WPB><<<grid, dim3(WPB * 32), 0, stream>>>(args);
            return;
        }
        case GroupedLayout::kWpb2: {
            // Barrier-free: 2 warps (channels) per CTA → grid (N/2, E) = 4x CTAs.
            constexpr int WPB = 2;
            dim3 grid((args.N + WPB - 1) / WPB, num_experts);
            gguf_mmvq_grouped_kernel<P, WPB><<<grid, dim3(WPB * 32), 0, stream>>>(args);
            return;
        }
        case GroupedLayout::kWpb8:
        default: {
            constexpr int WPB = 8;   // warps (output channels) per CTA (default)
            dim3 grid((args.N + WPB - 1) / WPB, num_experts);
            gguf_mmvq_grouped_kernel<P, WPB><<<grid, dim3(WPB * 32), 0, stream>>>(args);
            return;
        }
    }
}

void dispatch_mmvq_grouped(GgufType type, const GgufGroupedMmvqArgs& args,
                           int num_experts, int total_tokens, int* active_scratch,
                           cudaStream_t stream) {
    switch (type) {
        case GgufType::Q2_K: launch_mmvq_grouped<gguf_int::Q2K>(args, num_experts, total_tokens, active_scratch, stream);  break;
        case GgufType::Q3_K: launch_mmvq_grouped<gguf_int::Q3K>(args, num_experts, total_tokens, active_scratch, stream);  break;
        case GgufType::Q4_K: launch_mmvq_grouped<gguf_int::Q4K>(args, num_experts, total_tokens, active_scratch, stream);  break;
        case GgufType::Q5_K: launch_mmvq_grouped<gguf_int::Q5K>(args, num_experts, total_tokens, active_scratch, stream);  break;
        case GgufType::Q6_K: launch_mmvq_grouped<gguf_int::Q6K>(args, num_experts, total_tokens, active_scratch, stream);  break;
        case GgufType::Q8_0: launch_mmvq_grouped<gguf_int::Q8_0>(args, num_experts, total_tokens, active_scratch, stream); break;
        case GgufType::MXFP4: launch_mmvq_grouped<gguf_int::Mxfp4>(args, num_experts, total_tokens, active_scratch, stream); break;
    }
}

void dispatch_mmq_grouped(GgufType type, const GgufGroupedMmqArgs& args,
                          bool use_small, int T_max, cudaStream_t stream) {
    switch (type) {
        case GgufType::Q4_K: gguf_mmq_mma_grouped_launch_q4k(args, use_small, T_max, stream);  break;
        case GgufType::Q5_K: gguf_mmq_mma_grouped_launch_q5k(args, use_small, T_max, stream);  break;
        case GgufType::Q8_0: gguf_mmq_mma_grouped_launch_q8_0(args, use_small, T_max, stream); break;
        case GgufType::MXFP4: gguf_mmq_mma_grouped_launch_mxfp4(args, use_small, T_max, stream); break;
        case GgufType::Q6_K: gguf_mmq_mma_grouped_launch_q6k(args, use_small, T_max, stream);  break;
        case GgufType::Q2_K: gguf_mmq_mma_grouped_launch_q2k(args, use_small, T_max, stream);  break;
        case GgufType::Q3_K: gguf_mmq_mma_grouped_launch_q3k(args, use_small, T_max, stream);  break;
    }
}

// Per-expert mmvq/mmq cut-over: decode-like (avg M_e <= 8) -> mmvq, else mmq.
constexpr int kGroupedMmvqAvgM = 8;
// mmq small-tile (64x128) cut-over by average per-expert M.
constexpr int kGroupedMmqSmallAvgM = 64;

}  // namespace

void launch_gguf_grouped_int(
    GgufType type, int num_experts, int N, int K, int total_tokens,
    const __nv_bfloat16* A_base, __nv_bfloat16* D_base,
    const int32_t* expert_offsets, const void* const* B_ptrs,
    void* workspace, size_t workspace_bytes,
    GgufGroupedIntForce force, cudaStream_t stream) {

    if (num_experts <= 0 || total_tokens <= 0) return;   // nothing to do
    if (K % 32 != 0)
        throw std::runtime_error("GGUF grouped int: K must be divisible by 32");
    if (workspace == nullptr)
        throw std::runtime_error("GGUF grouped int: workspace required");

    // ── 1. Quantize all activations to Q8_1 ONCE (no per-expert loop). ──────
    auto* xq = reinterpret_cast<block_q8_1*>(workspace);
    const size_t q8_bytes = static_cast<size_t>(total_tokens) * (K / 32)
                          * sizeof(block_q8_1);
    if (workspace_bytes < q8_bytes)
        throw std::runtime_error("GGUF grouped int: workspace too small for Q8_1");
    gguf_quantize_q8_1_cuda(A_base, xq, total_tokens, K, stream);

    // ── 2. One up-front mmvq-vs-mmq choice by overall token count. ──────────
    const int avg_m = total_tokens / num_experts;
    bool use_mmq = (avg_m > kGroupedMmvqAvgM);
    if (force == GgufGroupedIntForce::Mmvq) use_mmq = false;
    if (force == GgufGroupedIntForce::Mmq)  use_mmq = true;

    if (!use_mmq) {
        GgufGroupedMmvqArgs args;
        args.N = N; args.K = K; args.xq = xq;
        args.expert_offsets = expert_offsets; args.B_ptrs = B_ptrs;
        args.D_base = D_base;
        // Active-expert scratch for the COMPACT layout (int[cap] in the ws tail).
        // cap = min(num_experts, total_tokens). Only carved/used when the layout
        // is kCompact AND the ws has room; else pass null (non-compact ignores it).
        const int cap = num_experts < total_tokens ? num_experts : total_tokens;
        const size_t act_off = (q8_bytes + 15) & ~static_cast<size_t>(15);
        int* active_scratch = nullptr;
        if (workspace_bytes >= act_off + static_cast<size_t>(cap) * sizeof(int))
            active_scratch = reinterpret_cast<int*>(
                static_cast<char*>(workspace) + act_off);
        dispatch_mmvq_grouped(type, args, num_experts, total_tokens,
                              active_scratch, stream);
        return;
    }

    // ── mmq: build the tile->expert map in the workspace tail, then route. ──
    const bool use_small = (avg_m <= kGroupedMmqSmallAvgM);
    const int tile_m = gguf_grouped_mmq_tile_m(use_small);
    const int T_max  = gguf_grouped_mmq_tmax(total_tokens, num_experts, tile_m);

    size_t map_off = (q8_bytes + 15) & ~static_cast<size_t>(15);  // align ints
    const size_t need = map_off + 2ull * static_cast<size_t>(T_max) * sizeof(int);
    if (workspace_bytes < need)
        throw std::runtime_error(
            "GGUF grouped int (mmq): workspace too small for tile map (have " +
            std::to_string(workspace_bytes) + ", need " + std::to_string(need) + ")");

    int* tile_expert = reinterpret_cast<int*>(static_cast<char*>(workspace) + map_off);
    int* tile_mstart = tile_expert + T_max;

    build_mmq_tile_map<<<1, 32, 0, stream>>>(
        expert_offsets, num_experts, tile_m, T_max, tile_expert, tile_mstart);

    GgufGroupedMmqArgs args;
    args.N = N; args.K = K; args.xq = xq;
    args.expert_offsets = expert_offsets; args.B_ptrs = B_ptrs;
    args.D_base = D_base;
    args.tile_expert = tile_expert; args.tile_mstart = tile_mstart;
    dispatch_mmq_grouped(type, args, use_small, T_max, stream);
}

}  // namespace layerstorm::compute
