// GGUF integer mat-mat (mmq), CuTe int8 tensor-core (MMA) variant.
//
// C[M,N] = A[M,K] @ dequant(W)^T. Activation quantized to Q8_1; weight kept
// packed and dequanted to centered int8 per K super-block into shared memory
// (verbatim loader from the dp4a gguf_mmq.cu). The dp4a inner loop is replaced
// by the SM80 int8 MMA atom (m16n8k16) with a per-16-K-chunk fragment rescale:
//   Acc[m,n] += d_x[m,c] * ( A_j[n,c]*P[m,n] + B_j[n,c]*sumqx[m,c] )
// with P[m,n] = Σ_{16k} Q[n,k]*qx[m,k] the int32 MMA partial for chunk c.

#include "sm120/gemm/gguf/gguf_mmq_cute.h"
#include "sm120/gemm/gguf/gguf_int_policy.h"
#include "sm120/gemm/gguf/gguf_dequant_gemm.h"
#include "formats/q8_1_format.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_traits_sm80.hpp>

namespace layerstorm::compute {

namespace {

using namespace cute;

namespace fmt = layerstorm::formats;
using fmt::block_q8_1;

static constexpr int TILE_M = 128;
static constexpr int TILE_N = 128;
static constexpr int BLOCK_SIZE = 256;   // 8 warps

// int8 tensor-core MMA: m16n8k16, int8 in / int32 accumulate, TN.
// 8 warps in a 4x2 layout over a 128x128x16 tile (one K-step == one 16-K chunk).
// Larger tile => 4x more output per weight load (arithmetic-intensity win) and
// 8 warps/SM to hide the smem-operand short-scoreboard stalls.
using MmaAtomS8 = MMA_Atom<SM80_16x8x16_S32S8S8S32_TN>;
using TiledMmaS8 = TiledMMA<
    MmaAtomS8,
    Layout<Shape<_4, _2, _1>>,
    Tile<_128, _128, _16>>;

// Shared-memory byte layout for a given VALS (per-instantiation).
template <int VALS>
struct SmemSizes {
    static constexpr int N32 = VALS / 32;       // q8_1 blocks per K super-block
    static constexpr int NC  = VALS / 16;       // 16-K chunks per K super-block
    static constexpr int qx_bytes = TILE_M * VALS;                 // int8
    static constexpr int qw_bytes = TILE_N * VALS;                 // int8
    static constexpr int dx_bytes = TILE_M * N32 * (int)sizeof(float);
    // A,B are constant within each 16-K MMA chunk (verified for all 6 GGUF
    // types), so we only store one (A,B) per chunk (NC) rather than per 4-group
    // (NG) — a 4x reduction of the weight-scale smem footprint vs per-group.
    static constexpr int a_bytes  = TILE_N * NC  * (int)sizeof(float);
    static constexpr int b_bytes  = TILE_N * NC  * (int)sizeof(float);
    static constexpr int sx_bytes = TILE_M * NC  * (int)sizeof(float);  // sumqx
    static constexpr int total = qx_bytes + qw_bytes + dx_bytes + a_bytes + b_bytes + sx_bytes;
};

template <class P>
__global__ void __launch_bounds__(BLOCK_SIZE)
gguf_mmq_cute_kernel(GgufMmvqParams params, const block_q8_1* __restrict__ xq) {
    constexpr int VALS = P::VALS;
    constexpr int N32 = VALS / 32;
    constexpr int NG  = VALS / 4;
    constexpr int NC  = VALS / 16;   // 16-K chunks per super-block
    using S = SmemSizes<VALS>;

    const int M = params.M, N = params.N, K = params.K;
    const int n_start = blockIdx.x * TILE_N;
    const int m_start = blockIdx.y * TILE_M;
    const int tid = threadIdx.x;

    extern __shared__ char smem[];
    int8_t* sQx = reinterpret_cast<int8_t*>(smem);
    int8_t* sQw = reinterpret_cast<int8_t*>(smem + S::qx_bytes);
    float*  sDx = reinterpret_cast<float*>(smem + S::qx_bytes + S::qw_bytes);
    float*  sA  = reinterpret_cast<float*>(smem + S::qx_bytes + S::qw_bytes + S::dx_bytes);
    float*  sB  = reinterpret_cast<float*>(smem + S::qx_bytes + S::qw_bytes + S::dx_bytes + S::a_bytes);
    float*  sSx = reinterpret_cast<float*>(smem + S::qx_bytes + S::qw_bytes + S::dx_bytes
                                                + S::a_bytes + S::b_bytes);

    const int x_bpr = K / 32;
    const int nblk = K / VALS;
    const uint8_t* Wbase = reinterpret_cast<const uint8_t*>(params.B);

    // Persistent float accumulator as a CuTe C-fragment (matches the MMA layout).
    TiledMmaS8 tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);
    // Identity coords for (m,n) of each accumulator element.
    auto cIdent = make_identity_tensor(make_shape(Int<TILE_M>{}, Int<TILE_N>{}));
    auto tCcAcc = thr_mma.partition_C(cIdent);          // -> (m,n) per frag elem
    const int frag_c = size(tCcAcc);

    // 128x128 output tile over 256 threads: C-fragment has 64 elements per thread.
    constexpr int FRAG_C = (TILE_M * TILE_N) / BLOCK_SIZE;
    float Acc[FRAG_C];
    #pragma unroll
    for (int i = 0; i < FRAG_C; ++i) Acc[i] = 0.0f;

    // smem tensors for MMA operand partitioning (row-major, k-major within row).
    auto sA_t = make_tensor(make_smem_ptr(sQx),
                            make_layout(make_shape(Int<TILE_M>{}, Int<VALS>{}),
                                        make_stride(Int<VALS>{}, _1{})));
    auto sB_t = make_tensor(make_smem_ptr(sQw),
                            make_layout(make_shape(Int<TILE_N>{}, Int<VALS>{}),
                                        make_stride(Int<VALS>{}, _1{})));

    for (int kb = 0; kb < nblk; ++kb) {
        // --- load activation quants + per-32 scales (verbatim from dp4a mmq) ---
        for (int idx = tid; idx < TILE_M * VALS; idx += BLOCK_SIZE) {
            const int mr = idx / VALS, v = idx % VALS;
            const int mg = m_start + mr;
            int8_t val = 0;
            if (mg < M) {
                const block_q8_1* b = &xq[(size_t)mg * x_bpr + kb * N32 + v / 32];
                val = b->qs[v % 32];
            }
            sQx[idx] = val;
        }
        for (int idx = tid; idx < TILE_M * N32; idx += BLOCK_SIZE) {
            const int mr = idx / N32, b = idx % N32;
            const int mg = m_start + mr;
            sDx[idx] = (mg < M)
                ? __half2float(__low2half(xq[(size_t)mg * x_bpr + kb * N32 + b].ds))
                : 0.0f;
        }
        // --- dequant weight tile to centered int8 + per-group (A,B) (verbatim) ---
        for (int idx = tid; idx < TILE_N * NG; idx += BLOCK_SIZE) {
            const int nr = idx / NG, g = idx % NG;
            const int ng = n_start + nr;
            int Q = 0; float A = 0.0f, B = 0.0f;
            if (ng < N) {
                const void* wblk = Wbase + ((size_t)ng * nblk + kb) * P::BYTES;
                float dw, dmw; P::scales(wblk, dw, dmw);
                P::group(wblk, g, dw, dmw, Q, A, B);
            }
            *reinterpret_cast<int*>(&sQw[nr * VALS + 4 * g]) = Q;
            // A,B are constant across the 4 groups of a 16-K chunk; store once
            // per chunk (g % 4 == 0 is the chunk's first group).
            if ((g & 3) == 0) {
                const int c = g >> 2;   // chunk index
                sA[nr * NC + c] = A;
                sB[nr * NC + c] = B;
            }
        }
        // --- sumqx[TILE_M][NC]: Σ of the 16 qx per chunk per m-row ---
        for (int idx = tid; idx < TILE_M * NC; idx += BLOCK_SIZE) {
            const int mr = idx / NC, c = idx % NC;
            // 16-K chunk = 4 consecutive 4-groups; sum via __dp4a of each group.
            const int base = mr * VALS + 16 * c;
            int s = 0;
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int qx4 = *reinterpret_cast<const int*>(&sQx[base + 4 * q]);
                s = __dp4a(qx4, 0x01010101, s);
            }
            sSx[idx] = (float)s;
        }
        __syncthreads();

        // --- compute: one MMA per 16-K chunk, rescale int32 partial into Acc ---
        // Full unroll so the compiler can software-pipeline: overlap the next
        // chunk's smem->reg operand loads with the current chunk's MMA+rescale
        // (the dominant stall is short-scoreboard on smem operand fetch).
        #pragma unroll 4
        for (int c = 0; c < NC; ++c) {
            // Slice the K-mode [16*c, 16*c+16).
            auto sAchunk = local_tile(sA_t, make_shape(Int<TILE_M>{}, Int<16>{}),
                                      make_coord(0, c));
            auto sBchunk = local_tile(sB_t, make_shape(Int<TILE_N>{}, Int<16>{}),
                                      make_coord(0, c));

            auto tCsA = thr_mma.partition_A(sAchunk);
            auto tCsB = thr_mma.partition_B(sBchunk);
            auto tCrA = thr_mma.partition_fragment_A(sAchunk);
            auto tCrB = thr_mma.partition_fragment_B(sBchunk);

            // Fresh int32 partial accumulator for this chunk.
            auto rP = partition_fragment_C(tiled_mma, Shape<Int<TILE_M>, Int<TILE_N>>{});
            clear(rP);

            cute::copy(tCsA(_, _, 0), tCrA(_, _, 0));
            cute::copy(tCsB(_, _, 0), tCrB(_, _, 0));
            cute::gemm(tiled_mma, tCrA(_, _, 0), tCrB(_, _, 0), rP);

            // Rescale this chunk's int32 partials into the float accumulator.
            const int b32 = (16 * c) / 32;     // which q8_1 32-block this chunk is in
            #pragma unroll
            for (int i = 0; i < FRAG_C; ++i) {
                if (i >= frag_c) break;
                const int m = (int)get<0>(tCcAcc(i));
                const int n = (int)get<1>(tCcAcc(i));
                const float dx = sDx[m * N32 + b32];
                const float A  = sA[n * NC + c];
                const float B  = sB[n * NC + c];
                const float sx = sSx[m * NC + c];
                const int   Pi = rP(i);
                Acc[i] += dx * (A * (float)Pi + B * sx);
            }
        }
        __syncthreads();
    }

    // --- epilogue: write BF16 via the partition_C (m,n) mapping ---
    #pragma unroll
    for (int i = 0; i < FRAG_C; ++i) {
        if (i >= frag_c) break;
        const int m = (int)get<0>(tCcAcc(i));
        const int n = (int)get<1>(tCcAcc(i));
        const int mg = m_start + m;
        const int ng = n_start + n;
        if (mg < M && ng < N)
            params.C[(size_t)mg * N + ng] = __float2bfloat16_rn(Acc[i]);
    }
}

template <class P>
void launch_for_policy(const GgufMmvqParams& params, const block_q8_1* xq, cudaStream_t stream) {
    constexpr int SMEM = SmemSizes<P::VALS>::total;
    dim3 grid((params.N + TILE_N - 1) / TILE_N, (params.M + TILE_M - 1) / TILE_M);
    // cudaFuncSetAttribute is DEVICE-scoped: latch per device, not
    // process-wide (a process-wide latch left every device after the
    // first at the 48 KB default dynamic-smem cap -> silent
    // cudaErrorInvalidValue launches on multi-GPU).
    int dev_ = 0;
    cudaGetDevice(&dev_);
    static bool configured[64] = {};
    if (dev_ >= 0 && dev_ < 64 && !configured[dev_]) {
        cudaFuncSetAttribute(gguf_mmq_cute_kernel<P>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        configured[dev_] = true;
    }
    gguf_mmq_cute_kernel<P><<<grid, dim3(BLOCK_SIZE), SMEM, stream>>>(params, xq);
}

}  // namespace

void launch_gguf_mmq_cute(const GgufMmvqParams& params, void* q8_1_workspace,
                          cudaStream_t stream) {
    const int qk = gguf_block_values(params.type);
    if (params.K % qk != 0) {
        throw std::invalid_argument(
            "gguf_mmq_cute: K must be divisible by " + std::to_string(qk) +
            ", got K=" + std::to_string(params.K));
    }
    block_q8_1* xq = reinterpret_cast<block_q8_1*>(q8_1_workspace);
    gguf_quantize_q8_1_cuda(params.A, xq, params.M, params.K, stream);

    switch (params.type) {
        case GgufType::Q2_K: launch_for_policy<gguf_int::Q2K>(params, xq, stream);  break;
        case GgufType::Q3_K: launch_for_policy<gguf_int::Q3K>(params, xq, stream);  break;
        case GgufType::Q4_K: launch_for_policy<gguf_int::Q4K>(params, xq, stream);  break;
        case GgufType::Q5_K: launch_for_policy<gguf_int::Q5K>(params, xq, stream);  break;
        case GgufType::Q6_K: launch_for_policy<gguf_int::Q6K>(params, xq, stream);  break;
        case GgufType::Q8_0: launch_for_policy<gguf_int::Q8_0>(params, xq, stream); break;
        case GgufType::MXFP4: launch_for_policy<gguf_int::Mxfp4>(params, xq, stream); break;
    }
}

}  // namespace layerstorm::compute
