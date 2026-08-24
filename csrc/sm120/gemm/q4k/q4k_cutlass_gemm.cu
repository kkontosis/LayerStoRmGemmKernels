// Q4_K tensor-core GEMM: shared-memory dequant + BF16 wmma MMA.
//
// M <= 8 (decode): delegates to Phase 2 GEMV (launch_q4k_dequant_gemm).
// M > 8 (prefill): wmma 16x16x16 BF16 MMA with cooperative Q4_K dequant.
//   Two tile configs dispatched by M:
//     M <= 256: 64x64 tiles (67 KB smem, 2 CTAs/SM for better occupancy)
//     M > 256:  128x64 tiles (99 KB smem, 2x compute per tile)
//
//   Per K-step of 256 (one Q4_K super-block):
//     1. cp.async A tile global→smem (non-blocking)
//     2. Dequant B tile → BF16 in smem (overlaps with A copy)
//     3. Wait for A copy, syncthreads
//     4. 16 wmma MMA steps (K=16 each)
//
// Dynamic shared memory: 67-99 KB.

#include "sm120/gemm/q4k/q4k_cutlass_gemm.h"
#include "sm120/gemm/q4k/q4k_dequant_gemm.h"  // for Phase 2 GEMV fallback
#include "formats/q4k_format.h"

#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>

namespace layerstorm::compute {

using layerstorm::formats::block_q4_K;
using layerstorm::formats::QK_K;
using layerstorm::formats::get_scale_min_k4;

// ===========================================================================
// Tile configurations
// ===========================================================================

struct TileConfig_64x64 {
    static constexpr int TILE_M = 64;
    static constexpr int TILE_N = 64;
    static constexpr int WARPS_M = 2;
    static constexpr int WARPS_N = 4;
};

struct TileConfig_128x64 {
    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = 64;
    static constexpr int WARPS_M = 4;
    static constexpr int WARPS_N = 2;
};

// Shared constants
static constexpr int TC_K_STEP = 256;
static constexpr int TC_PAD = 8;
static constexpr int TC_STRIDE = TC_K_STEP + TC_PAD;  // 264
static constexpr int TC_BLOCK_SIZE = 256;              // 8 warps

// wmma tile dimensions
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// ===========================================================================
// Async A tile loading via cp.async (overlaps with B dequant)
// ===========================================================================

template <int TILE_M>
__device__ __forceinline__
void tc_load_a_tile_async(
    const __nv_bfloat16* __restrict__ A,
    __nv_bfloat16* __restrict__ sA,
    int m_start, int k_start, int M, int K, int tid
) {
    constexpr int TOTAL = TILE_M * TC_K_STEP;
    constexpr int VEC_SIZE = 8;                              // 8 BF16 = 16 bytes
    constexpr int TOTAL_VECS = TOTAL / VEC_SIZE;
    constexpr int VECS_PER_THREAD = TOTAL_VECS / TC_BLOCK_SIZE;

    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; ++v) {
        int vec_idx = tid * VECS_PER_THREAD + v;
        int flat = vec_idx * VEC_SIZE;
        int m_local = flat / TC_K_STEP;
        int k_local = flat % TC_K_STEP;
        int m_global = m_start + m_local;

        __nv_bfloat16* dst = &sA[m_local * TC_STRIDE + k_local];
        uint32_t smem_addr = __cvta_generic_to_shared(dst);

        if (m_global < M) {
            const __nv_bfloat16* src = &A[m_global * K + k_start + k_local];
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(smem_addr), "l"(src)
            );
        } else {
            // Zero-fill OOB rows: cp.async with src_size=0
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16, 0;\n"
                :: "r"(smem_addr), "l"(A)
            );
        }
    }
}

// ===========================================================================
// Cooperative dequant B tile to BF16 (adapted from Phase 2)
// ===========================================================================

__device__ __forceinline__
void tc_dequant_b_tile(
    const void* __restrict__ B_q4k,
    __nv_bfloat16* __restrict__ sB,
    int n_start, int k_block, int N, int K, int tid
) {
    const int blocks_per_row = K / QK_K;
    const block_q4_K* B = reinterpret_cast<const block_q4_K*>(B_q4k);

    int block_id = tid / 4;      // which of 64 N rows
    int pair_id  = tid % 4;      // which pair of sub-blocks (0..3)
    int n_global = n_start + block_id;

    __nv_bfloat16* out_row = sB + block_id * TC_STRIDE;

    if (n_global >= N) {
        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            out_row[pair_id * 64 + l]      = __float2bfloat16(0.0f);
            out_row[pair_id * 64 + l + 32] = __float2bfloat16(0.0f);
        }
        return;
    }

    const block_q4_K* blk_ptr = &B[n_global * blocks_per_row + k_block];

    // Load dm (FP16 packed) and convert to FP32 for arithmetic
    half2 dm = __ldg(&blk_ptr->dm);
    float dall = __half2float(__low2half(dm));
    float dmin = __half2float(__high2half(dm));

    // Extract sub-block scales and mins
    const int is = pair_id * 2;
    uint8_t sc0, m0, sc1, m1;
    get_scale_min_k4(is,     blk_ptr->scales, sc0, m0);
    get_scale_min_k4(is + 1, blk_ptr->scales, sc1, m1);

    float d1 = dall * float(sc0);
    float m1_f = dmin * float(m0);
    float d2 = dall * float(sc1);
    float m2_f = dmin * float(m1);

    // Vectorized load of qs: 32 bytes = 2 x uint4
    const uint4* qs_vec = reinterpret_cast<const uint4*>(blk_ptr->qs + pair_id * 32);
    uint4 qs_lo = __ldg(&qs_vec[0]);
    uint4 qs_hi = __ldg(&qs_vec[1]);

    __nv_bfloat16* out = out_row + pair_id * 64;

    const uint8_t* qs_lo_bytes = reinterpret_cast<const uint8_t*>(&qs_lo);
    const uint8_t* qs_hi_bytes = reinterpret_cast<const uint8_t*>(&qs_hi);

    // Dequant with FP32 intermediate, store as BF16
    #pragma unroll
    for (int l = 0; l < 16; ++l) {
        uint8_t qbyte = qs_lo_bytes[l];
        out[l]      = __float2bfloat16_rn(d1 * float(qbyte & 0xF) - m1_f);
        out[l + 32] = __float2bfloat16_rn(d2 * float(qbyte >> 4)  - m2_f);
    }
    #pragma unroll
    for (int l = 0; l < 16; ++l) {
        uint8_t qbyte = qs_hi_bytes[l];
        out[16 + l]      = __float2bfloat16_rn(d1 * float(qbyte & 0xF) - m1_f);
        out[16 + l + 32] = __float2bfloat16_rn(d2 * float(qbyte >> 4)  - m2_f);
    }
}

// ===========================================================================
// Tensor-core GEMM kernel (wmma 16x16x16 BF16), templated on tile config
// ===========================================================================

template <typename Config>
__global__ void __launch_bounds__(TC_BLOCK_SIZE)
q4k_tc_gemm_kernel(Q4KCutlassGemmParams params) {
    using namespace nvcuda::wmma;

    constexpr int TILE_M = Config::TILE_M;
    constexpr int TILE_N = Config::TILE_N;
    constexpr int WARPS_M = Config::WARPS_M;
    constexpr int WARPS_N = Config::WARPS_N;
    constexpr int WARP_TILE_M = TILE_M / WARPS_M;   // rows per warp
    constexpr int WARP_TILE_N = TILE_N / WARPS_N;   // cols per warp
    constexpr int ATOMS_M = WARP_TILE_M / WMMA_M;   // wmma atoms per warp in M
    constexpr int ATOMS_N = WARP_TILE_N / WMMA_N;   // wmma atoms per warp in N
    constexpr int SMEM_A_BYTES = TILE_M * TC_STRIDE * sizeof(__nv_bfloat16);

    const int M = params.M;
    const int N = params.N;
    const int K = params.K;

    const int bx = blockIdx.x;   // N tile index
    const int by = blockIdx.y;   // M tile index
    const int tid = threadIdx.x;

    const int n_start = bx * TILE_N;
    const int m_start = by * TILE_M;

    const int warp_id = tid / 32;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;

    // Shared memory layout
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sB = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_A_BYTES);

    // Accumulator fragments: ATOMS_M × ATOMS_N per warp
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[ATOMS_M][ATOMS_N];
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am)
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an)
            fill_fragment(c_frag[am][an], 0.0f);

    const int n_k_blocks = K / TC_K_STEP;

    for (int kb = 0; kb < n_k_blocks; ++kb) {
        // Phase 1: Issue async A copy (non-blocking via cp.async)
        tc_load_a_tile_async<TILE_M>(params.A, sA, m_start, kb * TC_K_STEP, M, K, tid);
        asm volatile("cp.async.commit_group;\n" :::);

        // Phase 2: Dequant B tile (overlaps with async A copy)
        tc_dequant_b_tile(params.B_q4k, sB, n_start, kb, N, K, tid);

        // Wait for async A copy + sync all threads
        asm volatile("cp.async.wait_group 0;\n" :::);
        __syncthreads();

        // Phase 3: wmma MMA — 16 K-steps of 16
        // Load order: pre-load all A fragments per kk, then for each B fragment
        // reuse across all M-atoms. Minimizes fragment loads.
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag[ATOMS_M];
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;

        #pragma unroll
        for (int kk = 0; kk < TC_K_STEP / WMMA_K; ++kk) {
            // Pre-load all A fragments for this K-step
            #pragma unroll
            for (int am = 0; am < ATOMS_M; ++am) {
                load_matrix_sync(a_frag[am],
                    &sA[(warp_row * WARP_TILE_M + am * WMMA_M) * TC_STRIDE + kk * WMMA_K],
                    TC_STRIDE);
            }

            // For each N-atom: load B once, reuse across all M-atoms
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an) {
                load_matrix_sync(b_frag,
                    &sB[(warp_col * WARP_TILE_N + an * WMMA_N) * TC_STRIDE + kk * WMMA_K],
                    TC_STRIDE);

                #pragma unroll
                for (int am = 0; am < ATOMS_M; ++am) {
                    mma_sync(c_frag[am][an], a_frag[am], b_frag, c_frag[am][an]);
                }
            }
        }

        __syncthreads();
    }

    // ===========================================================================
    // Epilogue: store accumulators as BF16
    // ===========================================================================

    const int lane = tid % 32;
    float* c_stage = reinterpret_cast<float*>(smem_raw) + warp_id * WMMA_M * WMMA_N;

    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            int c_row = m_start + warp_row * WARP_TILE_M + am * WMMA_M;
            int c_col = n_start + warp_col * WARP_TILE_N + an * WMMA_N;

            store_matrix_sync(c_stage, c_frag[am][an], WMMA_N, mem_row_major);

            #pragma unroll
            for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 32) {
                int row = idx / WMMA_N;
                int col = idx % WMMA_N;
                int gm = c_row + row;
                int gn = c_col + col;
                if (gm < M && gn < N) {
                    params.C[gm * N + gn] = __float2bfloat16_rn(c_stage[idx]);
                }
            }

            __syncwarp();
        }
    }
}

// ===========================================================================
// Launcher with M-based dispatch
// ===========================================================================

void launch_q4k_cutlass_gemm(const Q4KCutlassGemmParams& params, cudaStream_t stream) {
    if (params.K % 256 != 0) {
        throw std::invalid_argument("Q4K TC GEMM: K must be divisible by 256, got K=" +
                                    std::to_string(params.K));
    }

    // M <= 8: delegate to Phase 2 GEMV (memory-bound, tensor cores don't help)
    if (params.M <= 8) {
        Q4KDequantGemmParams dequant_params;
        dequant_params.M = params.M;
        dequant_params.N = params.N;
        dequant_params.K = params.K;
        dequant_params.A = params.A;
        dequant_params.B_q4k = params.B_q4k;
        dequant_params.C = params.C;
        launch_q4k_dequant_gemm(dequant_params, stream);
        return;
    }

    using Config = TileConfig_64x64;
    constexpr int SMEM = Config::TILE_M * TC_STRIDE * sizeof(__nv_bfloat16)
                       + Config::TILE_N * TC_STRIDE * sizeof(__nv_bfloat16);

    static bool smem_configured = false;
    if (!smem_configured) {
        cudaFuncSetAttribute(
            q4k_tc_gemm_kernel<Config>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SMEM
        );
        smem_configured = true;
    }

    dim3 grid(
        (params.N + Config::TILE_N - 1) / Config::TILE_N,
        (params.M + Config::TILE_M - 1) / Config::TILE_M
    );
    q4k_tc_gemm_kernel<Config><<<grid, TC_BLOCK_SIZE, SMEM, stream>>>(params);
}

}  // namespace layerstorm::compute
