// Q4_K dequant-GEMM: two kernels dispatched by M.
//
// M <= 8 (decode): GEMV kernel — 1 warp per output row, threads stride over K
//   super-blocks, dequant in float32 registers, warp-shuffle reduction.
//   Templated on M for compile-time unrolling. ~128 threads per block.
//
// M > 8 (prefill): Tiled GEMM — 64×64 tiles, 256 threads, per-tile Q4_K
//   dequant into shared memory, FP32 FMA accumulation. Vectorized loads
//   with __ldg hints for A (float4) and B (uint4 for qs, uint32 for scales).
//
// Dynamic shared memory: GEMV ~1-8 KB, GEMM ~67 KB.

#include "sm120/gemm/q4k/q4k_dequant_gemm.h"
#include "formats/q4k_format.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>

namespace layerstorm::compute {

using layerstorm::formats::block_q4_K;
using layerstorm::formats::QK_K;
using layerstorm::formats::get_scale_min_k4;

// ===========================================================================
// GEMV kernel for M <= 8 (decode path)
// ===========================================================================

// GEMV configuration
static constexpr int GEMV_ROWS_PER_BLOCK = 4;  // N output rows per block
static constexpr int GEMV_WARP_SIZE = 32;
static constexpr int GEMV_BLOCK_SIZE = GEMV_ROWS_PER_BLOCK * GEMV_WARP_SIZE;  // 128

template <int M_STATIC>
__global__ void __launch_bounds__(GEMV_BLOCK_SIZE)
q4k_gemv_kernel(Q4KDequantGemmParams params) {
    const int N = params.N;
    const int K = params.K;
    const int blocks_per_row = K / QK_K;

    const int tid = threadIdx.x;
    const int warp_id = tid / GEMV_WARP_SIZE;    // 0..3, which output row in block
    const int lane = tid % GEMV_WARP_SIZE;        // 0..31, K-reduction lane

    const int n_global = blockIdx.x * GEMV_ROWS_PER_BLOCK + warp_id;
    if (n_global >= N) return;

    const block_q4_K* B = reinterpret_cast<const block_q4_K*>(params.B_q4k);

    // Accumulators: M partial sums per thread
    float partial[M_STATIC];
    #pragma unroll
    for (int m = 0; m < M_STATIC; ++m)
        partial[m] = 0.0f;

    // Each lane strides over K super-blocks. A is read directly from global
    // memory — L1 cache handles the broadcast (all warps read the same A rows).
    for (int kb = lane; kb < blocks_per_row; kb += GEMV_WARP_SIZE) {
        const block_q4_K& blk = B[n_global * blocks_per_row + kb];

        // Extract super-block scales as float32
        float dall = __half2float(__low2half(blk.dm));
        float dmin = __half2float(__high2half(blk.dm));

        int k_base = kb * QK_K;

        // Process 4 pairs of sub-blocks
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            int is = pair * 2;
            uint8_t sc0_u8, m0_u8, sc1_u8, m1_u8;
            get_scale_min_k4(is,     blk.scales, sc0_u8, m0_u8);
            get_scale_min_k4(is + 1, blk.scales, sc1_u8, m1_u8);

            float d1 = dall * float(sc0_u8);
            float m1_f = dmin * float(m0_u8);
            float d2 = dall * float(sc1_u8);
            float m2_f = dmin * float(m1_u8);

            const uint8_t* qs = blk.qs + pair * 32;

            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                uint8_t qbyte = qs[l];
                float v_lo = d1 * float(qbyte & 0xF) - m1_f;
                float v_hi = d2 * float(qbyte >> 4)  - m2_f;

                int k_lo = k_base + pair * 64 + l;
                int k_hi = k_lo + 32;

                #pragma unroll
                for (int m = 0; m < M_STATIC; ++m) {
                    float a_lo = __bfloat162float(params.A[m * K + k_lo]);
                    float a_hi = __bfloat162float(params.A[m * K + k_hi]);
                    partial[m] = fmaf(v_lo, a_lo, partial[m]);
                    partial[m] = fmaf(v_hi, a_hi, partial[m]);
                }
            }
        }
    }

    // Warp reduction
    #pragma unroll
    for (int m = 0; m < M_STATIC; ++m) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial[m] += __shfl_down_sync(0xffffffff, partial[m], offset);
        }
    }

    // Lane 0 writes output
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < M_STATIC; ++m) {
            if (m < params.M) {
                params.C[m * N + n_global] = __float2bfloat16_rn(partial[m]);
            }
        }
    }
}

static void launch_q4k_gemv(const Q4KDequantGemmParams& params, cudaStream_t stream) {
    dim3 grid((params.N + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK);
    dim3 block(GEMV_BLOCK_SIZE);

    // No shared memory needed — A is read from global via L1 cache
    switch (params.M) {
        case 1: q4k_gemv_kernel<1><<<grid, block, 0, stream>>>(params); break;
        case 2: q4k_gemv_kernel<2><<<grid, block, 0, stream>>>(params); break;
        case 3: q4k_gemv_kernel<3><<<grid, block, 0, stream>>>(params); break;
        case 4: q4k_gemv_kernel<4><<<grid, block, 0, stream>>>(params); break;
        case 5: q4k_gemv_kernel<5><<<grid, block, 0, stream>>>(params); break;
        case 6: q4k_gemv_kernel<6><<<grid, block, 0, stream>>>(params); break;
        case 7: q4k_gemv_kernel<7><<<grid, block, 0, stream>>>(params); break;
        case 8: q4k_gemv_kernel<8><<<grid, block, 0, stream>>>(params); break;
    }
}


// ===========================================================================
// Tiled GEMM kernel for M > 8 (prefill path)
// ===========================================================================

// Tile configuration
static constexpr int TILE_M = 64;
static constexpr int TILE_N = 64;
static constexpr int K_STEP = 256;            // = QK_K, one Q4_K super-block
static constexpr int PAD = 8;                 // bank-conflict avoidance padding
static constexpr int SA_STRIDE = K_STEP + PAD;  // 264
static constexpr int SB_STRIDE = K_STEP + PAD;  // 264
static constexpr int BLOCK_SIZE = 256;        // threads per block

// Thread tile: each thread computes TM x TN output elements
static constexpr int TM = 4;
static constexpr int TN = 4;
// Thread layout: 16 threads along N, 16 threads along M => 256 total
static constexpr int THREAD_N = TILE_N / TN;  // 16

// Shared memory size in bytes
static constexpr int SMEM_A_BYTES = TILE_M * SA_STRIDE * sizeof(__nv_bfloat16);
static constexpr int SMEM_B_BYTES = TILE_N * SB_STRIDE * sizeof(half);
static constexpr int SMEM_BYTES = SMEM_A_BYTES + SMEM_B_BYTES;  // 67584

// Cooperative load A tile with vectorized float4 loads (8 BF16 per load)
__device__ __forceinline__
void load_a_tile(
    const __nv_bfloat16* __restrict__ A,
    __nv_bfloat16* __restrict__ sA,
    int m_start, int k_start, int M, int K, int tid
) {
    constexpr int TOTAL = TILE_M * K_STEP;            // 16384
    constexpr int VEC_SIZE = 8;                        // 8 BF16 per float4
    constexpr int TOTAL_VECS = TOTAL / VEC_SIZE;       // 2048
    constexpr int VECS_PER_THREAD = TOTAL_VECS / BLOCK_SIZE;  // 8

    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; ++v) {
        int vec_idx = tid * VECS_PER_THREAD + v;
        int flat = vec_idx * VEC_SIZE;
        int m_local = flat / K_STEP;
        int k_local = flat % K_STEP;
        int m_global = m_start + m_local;

        if (m_global < M) {
            // Vectorized 16-byte load (8 contiguous BF16 values)
            float4 vec = __ldg(reinterpret_cast<const float4*>(
                &A[m_global * K + k_start + k_local]));
            *reinterpret_cast<float4*>(&sA[m_local * SA_STRIDE + k_local]) = vec;
        } else {
            // Zero-fill 8 values
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i)
                sA[m_local * SA_STRIDE + k_local + i] = __float2bfloat16(0.0f);
        }
    }
}

// Cooperative load + dequant B tile with vectorized Q4_K reads
__device__ __forceinline__
void dequant_b_tile(
    const void* __restrict__ B_q4k,
    half* __restrict__ sB,
    int n_start, int k_block, int N, int K, int tid
) {
    const int blocks_per_row = K / QK_K;
    const block_q4_K* B = reinterpret_cast<const block_q4_K*>(B_q4k);

    int block_id = tid / 4;      // which of 64 N rows
    int pair_id  = tid % 4;      // which pair of sub-blocks (0..3)
    int n_global = n_start + block_id;

    half* out_row = sB + block_id * SB_STRIDE;

    if (n_global >= N) {
        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            out_row[pair_id * 64 + l]      = __float2half(0.0f);
            out_row[pair_id * 64 + l + 32] = __float2half(0.0f);
        }
        return;
    }

    const block_q4_K* blk_ptr = &B[n_global * blocks_per_row + k_block];

    // Vectorized load of dm (4 bytes as uint32)
    half2 dm = __ldg(&blk_ptr->dm);
    const half dall = __low2half(dm);
    const half dmin = __high2half(dm);

    // Load scales with __ldg
    const int is = pair_id * 2;
    uint8_t sc0, m0, sc1, m1;
    get_scale_min_k4(is,     blk_ptr->scales, sc0, m0);
    get_scale_min_k4(is + 1, blk_ptr->scales, sc1, m1);

    const half d1 = __hmul(dall, __int2half_rn(sc0));
    const half m1_h = __hmul(dmin, __int2half_rn(m0));
    const half d2 = __hmul(dall, __int2half_rn(sc1));
    const half m2_h = __hmul(dmin, __int2half_rn(m1));

    // Vectorized load of qs: 32 bytes = 2 × uint4 (16 bytes each)
    const uint4* qs_vec = reinterpret_cast<const uint4*>(blk_ptr->qs + pair_id * 32);
    uint4 qs_lo = __ldg(&qs_vec[0]);  // bytes 0-15
    uint4 qs_hi = __ldg(&qs_vec[1]);  // bytes 16-31

    half* out = out_row + pair_id * 64;

    // Unpack from the two uint4 vectors
    // qs_lo contains bytes 0-15, qs_hi contains bytes 16-31
    const uint8_t* qs_lo_bytes = reinterpret_cast<const uint8_t*>(&qs_lo);
    const uint8_t* qs_hi_bytes = reinterpret_cast<const uint8_t*>(&qs_hi);

    #pragma unroll
    for (int l = 0; l < 16; ++l) {
        uint8_t qbyte = qs_lo_bytes[l];
        out[l]      = __hsub(__hmul(d1, __int2half_rn(qbyte & 0xF)), m1_h);
        out[l + 32] = __hsub(__hmul(d2, __int2half_rn(qbyte >> 4)),  m2_h);
    }
    #pragma unroll
    for (int l = 0; l < 16; ++l) {
        uint8_t qbyte = qs_hi_bytes[l];
        out[16 + l]      = __hsub(__hmul(d1, __int2half_rn(qbyte & 0xF)), m1_h);
        out[16 + l + 32] = __hsub(__hmul(d2, __int2half_rn(qbyte >> 4)),  m2_h);
    }
}

// Main tiled GEMM kernel
__global__ void __launch_bounds__(BLOCK_SIZE)
q4k_dequant_gemm_kernel(Q4KDequantGemmParams params) {
    const int M = params.M;
    const int N = params.N;
    const int K = params.K;

    const int bx = blockIdx.x;   // N tile index
    const int by = blockIdx.y;   // M tile index
    const int tid = threadIdx.x;

    const int n_start = bx * TILE_N;
    const int m_start = by * TILE_M;

    const int tx = tid % THREAD_N;   // 0..15, N dimension
    const int ty = tid / THREAD_N;   // 0..15, M dimension

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    half* sB = reinterpret_cast<half*>(smem_raw + SMEM_A_BYTES);

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    const int n_k_blocks = K / K_STEP;

    for (int kb = 0; kb < n_k_blocks; ++kb) {
        load_a_tile(params.A, sA, m_start, kb * K_STEP, M, K, tid);
        dequant_b_tile(params.B_q4k, sB, n_start, kb, N, K, tid);

        __syncthreads();

        // Compute 4x4 register tile with k-loop sub-stepping by 4
        #pragma unroll
        for (int k = 0; k < K_STEP; k += 4) {
            float a_frag[TM][4];
            float b_frag[TN][4];

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk)
                    a_frag[i][kk] = __bfloat162float(sA[(ty * TM + i) * SA_STRIDE + k + kk]);

            #pragma unroll
            for (int j = 0; j < TN; ++j)
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk)
                    b_frag[j][kk] = __half2float(sB[(tx * TN + j) * SB_STRIDE + k + kk]);

            #pragma unroll
            for (int kk = 0; kk < 4; ++kk)
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    #pragma unroll
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] = fmaf(a_frag[i][kk], b_frag[j][kk], acc[i][j]);
        }

        __syncthreads();
    }

    // Write output as BF16
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int m_global = m_start + ty * TM + i;
        if (m_global >= M) break;

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int n_global = n_start + tx * TN + j;
            if (n_global < N) {
                params.C[m_global * N + n_global] = __float2bfloat16_rn(acc[i][j]);
            }
        }
    }
}

// ===========================================================================
// Launcher with M-based dispatch
// ===========================================================================

void launch_q4k_dequant_gemm(const Q4KDequantGemmParams& params, cudaStream_t stream) {
    if (params.K % 256 != 0) {
        throw std::invalid_argument("Q4K GEMM: K must be divisible by 256, got K=" + std::to_string(params.K));
    }

    if (params.M <= 8) {
        launch_q4k_gemv(params, stream);
    } else {
        dim3 grid(
            (params.N + TILE_N - 1) / TILE_N,
            (params.M + TILE_M - 1) / TILE_M
        );
        dim3 block(BLOCK_SIZE);

        static bool smem_configured = false;
        if (!smem_configured) {
            cudaFuncSetAttribute(
                q4k_dequant_gemm_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_BYTES
            );
            smem_configured = true;
        }

        q4k_dequant_gemm_kernel<<<grid, block, SMEM_BYTES, stream>>>(params);
    }
}

}  // namespace layerstorm::compute
