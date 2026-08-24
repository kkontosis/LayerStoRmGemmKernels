// GGUF dequant-to-float GEMM — strategy 3 (dequant + FP32 GEMM), format-agnostic.
//
// Two kernels dispatched by M, both templated on a per-type "format policy" that
// exposes the packed block size and a whole-block dequant (from the formats/
// headers, which are faithful ports of ggml's convert.cu math):
//   M <= 8 : GEMV  — 1 warp per output channel, lane strides over K blocks.
//   M  > 8 : 32x32 tiled GEMM — dequant B tile to shared memory, FP32 FMA.
//
// This is the correctness/fallback path; mmvq (decode) and mmq (prefill) provide
// the faster integer paths.

#include "sm120/gemm/gguf/gguf_dequant_gemm.h"

#include "formats/mxfp4_format.h"
#include "formats/q2k_format.h"
#include "formats/q3k_format.h"
#include "formats/q4k_format.h"
#include "formats/q5k_format.h"
#include "formats/q6k_format.h"
#include "formats/q8_0_format.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

namespace layerstorm::compute {

// ===========================================================================
// Format policies — map a GgufType to its block struct, sizes, and dequant.
// ===========================================================================

namespace {

namespace fmt = layerstorm::formats;

struct PolicyQ2K {
    static constexpr int VALS = 256;
    static constexpr int BYTES = fmt::BLOCK_Q2K_SIZE;  // 84
    __device__ static void dequant(const void* blk, float* out) {
        fmt::dequant_q2k_block(*reinterpret_cast<const fmt::block_q2_K*>(blk), out);
    }
};
struct PolicyQ3K {
    static constexpr int VALS = 256;
    static constexpr int BYTES = fmt::BLOCK_Q3K_SIZE;  // 110
    __device__ static void dequant(const void* blk, float* out) {
        fmt::dequant_q3k_block(*reinterpret_cast<const fmt::block_q3_K*>(blk), out);
    }
};
struct PolicyQ4K {
    static constexpr int VALS = 256;
    static constexpr int BYTES = fmt::BLOCK_Q4K_SIZE;  // 144
    __device__ static void dequant(const void* blk, float* out) {
        fmt::dequant_q4k_block(*reinterpret_cast<const fmt::block_q4_K*>(blk), out);
    }
};
struct PolicyQ5K {
    static constexpr int VALS = 256;
    static constexpr int BYTES = fmt::BLOCK_Q5K_SIZE;  // 176
    __device__ static void dequant(const void* blk, float* out) {
        fmt::dequant_q5k_block(*reinterpret_cast<const fmt::block_q5_K*>(blk), out);
    }
};
struct PolicyQ6K {
    static constexpr int VALS = 256;
    static constexpr int BYTES = fmt::BLOCK_Q6K_SIZE;  // 210
    __device__ static void dequant(const void* blk, float* out) {
        fmt::dequant_q6k_block(*reinterpret_cast<const fmt::block_q6_K*>(blk), out);
    }
};
struct PolicyQ8_0 {
    static constexpr int VALS = 32;
    static constexpr int BYTES = fmt::BLOCK_Q8_0_SIZE;  // 34
    __device__ static void dequant(const void* blk, float* out) {
        fmt::dequant_q8_0_block(*reinterpret_cast<const fmt::block_q8_0*>(blk), out);
    }
};
struct PolicyMxfp4 {
    static constexpr int VALS = 32;
    static constexpr int BYTES = fmt::BLOCK_MXFP4_SIZE;  // 17
    __device__ static void dequant(const void* blk, float* out) {
        fmt::dequant_mxfp4_block(*reinterpret_cast<const fmt::block_mxfp4*>(blk), out);
    }
};

// ===========================================================================
// GEMV kernel for M <= 8 (decode path)
// ===========================================================================

static constexpr int GEMV_ROWS_PER_BLOCK = 4;
static constexpr int GEMV_BLOCK_SIZE = GEMV_ROWS_PER_BLOCK * 32;  // 128

template <class P, int M_STATIC>
__global__ void __launch_bounds__(GEMV_BLOCK_SIZE)
gguf_gemv_kernel(GgufDequantGemmParams params) {
    const int N = params.N;
    const int K = params.K;
    const int blocks_per_row = K / P::VALS;

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;

    const int n = blockIdx.x * GEMV_ROWS_PER_BLOCK + warp;
    if (n >= N) return;

    const uint8_t* B = reinterpret_cast<const uint8_t*>(params.B)
                     + static_cast<size_t>(n) * blocks_per_row * P::BYTES;

    float partial[M_STATIC];
    #pragma unroll
    for (int m = 0; m < M_STATIC; ++m) partial[m] = 0.0f;

    float buf[P::VALS];
    for (int kb = lane; kb < blocks_per_row; kb += 32) {
        P::dequant(B + static_cast<size_t>(kb) * P::BYTES, buf);
        const int kbase = kb * P::VALS;
        for (int i = 0; i < P::VALS; ++i) {
            const float w = buf[i];
            const int k = kbase + i;
            #pragma unroll
            for (int m = 0; m < M_STATIC; ++m) {
                partial[m] = fmaf(w, __bfloat162float(params.A[m * K + k]), partial[m]);
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < M_STATIC; ++m) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            partial[m] += __shfl_down_sync(0xffffffff, partial[m], offset);
    }

    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < M_STATIC; ++m)
            if (m < params.M)
                params.C[m * N + n] = __float2bfloat16_rn(partial[m]);
    }
}

template <class P>
void launch_gemv(const GgufDequantGemmParams& params, cudaStream_t stream) {
    dim3 grid((params.N + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK);
    dim3 block(GEMV_BLOCK_SIZE);
    switch (params.M) {
        case 1: gguf_gemv_kernel<P, 1><<<grid, block, 0, stream>>>(params); break;
        case 2: gguf_gemv_kernel<P, 2><<<grid, block, 0, stream>>>(params); break;
        case 3: gguf_gemv_kernel<P, 3><<<grid, block, 0, stream>>>(params); break;
        case 4: gguf_gemv_kernel<P, 4><<<grid, block, 0, stream>>>(params); break;
        case 5: gguf_gemv_kernel<P, 5><<<grid, block, 0, stream>>>(params); break;
        case 6: gguf_gemv_kernel<P, 6><<<grid, block, 0, stream>>>(params); break;
        case 7: gguf_gemv_kernel<P, 7><<<grid, block, 0, stream>>>(params); break;
        case 8: gguf_gemv_kernel<P, 8><<<grid, block, 0, stream>>>(params); break;
    }
}

// ===========================================================================
// Tiled GEMM kernel for M > 8 (prefill correctness/fallback path)
// ===========================================================================

static constexpr int TILE_M = 32;
static constexpr int TILE_N = 32;
static constexpr int PAD = 8;
static constexpr int GEMM_BLOCK_SIZE = 256;  // 16x16 threads, 2x2 per thread
static constexpr int TM = 2;
static constexpr int TN = 2;
static constexpr int THREAD_N = TILE_N / TN;  // 16

template <class P>
__global__ void __launch_bounds__(GEMM_BLOCK_SIZE)
gguf_tiled_gemm_kernel(GgufDequantGemmParams params) {
    constexpr int K_STEP = P::VALS;
    constexpr int SA_STRIDE = K_STEP + PAD;
    constexpr int SB_STRIDE = K_STEP + PAD;

    const int M = params.M;
    const int N = params.N;
    const int K = params.K;

    const int n_start = blockIdx.x * TILE_N;
    const int m_start = blockIdx.y * TILE_M;
    const int tid = threadIdx.x;
    const int tx = tid % THREAD_N;  // 0..15 (N)
    const int ty = tid / THREAD_N;  // 0..15 (M)

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    float* sB = reinterpret_cast<float*>(smem_raw + TILE_M * SA_STRIDE * sizeof(__nv_bfloat16));

    const int blocks_per_row = K / K_STEP;
    const uint8_t* B = reinterpret_cast<const uint8_t*>(params.B);

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int kb = 0; kb < blocks_per_row; ++kb) {
        // Cooperative load of the A tile (TILE_M x K_STEP) into shared memory.
        for (int idx = tid; idx < TILE_M * K_STEP; idx += GEMM_BLOCK_SIZE) {
            const int mr = idx / K_STEP;
            const int kc = idx % K_STEP;
            const int mg = m_start + mr;
            sA[mr * SA_STRIDE + kc] =
                (mg < M) ? params.A[mg * K + kb * K_STEP + kc] : __float2bfloat16(0.0f);
        }
        // Dequant the B tile: one thread per N row dequants its block into sB.
        if (tid < TILE_N) {
            const int ng = n_start + tid;
            float* dst = sB + tid * SB_STRIDE;
            if (ng < N) {
                P::dequant(B + (static_cast<size_t>(ng) * blocks_per_row + kb) * P::BYTES, dst);
            } else {
                #pragma unroll
                for (int i = 0; i < K_STEP; ++i) dst[i] = 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll 4
        for (int k = 0; k < K_STEP; ++k) {
            float a_frag[TM], b_frag[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                a_frag[i] = __bfloat162float(sA[(ty * TM + i) * SA_STRIDE + k]);
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                b_frag[j] = sB[(tx * TN + j) * SB_STRIDE + k];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int mg = m_start + ty * TM + i;
        if (mg >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int ng = n_start + tx * TN + j;
            if (ng < N) params.C[mg * N + ng] = __float2bfloat16_rn(acc[i][j]);
        }
    }
}

template <class P>
void launch_tiled(const GgufDequantGemmParams& params, cudaStream_t stream) {
    constexpr int K_STEP = P::VALS;
    constexpr int SA_BYTES = TILE_M * (K_STEP + PAD) * sizeof(__nv_bfloat16);
    constexpr int SB_BYTES = TILE_N * (K_STEP + PAD) * sizeof(float);
    constexpr int SMEM_BYTES = SA_BYTES + SB_BYTES;

    dim3 grid((params.N + TILE_N - 1) / TILE_N, (params.M + TILE_M - 1) / TILE_M);
    dim3 block(GEMM_BLOCK_SIZE);

    // cudaFuncSetAttribute is DEVICE-scoped: latch per device, not
    // process-wide (a process-wide latch left every device after the
    // first at the 48 KB default dynamic-smem cap -> silent
    // cudaErrorInvalidValue launches on multi-GPU).
    int dev_ = 0;
    cudaGetDevice(&dev_);
    static bool configured[64] = {};
    if (dev_ >= 0 && dev_ < 64 && !configured[dev_]) {
        cudaFuncSetAttribute(gguf_tiled_gemm_kernel<P>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        configured[dev_] = true;
    }
    gguf_tiled_gemm_kernel<P><<<grid, block, SMEM_BYTES, stream>>>(params);
}

// Dispatch over M for one format policy.
template <class P>
void launch_for_policy(const GgufDequantGemmParams& params, cudaStream_t stream) {
    if (params.M <= 8) launch_gemv<P>(params, stream);
    else               launch_tiled<P>(params, stream);
}

}  // namespace

// ===========================================================================
// Host entry points
// ===========================================================================

int gguf_block_values(GgufType type) {
    return (type == GgufType::Q8_0 || type == GgufType::MXFP4) ? 32 : 256;
}

int gguf_block_bytes(GgufType type) {
    switch (type) {
        case GgufType::Q2_K: return fmt::BLOCK_Q2K_SIZE;
        case GgufType::Q3_K: return fmt::BLOCK_Q3K_SIZE;
        case GgufType::Q4_K: return fmt::BLOCK_Q4K_SIZE;
        case GgufType::Q5_K: return fmt::BLOCK_Q5K_SIZE;
        case GgufType::Q6_K: return fmt::BLOCK_Q6K_SIZE;
        case GgufType::Q8_0: return fmt::BLOCK_Q8_0_SIZE;
        case GgufType::MXFP4: return fmt::BLOCK_MXFP4_SIZE;
    }
    return 0;
}

void launch_gguf_dequant_gemm(const GgufDequantGemmParams& params, cudaStream_t stream) {
    const int qk = gguf_block_values(params.type);
    if (params.K % qk != 0) {
        throw std::invalid_argument(
            "gguf_dequant_gemm: K must be divisible by " + std::to_string(qk) +
            " for this type, got K=" + std::to_string(params.K));
    }
    switch (params.type) {
        case GgufType::Q2_K: launch_for_policy<PolicyQ2K>(params, stream);  break;
        case GgufType::Q3_K: launch_for_policy<PolicyQ3K>(params, stream);  break;
        case GgufType::Q4_K: launch_for_policy<PolicyQ4K>(params, stream);  break;
        case GgufType::Q5_K: launch_for_policy<PolicyQ5K>(params, stream);  break;
        case GgufType::Q6_K: launch_for_policy<PolicyQ6K>(params, stream);  break;
        case GgufType::Q8_0: launch_for_policy<PolicyQ8_0>(params, stream); break;
        case GgufType::MXFP4: launch_for_policy<PolicyMxfp4>(params, stream); break;
    }
}

}  // namespace layerstorm::compute
