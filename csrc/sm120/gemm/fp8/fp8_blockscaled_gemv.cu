// FP8 E4M3 blockwise-scaled GEMV (M == 1) for SM120.
//
// Hand-written CUDA-core dequant-GEMV for the decode path. Bandwidth-bound:
// the dominant traffic is the FP8 weight B [N, K]. One warp computes one output
// row D[0, n]; lanes stride over K with vectorized 16-byte (uint4 = 16 FP8)
// loads of B. The activation row (K FP8 bytes + ceil(K/128) scale_A floats) is
// staged once per block in shared memory.
//
// Numerics match launch_fp8_gemm (alpha == 1.0):
//   D[0,n] = sum_kb scale_A[kb]*scale_B[kb,nb] * sum_{k in block kb} A_f[k]*B_f[n,k]
// Scales are per-128-wide-K-block constants, so the per-element products are
// accumulated in float within a K-block, then folded into the row total scaled
// by scale_A[kb]*scale_B[kb,nb]. This is mathematically identical to scaling
// each element (block scale is a common factor) and avoids per-element
// multiplies by the scales.

#include "sm120/gemm/fp8/fp8_blockscaled_gemv.h"

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>

namespace layerstorm::compute {

namespace {

constexpr int WARP_SIZE       = 32;
constexpr int ROWS_PER_BLOCK  = 8;                       // warps per block = N rows per block
constexpr int BLOCK_SIZE      = ROWS_PER_BLOCK * WARP_SIZE;  // 256
constexpr int K_BLOCK         = 128;                     // scale block size in K
constexpr int VEC_FP8         = 16;                      // FP8 per uint4 load

// Float -> output dtype, type-specialized (avoids constexpr both-branch checking).
__forceinline__ __device__ void store_out(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16_rn(v);
}
__forceinline__ __device__ void store_out(__half* p, float v) {
    *p = __float2half_rn(v);
}

// Reinterpret a uint4 as 16 FP8 E4M3 values and convert to float.
__forceinline__ __device__ void unpack_u4_fp8(const uint4& v, float out[VEC_FP8]) {
    const __nv_fp8_storage_t* bytes =
        reinterpret_cast<const __nv_fp8_storage_t*>(&v);
    #pragma unroll
    for (int i = 0; i < VEC_FP8; ++i) {
        out[i] = __half2float(__nv_cvt_fp8_to_halfraw(bytes[i], __NV_E4M3));
    }
}

template <typename OutType>
__global__ void __launch_bounds__(BLOCK_SIZE)
fp8_blockscaled_gemv_kernel(const __nv_fp8_e4m3* __restrict__ A,
                            const __nv_fp8_e4m3* __restrict__ B,
                            OutType* __restrict__ D,
                            const float* __restrict__ scale_A,
                            const float* __restrict__ scale_B,
                            int N, int K) {
    const int nb_k = (K + K_BLOCK - 1) / K_BLOCK;   // K-blocks
    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;               // 0..ROWS_PER_BLOCK-1
    const int lane = tid % WARP_SIZE;

    // Shared staging of the (single) activation row: K FP8 bytes + nb_k scales.
    extern __shared__ char smem_raw[];
    __nv_fp8_e4m3* sA = reinterpret_cast<__nv_fp8_e4m3*>(smem_raw);
    // Align scale array to float (K is %16==0, so K bytes is 16-aligned).
    float* sScaleA = reinterpret_cast<float*>(smem_raw + K);

    // Cooperative load of A row (coalesced, 16 bytes per thread per step).
    const uint4* A_vec = reinterpret_cast<const uint4*>(A);
    uint4* sA_vec = reinterpret_cast<uint4*>(sA);
    const int n_vecs = K / VEC_FP8;  // K % 16 == 0 guaranteed by caller
    for (int v = tid; v < n_vecs; v += BLOCK_SIZE) {
        sA_vec[v] = A_vec[v];
    }
    for (int kb = tid; kb < nb_k; kb += BLOCK_SIZE) {
        sScaleA[kb] = scale_A[kb];
    }
    __syncthreads();

    const int n = blockIdx.x * ROWS_PER_BLOCK + warp;
    if (n >= N) return;

    const __nv_fp8_e4m3* Brow = B + static_cast<size_t>(n) * K;
    const uint4* Brow_vec = reinterpret_cast<const uint4*>(Brow);
    const int nb = n / K_BLOCK;                 // N-block index for scale_B
    const float* sB_col = scale_B + static_cast<size_t>(nb) * nb_k;  // K-major column

    const int n_vecs_row = K / VEC_FP8;         // 16 elems per vec; K%16==0
    constexpr int VECS_PER_KBLOCK = K_BLOCK / VEC_FP8;  // 8 vectors per 128-K-block
    constexpr int UNROLL = 4;                   // outstanding loads (memory-level parallelism)

    float row_acc = 0.0f;  // accumulated, fully scaled

    // Stream the full row contiguously with all 32 lanes active. A 16-wide
    // vector never straddles a 128-K-block boundary (128/16 == 8), so each
    // vector's scale is the single (scale_A[kb], scale_B[kb,nb]) pair for
    // kb = vec_idx / 8 — folded per-vector. UNROLL outstanding loads hide
    // global-memory latency (the dominant cost at M=1).
    int v = lane;
    for (; v + (UNROLL - 1) * WARP_SIZE < n_vecs_row; v += UNROLL * WARP_SIZE) {
        uint4 bv[UNROLL], av[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int vidx = v + u * WARP_SIZE;
            bv[u] = Brow_vec[vidx];
            av[u] = sA_vec[vidx];
        }
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int vidx = v + u * WARP_SIZE;
            const int kb = vidx / VECS_PER_KBLOCK;
            const float scale = sScaleA[kb] * sB_col[kb];
            float bf[VEC_FP8], af[VEC_FP8];
            unpack_u4_fp8(bv[u], bf);
            unpack_u4_fp8(av[u], af);
            #pragma unroll
            for (int i = 0; i < VEC_FP8; ++i)
                row_acc = fmaf(af[i] * scale, bf[i], row_acc);
        }
    }
    // Remainder vectors (fewer than UNROLL*WARP_SIZE left).
    for (; v < n_vecs_row; v += WARP_SIZE) {
        const int kb = v / VECS_PER_KBLOCK;
        const float scale = sScaleA[kb] * sB_col[kb];
        float bf[VEC_FP8], af[VEC_FP8];
        unpack_u4_fp8(Brow_vec[v], bf);
        unpack_u4_fp8(sA_vec[v], af);
        #pragma unroll
        for (int i = 0; i < VEC_FP8; ++i)
            row_acc = fmaf(af[i] * scale, bf[i], row_acc);
    }

    // Warp reduction over lanes.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        row_acc += __shfl_down_sync(0xffffffffu, row_acc, offset);

    if (lane == 0) {
        store_out(&D[n], row_acc);
    }
}

}  // anonymous namespace

void launch_fp8_blockscaled_gemv(const Fp8GemmParams& params, void* stream) {
    if (params.M != 1)
        throw std::runtime_error("fp8_blockscaled_gemv requires M == 1, got M=" +
                                 std::to_string(params.M));
    if (params.K % 16 != 0)
        throw std::runtime_error("fp8_blockscaled_gemv: K must be divisible by 16");

    auto cuda_stream = static_cast<cudaStream_t>(stream);
    const int N = params.N;
    const int K = params.K;
    const int nb_k = (K + K_BLOCK - 1) / K_BLOCK;

    dim3 grid((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    dim3 block(BLOCK_SIZE);
    // smem: K FP8 bytes + nb_k float scales.
    size_t smem_bytes = static_cast<size_t>(K) + static_cast<size_t>(nb_k) * sizeof(float);

    const auto* A  = static_cast<const __nv_fp8_e4m3*>(params.A);
    const auto* B  = static_cast<const __nv_fp8_e4m3*>(params.B);
    const auto* sA = static_cast<const float*>(params.scale_A);
    const auto* sB = static_cast<const float*>(params.scale_B);

    switch (params.output_dtype) {
        case GemmOutputDtype::kBFloat16:
            fp8_blockscaled_gemv_kernel<__nv_bfloat16>
                <<<grid, block, smem_bytes, cuda_stream>>>(
                    A, B, static_cast<__nv_bfloat16*>(params.D), sA, sB, N, K);
            break;
        case GemmOutputDtype::kFloat16:
            fp8_blockscaled_gemv_kernel<__half>
                <<<grid, block, smem_bytes, cuda_stream>>>(
                    A, B, static_cast<__half*>(params.D), sA, sB, N, K);
            break;
        default:
            throw std::runtime_error("fp8_blockscaled_gemv: unsupported output dtype");
    }
}

}  // namespace layerstorm::compute
