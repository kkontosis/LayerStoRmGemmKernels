// GGUF integer mat-mat (mmq), dp4a tiled variant.
//
// C[M,N] = A[M,K] @ dequant(W)^T, with the activation quantized to Q8_1 and the
// weight kept packed (dequanted to centered int8 per K super-block into shared
// memory). 64x64 output tile, 256 threads (4x4 per thread). Per 4-value group:
//   acc += d_x * (A_j * dp4a(Qw, qx) + B_j * Σqx)
// matching the mmvq math, but with tile data-reuse for prefill throughput.

#include "sm120/gemm/gguf/gguf_mmq.h"
#include "sm120/gemm/gguf/gguf_int_policy.h"
#include "formats/q8_1_format.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

namespace layerstorm::compute {

namespace {

namespace fmt = layerstorm::formats;
using fmt::block_q8_1;

static constexpr int TILE_M = 64;
static constexpr int TILE_N = 64;
static constexpr int BLOCK_SIZE = 256;
static constexpr int TM = 4;
static constexpr int TN = 4;
static constexpr int THREAD_N = TILE_N / TN;  // 16

// Shared-memory byte size for a given VALS (per-instantiation).
template <int VALS>
struct SmemSizes {
    static constexpr int N32 = VALS / 32;       // q8_1 blocks per K-step
    static constexpr int NG  = VALS / 4;        // 4-value groups per K-step
    static constexpr int qx_bytes = TILE_M * VALS;                 // int8
    static constexpr int qw_bytes = TILE_N * VALS;                 // int8
    static constexpr int dx_bytes = TILE_M * N32 * (int)sizeof(float);
    static constexpr int a_bytes  = TILE_N * NG  * (int)sizeof(float);
    static constexpr int b_bytes  = TILE_N * NG  * (int)sizeof(float);
    static constexpr int total = qx_bytes + qw_bytes + dx_bytes + a_bytes + b_bytes;
};

template <class P>
__global__ void __launch_bounds__(BLOCK_SIZE)
gguf_mmq_kernel(GgufMmvqParams params, const block_q8_1* __restrict__ xq) {
    constexpr int VALS = P::VALS;
    constexpr int N32 = VALS / 32;
    constexpr int NG = VALS / 4;
    using S = SmemSizes<VALS>;

    const int M = params.M, N = params.N, K = params.K;
    const int n_start = blockIdx.x * TILE_N;
    const int m_start = blockIdx.y * TILE_M;
    const int tid = threadIdx.x;
    const int tx = tid % THREAD_N;   // 0..15 (N)
    const int ty = tid / THREAD_N;   // 0..15 (M)

    extern __shared__ char smem[];
    int8_t* sQx = reinterpret_cast<int8_t*>(smem);
    int8_t* sQw = reinterpret_cast<int8_t*>(smem + S::qx_bytes);
    float*  sDx = reinterpret_cast<float*>(smem + S::qx_bytes + S::qw_bytes);
    float*  sA  = reinterpret_cast<float*>(smem + S::qx_bytes + S::qw_bytes + S::dx_bytes);
    float*  sB  = reinterpret_cast<float*>(smem + S::qx_bytes + S::qw_bytes + S::dx_bytes + S::a_bytes);

    const int x_bpr = K / 32;
    const int nblk = K / VALS;
    const uint8_t* Wbase = reinterpret_cast<const uint8_t*>(params.B);

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int kb = 0; kb < nblk; ++kb) {
        // --- load activation quants + per-32 scales ---
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
        // --- dequant weight tile to centered int8 + per-group (A,B) ---
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
            sA[nr * NG + g] = A;
            sB[nr * NG + g] = B;
        }
        __syncthreads();

        // --- compute ---
        #pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int b32 = (4 * g) / 32;
            int qx[TM]; float dx[TM]; int sumqx[TM];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int mr = ty * TM + i;
                qx[i] = *reinterpret_cast<const int*>(&sQx[mr * VALS + 4 * g]);
                dx[i] = sDx[mr * N32 + b32];
                sumqx[i] = __dp4a(qx[i], 0x01010101, 0);
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int nr = tx * TN + j;
                const int Qw = *reinterpret_cast<const int*>(&sQw[nr * VALS + 4 * g]);
                const float A = sA[nr * NG + g], B = sB[nr * NG + g];
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const int dotQ = __dp4a(Qw, qx[i], 0);
                    acc[i][j] += dx[i] * (A * dotQ + B * sumqx[i]);
                }
            }
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
            if (ng < N) params.C[(size_t)mg * N + ng] = __float2bfloat16_rn(acc[i][j]);
        }
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
        cudaFuncSetAttribute(gguf_mmq_kernel<P>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        configured[dev_] = true;
    }
    gguf_mmq_kernel<P><<<grid, dim3(BLOCK_SIZE), SMEM, stream>>>(params, xq);
}

}  // namespace

void launch_gguf_mmq(const GgufMmvqParams& params, void* q8_1_workspace, cudaStream_t stream) {
    const int qk = gguf_block_values(params.type);
    if (params.K % qk != 0) {
        throw std::invalid_argument(
            "gguf_mmq: K must be divisible by " + std::to_string(qk) +
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
