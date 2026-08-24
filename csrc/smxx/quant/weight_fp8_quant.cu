// Offline BF16 → FP8 E4M3 weight quantization with tile-level scales.
//
// Quantizes [N, K] row-major BF16 to [N, K] row-major FP8 E4M3 with
// per-tile float32 scales matching CUTLASS SM120 blockwise FP8 GEMM layout.
// Output is row-major (same as input) — the CUTLASS TN kernel reads it
// correctly despite LayoutBTag = ColumnMajor (via make_cute_packed_stride).
// Architecture-agnostic.

#ifndef WEIGHT_FP8_QUANT_CU_INCLUDED
#define WEIGHT_FP8_QUANT_CU_INCLUDED
#include "smxx/quant/weight_fp8_quant.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

// FP8 E4M3 max representable value.
static constexpr float kWqFp8E4M3Max = 448.0f;

// Tile dimensions (matching CUTLASS SM120 blockwise scale config: MmaTileShape 128x128x128).
static constexpr int kTileN = 128;
static constexpr int kTileK = 128;

// Threads per CUDA block.  Each thread processes multiple elements of the tile.
static constexpr int kWqThreadsPerBlock = 256;

// Elements per tile.
static constexpr int kTileElems = kTileN * kTileK;  // 16384

// ── Tile-level weight quantization kernel ────────────────────────────────
//
// Grid: (N_blocks, K_blocks) where N_blocks = ceil(N/128), K_blocks = ceil(K/128)
// Block: kWqThreadsPerBlock threads
//
// Each CUDA block handles one 128x128 tile of the weight matrix:
//   1. Find tile-wide amax via warp shuffle reduction
//   2. Compute scale = amax / fp8_max
//   3. Quantize each element and write in row-major order (no transpose)
//   4. Store tile scale in K-major layout (scale_B contract of launch_fp8_gemm)
//
// NOT in-place safe: step 3 re-reads input elements after writing output —
// with output==input the FP8 byte for element e overwrites the BF16 bytes of
// element e/2, which other warps/blocks may not have read yet (scheduling-
// dependent corruption). Callers must use a separate output buffer.

__global__ void weight_fp8_quant_kernel(
    __nv_fp8_e4m3* __restrict__ output,       // [N, K] row-major (K stride-1)
    const __nv_bfloat16* __restrict__ input,   // [N, K] row-major (K stride-1)
    float* __restrict__ scales,                // [K_blocks, N_blocks] K-major
    int N, int K, int K_blocks) {

    const int n_blk = blockIdx.x;   // tile row index
    const int k_blk = blockIdx.y;   // tile col index

    const int n_start = n_blk * kTileN;
    const int k_start = k_blk * kTileK;
    const int n_end = min(n_start + kTileN, N);
    const int k_end = min(k_start + kTileK, K);
    const int tile_n = n_end - n_start;  // actual rows in this tile (<=128)
    const int tile_k = k_end - k_start;  // actual cols in this tile (<=128)
    const int tile_count = tile_n * tile_k;

    // ── Step 1: Find tile-wide amax ──────────────────────────────────────
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < tile_count; i += kWqThreadsPerBlock) {
        int local_n = i / tile_k;
        int local_k = i % tile_k;
        int global_n = n_start + local_n;
        int global_k = k_start + local_k;
        // Row-major read: input[global_n * K + global_k]
        float val = __bfloat162float(__ldg(&input[global_n * K + global_k]));
        local_max = fmaxf(local_max, fabsf(val));
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }

    // Cross-warp reduction via shared memory
    __shared__ float warp_maxes[kWqThreadsPerBlock / 32];
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    if (lane == 0) warp_maxes[warp_id] = local_max;
    __syncthreads();

    float tile_max = 0.0f;
    if (warp_id == 0) {
        const int num_warps = kWqThreadsPerBlock / 32;
        tile_max = (lane < num_warps) ? warp_maxes[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, offset));
        }
    }

    // Broadcast scale to all threads
    __shared__ float scale_shared;
    if (threadIdx.x == 0) {
        scale_shared = (tile_max > 0.0f) ? (tile_max / kWqFp8E4M3Max) : 1.0f;
    }
    __syncthreads();

    const float scale = scale_shared;
    const float inv_scale = 1.0f / scale;

    // ── Step 2: Store tile scale ─────────────────────────────────────────
    // K-major: scales[k_blk + n_blk * K_blocks], i.e. [K_blocks, N_blocks]
    // column-major — the layout launch_fp8_gemm consumes for scale_B (see
    // fp8_gemm.h and the pad-N path in fp8_gemm.cu, which copies scale_B
    // column-by-column with sB_rows = ceil(K/128)). The previous N-major
    // store (scales[n_blk + k_blk * N_blocks]) sprayed off-diagonal tiles
    // with the wrong scales — empirically cos ≈ 0.95 per GEMM vs reference,
    // enough to flatten DeepSeek V3.2 logits over 61 layers (TD-GOLDEN).
    if (threadIdx.x == 0) {
        scales[k_blk + n_blk * K_blocks] = scale;
    }

    // ── Step 3: Quantize (row-major in, row-major out) ─────────────────
    for (int i = threadIdx.x; i < tile_count; i += kWqThreadsPerBlock) {
        int local_n = i / tile_k;
        int local_k = i % tile_k;
        int global_n = n_start + local_n;
        int global_k = k_start + local_k;

        // Row-major read: input[global_n * K + global_k]
        float val = __bfloat162float(__ldg(&input[global_n * K + global_k]));
        float scaled = val * inv_scale;
        scaled = fminf(fmaxf(scaled, -kWqFp8E4M3Max), kWqFp8E4M3Max);

        // Row-major write: output[global_n * K + global_k]  (K stride-1)
        output[global_n * K + global_k] = __nv_fp8_e4m3(scaled);
    }
}

}  // namespace

namespace layerstorm::compute {

void launch_weight_fp8_quant(const WeightFp8QuantParams& params,
                              void* stream) {
    if (params.N <= 0 || params.K <= 0) return;
    if (!params.input || !params.output || !params.scales) {
        throw std::runtime_error(
            "launch_weight_fp8_quant: null pointer in params");
    }

    const int N_blocks = (params.N + kTileN - 1) / kTileN;
    const int K_blocks = (params.K + kTileK - 1) / kTileK;

    dim3 grid(N_blocks, K_blocks);
    dim3 block(kWqThreadsPerBlock);

    weight_fp8_quant_kernel<<<grid, block, 0,
                               static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_fp8_e4m3*>(params.output),
        static_cast<const __nv_bfloat16*>(params.input),
        static_cast<float*>(params.scales),
        params.N, params.K, K_blocks);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_weight_fp8_quant failed: ") +
            cudaGetErrorString(err));
    }
}

}  // namespace layerstorm::compute

#endif  // WEIGHT_FP8_QUANT_CU_INCLUDED
