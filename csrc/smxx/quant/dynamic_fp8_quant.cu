// Dynamic FP8 E4M3 quantization kernel: BF16 → FP8 with per-block scales.
//
// Adapted from vLLM csrc/quantization/fp8/scaled_fp8_quant_kernel.cu (Apache-2.0).
// Architecture-agnostic. Block size = 128 elements (matches FP8 GEMM scale layout).

#include "smxx/quant/dynamic_fp8_quant.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

// FP8 E4M3 max representable value.
static constexpr float kFp8E4M3Max = 448.0f;

// Elements per quantization block (matches Fp8GemmParams scale layout).
static constexpr int kBlockSize = 128;

// Threads per CUDA block.
static constexpr int kThreadsPerBlock = 256;

// ── Per-block dynamic quantization kernel ──────────────────────────────────
//
// Grid: (num_blocks_per_row, num_tokens)   where num_blocks_per_row = ceil(K/128)
// Block: kThreadsPerBlock threads
//
// Each CUDA block handles one 128-element quantization block:
//   1. Find block-wide amax via warp shuffle reduction
//   2. Compute scale = amax / fp8_max
//   3. Quantize: output[i] = clamp(input[i] / scale, -fp8_max, fp8_max)
//   4. Store scale

__global__ void dynamic_fp8_quant_kernel(
    __nv_fp8_e4m3* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ scales,
    int hidden_size,
    int num_blocks_per_row,
    int num_tokens,
    bool m_major_scales) {

    const int block_col = blockIdx.x;   // which 128-element block in the row
    const int token_idx = blockIdx.y;   // which token (row)

    const int row_offset = token_idx * hidden_size;
    const int block_start = block_col * kBlockSize;
    const int block_end = min(block_start + kBlockSize, hidden_size);
    const int block_len = block_end - block_start;

    // ── Step 1: Find block-wide amax ──────────────────────────────────────
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < block_len; i += kThreadsPerBlock) {
        float val = __bfloat162float(__ldg(&input[row_offset + block_start + i]));
        local_max = fmaxf(local_max, fabsf(val));
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }

    // Cross-warp reduction via shared memory
    __shared__ float warp_maxes[kThreadsPerBlock / 32];
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    if (lane == 0) warp_maxes[warp_id] = local_max;
    __syncthreads();

    // First warp reduces across warps
    float block_max = 0.0f;
    if (warp_id == 0) {
        const int num_warps = kThreadsPerBlock / 32;
        block_max = (lane < num_warps) ? warp_maxes[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, offset));
        }
    }

    // Broadcast to all threads
    __shared__ float scale_shared;
    if (threadIdx.x == 0) {
        // scale = amax / fp8_max.  If amax == 0, use 1.0 to avoid division by zero.
        scale_shared = (block_max > 0.0f) ? (block_max / kFp8E4M3Max) : 1.0f;
    }
    __syncthreads();

    const float scale = scale_shared;
    const float inv_scale = 1.0f / scale;

    // ── Step 2: Store scale ───────────────────────────────────────────────
    if (threadIdx.x == 0) {
        // M-major (scales[m + k_blk*M]) is the fp8_gemm SFA contract; the
        // row-major layout is kept as default for standalone consumers.
        int idx = m_major_scales ? (token_idx + block_col * num_tokens)
                                 : (token_idx * num_blocks_per_row + block_col);
        scales[idx] = scale;
    }

    // ── Step 3: Quantize ──────────────────────────────────────────────────
    for (int i = threadIdx.x; i < block_len; i += kThreadsPerBlock) {
        float val = __bfloat162float(__ldg(&input[row_offset + block_start + i]));
        float scaled = val * inv_scale;
        // Clamp to FP8 E4M3 range
        scaled = fminf(fmaxf(scaled, -kFp8E4M3Max), kFp8E4M3Max);
        output[row_offset + block_start + i] = __nv_fp8_e4m3(scaled);
    }
}

}  // namespace

namespace layerstorm::compute {

void launch_dynamic_fp8_quant(const DynamicFp8QuantParams& params,
                               void* stream) {
    if (params.num_tokens <= 0 || params.hidden_size <= 0) return;
    if (!params.input || !params.output || !params.scales) {
        throw std::runtime_error(
            "launch_dynamic_fp8_quant: null pointer in params");
    }

    const int num_blocks_per_row = (params.hidden_size + kBlockSize - 1) / kBlockSize;

    dim3 grid(num_blocks_per_row, params.num_tokens);
    dim3 block(kThreadsPerBlock);

    dynamic_fp8_quant_kernel<<<grid, block, 0,
                                static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_fp8_e4m3*>(params.output),
        static_cast<const __nv_bfloat16*>(params.input),
        static_cast<float*>(params.scales),
        params.hidden_size,
        num_blocks_per_row,
        params.num_tokens,
        params.m_major_scales);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_dynamic_fp8_quant failed: ") +
            cudaGetErrorString(err));
    }
}

}  // namespace layerstorm::compute
