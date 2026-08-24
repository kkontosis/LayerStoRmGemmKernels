// FP8 E4M3 blockwise-scaled grouped GEMM for SM120 using CUTLASS 3.x.
//
// Adapted from:
//   sglang csrc/moe/fp8_blockwise_moe_kernel.cu (Apache-2.0)
//   CUTLASS example 87a blackwell_geforce_fp8_bf16_gemm_blockwise (BSD-3-Clause)
//
// Uses per-expert dispatch loop calling the single-expert FP8 GEMM.
// A native CUTLASS grouped GEMM with blockwise scales on SM120 requires
// tuple<Layout*, ScaleLayout*> pointers in the CollectiveBuilder, which is
// not yet supported in CUTLASS for OpClassTensorOp+blockwise scales.
// The dispatch loop reuses the proven single-expert fp8_gemm.cu code.

#include "sm120/gemm/grouped_gemm.h"
#include "sm120/gemm/fp8/fp8_gemm.h"

#include "cutlass/cutlass.h"
#include "cutlass/arch/config.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM120A_SUPPORTED) || \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

namespace {

// ── GPU kernel: read expert_offsets and problem_sizes to host ───────────────
// (We need these on host to dispatch per-expert GEMMs.)

struct Fp8ExpertInfo {
    int32_t token_offset;
    int32_t M;
    int32_t N;
    int32_t K;
};

}  // anonymous namespace

namespace layerstorm::compute {

size_t query_fp8_grouped_gemm_workspace_size(
    int max_experts, int N, int K,
    GemmOutputDtype output_dtype) {
    // Workspace for per-expert dispatch:
    // - Single-expert workspace (max across all experts) — query for M=max_tokens
    //   but M doesn't affect workspace much for FP8; use a reasonable upper bound
    size_t single_ws = query_fp8_gemm_workspace_size(4096, N, K, output_dtype);
    return single_ws;
}

void launch_fp8_grouped_gemm(
    const Fp8GroupedGemmParams& params,
    void* workspace, size_t workspace_bytes,
    void* stream) {

    if ((params.B_ptrs == nullptr) != (params.scale_B_ptrs == nullptr)) {
        throw std::runtime_error(
            "FP8 grouped GEMM: B_ptrs and scale_B_ptrs must be both null or both non-null");
    }
    if (params.K % 16 != 0) {
        throw std::runtime_error(
            "FP8 grouped GEMM: K must be divisible by 16, got K=" +
            std::to_string(params.K));
    }
    if (params.N % 16 != 0) {
        throw std::runtime_error(
            "FP8 grouped GEMM: N must be divisible by 16, got N=" +
            std::to_string(params.N));
    }
    if (params.num_experts <= 0) {
        return;
    }

    auto cuda_stream = static_cast<cudaStream_t>(stream);

    // Copy expert_offsets and problem_sizes to host for dispatch loop.
    // This is a synchronization point, but acceptable for the fallback path.
    std::vector<int32_t> h_offsets(params.num_experts + 1);
    std::vector<int32_t> h_problems(params.num_experts * 3);

    cudaMemcpyAsync(h_offsets.data(), params.expert_offsets,
                     (params.num_experts + 1) * sizeof(int32_t),
                     cudaMemcpyDeviceToHost, cuda_stream);
    cudaMemcpyAsync(h_problems.data(), params.problem_sizes,
                     params.num_experts * 3 * sizeof(int32_t),
                     cudaMemcpyDeviceToHost, cuda_stream);

    // If per-expert B pointers provided, copy to host alongside metadata.
    std::vector<const void*> h_b_ptrs;
    std::vector<const void*> h_scb_ptrs;
    if (params.B_ptrs) {
        h_b_ptrs.resize(params.num_experts);
        cudaMemcpyAsync(h_b_ptrs.data(), params.B_ptrs,
                         params.num_experts * sizeof(void*),
                         cudaMemcpyDeviceToHost, cuda_stream);
    }
    if (params.scale_B_ptrs) {
        h_scb_ptrs.resize(params.num_experts);
        cudaMemcpyAsync(h_scb_ptrs.data(), params.scale_B_ptrs,
                         params.num_experts * sizeof(void*),
                         cudaMemcpyDeviceToHost, cuda_stream);
    }

    cudaStreamSynchronize(cuda_stream);

    int N = params.N;
    int K = params.K;
    int scale_k_blocks = (K + 127) / 128;  // FP8 blockwise: one scale per 128 elements

    auto* a_base = static_cast<const char*>(params.A_base);
    auto* b_base = static_cast<const char*>(params.B_base);
    auto* d_base = static_cast<char*>(params.D_base);
    auto* sca_base = static_cast<const char*>(params.scale_A_base);
    auto* scb_base = static_cast<const char*>(params.scale_B_base);

    for (int e = 0; e < params.num_experts; e++) {
        int32_t M_e = h_problems[e * 3];
        if (M_e <= 0) continue;  // Empty expert, skip

        int32_t token_off = h_offsets[e];

        Fp8GemmParams single_params;
        single_params.M = M_e;
        single_params.N = N;
        single_params.K = K;
        single_params.output_dtype = params.output_dtype;

        // A: [total_tokens, K] FP8, offset by token_off * K bytes
        single_params.A = a_base + static_cast<int64_t>(token_off) * K;
        // B: per-expert pointer if provided, else contiguous [num_experts, N, K]
        single_params.B = !h_b_ptrs.empty()
            ? static_cast<const char*>(h_b_ptrs[e])
            : b_base + static_cast<int64_t>(e) * N * K;
        // D: [total_tokens, N] output (2 bytes per elem for BF16/FP16)
        int out_elem_bytes = (params.output_dtype == GemmOutputDtype::kBFloat16) ? 2 : 2;
        single_params.D = d_base + static_cast<int64_t>(token_off) * N * out_elem_bytes;

        // scale_A: float32 [total_tokens, ceil(K/128)], M-major
        // offset by token_off * scale_k_blocks * sizeof(float)
        single_params.scale_A = sca_base +
            static_cast<int64_t>(token_off) * scale_k_blocks * sizeof(float);
        // scale_B: per-expert pointer if provided, else contiguous
        int scale_n_blocks = (N + 127) / 128;
        single_params.scale_B = !h_scb_ptrs.empty()
            ? static_cast<const char*>(h_scb_ptrs[e])
            : scb_base + static_cast<int64_t>(e) * scale_k_blocks * scale_n_blocks * sizeof(float);

        launch_fp8_gemm(single_params, workspace, stream);
    }
}

}  // namespace layerstorm::compute

#else  // !SM120 supported

namespace layerstorm::compute {

size_t query_fp8_grouped_gemm_workspace_size(int, int, int, GemmOutputDtype) {
    throw std::runtime_error(
        "FP8 grouped GEMM requires SM120 "
        "(CUTLASS_ARCH_MMA_SM120_SUPPORTED not defined)");
}

void launch_fp8_grouped_gemm(const Fp8GroupedGemmParams&, void*, size_t, void*) {
    throw std::runtime_error(
        "FP8 grouped GEMM requires SM120 "
        "(CUTLASS_ARCH_MMA_SM120_SUPPORTED not defined)");
}

}  // namespace layerstorm::compute

#endif
