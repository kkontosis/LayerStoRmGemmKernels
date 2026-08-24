// GGUF grouped GEMM for MoE expert FFN on SM120.
//
// Int strategy: DEVICE-ROUTED (TD-GG5) — one Q8_1 activation quant for all
// tokens, then on-device expert routing (mmvq decode / mmq prefill) with NO
// host<-device sync, so the path is CUDA-graph capturable. See gguf_grouped_int
// .{h,cu} for the routing (a tile->expert map / per-CTA expert lookup adapted
// from llama.cpp mul_mat_id).
//
// Dequant strategy: the original per-expert HOST dispatch loop (copies
// expert_offsets+B_ptrs to host, one sync, loops the single-expert launchers).
// Retained as the correctness golden (launch_gguf_grouped_gemm_hostloop) and as
// the Dequant implementation. One GgufType + one strategy per call (GG-6).

#include "sm120/gemm/gguf/gguf_grouped_gemm.h"
#include "sm120/gemm/gguf/gguf_grouped_int.h"   // device-routed int path
#include "sm120/gemm/gguf/gguf_mmvq.h"          // GgufMmvqParams, workspace + quantizer
#include "sm120/gemm/gguf/gguf_dequant_gemm.h"  // GgufDequantGemmParams
#include "sm120/gemm/gguf/mmq_mma/mmq_mma.h"    // launch_gguf_mmq_mma (int8 tensor-core)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace layerstorm::compute {

size_t gguf_grouped_gemm_workspace_bytes(int total_tokens, int K, int num_experts) {
    if (total_tokens <= 0 || K <= 0) return 0;
    // Q8_1 scratch for all tokens, + (mmq) a tile->expert map sized for the
    // worst-case 64-tall tile. The Q8_1 region also covers the host-loop's
    // per-expert reuse (M_e <= total_tokens), so this is a superset of both.
    const size_t q8 = gguf_mmvq_workspace_bytes(total_tokens, K);
    const size_t map_off = (q8 + 15) & ~static_cast<size_t>(15);
    const int Tmax = gguf_grouped_mmq_tmax(total_tokens, num_experts, /*tile_m=*/64);
    return map_off + 2ull * static_cast<size_t>(Tmax) * sizeof(int);
}

// ── Public launcher: device-routed Int, host-loop Dequant ──────────────────
void launch_gguf_grouped_gemm(
    const GgufGroupedGemmKernelParams& params, int total_tokens,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream) {

    if (params.num_experts <= 0 || total_tokens <= 0) return;  // nothing to do
    if (params.strategy == GgufGroupedStrategy::Int) {
        launch_gguf_grouped_int(
            params.type, params.num_experts, params.N, params.K, total_tokens,
            params.A_base, params.D_base, params.expert_offsets, params.B_ptrs,
            workspace, workspace_bytes, GgufGroupedIntForce::Auto, stream);
        return;
    }
    // Dequant: the host per-expert dispatch loop (correctness reference).
    launch_gguf_grouped_gemm_hostloop(params, workspace, workspace_bytes, stream);
}

void launch_gguf_grouped_gemm_int_forced(
    const GgufGroupedGemmKernelParams& params, int total_tokens,
    void* workspace, size_t workspace_bytes,
    GgufGroupedIntForce force, cudaStream_t stream) {
    if (params.num_experts <= 0 || total_tokens <= 0) return;
    if (params.strategy != GgufGroupedStrategy::Int)
        throw std::runtime_error("launch_gguf_grouped_gemm_int_forced: Int only");
    launch_gguf_grouped_int(
        params.type, params.num_experts, params.N, params.K, total_tokens,
        params.A_base, params.D_base, params.expert_offsets, params.B_ptrs,
        workspace, workspace_bytes, force, stream);
}

// ── Golden / Dequant: original host per-expert dispatch loop ───────────────
void launch_gguf_grouped_gemm_hostloop(
    const GgufGroupedGemmKernelParams& params,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream) {

    if (params.num_experts <= 0) return;  // nothing to do
    if (params.B_ptrs == nullptr) {
        throw std::runtime_error(
            "GGUF grouped GEMM: B_ptrs required (scattered per-expert blocks)");
    }
    if (params.A_base == nullptr || params.D_base == nullptr ||
        params.expert_offsets == nullptr) {
        throw std::runtime_error("GGUF grouped GEMM: null A/D/expert_offsets");
    }
    const int qk = gguf_block_values(params.type);  // 32 (Q8_0) or 256 (k-quants)
    if (params.K % qk != 0) {
        throw std::runtime_error(
            "GGUF grouped GEMM: K=" + std::to_string(params.K) +
            " must be divisible by the block size " + std::to_string(qk));
    }
    if (params.strategy == GgufGroupedStrategy::Int && (params.K % 32 != 0)) {
        // The Q8_1 activation quantizer works on 32-element blocks.
        throw std::runtime_error(
            "GGUF grouped GEMM (Int): K must be divisible by 32");
    }

    // ── Copy expert_offsets + per-expert B pointers to host (one sync) ─────────
    std::vector<int32_t> h_offsets(params.num_experts + 1);
    std::vector<const void*> h_b_ptrs(params.num_experts);

    cudaMemcpyAsync(h_offsets.data(), params.expert_offsets,
                    (params.num_experts + 1) * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_b_ptrs.data(), params.B_ptrs,
                    params.num_experts * sizeof(void*),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    const int N = params.N;
    const int K = params.K;
    const auto* a_base = params.A_base;
    auto*       d_base = params.D_base;

    for (int e = 0; e < params.num_experts; ++e) {
        const int32_t off = h_offsets[e];
        const int32_t M_e = h_offsets[e + 1] - off;
        if (M_e <= 0) continue;  // empty expert

        const __nv_bfloat16* A_e = a_base + static_cast<int64_t>(off) * K;
        __nv_bfloat16*       D_e = d_base + static_cast<int64_t>(off) * N;
        const void*          B_e = h_b_ptrs[e];

        if (params.strategy == GgufGroupedStrategy::Dequant) {
            GgufDequantGemmParams gp;
            gp.M = M_e; gp.N = N; gp.K = K;
            gp.A = A_e; gp.B = B_e; gp.C = D_e;
            gp.type = params.type;
            launch_gguf_dequant_gemm(gp, stream);
        } else {  // Int
            GgufMmvqParams gp;
            gp.M = M_e; gp.N = N; gp.K = K;
            gp.A = A_e; gp.B = B_e; gp.C = D_e;
            gp.type = params.type;
            // Per-expert Q8_1 scratch sub-allocated from the shared workspace.
            // M_e <= total_tokens, so a workspace sized for total_tokens fits.
            const size_t need = gguf_mmvq_workspace_bytes(M_e, K);
            if (workspace == nullptr || workspace_bytes < need) {
                throw std::runtime_error(
                    "GGUF grouped GEMM (Int): workspace too small (have " +
                    std::to_string(workspace_bytes) + ", need " +
                    std::to_string(need) + ")");
            }
            if (M_e <= 8) {
                launch_gguf_mmvq(gp, workspace, stream);
            } else {
                launch_gguf_mmq_mma(gp, workspace, stream);
            }
        }
    }
}

}  // namespace layerstorm::compute
