// Q6_K instance of the FAST GGUF mmq int8 tensor-core path (mmq_mma).
//
// Q6_K is the 16-subblock representative: VALS=256, 16 sub-blocks of 16,
// center-quant (B=0, A = d_w * scales[j], Q = q6 - 32). It uses the k=16 kernel
// body (gguf_mmq_mma_16subblock) — one MMA per 16-element sub-block — with the
// per-16-chunk affine rescale. The activation is already quantized to Q8_1 by
// the dispatch layer; this just configures the dynamic smem cap and launches.
//
// Next agent (Q2_K/Q3_K): this file is the template. Q3_K is a near-copy with
// P = gguf_int::Q3K (also center-quant, B=0). Q2_K is a near-copy with
// P = gguf_int::Q2K (min-type, B != 0 — already handled generically by the
// per-16-chunk rescale, which folds B*sumqx). No kernel change needed for any
// of them; only the policy and the launch wrapper name differ.

#include "sm120/gemm/gguf/mmq_mma/mmq_kernel_16subblock.cuh"
#include "sm120/gemm/gguf/gguf_int_policy.h"
#include "sm120/gemm/gguf/gguf_grouped_int.h"   // GgufGroupedMmqArgs

namespace layerstorm::compute {

template <bool USE_CPASYNC, class Cfg = mmq_mma::DefaultTileCfg>
static void launch_q6k_impl(const GgufMmvqParams& params,
                            layerstorm::formats::block_q8_1* xq,
                            cudaStream_t stream) {
    using namespace mmq_mma;
    using P = gguf_int::Q6K;
    constexpr int SMEM = SmemSizes16<P::VALS, Cfg::kTILE_M, Cfg::kTILE_N>::total;

    dim3 grid((params.N + Cfg::kTILE_N - 1) / Cfg::kTILE_N,
              (params.M + Cfg::kTILE_M - 1) / Cfg::kTILE_M);

    // cudaFuncSetAttribute is DEVICE-scoped: latch per device, not
    // process-wide (a process-wide latch left every device after the
    // first at the 48 KB default dynamic-smem cap -> silent
    // cudaErrorInvalidValue launches on multi-GPU).
    int dev_ = 0;
    cudaGetDevice(&dev_);
    static bool configured[64] = {};
    if (dev_ >= 0 && dev_ < 64 && !configured[dev_]) {
        cudaFuncSetAttribute(gguf_mmq_mma_16subblock<P, USE_CPASYNC, Cfg>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        configured[dev_] = true;
    }
    gguf_mmq_mma_16subblock<P, USE_CPASYNC, Cfg>
        <<<grid, dim3(Cfg::kBLOCK_SIZE), SMEM, stream>>>(params, xq);
}

void gguf_mmq_mma_launch_q6k(const GgufMmvqParams& params,
                             layerstorm::formats::block_q8_1* xq,
                             cudaStream_t stream) {
    // cp.async + M-tile heuristic (see mmq_mma_use_small_m).
    using namespace mmq_mma;
    if (mmq_mma_use_small_m(params.M, params.N))
        launch_q6k_impl<true, SmallMTileCfg>(params, xq, stream);
    else
        launch_q6k_impl<true, DefaultTileCfg>(params, xq, stream);
}

// Benchmark/diagnostic hook: force the non-cp.async path.
void gguf_mmq_mma_launch_q6k_baseline(const GgufMmvqParams& params,
                                      layerstorm::formats::block_q8_1* xq,
                                      cudaStream_t stream) {
    launch_q6k_impl</*USE_CPASYNC=*/false>(params, xq, stream);
}

// Benchmark/diagnostic hook: force the small-M tile config (64x128).
void gguf_mmq_mma_launch_q6k_smallm(const GgufMmvqParams& params,
                                    layerstorm::formats::block_q8_1* xq,
                                    cudaStream_t stream) {
    launch_q6k_impl<true, mmq_mma::SmallMTileCfg>(params, xq, stream);
}

// ── Device-routed grouped mmq (TD-GG5) ─────────────────────────────────────
template <bool USE_CPASYNC, class Cfg>
static void launch_q6k_grouped_impl(const GgufGroupedMmqArgs& args, int T_max,
                                    cudaStream_t stream) {
    using namespace mmq_mma;
    using P = gguf_int::Q6K;
    constexpr int SMEM = SmemSizes16<P::VALS, Cfg::kTILE_M, Cfg::kTILE_N>::total;
    dim3 grid((args.N + Cfg::kTILE_N - 1) / Cfg::kTILE_N, T_max);
    // cudaFuncSetAttribute is DEVICE-scoped: latch per device, not
    // process-wide (a process-wide latch left every device after the
    // first at the 48 KB default dynamic-smem cap -> silent
    // cudaErrorInvalidValue launches on multi-GPU).
    int dev_ = 0;
    cudaGetDevice(&dev_);
    static bool configured[64] = {};
    if (dev_ >= 0 && dev_ < 64 && !configured[dev_]) {
        cudaFuncSetAttribute(gguf_mmq_mma_16subblock_grouped<P, USE_CPASYNC, Cfg>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        configured[dev_] = true;
    }
    gguf_mmq_mma_16subblock_grouped<P, USE_CPASYNC, Cfg>
        <<<grid, dim3(Cfg::kBLOCK_SIZE), SMEM, stream>>>(args);
}

void gguf_mmq_mma_grouped_launch_q6k(const GgufGroupedMmqArgs& args, bool use_small,
                                     int T_max, cudaStream_t stream) {
    using namespace mmq_mma;
    if (use_small) launch_q6k_grouped_impl<true, SmallMTileCfg>(args, T_max, stream);
    else           launch_q6k_grouped_impl<true, DefaultTileCfg>(args, T_max, stream);
}

}  // namespace layerstorm::compute
