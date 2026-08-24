// Q4_K instance of the FAST GGUF mmq int8 tensor-core path (mmq_mma).
//
// Instantiates + launches the 32-aligned kernel body for Q4_K. The activation
// is already quantized to Q8_1 by the dispatch layer; this just configures the
// dynamic smem cap and launches.

#include "sm120/gemm/gguf/mmq_mma/mmq_kernel_32aligned.cuh"
#include "sm120/gemm/gguf/gguf_int_policy.h"
#include "sm120/gemm/gguf/gguf_grouped_int.h"   // GgufGroupedMmqArgs

namespace layerstorm::compute {

template <bool USE_CPASYNC, bool USE_LDMATRIX, class Cfg = mmq_mma::DefaultTileCfg>
static void launch_q4k_impl(const GgufMmvqParams& params,
                            layerstorm::formats::block_q8_1* xq,
                            cudaStream_t stream) {
    using namespace mmq_mma;
    using P = gguf_int::Q4K;
    constexpr int SMEM = SmemSizes32<P::VALS, Cfg::kTILE_M, Cfg::kTILE_N>::total;

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
        cudaFuncSetAttribute(gguf_mmq_mma_32aligned<P, USE_CPASYNC, USE_LDMATRIX, Cfg>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        configured[dev_] = true;
    }
    gguf_mmq_mma_32aligned<P, USE_CPASYNC, USE_LDMATRIX, Cfg>
        <<<grid, dim3(Cfg::kBLOCK_SIZE), SMEM, stream>>>(params, xq);
}

void gguf_mmq_mma_launch_q4k(const GgufMmvqParams& params,
                             layerstorm::formats::block_q8_1* xq,
                             cudaStream_t stream) {
    // cp.async activation streaming is a net win on this tile (see report);
    // it is the default path. ldmatrix did not beat it (see report) — kept off.
    // M-tile heuristic: pick the 64x128 small-M tile when the default 128x128
    // grid underfills the GPU (see mmq_mma_use_small_m).
    using namespace mmq_mma;
    if (mmq_mma_use_small_m(params.M, params.N))
        launch_q4k_impl<true, false, SmallMTileCfg>(params, xq, stream);
    else
        launch_q4k_impl<true, false, DefaultTileCfg>(params, xq, stream);
}

// Benchmark/diagnostic hook: force the non-cp.async (step-1) path.
void gguf_mmq_mma_launch_q4k_baseline(const GgufMmvqParams& params,
                                      layerstorm::formats::block_q8_1* xq,
                                      cudaStream_t stream) {
    launch_q4k_impl</*USE_CPASYNC=*/false, /*USE_LDMATRIX=*/false>(params, xq, stream);
}

// Benchmark/diagnostic hook: cp.async + ldmatrix operand loads.
void gguf_mmq_mma_launch_q4k_ldmatrix(const GgufMmvqParams& params,
                                      layerstorm::formats::block_q8_1* xq,
                                      cudaStream_t stream) {
    launch_q4k_impl</*USE_CPASYNC=*/true, /*USE_LDMATRIX=*/true>(params, xq, stream);
}

// Benchmark/diagnostic hook: force the small-M tile config (64x128).
void gguf_mmq_mma_launch_q4k_smallm(const GgufMmvqParams& params,
                                    layerstorm::formats::block_q8_1* xq,
                                    cudaStream_t stream) {
    launch_q4k_impl<true, false, mmq_mma::SmallMTileCfg>(params, xq, stream);
}

// ── Device-routed grouped mmq (TD-GG5) ─────────────────────────────────────
template <bool USE_CPASYNC, bool USE_LDMATRIX, class Cfg>
static void launch_q4k_grouped_impl(const GgufGroupedMmqArgs& args, int T_max,
                                    cudaStream_t stream) {
    using namespace mmq_mma;
    using P = gguf_int::Q4K;
    constexpr int SMEM = SmemSizes32<P::VALS, Cfg::kTILE_M, Cfg::kTILE_N>::total;
    dim3 grid((args.N + Cfg::kTILE_N - 1) / Cfg::kTILE_N, T_max);
    // cudaFuncSetAttribute is DEVICE-scoped: latch per device, not
    // process-wide (a process-wide latch left every device after the
    // first at the 48 KB default dynamic-smem cap -> silent
    // cudaErrorInvalidValue launches on multi-GPU).
    int dev_ = 0;
    cudaGetDevice(&dev_);
    static bool configured[64] = {};
    if (dev_ >= 0 && dev_ < 64 && !configured[dev_]) {
        cudaFuncSetAttribute(gguf_mmq_mma_32aligned_grouped<P, USE_CPASYNC, USE_LDMATRIX, Cfg>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        configured[dev_] = true;
    }
    gguf_mmq_mma_32aligned_grouped<P, USE_CPASYNC, USE_LDMATRIX, Cfg>
        <<<grid, dim3(Cfg::kBLOCK_SIZE), SMEM, stream>>>(args);
}

void gguf_mmq_mma_grouped_launch_q4k(const GgufGroupedMmqArgs& args, bool use_small,
                                     int T_max, cudaStream_t stream) {
    using namespace mmq_mma;
    if (use_small) launch_q4k_grouped_impl<true, false, SmallMTileCfg>(args, T_max, stream);
    else           launch_q4k_grouped_impl<true, false, DefaultTileCfg>(args, T_max, stream);
}

}  // namespace layerstorm::compute
