// FP8 E4M3 blockwise-scaled GEMM for SM120 using CUTLASS 3.x.
//
// Adapted from:
//   sglang fp8_blockwise_gemm_kernel.cu (Apache-2.0)
//   CUTLASS example 87a blackwell_geforce_fp8_bf16_gemm_blockwise (BSD-3-Clause)
//
// Uses OpClassTensorOp with Sm120BlockwiseScaleConfig (per-row in M,
// per-128-element-block in N and K). Float32 per-block scales.
//
// Two tile configurations for M-dimension dispatch:
//   M <  128: 64x128x128 (pingpong schedule, no M>=128 constraint)
//   M >= 128: 128x128x128 (cooperative schedule, better throughput)

#include "sm120/gemm/fp8/fp8_gemm.h"
#include "sm120/gemm/fp8/fp8_blockscaled_gemv.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <string>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM120A_SUPPORTED) || \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

namespace {

// ── Tile configurations ─────────────────────────────────────────────────────

struct TileConfig_Cooperative {
    using MmaTileShape   = Shape<_128, _128, _128>;
    using PerSmTileShape = Shape<_128, _128, _128>;
    using ClusterShape   = Shape<_1, _1, _1>;
    // Per-row in M: ScaleGranularityM = TileM / ScalesPerTileM = 128/128 = 1
    using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
        1,   // ScaleGranularityM: one scale per row
        128, // ScaleGranularityN: one scale per 128 columns
        128, // ScaleGranularityK: one scale per 128 elements in K
        cute::UMMA::Major::MN,
        cute::UMMA::Major::K>;
    using KernelScheduleTag = cutlass::gemm::KernelScheduleSm120Blockwise;
};

struct TileConfig_Pingpong {
    using MmaTileShape   = Shape<_64, _128, _128>;
    using PerSmTileShape = Shape<_64, _128, _128>;
    using ClusterShape   = Shape<_1, _1, _1>;
    // Per-row in M: ScaleGranularityM = TileM / ScalesPerTileM = 64/64 = 1
    using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
        1,   // ScaleGranularityM
        128, // ScaleGranularityN
        128, // ScaleGranularityK
        cute::UMMA::Major::MN,
        cute::UMMA::Major::K>;
    using KernelScheduleTag = cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
};

// ── GEMM type alias template ────────────────────────────────────────────────

template <typename Config, typename OutType>
struct Fp8GemmType {
    using ElementA    = cutlass::float_e4m3_t;
    using LayoutATag  = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16

    using ElementB    = cutlass::float_e4m3_t;
    using LayoutBTag  = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16

    using ElementD    = OutType;
    using ElementC    = void;  // No residual input
    using LayoutCTag  = cutlass::layout::RowMajor;
    using LayoutDTag  = cutlass::layout::RowMajor;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentC = AlignmentD;

    using ElementAccumulator = float;
    using ElementCompute     = float;
    using ArchTag       = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    using MmaTileShape     = typename Config::MmaTileShape;
    using PerSmTileShape   = typename Config::PerSmTileShape;
    using ClusterShape     = typename Config::ClusterShape;
    using ScaleConfig      = typename Config::ScaleConfig;
    using KernelScheduleTag = typename Config::KernelScheduleTag;

    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        PerSmTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, cute::tuple<LayoutATag, LayoutSFA>, AlignmentA,
        ElementB, cute::tuple<LayoutBTag, LayoutSFB>, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelScheduleTag
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop, CollectiveEpilogue, void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// ── Run a single GEMM ──────────────────────────────────────────────────────

template <typename Config, typename OutType>
void run_fp8_gemm(const layerstorm::compute::Fp8GemmParams& params,
                   void* workspace, cudaStream_t stream) {
    using GT = Fp8GemmType<Config, OutType>;
    using Gemm       = typename GT::Gemm;
    using GemmKernel = typename GT::GemmKernel;
    using ElementA   = typename GT::ElementA;
    using ElementB   = typename GT::ElementB;
    using ElementD   = typename GT::ElementD;
    using ScaleConfig = typename GT::ScaleConfig;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideD = typename GemmKernel::StrideD;
    using LayoutSFA = typename GT::LayoutSFA;
    using LayoutSFB = typename GT::LayoutSFB;

    int M = params.M, N = params.N, K = params.K;

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    typename GemmKernel::MainloopArguments mainloop_args{
        static_cast<ElementA const*>(params.A), stride_a,
        static_cast<ElementB const*>(params.B), stride_b,
        static_cast<float const*>(params.scale_A), layout_SFA,
        static_cast<float const*>(params.scale_B), layout_SFB};

    typename GemmKernel::EpilogueArguments epilogue_args{
        {},
        static_cast<ElementD const*>(params.D), stride_d,
        static_cast<ElementD*>(params.D), stride_d};
    epilogue_args.thread.alpha = 1.0f;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        mainloop_args,
        epilogue_args};

    Gemm gemm;

    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("FP8 GEMM can_implement failed (M=") +
            std::to_string(M) + " N=" + std::to_string(N) +
            " K=" + std::to_string(K) + "): " +
            cutlassGetStatusString(status));
    }

    status = gemm.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("FP8 GEMM initialize failed: ") +
            cutlassGetStatusString(status));
    }

    status = gemm.run(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("FP8 GEMM run failed: ") +
            cutlassGetStatusString(status));
    }
}

// ── Padded cooperative fallback ──────────────────────────────────────────────
//
// SM120 SFA rejects shapes where M or N don't satisfy scale-factor atom
// alignment (e.g. M=1 decode, N=576 kv_a_proj). Pad both to multiples of 128,
// run cooperative, copy valid output back. Synchronous cudaMalloc/cudaFree
// keep the padding self-contained (TD-70a tracks switching to async).

// Helper: checked cudaMalloc with throw on failure.
static void* checked_alloc(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("FP8 GEMM padding alloc failed (") +
            std::to_string(bytes) + " bytes): " + cudaGetErrorString(err));
    return ptr;
}

// TD-70d: RAII guard that frees all tracked device pointers on destruction.
// Prevents leaks when checked_alloc or run_fp8_gemm throws mid-sequence.
struct PadBufferGuard {
    void* ptrs[5] = {};
    int count = 0;

    void* alloc(size_t bytes) {
        if (count >= 5)
            throw std::runtime_error("PadBufferGuard: exceeded 5-pointer capacity");
        void* p = checked_alloc(bytes);
        ptrs[count++] = p;
        return p;
    }

    void release_all() {
        for (int i = 0; i < count; ++i) {
            cudaFree(ptrs[i]);
            ptrs[i] = nullptr;
        }
        count = 0;
    }

    ~PadBufferGuard() { release_all(); }

    PadBufferGuard() = default;
    PadBufferGuard(const PadBufferGuard&) = delete;
    PadBufferGuard& operator=(const PadBufferGuard&) = delete;
};

template <typename OutType>
void run_padded_cooperative(const layerstorm::compute::Fp8GemmParams& params,
                            void* workspace, cudaStream_t stream) {
    const int M_orig = params.M;
    const int N_orig = params.N;
    const int K_val  = params.K;
    const int M_pad  = ((M_orig + 127) / 128) * 128;
    const int N_pad  = ((N_orig + 127) / 128) * 128;
    const bool need_pad_m = (M_pad != M_orig);
    const bool need_pad_n = (N_pad != N_orig);

    const size_t sA_cols = (K_val + 127) / 128;
    const size_t sB_rows = sA_cols;
    const size_t sB_cols_orig = (N_orig + 127) / 128;
    const size_t D_elem = sizeof(OutType);

    // TD-70d: RAII guard frees all temporaries on any exception or early exit.
    PadBufferGuard guard;

    void *pad_A = nullptr, *pad_sA = nullptr, *pad_D = nullptr;
    void *pad_B = nullptr, *pad_sB = nullptr;

    // Pad A + scale_A if M needs padding
    if (need_pad_m) {
        const size_t A_bytes  = static_cast<size_t>(M_pad) * K_val;
        const size_t sA_bytes = static_cast<size_t>(M_pad) * sA_cols * sizeof(float);
        pad_A  = guard.alloc(A_bytes);
        pad_sA = guard.alloc(sA_bytes);
        cudaMemsetAsync(pad_A, 0, A_bytes, stream);
        cudaMemsetAsync(pad_sA, 0, sA_bytes, stream);
        cudaMemcpyAsync(pad_A, params.A,
                         static_cast<size_t>(M_orig) * K_val,
                         cudaMemcpyDeviceToDevice, stream);
        // scale_A [M, ceil(K/128)] M-major: column k_blk lives at
        // buf[m + k_blk*M], so the column stride changes M_orig → M_pad and
        // the copy must go column by column. A single contiguous copy lands
        // every column at the wrong stride — for M=1 decode it left scale_A
        // zero for all k_blk > 0, truncating the GEMM to the first 128-wide
        // K block (LayerStoRm3 TD-GOLDEN: kv_a_proj N=576 is the only
        // attention GEMM that takes this pad path; its output was a 1/56th
        // partial sum).
        for (size_t c = 0; c < sA_cols; ++c) {
            cudaMemcpyAsync(
                static_cast<char*>(pad_sA) + c * M_pad * sizeof(float),
                static_cast<const char*>(params.scale_A) + c * M_orig * sizeof(float),
                static_cast<size_t>(M_orig) * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
        }
    }

    // Pad B + scale_B if N needs padding
    if (need_pad_n) {
        const size_t B_bytes  = static_cast<size_t>(N_pad) * K_val;
        const size_t sB_cols_pad = (N_pad + 127) / 128;
        const size_t sB_bytes = sB_rows * sB_cols_pad * sizeof(float);
        pad_B  = guard.alloc(B_bytes);
        pad_sB = guard.alloc(sB_bytes);
        cudaMemsetAsync(pad_B, 0, B_bytes, stream);
        cudaMemsetAsync(pad_sB, 0, sB_bytes, stream);
        // B is col-major [N, K]: N_orig rows of K elements contiguous
        cudaMemcpyAsync(pad_B, params.B,
                         static_cast<size_t>(N_orig) * K_val,
                         cudaMemcpyDeviceToDevice, stream);
        // scale_B [ceil(K/128), ceil(N/128)] K-major: copy column by column
        for (size_t c = 0; c < sB_cols_orig; ++c) {
            cudaMemcpyAsync(
                static_cast<char*>(pad_sB) + c * sB_rows * sizeof(float),
                static_cast<const char*>(params.scale_B) + c * sB_rows * sizeof(float),
                sB_rows * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
        }
    }

    // D always needs allocation when any dimension is padded
    const size_t D_bytes = static_cast<size_t>(M_pad) * N_pad * D_elem;
    pad_D = guard.alloc(D_bytes);

    layerstorm::compute::Fp8GemmParams padded = params;
    padded.M = M_pad;
    padded.N = N_pad;
    if (pad_A)  padded.A = pad_A;
    if (pad_sA) padded.scale_A = pad_sA;
    if (pad_B)  padded.B = pad_B;
    if (pad_sB) padded.scale_B = pad_sB;
    padded.D = pad_D;

    run_fp8_gemm<TileConfig_Cooperative, OutType>(padded, workspace, stream);

    // Verify no kernel crash from the padded GEMM before copying results.
    {
        cudaError_t kerr = cudaStreamSynchronize(stream);
        if (kerr != cudaSuccess) {
            // guard destructor frees all allocated buffers.
            throw std::runtime_error(
                std::string("FP8 padded GEMM kernel failed (M_pad=") +
                std::to_string(M_pad) + " N_pad=" + std::to_string(N_pad) +
                " K=" + std::to_string(K_val) + "): " +
                cudaGetErrorString(kerr));
        }
    }

    // Copy valid output: first M_orig rows × first N_orig cols from [M_pad, N_pad]
    if (!need_pad_n) {
        // Same row stride — single copy
        cudaMemcpyAsync(params.D, pad_D,
                         static_cast<size_t>(M_orig) * N_orig * D_elem,
                         cudaMemcpyDeviceToDevice, stream);
    } else {
        // Different row strides — row-by-row copy
        for (int m = 0; m < M_orig; ++m) {
            cudaMemcpyAsync(
                static_cast<char*>(params.D) + static_cast<size_t>(m) * N_orig * D_elem,
                static_cast<const char*>(pad_D) + static_cast<size_t>(m) * N_pad * D_elem,
                N_orig * D_elem,
                cudaMemcpyDeviceToDevice, stream);
        }
    }

    // guard destructor frees all pad buffers.
}

// ── Check if cooperative can handle the shape directly ──────────────────────

template <typename OutType>
bool try_cooperative(const layerstorm::compute::Fp8GemmParams& params,
                     void* workspace, cudaStream_t stream) {
    using GT = Fp8GemmType<TileConfig_Cooperative, OutType>;
    using Gemm = typename GT::Gemm;
    using GemmKernel = typename GT::GemmKernel;
    using ScaleConfig = typename GT::ScaleConfig;
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideD = typename GemmKernel::StrideD;

    int M = params.M, N = params.N, K = params.K;
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    typename GemmKernel::MainloopArguments mainloop_args{
        static_cast<typename GT::ElementA const*>(params.A), stride_a,
        static_cast<typename GT::ElementB const*>(params.B), stride_b,
        static_cast<float const*>(params.scale_A), layout_SFA,
        static_cast<float const*>(params.scale_B), layout_SFB};
    typename GemmKernel::EpilogueArguments epilogue_args{
        {},
        static_cast<typename GT::ElementD const*>(params.D), stride_d,
        static_cast<typename GT::ElementD*>(params.D), stride_d};
    epilogue_args.thread.alpha = 1.0f;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1}, mainloop_args, epilogue_args};

    Gemm gemm;
    auto s = gemm.can_implement(args);
    if (s != cutlass::Status::kSuccess) return false;
    s = gemm.initialize(args, workspace, stream);
    if (s != cutlass::Status::kSuccess) return false;
    s = gemm.run(args, workspace, stream);
    return s == cutlass::Status::kSuccess;
}

// ── M-dispatch: pick tile config based on M ─────────────────────────────────

template <typename OutType>
void fp8_dispatch_m(const layerstorm::compute::Fp8GemmParams& params,
                    void* workspace, cudaStream_t stream) {
    if (params.M == 1) {
        // Decode path: hand-written bandwidth-bound dequant-GEMV. The CUTLASS
        // pingpong/cooperative GEMM under-occupies at M=1 and streams the weight
        // at a fraction of peak bandwidth. The GEMV's numerics match the CUTLASS
        // path exactly (alpha == 1.0, per-128-block scales). output_dtype in
        // params selects BF16/FP16 inside the launcher.
        layerstorm::compute::launch_fp8_blockscaled_gemv(params, stream);
        return;
    }
    if (params.M < 128) {
        // Try pingpong schedule (64x128x128 tile) first.
        auto status = [&]() -> cutlass::Status {
            using GT = Fp8GemmType<TileConfig_Pingpong, OutType>;
            using Gemm = typename GT::Gemm;
            using GemmKernel = typename GT::GemmKernel;
            using ScaleConfig = typename GT::ScaleConfig;
            using StrideA = typename GemmKernel::StrideA;
            using StrideB = typename GemmKernel::StrideB;
            using StrideD = typename GemmKernel::StrideD;

            int M = params.M, N = params.N, K = params.K;
            StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
            StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
            StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
            auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
            auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

            typename GemmKernel::MainloopArguments mainloop_args{
                static_cast<typename GT::ElementA const*>(params.A), stride_a,
                static_cast<typename GT::ElementB const*>(params.B), stride_b,
                static_cast<float const*>(params.scale_A), layout_SFA,
                static_cast<float const*>(params.scale_B), layout_SFB};
            typename GemmKernel::EpilogueArguments epilogue_args{
                {},
                static_cast<typename GT::ElementD const*>(params.D), stride_d,
                static_cast<typename GT::ElementD*>(params.D), stride_d};
            epilogue_args.thread.alpha = 1.0f;

            typename Gemm::Arguments args{
                cutlass::gemm::GemmUniversalMode::kGemm,
                {M, N, K, 1}, mainloop_args, epilogue_args};

            Gemm gemm;
            auto s = gemm.can_implement(args);
            if (s != cutlass::Status::kSuccess) return s;
            s = gemm.initialize(args, workspace, stream);
            if (s != cutlass::Status::kSuccess) return s;
            return gemm.run(args, workspace, stream);
        }();

        if (status == cutlass::Status::kSuccess) return;

        // Pingpong rejected — fall through to padded cooperative.
    } else {
        // M >= 128: try cooperative directly.
        if (try_cooperative<OutType>(params, workspace, stream)) return;

        // Cooperative rejected (e.g. non-128-aligned N) — fall through.
    }

    // Padded cooperative fallback: pad M and N to 128-aligned.
    run_padded_cooperative<OutType>(params, workspace, stream);
}

// ── Workspace query helper ──────────────────────────────────────────────────

template <typename Config, typename OutType>
size_t query_workspace_for(int M, int N, int K) {
    using GT = Fp8GemmType<Config, OutType>;
    using Gemm       = typename GT::Gemm;
    using GemmKernel = typename GT::GemmKernel;
    using ScaleConfig = typename GT::ScaleConfig;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideD = typename GemmKernel::StrideD;
    using LayoutSFA = typename GT::LayoutSFA;
    using LayoutSFB = typename GT::LayoutSFB;

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    typename GemmKernel::MainloopArguments mainloop_args{
        nullptr, stride_a, nullptr, stride_b,
        nullptr, layout_SFA, nullptr, layout_SFB};
    typename GemmKernel::EpilogueArguments epilogue_args{
        {}, nullptr, stride_d, nullptr, stride_d};
    epilogue_args.thread.alpha = 1.0f;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        mainloop_args,
        epilogue_args};

    return Gemm::get_workspace_size(args);
}

}  // anonymous namespace

// ── Public API ──────────────────────────────────────────────────────────────

namespace layerstorm::compute {

size_t query_fp8_gemm_workspace_size(int M, int N, int K,
                                      GemmOutputDtype output_dtype) {
    // When M < 128, the dispatch may pad M and N to 128-aligned for cooperative
    // fallback. Query workspace for the padded dimensions to ensure enough space.
    const int M_query = ((M + 127) / 128) * 128;
    const int N_query = ((N + 127) / 128) * 128;
    if (output_dtype == GemmOutputDtype::kBFloat16) {
        return std::max(
            query_workspace_for<TileConfig_Cooperative, cutlass::bfloat16_t>(M_query, N_query, K),
            query_workspace_for<TileConfig_Pingpong, cutlass::bfloat16_t>(M, N, K));
    } else {
        return std::max(
            query_workspace_for<TileConfig_Cooperative, cutlass::half_t>(M_query, N_query, K),
            query_workspace_for<TileConfig_Pingpong, cutlass::half_t>(M, N, K));
    }
}

void launch_fp8_gemm(const Fp8GemmParams& params,
                      void* workspace,
                      void* stream) {
    if (params.K % 16 != 0) {
        throw std::runtime_error("FP8 GEMM: K must be divisible by 16, got K=" +
                                  std::to_string(params.K));
    }
    if (params.N % 16 != 0) {
        throw std::runtime_error("FP8 GEMM: N must be divisible by 16, got N=" +
                                  std::to_string(params.N));
    }

    auto cuda_stream = static_cast<cudaStream_t>(stream);

    switch (params.output_dtype) {
        case GemmOutputDtype::kBFloat16:
            fp8_dispatch_m<cutlass::bfloat16_t>(params, workspace, cuda_stream);
            break;
        case GemmOutputDtype::kFloat16:
            fp8_dispatch_m<cutlass::half_t>(params, workspace, cuda_stream);
            break;
        default:
            throw std::runtime_error("FP8 GEMM: unsupported output dtype");
    }
}

}  // namespace layerstorm::compute

#else  // !SM120 supported

namespace layerstorm::compute {

size_t query_fp8_gemm_workspace_size(int, int, int, GemmOutputDtype) {
    throw std::runtime_error("FP8 GEMM requires SM120 (CUTLASS_ARCH_MMA_SM120_SUPPORTED not defined)");
}

void launch_fp8_gemm(const Fp8GemmParams&, void*, void*) {
    throw std::runtime_error("FP8 GEMM requires SM120 (CUTLASS_ARCH_MMA_SM120_SUPPORTED not defined)");
}

}  // namespace layerstorm::compute

#endif
