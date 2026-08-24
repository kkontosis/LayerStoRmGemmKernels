// NVFP4 (FP4 E2M1) GEMM for SM120 using CUTLASS 3.x BlockScaledTensorOp.
//
// Adapted from:
//   vLLM nvfp4_scaled_mm_sm120_kernels.cu (Apache-2.0)
//   CUTLASS example 79a blackwell_geforce_nvfp4_bf16_gemm (BSD-3-Clause)
//
// Two tile configurations for M-dimension dispatch:
//   M <= 256: 128x128x128 (better occupancy for MoE decode)
//   M >  256: 256x128x128 (better throughput for prefill)

#include "sm120/gemm/nvfp4/nvfp4_gemm.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

namespace {

// ── Tile configurations ─────────────────────────────────────────────────────

struct TileConfig_M256 {
    using ClusterShape     = Shape<_1, _1, _1>;
    using MmaTileShape     = Shape<_128, _128, _128>;
    using PerSmTileShape   = Shape<_128, _128, _128>;
};

struct TileConfig_Default {
    using ClusterShape     = Shape<_1, _1, _1>;
    using MmaTileShape     = Shape<_256, _128, _128>;
    using PerSmTileShape   = Shape<_256, _128, _128>;
};

// ── GEMM type alias template ────────────────────────────────────────────────

template <typename Config, typename OutType>
struct Nvfp4GemmType {
    using ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutATag  = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 32;

    using ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutBTag  = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 32;

    using ElementD    = OutType;
    using ElementC    = OutType;
    using LayoutCTag  = cutlass::layout::RowMajor;
    using LayoutDTag  = cutlass::layout::RowMajor;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using ElementAccumulator = float;
    using ArchTag       = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    using MmaTileShape     = typename Config::MmaTileShape;
    using ClusterShape     = typename Config::ClusterShape;
    using PerSmTileShape   = typename Config::PerSmTileShape;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, PerSmTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop, CollectiveEpilogue, void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// ── Helper: next power of 2 ────────────────────────────────────────────────

inline uint32_t next_pow2(uint32_t v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

// ── Run a single GEMM ──────────────────────────────────────────────────────

template <typename GemmType>
void run_nvfp4_gemm(const layerstorm::compute::Nvfp4GemmParams& params,
                     void* workspace, cudaStream_t stream) {
    using Gemm = typename GemmType::Gemm;
    using GemmKernel = typename GemmType::GemmKernel;
    using ElementA   = typename GemmType::ElementA;
    using ElementB   = typename GemmType::ElementB;
    using ElementD   = typename GemmType::ElementD;

    // Block-scaled ops use raw data type pointers, not nv_float4_t pointers.
    // CUTLASS 79a: block_A is HostTensor<ElementA::DataType, ...>
    using DataTypeA  = typename ElementA::DataType;       // float_e2m1_t
    using DataTypeB  = typename ElementB::DataType;
    using ScaleTypeA = typename ElementA::ScaleFactorType; // float_ue4m3_t (or similar)
    using ScaleTypeB = typename ElementB::ScaleFactorType;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    int M = params.M, N = params.N, K = params.K;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    // Follow CUTLASS example 79a argument construction pattern exactly.
    // Mainloop: raw data pointers + scale factor pointers + layouts.
    // Epilogue: {alpha, beta} fusion args + C/D pointers + strides.
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        // Mainloop args (raw data type pointers, not nv_float4_t)
        {static_cast<DataTypeA const*>(params.A), stride_A,
         static_cast<DataTypeB const*>(params.B), stride_B,
         static_cast<ScaleTypeA const*>(params.scale_A), layout_SFA,
         static_cast<ScaleTypeB const*>(params.scale_B), layout_SFB},
        // Epilogue args: {fusion_args, C_ptr, stride_C, D_ptr, stride_D}
        {{params.alpha, 0.0f},
         static_cast<ElementD const*>(params.D), stride_C,
         static_cast<ElementD*>(params.D), stride_D}
    };

    Gemm gemm;

    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("NVFP4 GEMM can_implement failed: ") +
            cutlassGetStatusString(status));
    }

    status = gemm.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("NVFP4 GEMM initialize failed: ") +
            cutlassGetStatusString(status));
    }

    status = gemm.run(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("NVFP4 GEMM run failed: ") +
            cutlassGetStatusString(status));
    }
}

// ── M-dispatch: pick tile config based on M ─────────────────────────────────

template <typename OutType>
void nvfp4_dispatch_m(const layerstorm::compute::Nvfp4GemmParams& params,
                       void* workspace, cudaStream_t stream) {
    uint32_t mp2 = std::max(16u, next_pow2(static_cast<uint32_t>(params.M)));
    if (mp2 <= 256) {
        run_nvfp4_gemm<Nvfp4GemmType<TileConfig_M256, OutType>>(
            params, workspace, stream);
    } else {
        run_nvfp4_gemm<Nvfp4GemmType<TileConfig_Default, OutType>>(
            params, workspace, stream);
    }
}

// ── Workspace query helper ──────────────────────────────────────────────────

template <typename GemmType>
size_t query_workspace_for(int M, int N, int K) {
    using Gemm = typename GemmType::Gemm;
    using GemmKernel = typename GemmType::GemmKernel;
    using ElementA   = typename GemmType::ElementA;
    using ElementB   = typename GemmType::ElementB;
    using ElementD   = typename GemmType::ElementD;
    using DataTypeA  = typename ElementA::DataType;
    using DataTypeB  = typename ElementB::DataType;
    using ScaleTypeA = typename ElementA::ScaleFactorType;
    using ScaleTypeB = typename ElementB::ScaleFactorType;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {static_cast<DataTypeA const*>(nullptr), stride_A,
         static_cast<DataTypeB const*>(nullptr), stride_B,
         static_cast<ScaleTypeA const*>(nullptr), layout_SFA,
         static_cast<ScaleTypeB const*>(nullptr), layout_SFB},
        {{1.0f, 0.0f},
         static_cast<ElementD const*>(nullptr), stride_C,
         static_cast<ElementD*>(nullptr), stride_D}
    };

    return Gemm::get_workspace_size(arguments);
}

// ── SF buffer size query helper ──────────────────────────────────────────────

template <typename GemmType>
size_t query_sf_size(int M, int N, int K, bool is_a) {
    using GemmKernel = typename GemmType::GemmKernel;
    using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    if (is_a) {
        auto layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
            cute::make_shape(M, N, K, 1));
        return size(filter_zeros(layout));
    } else {
        auto layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
            cute::make_shape(M, N, K, 1));
        return size(filter_zeros(layout));
    }
}

}  // anonymous namespace

// ── Public API ──────────────────────────────────────────────────────────────

namespace layerstorm::compute {

size_t query_nvfp4_gemm_workspace_size(int M, int N, int K,
                                        GemmOutputDtype output_dtype) {
    // Return max workspace across both tile configs for the given output type.
    size_t ws = 0;
    if (output_dtype == GemmOutputDtype::kBFloat16) {
        ws = std::max(
            query_workspace_for<Nvfp4GemmType<TileConfig_M256, cutlass::bfloat16_t>>(M, N, K),
            query_workspace_for<Nvfp4GemmType<TileConfig_Default, cutlass::bfloat16_t>>(M, N, K));
    } else {
        ws = std::max(
            query_workspace_for<Nvfp4GemmType<TileConfig_M256, cutlass::half_t>>(M, N, K),
            query_workspace_for<Nvfp4GemmType<TileConfig_Default, cutlass::half_t>>(M, N, K));
    }
    return ws;
}

void launch_nvfp4_gemm(const Nvfp4GemmParams& params,
                        void* workspace,
                        void* stream) {
    if (params.K % 32 != 0) {
        throw std::runtime_error("NVFP4 GEMM: K must be divisible by 32, got K=" +
                                  std::to_string(params.K));
    }
    if (params.N % 32 != 0) {
        throw std::runtime_error("NVFP4 GEMM: N must be divisible by 32, got N=" +
                                  std::to_string(params.N));
    }

    auto cuda_stream = static_cast<cudaStream_t>(stream);

    switch (params.output_dtype) {
        case GemmOutputDtype::kBFloat16:
            nvfp4_dispatch_m<cutlass::bfloat16_t>(params, workspace, cuda_stream);
            break;
        case GemmOutputDtype::kFloat16:
            nvfp4_dispatch_m<cutlass::half_t>(params, workspace, cuda_stream);
            break;
        default:
            throw std::runtime_error("NVFP4 GEMM: unsupported output dtype");
    }
}

size_t query_sf_buffer_size_a(int M, int N, int K) {
    return query_sf_size<Nvfp4GemmType<TileConfig_M256, cutlass::bfloat16_t>>(M, N, K, true);
}

size_t query_sf_buffer_size_b(int M, int N, int K) {
    return query_sf_size<Nvfp4GemmType<TileConfig_M256, cutlass::bfloat16_t>>(M, N, K, false);
}

}  // namespace layerstorm::compute

#else  // !CUTLASS_ARCH_MMA_SM120_SUPPORTED

namespace layerstorm::compute {

size_t query_nvfp4_gemm_workspace_size(int, int, int, GemmOutputDtype) {
    throw std::runtime_error("NVFP4 GEMM requires SM120 (CUTLASS_ARCH_MMA_SM120_SUPPORTED not defined)");
}

void launch_nvfp4_gemm(const Nvfp4GemmParams&, void*, void*) {
    throw std::runtime_error("NVFP4 GEMM requires SM120 (CUTLASS_ARCH_MMA_SM120_SUPPORTED not defined)");
}

size_t query_sf_buffer_size_a(int, int, int) {
    throw std::runtime_error("NVFP4 GEMM requires SM120");
}

size_t query_sf_buffer_size_b(int, int, int) {
    throw std::runtime_error("NVFP4 GEMM requires SM120");
}

}  // namespace layerstorm::compute

#endif
