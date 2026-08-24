// NVFP4 (FP4 E2M1) grouped GEMM for SM120 using CUTLASS 3.x.
//
// Adapted from:
//   sglang jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh (Apache-2.0)
//   CUTLASS example 79a blackwell_geforce_nvfp4_bf16_gemm (BSD-3-Clause)
//
// Uses CUTLASS GroupProblemShape with KernelPtrArrayTmaWarpSpecializedPingpong
// for batched expert execution. Per-expert pointer arrays computed by a GPU
// kernel (__get_group_gemm_starts), then dispatched via GemmUniversalAdapter.

#include "sm120/gemm/grouped_gemm.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <limits>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

namespace nvfp4_grouped_detail {

// ── CUTLASS grouped GEMM type definition ───────────────────────────────────

template <typename OutType>
struct Nvfp4GroupedGemmType {
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;

    using ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementType = cutlass::float_e2m1_t;  // raw data type
    using ElementSF   = cutlass::float_ue4m3_t;
    using ElementC    = OutType;
    using ElementD    = OutType;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 32;
    static constexpr int AlignmentB = 32;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using ArchTag       = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    using ThreadBlockShape = Shape<_128, _128, _128>;
    using ClusterShape     = Shape<_1, _1, _1>;

    using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementAccumulator, ElementC, ElementAccumulator>;

    // Grouped GEMM uses pointer-based layouts (LayoutA*, not LayoutA)
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC*, AlignmentC,
        ElementD, LayoutC*, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutA*, AlignmentA,
        ElementB, LayoutB*, AlignmentB,
        ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Internal types extracted from the kernel
    using StrideA   = typename GemmKernel::InternalStrideA;
    using StrideB   = typename GemmKernel::InternalStrideB;
    using StrideC   = typename GemmKernel::InternalStrideC;
    using StrideD   = typename GemmKernel::InternalStrideD;
    using LayoutSFA = typename GemmKernel::CollectiveMainloop::InternalLayoutSFA;
    using LayoutSFB = typename GemmKernel::CollectiveMainloop::InternalLayoutSFB;
    using ScaleConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
};

// ── Workspace layout for grouped GEMM ──────────────────────────────────────

template <typename GemmType>
struct GroupedWorkspaceLayout {
    size_t a_ptrs_offset;         // ElementType*[max_experts]
    size_t b_ptrs_offset;         // ElementType*[max_experts]
    size_t d_ptrs_offset;         // ElementD*[max_experts]
    size_t sfa_ptrs_offset;       // ElementSF*[max_experts]
    size_t sfb_ptrs_offset;       // ElementSF*[max_experts]
    size_t alpha_ptrs_offset;     // float*[max_experts]
    size_t layout_sfa_offset;     // LayoutSFA[max_experts]
    size_t layout_sfb_offset;     // LayoutSFB[max_experts]
    size_t strides_ab_offset;     // StrideA[max_experts] (A and B share stride layout)
    size_t strides_c_offset;      // StrideC[max_experts]
    size_t cutlass_ws_offset;     // CUTLASS internal workspace
    size_t total_bytes;

    static GroupedWorkspaceLayout compute(int max_experts) {
        auto align256 = [](size_t x) { return (x + 255) & ~size_t(255); };
        size_t ptr_arr = max_experts * sizeof(void*);

        GroupedWorkspaceLayout layout;
        size_t offset = 0;
        layout.a_ptrs_offset     = offset; offset += align256(ptr_arr);
        layout.b_ptrs_offset     = offset; offset += align256(ptr_arr);
        layout.d_ptrs_offset     = offset; offset += align256(ptr_arr);
        layout.sfa_ptrs_offset   = offset; offset += align256(ptr_arr);
        layout.sfb_ptrs_offset   = offset; offset += align256(ptr_arr);
        layout.alpha_ptrs_offset = offset; offset += align256(ptr_arr);

        using LayoutSFA = typename GemmType::LayoutSFA;
        using LayoutSFB = typename GemmType::LayoutSFB;
        using StrideA = typename GemmType::StrideA;
        using StrideC = typename GemmType::StrideC;

        layout.layout_sfa_offset = offset; offset += align256(max_experts * sizeof(LayoutSFA));
        layout.layout_sfb_offset = offset; offset += align256(max_experts * sizeof(LayoutSFB));
        layout.strides_ab_offset = offset; offset += align256(max_experts * sizeof(StrideA));
        layout.strides_c_offset  = offset; offset += align256(max_experts * sizeof(StrideC));

        // CUTLASS grouped GEMM workspace (~150 KB typical for SM120 PtrArray)
        layout.cutlass_ws_offset = offset; offset += align256(256 * 1024);
        layout.total_bytes = offset;
        return layout;
    }
};

// ── GPU kernel: compute per-expert pointers and layouts ────────────────────
// One thread per expert. Runs as <<<1, num_experts>>>.

template <typename ElementType, typename ElementSF, typename ElementD,
          typename LayoutSFA, typename LayoutSFB, typename StrideA,
          typename StrideC, typename ScaleConfig>
__global__ void get_nvfp4_group_starts(
    const ElementType** a_ptrs,
    const ElementType** b_ptrs,
    ElementD** d_ptrs,
    const ElementSF** sfa_ptrs,
    const ElementSF** sfb_ptrs,
    float** alpha_ptrs,
    LayoutSFA* layout_sfa_arr,
    LayoutSFB* layout_sfb_arr,
    StrideA* strides_ab,
    StrideC* strides_c,
    const ElementType* a_base,
    const ElementType* b_base,
    ElementD* d_base,
    const ElementSF* sfa_base,
    const ElementSF* sfb_base,
    float* alphas_base,
    const int32_t* expert_offsets,
    const int32_t* sf_offsets,
    const int32_t* problem_sizes,
    int K, int N,
    const void** b_ptrs_in,      // optional: pre-computed per-expert B pointers
    const void** sfb_ptrs_in) {  // optional: pre-computed per-expert scale_B pointers

    int expert_id = threadIdx.x;
    int64_t token_offset = static_cast<int64_t>(expert_offsets[expert_id]);
    int64_t sf_offset = static_cast<int64_t>(sf_offsets[expert_id]);

    int32_t m = problem_sizes[expert_id * 3];
    int32_t n = problem_sizes[expert_id * 3 + 1];
    int32_t k = problem_sizes[expert_id * 3 + 2];

    int64_t half_k = static_cast<int64_t>(k / 2);
    int64_t group_k = static_cast<int64_t>(k / 16);  // group_size = 16 for NVFP4

    // A: [total_tokens, K/2] packed FP4 — offset by token_offset * K/2
    a_ptrs[expert_id] = a_base + token_offset * half_k;
    // B: per-expert pointer if provided, else contiguous [num_experts, N, K/2]
    b_ptrs[expert_id] = b_ptrs_in
        ? reinterpret_cast<const ElementType*>(b_ptrs_in[expert_id])
        : b_base + static_cast<int64_t>(expert_id) * n * half_k;
    // D: [total_tokens, N] — offset by token_offset * N
    d_ptrs[expert_id] = d_base + token_offset * n;
    // Scale A: [sum_sf_sizes, K/group_size] — offset by sf_offset * K/group_size
    sfa_ptrs[expert_id] = sfa_base + sf_offset * group_k;
    // Scale B: per-expert pointer if provided, else contiguous [num_experts, N, K/group_size]
    sfb_ptrs[expert_id] = sfb_ptrs_in
        ? reinterpret_cast<const ElementSF*>(sfb_ptrs_in[expert_id])
        : sfb_base + static_cast<int64_t>(expert_id) * n * group_k;
    // Alpha: [num_experts]
    alpha_ptrs[expert_id] = alphas_base + expert_id;

    // Strides (per CUTLASS convention: {leading_dim, batch_stride})
    strides_ab[expert_id] = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    strides_c[expert_id] = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});

    // Scale factor layouts (per Sm1xxBlkScaledConfig)
    layout_sfa_arr[expert_id] = ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape(static_cast<int>(m), static_cast<int>(n),
                          static_cast<int>(k), 1));
    layout_sfb_arr[expert_id] = ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape(static_cast<int>(m), static_cast<int>(n),
                          static_cast<int>(k), 1));
}

// ── Run grouped GEMM ──────────────────────────────────────────────────────

template <typename OutType>
void run_nvfp4_grouped_gemm(
    const layerstorm::compute::Nvfp4GroupedGemmParams& params,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream) {

    using GemmType = nvfp4_grouped_detail::Nvfp4GroupedGemmType<OutType>;
    using Gemm = typename GemmType::Gemm;
    using GemmKernel = typename GemmType::GemmKernel;
    using ElementType = typename GemmType::ElementType;
    using ElementSF = typename GemmType::ElementSF;
    using ElementD = typename GemmType::ElementD;
    using StrideA = typename GemmType::StrideA;
    using StrideC = typename GemmType::StrideC;
    using LayoutSFA = typename GemmType::LayoutSFA;
    using LayoutSFB = typename GemmType::LayoutSFB;
    using ScaleConfig = typename GemmType::ScaleConfig;
    using UnderlyingProblemShape = typename GemmType::UnderlyingProblemShape;

    using WsLayout = GroupedWorkspaceLayout<GemmType>;
    auto ws_layout = WsLayout::compute(params.num_experts);

    if (workspace_bytes < ws_layout.total_bytes) {
        throw std::runtime_error(
            "NVFP4 grouped GEMM: workspace too small (" +
            std::to_string(workspace_bytes) + " < " +
            std::to_string(ws_layout.total_bytes) + ")");
    }

    auto* ws = static_cast<char*>(workspace);

    auto* a_ptrs     = reinterpret_cast<const ElementType**>(ws + ws_layout.a_ptrs_offset);
    auto* b_ptrs     = reinterpret_cast<const ElementType**>(ws + ws_layout.b_ptrs_offset);
    auto* d_ptrs     = reinterpret_cast<ElementD**>(ws + ws_layout.d_ptrs_offset);
    auto* sfa_ptrs   = reinterpret_cast<const ElementSF**>(ws + ws_layout.sfa_ptrs_offset);
    auto* sfb_ptrs   = reinterpret_cast<const ElementSF**>(ws + ws_layout.sfb_ptrs_offset);
    auto* alpha_ptrs = reinterpret_cast<float**>(ws + ws_layout.alpha_ptrs_offset);
    auto* layout_sfa = reinterpret_cast<LayoutSFA*>(ws + ws_layout.layout_sfa_offset);
    auto* layout_sfb = reinterpret_cast<LayoutSFB*>(ws + ws_layout.layout_sfb_offset);
    auto* strides_ab = reinterpret_cast<StrideA*>(ws + ws_layout.strides_ab_offset);
    auto* strides_c  = reinterpret_cast<StrideC*>(ws + ws_layout.strides_c_offset);
    void* cutlass_ws = ws + ws_layout.cutlass_ws_offset;

    // Launch pointer setup kernel
    get_nvfp4_group_starts<ElementType, ElementSF, ElementD,
                            LayoutSFA, LayoutSFB, StrideA, StrideC, ScaleConfig>
        <<<1, params.num_experts, 0, stream>>>(
            a_ptrs, b_ptrs, d_ptrs, sfa_ptrs, sfb_ptrs, alpha_ptrs,
            layout_sfa, layout_sfb, strides_ab, strides_c,
            static_cast<const ElementType*>(params.A_base),
            static_cast<const ElementType*>(params.B_base),
            static_cast<ElementD*>(params.D_base),
            static_cast<const ElementSF*>(params.scale_A_base),
            static_cast<const ElementSF*>(params.scale_B_base),
            const_cast<float*>(params.alphas),
            params.expert_offsets, params.sf_offsets, params.problem_sizes,
            params.K, params.N,
            params.B_ptrs, params.scale_B_ptrs);

    // Set up hardware info
    cutlass::KernelHardwareInfo hw_info;
    int device_id = 0;
    cudaGetDevice(&device_id);
    hw_info.device_id = device_id;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    // Scheduler
    using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;
    typename GemmKernel::TileSchedulerArguments scheduler;
    scheduler.raster_order = RasterOrderOptions::AlongM;

    // Problem shapes (device pointer to [num_experts, 3] packed int32)
    auto* problem_shapes = reinterpret_cast<UnderlyingProblemShape*>(
        const_cast<int32_t*>(params.problem_sizes));

    // Mainloop arguments
    typename GemmKernel::MainloopArguments mainloop_args{
        a_ptrs,
        strides_ab,
        b_ptrs,
        strides_ab,  // A and B share the same stride type
        sfa_ptrs,
        layout_sfa,
        sfb_ptrs,
        layout_sfb};

    // Epilogue arguments
    typename GemmKernel::EpilogueArguments epilogue_args{
        {},       // epilogue.thread (filled below)
        nullptr,  // C ptr (unused, beta=0)
        strides_c,
        d_ptrs,
        strides_c};
    auto& fusion_args = epilogue_args.thread;
    fusion_args.alpha_ptr_array = alpha_ptrs;
    fusion_args.dAlpha = {_0{}, _0{}, 1};
    fusion_args.beta = 0.0f;

    // Gemm arguments
    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {params.num_experts, problem_shapes, nullptr},
        mainloop_args,
        epilogue_args,
        hw_info,
        scheduler};

    Gemm gemm_op;

    size_t cutlass_ws_needed = Gemm::get_workspace_size(args);
    size_t cutlass_ws_available = ws_layout.total_bytes - ws_layout.cutlass_ws_offset;
    if (cutlass_ws_needed > cutlass_ws_available) {
        throw std::runtime_error(
            "NVFP4 grouped GEMM: CUTLASS workspace too small (" +
            std::to_string(cutlass_ws_available) + " < " +
            std::to_string(cutlass_ws_needed) + ")");
    }

    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("NVFP4 grouped GEMM can_implement failed: ") +
            cutlassGetStatusString(status));
    }

    status = gemm_op.initialize(args, cutlass_ws, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("NVFP4 grouped GEMM initialize failed: ") +
            cutlassGetStatusString(status));
    }

    status = gemm_op.run(args, cutlass_ws, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("NVFP4 grouped GEMM run failed: ") +
            cutlassGetStatusString(status));
    }
}

}  // namespace nvfp4_grouped_detail

// ── Public API ─────────────────────────────────────────────────────────────

namespace layerstorm::compute {

size_t query_nvfp4_grouped_gemm_workspace_size(
    int max_experts, int N, int K,
    GemmOutputDtype output_dtype) {

    if (output_dtype == GemmOutputDtype::kBFloat16) {
        using GemmType = nvfp4_grouped_detail::Nvfp4GroupedGemmType<cutlass::bfloat16_t>;
        return nvfp4_grouped_detail::GroupedWorkspaceLayout<GemmType>::compute(max_experts).total_bytes;
    } else {
        using GemmType = nvfp4_grouped_detail::Nvfp4GroupedGemmType<cutlass::half_t>;
        return nvfp4_grouped_detail::GroupedWorkspaceLayout<GemmType>::compute(max_experts).total_bytes;
    }
}

void launch_nvfp4_grouped_gemm(
    const Nvfp4GroupedGemmParams& params,
    void* workspace, size_t workspace_bytes,
    void* stream) {

    if (params.K % 32 != 0) {
        throw std::runtime_error(
            "NVFP4 grouped GEMM: K must be divisible by 32, got K=" +
            std::to_string(params.K));
    }
    if ((params.B_ptrs == nullptr) != (params.scale_B_ptrs == nullptr)) {
        throw std::runtime_error(
            "NVFP4 grouped GEMM: B_ptrs and scale_B_ptrs must be both null or both non-null");
    }
    if (params.N % 32 != 0) {
        throw std::runtime_error(
            "NVFP4 grouped GEMM: N must be divisible by 32, got N=" +
            std::to_string(params.N));
    }
    if (params.num_experts <= 0) {
        return;  // Nothing to do
    }

    auto cuda_stream = static_cast<cudaStream_t>(stream);

    switch (params.output_dtype) {
        case GemmOutputDtype::kBFloat16:
            nvfp4_grouped_detail::run_nvfp4_grouped_gemm<cutlass::bfloat16_t>(
                params, workspace, workspace_bytes, cuda_stream);
            break;
        case GemmOutputDtype::kFloat16:
            nvfp4_grouped_detail::run_nvfp4_grouped_gemm<cutlass::half_t>(
                params, workspace, workspace_bytes, cuda_stream);
            break;
        default:
            throw std::runtime_error("NVFP4 grouped GEMM: unsupported output dtype");
    }
}

}  // namespace layerstorm::compute

#else  // !CUTLASS_ARCH_MMA_SM120_SUPPORTED

namespace layerstorm::compute {

size_t query_nvfp4_grouped_gemm_workspace_size(int, int, int, GemmOutputDtype) {
    throw std::runtime_error(
        "NVFP4 grouped GEMM requires SM120 "
        "(CUTLASS_ARCH_MMA_SM120_SUPPORTED not defined)");
}

void launch_nvfp4_grouped_gemm(const Nvfp4GroupedGemmParams&, void*, size_t, void*) {
    throw std::runtime_error(
        "NVFP4 grouped GEMM requires SM120 "
        "(CUTLASS_ARCH_MMA_SM120_SUPPORTED not defined)");
}

}  // namespace layerstorm::compute

#endif
