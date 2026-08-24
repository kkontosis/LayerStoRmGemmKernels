// Pybind11 Python wrappers for SM120 Quantized GEMM CUDA kernels.
// #included from bindings.cu — not compiled standalone.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <string>
#include <cstdint>

// All kernel headers already visible via bindings.cu includes.
namespace lc = layerstorm::compute;

// ── Helpers ──────────────────────────────────────────────────────────────────

static lc::GemmOutputDtype parse_output_dtype(const std::string& s) {
    if (s == "bf16" || s == "bfloat16")
        return lc::GemmOutputDtype::kBFloat16;
    if (s == "fp16" || s == "float16")
        return lc::GemmOutputDtype::kFloat16;
    TORCH_CHECK(false, "Unknown output dtype: ", s,
                ". Expected 'bf16' or 'fp16'.");
    return lc::GemmOutputDtype::kBFloat16;
}

static void* current_stream() {
    return static_cast<void*>(at::cuda::getCurrentCUDAStream().stream());
}

// ── Dynamic FP8 Quantization ────────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor> dynamic_fp8_quant(torch::Tensor input,
                                                            bool m_major_scales) {
    TORCH_CHECK(input.is_cuda(), "dynamic_fp8_quant: input must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "dynamic_fp8_quant: input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "dynamic_fp8_quant: input must be 2D [M, K]");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16,
                "dynamic_fp8_quant: input must be BF16");

    int M = input.size(0);
    int K = input.size(1);
    int num_blocks = (K + 127) / 128;

    auto output = torch::empty({M, K}, input.options().dtype(at::kFloat8_e4m3fn));
    // m_major: [num_blocks, M] row-major == scales[m + k_blk*M] (fp8_gemm SFA).
    auto scales = m_major_scales
        ? torch::empty({num_blocks, M}, input.options().dtype(at::kFloat))
        : torch::empty({M, num_blocks}, input.options().dtype(at::kFloat));

    lc::DynamicFp8QuantParams params;
    params.num_tokens = M;
    params.hidden_size = K;
    params.input = input.data_ptr();
    params.output = output.data_ptr();
    params.scales = scales.data_ptr();
    params.m_major_scales = m_major_scales;

    lc::launch_dynamic_fp8_quant(params, current_stream());

    return {output, scales};
}

// ── NVFP4 Single GEMM (low-level: pre-quantized FP4 inputs) ────────────────

torch::Tensor nvfp4_gemm(
    torch::Tensor A,        // [M, K/2] uint8 packed FP4
    torch::Tensor B,        // [N, K/2] uint8 packed FP4
    torch::Tensor scale_A,  // UE4M3 scales (padded+swizzled)
    torch::Tensor scale_B,  // UE4M3 scales (padded+swizzled)
    int64_t M, int64_t N, int64_t K,
    double alpha,
    const std::string& output_dtype) {

    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "nvfp4_gemm: A and B must be on CUDA");
    TORCH_CHECK(scale_A.is_cuda() && scale_B.is_cuda(),
                "nvfp4_gemm: scales must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "nvfp4_gemm: A and B must be contiguous");

    auto odtype = parse_output_dtype(output_dtype);
    auto torch_odtype = (odtype == lc::GemmOutputDtype::kBFloat16)
                        ? at::kBFloat16 : at::kHalf;

    auto D = torch::empty({M, N}, A.options().dtype(torch_odtype));

    size_t ws_bytes = lc::query_nvfp4_gemm_workspace_size(
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), odtype);
    torch::Tensor workspace;
    void* ws_ptr = nullptr;
    if (ws_bytes > 0) {
        workspace = torch::empty({static_cast<int64_t>(ws_bytes)},
                                  A.options().dtype(at::kByte));
        ws_ptr = workspace.data_ptr();
    }

    lc::Nvfp4GemmParams params;
    params.M = static_cast<int>(M);
    params.N = static_cast<int>(N);
    params.K = static_cast<int>(K);
    params.A = A.data_ptr();
    params.B = B.data_ptr();
    params.D = D.data_ptr();
    params.scale_A = scale_A.data_ptr();
    params.scale_B = scale_B.data_ptr();
    params.alpha = static_cast<float>(alpha);
    params.output_dtype = odtype;

    lc::launch_nvfp4_gemm(params, ws_ptr, current_stream());

    return D;
}

// ── NVFP4 High-Level GEMM (BF16 input → quantize → GEMM) ──────────────────

torch::Tensor nvfp4_gemm_preprocess(
    torch::Tensor weight_scales,
    int64_t N,
    int64_t K
) {
    TORCH_CHECK(weight_scales.is_cuda(), "weight_scales must be on CUDA");
    TORCH_CHECK(weight_scales.dim() == 2 && weight_scales.is_contiguous(),
        "weight_scales must be contiguous [N, K/16]");
    TORCH_CHECK(weight_scales.size(0) == N && weight_scales.size(1) == K / 16,
        "weight_scales shape mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto weight_scales_f32 = weight_scales.to(torch::kFloat32);

    size_t wt_sf_size = lc::query_sf_buffer_size_b(1, N, K);
    auto wt_scales_buf = torch::zeros({static_cast<int64_t>(wt_sf_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(weight_scales.device()));

    lc::ReformatScalesParams reformat_params;
    reformat_params.src_scales = reinterpret_cast<const float*>(weight_scales_f32.data_ptr());
    reformat_params.dst_scales = reinterpret_cast<uint8_t*>(wt_scales_buf.data_ptr());
    reformat_params.rows = N;
    reformat_params.groups_per_row = K / 16;
    reformat_params.M = 1;
    reformat_params.N = N;
    reformat_params.K = K;
    reformat_params.is_scale_a = false;
    lc::launch_reformat_scales(reformat_params, stream);

    return wt_scales_buf;
}

torch::Tensor nvfp4_gemm_from_bf16(
    torch::Tensor x_bf16,
    torch::Tensor weight_packed_uint8,
    torch::Tensor weight_scales,
    double weight_scale_global
) {
    TORCH_CHECK(x_bf16.is_cuda() && x_bf16.dtype() == torch::kBFloat16,
        "x_bf16 must be BF16 on CUDA");
    TORCH_CHECK(x_bf16.dim() == 2 && x_bf16.is_contiguous(),
        "x_bf16 must be contiguous [M, K]");
    TORCH_CHECK(weight_packed_uint8.is_cuda() && weight_packed_uint8.dtype() == torch::kUInt8,
        "weight_packed_uint8 must be uint8 on CUDA");
    TORCH_CHECK(weight_packed_uint8.dim() == 2 && weight_packed_uint8.is_contiguous(),
        "weight_packed_uint8 must be contiguous [N, K/2]");
    TORCH_CHECK(weight_scales.is_cuda(),
        "weight_scales must be on CUDA");

    int M = x_bf16.size(0);
    int K = x_bf16.size(1);
    int N = weight_packed_uint8.size(0);
    TORCH_CHECK(weight_packed_uint8.size(1) == K / 2,
        "weight shape mismatch: expected [N, K/2]");
    TORCH_CHECK(K % 32 == 0, "K must be divisible by 32, got K=" + std::to_string(K));
    TORCH_CHECK(N % 32 == 0, "N must be divisible by 32, got N=" + std::to_string(N));

    auto stream = at::cuda::getCurrentCUDAStream();

    bool scales_preprocessed = (weight_scales.dtype() == torch::kUInt8 && weight_scales.dim() == 1);

    if (!scales_preprocessed) {
        TORCH_CHECK(weight_scales.dim() == 2 && weight_scales.is_contiguous(),
            "weight_scales must be contiguous [N, K/16] or preprocessed 1D uint8");
        TORCH_CHECK(weight_scales.size(0) == N && weight_scales.size(1) == K / 16,
            "weight_scales shape mismatch: expected [N, K/16]");
    }

    size_t act_packed_bytes = static_cast<size_t>(M) * (K / 2);
    size_t act_sf_size = lc::query_sf_buffer_size_a(M, N, K);
    size_t wt_sf_size = scales_preprocessed ? 0 : lc::query_sf_buffer_size_b(M, N, K);
    size_t ws_size = lc::query_nvfp4_gemm_workspace_size(M, N, K, lc::GemmOutputDtype::kBFloat16);

    size_t total_temp = act_packed_bytes + act_sf_size + wt_sf_size + ws_size;
    auto temp_buf = torch::empty({static_cast<int64_t>(total_temp)},
        torch::TensorOptions().dtype(torch::kUInt8).device(x_bf16.device()));
    uint8_t* temp_ptr = reinterpret_cast<uint8_t*>(temp_buf.data_ptr());

    uint8_t* act_packed_ptr  = temp_ptr;
    uint8_t* act_scales_ptr  = temp_ptr + act_packed_bytes;
    uint8_t* wt_scales_ptr   = scales_preprocessed
        ? reinterpret_cast<uint8_t*>(weight_scales.data_ptr())
        : temp_ptr + act_packed_bytes + act_sf_size;
    void*    workspace_ptr   = (ws_size > 0)
        ? temp_ptr + act_packed_bytes + act_sf_size + wt_sf_size
        : nullptr;

    cudaMemsetAsync(act_scales_ptr, 0, act_sf_size, stream);

    lc::Bf16ToNvfp4Params quant_params;
    quant_params.input = reinterpret_cast<const __nv_bfloat16*>(x_bf16.data_ptr());
    quant_params.output_packed = act_packed_ptr;
    quant_params.output_scales = act_scales_ptr;
    quant_params.M = M;
    quant_params.N = N;
    quant_params.K = K;
    lc::launch_bf16_to_nvfp4(quant_params, stream);

    if (!scales_preprocessed) {
        auto weight_scales_f32 = weight_scales.to(torch::kFloat32);
        lc::ReformatScalesParams reformat_params;
        reformat_params.src_scales = reinterpret_cast<const float*>(weight_scales_f32.data_ptr());
        reformat_params.dst_scales = wt_scales_ptr;
        reformat_params.rows = N;
        reformat_params.groups_per_row = K / 16;
        reformat_params.M = M;
        reformat_params.N = N;
        reformat_params.K = K;
        reformat_params.is_scale_a = false;
        lc::launch_reformat_scales(reformat_params, stream);
    }

    auto output = torch::empty({M, N}, x_bf16.options());

    lc::Nvfp4GemmParams gemm_params;
    gemm_params.M = M;
    gemm_params.N = N;
    gemm_params.K = K;
    gemm_params.A = act_packed_ptr;
    gemm_params.B = weight_packed_uint8.data_ptr();
    gemm_params.D = output.data_ptr();
    gemm_params.scale_A = act_scales_ptr;
    gemm_params.scale_B = wt_scales_ptr;
    gemm_params.alpha = static_cast<float>(weight_scale_global);
    gemm_params.output_dtype = lc::GemmOutputDtype::kBFloat16;

    lc::launch_nvfp4_gemm(gemm_params, workspace_ptr, current_stream());

    return output;
}

// ── NVFP4 Grouped GEMM ────────────────────────────────────────────────────

torch::Tensor nvfp4_grouped_gemm(
    torch::Tensor A_base, torch::Tensor B_base, torch::Tensor D_base,
    torch::Tensor scale_A_base, torch::Tensor scale_B_base,
    torch::Tensor alphas,
    torch::Tensor expert_offsets, torch::Tensor sf_offsets,
    torch::Tensor problem_sizes,
    int64_t N, int64_t K,
    const std::string& output_dtype) {

    TORCH_CHECK(A_base.is_cuda(), "nvfp4_grouped_gemm: A_base must be on CUDA");
    auto odtype = parse_output_dtype(output_dtype);
    int num_experts = expert_offsets.size(0) - 1;

    size_t ws_bytes = lc::query_nvfp4_grouped_gemm_workspace_size(
        num_experts, static_cast<int>(N), static_cast<int>(K), odtype);
    auto workspace = torch::empty({static_cast<int64_t>(ws_bytes)},
                                   A_base.options().dtype(at::kByte));

    lc::Nvfp4GroupedGemmParams params;
    params.num_experts = num_experts;
    params.N = static_cast<int>(N);
    params.K = static_cast<int>(K);
    params.A_base = A_base.data_ptr();
    params.B_base = B_base.data_ptr();
    params.D_base = D_base.data_ptr();
    params.scale_A_base = scale_A_base.data_ptr();
    params.scale_B_base = scale_B_base.data_ptr();
    params.alphas = alphas.data_ptr<float>();
    params.expert_offsets = expert_offsets.data_ptr<int32_t>();
    params.sf_offsets = sf_offsets.data_ptr<int32_t>();
    params.problem_sizes = problem_sizes.data_ptr<int32_t>();
    params.output_dtype = odtype;

    lc::launch_nvfp4_grouped_gemm(params, workspace.data_ptr(), ws_bytes, current_stream());
    return D_base;
}

// ── Grouped BF16 → NVFP4 activation quantization ──────────────────────────

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bf16_to_nvfp4_grouped(
    torch::Tensor input,           // [total_tokens, K] BF16
    torch::Tensor expert_offsets,  // [num_experts + 1] int32 cumulative tokens
    c10::optional<torch::Tensor> input_scales) {  // [num_experts] float32

    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16,
                "bf16_to_nvfp4_grouped: input must be BF16 on CUDA");
    TORCH_CHECK(input.dim() == 2 && input.is_contiguous(),
                "bf16_to_nvfp4_grouped: input must be contiguous [T, K]");
    TORCH_CHECK(expert_offsets.dim() == 1 &&
                expert_offsets.scalar_type() == at::kInt,
                "bf16_to_nvfp4_grouped: expert_offsets must be 1D int32");

    int total_tokens = input.size(0);
    int K = input.size(1);
    int num_experts = expert_offsets.size(0) - 1;
    TORCH_CHECK(K % 64 == 0,
                "bf16_to_nvfp4_grouped: K must be divisible by 64 "
                "(K/16 groups must align to the 4-group Sm1xx atom)");

    auto offs_cpu = expert_offsets.to(torch::kCPU).contiguous();
    const int32_t* offs = offs_cpu.data_ptr<int32_t>();
    TORCH_CHECK(offs[num_experts] == total_tokens,
                "bf16_to_nvfp4_grouped: expert_offsets must end at total_tokens");

    // Per-expert 128-row-aligned scale sections (grouped GEMM scale_A contract):
    // sf_offsets[e] = cumulative padded rows; section e spans
    // padded_rows_e * (K/16) bytes.
    int groups_per_row = K / 16;
    auto sf_offsets_cpu = torch::empty({num_experts + 1},
                                       torch::TensorOptions().dtype(at::kInt));
    int32_t* sf = sf_offsets_cpu.data_ptr<int32_t>();
    int padded_total = 0;
    for (int e = 0; e <= num_experts; e++) {
        sf[e] = padded_total;
        if (e < num_experts) {
            int M_e = offs[e + 1] - offs[e];
            padded_total += ((M_e + 127) / 128) * 128;
        }
    }

    auto packed = torch::empty({total_tokens, K / 2},
                               input.options().dtype(torch::kUInt8));
    auto scales = torch::zeros({static_cast<int64_t>(padded_total) * groups_per_row},
                               input.options().dtype(torch::kUInt8));
    auto expert_offsets_dev = expert_offsets.is_cuda()
        ? expert_offsets.contiguous()
        : expert_offsets.to(input.device()).contiguous();
    auto sf_offsets_dev = sf_offsets_cpu.to(input.device());

    lc::Bf16ToNvfp4GroupedParams params{};
    params.input = input.data_ptr();
    params.output_packed = packed.data_ptr();
    params.output_scales = scales.data_ptr();
    params.expert_offsets = expert_offsets_dev.data_ptr<int32_t>();
    params.sf_offsets = sf_offsets_dev.data_ptr<int32_t>();
    params.total_tokens = total_tokens;
    params.num_experts = num_experts;
    params.K = K;
    if (input_scales.has_value()) {
        auto& is = input_scales.value();
        TORCH_CHECK(is.is_cuda() && is.scalar_type() == at::kFloat &&
                    is.numel() == num_experts,
                    "bf16_to_nvfp4_grouped: input_scales must be [num_experts] "
                    "float32 on CUDA");
        params.input_scales = is.data_ptr<float>();
    }

    lc::launch_bf16_to_nvfp4_grouped(params, current_stream());

    return {packed, scales, sf_offsets_dev};
}

// ── Fused SiLU-mul + grouped BF16 → NVFP4 activation quantization ─────────

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> silu_mul_to_nvfp4_grouped(
    torch::Tensor gate,            // [total_tokens, K] BF16 (gate GEMM output)
    torch::Tensor up,              // [total_tokens, K] BF16 (up GEMM output)
    torch::Tensor expert_offsets,  // [num_experts + 1] int32 cumulative tokens
    c10::optional<torch::Tensor> input_scales) {  // [num_experts] float32

    TORCH_CHECK(gate.is_cuda() && gate.dtype() == torch::kBFloat16,
                "silu_mul_to_nvfp4_grouped: gate must be BF16 on CUDA");
    TORCH_CHECK(up.is_cuda() && up.dtype() == torch::kBFloat16,
                "silu_mul_to_nvfp4_grouped: up must be BF16 on CUDA");
    TORCH_CHECK(gate.dim() == 2 && gate.is_contiguous(),
                "silu_mul_to_nvfp4_grouped: gate must be contiguous [T, K]");
    TORCH_CHECK(up.dim() == 2 && up.is_contiguous(),
                "silu_mul_to_nvfp4_grouped: up must be contiguous [T, K]");
    TORCH_CHECK(gate.size(0) == up.size(0) && gate.size(1) == up.size(1),
                "silu_mul_to_nvfp4_grouped: gate and up must have the same shape");
    TORCH_CHECK(expert_offsets.dim() == 1 &&
                expert_offsets.scalar_type() == at::kInt,
                "silu_mul_to_nvfp4_grouped: expert_offsets must be 1D int32");

    int total_tokens = gate.size(0);
    int K = gate.size(1);
    int num_experts = expert_offsets.size(0) - 1;
    TORCH_CHECK(K % 64 == 0,
                "silu_mul_to_nvfp4_grouped: K must be divisible by 64 "
                "(K/16 groups must align to the 4-group Sm1xx atom)");

    auto offs_cpu = expert_offsets.to(torch::kCPU).contiguous();
    const int32_t* offs = offs_cpu.data_ptr<int32_t>();
    TORCH_CHECK(offs[num_experts] == total_tokens,
                "silu_mul_to_nvfp4_grouped: expert_offsets must end at total_tokens");

    // Per-expert 128-row-aligned scale sections — IDENTICAL layout to
    // bf16_to_nvfp4_grouped (the down GEMM consumes the same scale_A contract).
    int groups_per_row = K / 16;
    auto sf_offsets_cpu = torch::empty({num_experts + 1},
                                       torch::TensorOptions().dtype(at::kInt));
    int32_t* sf = sf_offsets_cpu.data_ptr<int32_t>();
    int padded_total = 0;
    for (int e = 0; e <= num_experts; e++) {
        sf[e] = padded_total;
        if (e < num_experts) {
            int M_e = offs[e + 1] - offs[e];
            padded_total += ((M_e + 127) / 128) * 128;
        }
    }

    auto packed = torch::empty({total_tokens, K / 2},
                               gate.options().dtype(torch::kUInt8));
    auto scales = torch::zeros({static_cast<int64_t>(padded_total) * groups_per_row},
                               gate.options().dtype(torch::kUInt8));
    auto expert_offsets_dev = expert_offsets.is_cuda()
        ? expert_offsets.contiguous()
        : expert_offsets.to(gate.device()).contiguous();
    auto sf_offsets_dev = sf_offsets_cpu.to(gate.device());

    lc::SiluMulToNvfp4GroupedParams params{};
    params.gate = gate.data_ptr();
    params.up = up.data_ptr();
    params.output_packed = packed.data_ptr();
    params.output_scales = scales.data_ptr();
    params.expert_offsets = expert_offsets_dev.data_ptr<int32_t>();
    params.sf_offsets = sf_offsets_dev.data_ptr<int32_t>();
    params.total_tokens = total_tokens;
    params.num_experts = num_experts;
    params.K = K;
    if (input_scales.has_value()) {
        auto& is = input_scales.value();
        TORCH_CHECK(is.is_cuda() && is.scalar_type() == at::kFloat &&
                    is.numel() == num_experts,
                    "silu_mul_to_nvfp4_grouped: input_scales must be [num_experts] "
                    "float32 on CUDA");
        params.input_scales = is.data_ptr<float>();
    }

    lc::launch_silu_mul_to_nvfp4_grouped(params, current_stream());

    return {packed, scales, sf_offsets_dev};
}

// ── Offline BF16 → FP8 weight quantization ────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor> weight_fp8_quant(torch::Tensor weight) {
    TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kBFloat16,
                "weight_fp8_quant: weight must be BF16 on CUDA");
    TORCH_CHECK(weight.dim() == 2 && weight.is_contiguous(),
                "weight_fp8_quant: weight must be contiguous [N, K]");

    int N = weight.size(0);
    int K = weight.size(1);
    int K_blocks = (K + 127) / 128;
    int N_blocks = (N + 127) / 128;

    auto output = torch::empty({N, K}, weight.options().dtype(at::kFloat8_e4m3fn));
    // Returned shape [N_blocks, K_blocks] row-major == the K-major
    // scales[k_blk + n_blk*K_blocks] buffer launch_fp8_gemm consumes as scale_B.
    auto scales = torch::empty({N_blocks, K_blocks},
                               weight.options().dtype(at::kFloat));

    lc::WeightFp8QuantParams params{};
    params.N = N;
    params.K = K;
    params.input = weight.data_ptr();
    params.output = output.data_ptr();
    params.scales = scales.data_ptr();

    lc::launch_weight_fp8_quant(params, current_stream());

    return {output, scales};
}

// ── FP8 Single GEMM ────────────────────────────────────────────────────────

torch::Tensor fp8_gemm(
    torch::Tensor A, torch::Tensor B,
    torch::Tensor scale_A, torch::Tensor scale_B,
    int64_t M, int64_t N, int64_t K,
    const std::string& output_dtype) {

    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "fp8_gemm: A and B must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "fp8_gemm: must be contiguous");

    auto odtype = parse_output_dtype(output_dtype);
    auto torch_odtype = (odtype == lc::GemmOutputDtype::kBFloat16) ? at::kBFloat16 : at::kHalf;
    auto D = torch::empty({M, N}, A.options().dtype(torch_odtype));

    size_t ws_bytes = lc::query_fp8_gemm_workspace_size(
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), odtype);
    torch::Tensor workspace;
    void* ws_ptr = nullptr;
    if (ws_bytes > 0) {
        workspace = torch::empty({static_cast<int64_t>(ws_bytes)}, A.options().dtype(at::kByte));
        ws_ptr = workspace.data_ptr();
    }

    lc::Fp8GemmParams params;
    params.M = static_cast<int>(M);
    params.N = static_cast<int>(N);
    params.K = static_cast<int>(K);
    params.A = A.data_ptr();
    params.B = B.data_ptr();
    params.D = D.data_ptr();
    params.scale_A = scale_A.data_ptr();
    params.scale_B = scale_B.data_ptr();
    params.output_dtype = odtype;

    lc::launch_fp8_gemm(params, ws_ptr, current_stream());
    return D;
}

// ── FP8 Grouped GEMM ───────────────────────────────────────────────────────

torch::Tensor fp8_grouped_gemm(
    torch::Tensor A_base, torch::Tensor B_base, torch::Tensor D_base,
    torch::Tensor scale_A_base, torch::Tensor scale_B_base,
    torch::Tensor expert_offsets, torch::Tensor problem_sizes,
    int64_t N, int64_t K,
    const std::string& output_dtype) {

    TORCH_CHECK(A_base.is_cuda(), "fp8_grouped_gemm: A_base must be on CUDA");
    auto odtype = parse_output_dtype(output_dtype);
    int num_experts = expert_offsets.size(0) - 1;

    size_t ws_bytes = lc::query_fp8_grouped_gemm_workspace_size(
        num_experts, static_cast<int>(N), static_cast<int>(K), odtype);
    auto workspace = torch::empty({static_cast<int64_t>(ws_bytes)},
                                   A_base.options().dtype(at::kByte));

    lc::Fp8GroupedGemmParams params;
    params.num_experts = num_experts;
    params.N = static_cast<int>(N);
    params.K = static_cast<int>(K);
    params.A_base = A_base.data_ptr();
    params.B_base = B_base.data_ptr();
    params.D_base = D_base.data_ptr();
    params.scale_A_base = scale_A_base.data_ptr();
    params.scale_B_base = scale_B_base.data_ptr();
    params.expert_offsets = expert_offsets.data_ptr<int32_t>();
    params.problem_sizes = problem_sizes.data_ptr<int32_t>();
    params.output_dtype = odtype;

    lc::launch_fp8_grouped_gemm(params, workspace.data_ptr(), ws_bytes, current_stream());
    return D_base;
}

// ── GGUF Grouped GEMM (MoE expert FFN) ─────────────────────────────────────
//
// Device-routed Int path (mmvq/mmq) is CUDA-graph capturable (TD-GG5); Dequant
// uses the host-loop. `b_ptrs` is a CUDA int64 tensor of per-expert packed-GGUF
// weight device pointers ([num_experts]); `expert_offsets` is [num_experts+1]
// int32 cumulative token counts. `force`: 0=auto, 1=mmvq, 2=mmq, 3=hostloop.
torch::Tensor gguf_grouped_gemm(
    torch::Tensor A_base, torch::Tensor expert_offsets, torch::Tensor b_ptrs,
    int64_t type_i, int64_t strategy_i, int64_t N, int64_t K, int64_t force_i) {

    TORCH_CHECK(A_base.is_cuda() && A_base.dtype() == torch::kBFloat16,
                "gguf_grouped_gemm: A_base must be BF16 on CUDA");
    TORCH_CHECK(A_base.dim() == 2 && A_base.is_contiguous(),
                "gguf_grouped_gemm: A_base must be contiguous [T, K]");
    TORCH_CHECK(expert_offsets.is_cuda() && expert_offsets.dtype() == torch::kInt32,
                "gguf_grouped_gemm: expert_offsets must be int32 on CUDA");
    TORCH_CHECK(b_ptrs.is_cuda() && b_ptrs.dtype() == torch::kInt64,
                "gguf_grouped_gemm: b_ptrs must be int64 (device pointers) on CUDA");

    const int T = static_cast<int>(A_base.size(0));
    const int E = static_cast<int>(b_ptrs.numel());
    TORCH_CHECK(expert_offsets.numel() == E + 1,
                "gguf_grouped_gemm: expert_offsets must be [num_experts+1]");

    auto D = torch::empty({T, N}, A_base.options());

    lc::GgufGroupedGemmKernelParams p;
    p.type = static_cast<lc::GgufType>(type_i);
    p.strategy = static_cast<lc::GgufGroupedStrategy>(strategy_i);
    p.num_experts = E;
    p.N = static_cast<int>(N);
    p.K = static_cast<int>(K);
    p.A_base = reinterpret_cast<const __nv_bfloat16*>(A_base.data_ptr());
    p.D_base = reinterpret_cast<__nv_bfloat16*>(D.data_ptr());
    p.expert_offsets = expert_offsets.data_ptr<int32_t>();
    p.B_ptrs = reinterpret_cast<const void**>(b_ptrs.data_ptr());

    const size_t ws_bytes = lc::gguf_grouped_gemm_workspace_bytes(T, static_cast<int>(K), E);
    torch::Tensor ws;
    void* ws_ptr = nullptr;
    if (ws_bytes > 0) {
        ws = torch::empty({static_cast<int64_t>(ws_bytes)},
                          A_base.options().dtype(at::kByte));
        ws_ptr = ws.data_ptr();
    }
    auto stream = static_cast<cudaStream_t>(current_stream());

    if (p.strategy == lc::GgufGroupedStrategy::Dequant || force_i == 3) {
        lc::launch_gguf_grouped_gemm_hostloop(p, ws_ptr, ws_bytes, stream);
    } else if (force_i == 0) {
        lc::launch_gguf_grouped_gemm(p, T, ws_ptr, ws_bytes, stream);
    } else {
        lc::launch_gguf_grouped_gemm_int_forced(
            p, T, ws_ptr, ws_bytes,
            static_cast<lc::GgufGroupedIntForce>(force_i), stream);
    }
    return D;
}

// ── GGUF single-weight mmvq (decode GEMV) ──────────────────────────────────
//
// Direct binding for the single-weight integer mat-vec (Q8_1 activation quant
// + dp4a dot against the packed GGUF weight) — the decode GEMV the engine's
// projection graph launches for every attention/indexer projection. Exposed so
// the kernel is testable/benchmarkable in isolation (the grouped path reuses
// the same inner math but different routing).
torch::Tensor gguf_mmvq(
    torch::Tensor A, torch::Tensor B_packed, int64_t type_i, int64_t N, int64_t K) {

    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kBFloat16,
                "gguf_mmvq: A must be BF16 on CUDA");
    TORCH_CHECK(A.dim() == 2 && A.is_contiguous(),
                "gguf_mmvq: A must be contiguous [M, K]");
    TORCH_CHECK(A.size(1) == K, "gguf_mmvq: A.size(1) != K");
    TORCH_CHECK(B_packed.is_cuda() && B_packed.dtype() == torch::kUInt8 &&
                B_packed.is_contiguous(),
                "gguf_mmvq: B_packed must be contiguous uint8 on CUDA");

    const int M = static_cast<int>(A.size(0));
    auto C = torch::empty({M, N}, A.options());

    lc::GgufMmvqParams p;
    p.M = M;
    p.N = static_cast<int>(N);
    p.K = static_cast<int>(K);
    p.A = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr());
    p.B = B_packed.data_ptr();
    p.C = reinterpret_cast<__nv_bfloat16*>(C.data_ptr());
    p.type = static_cast<lc::GgufType>(type_i);

    const size_t ws_bytes = lc::gguf_mmvq_workspace_bytes(M, static_cast<int>(K));
    auto ws = torch::empty({static_cast<int64_t>(ws_bytes)},
                           A.options().dtype(at::kByte));
    lc::launch_gguf_mmvq(p, ws.data_ptr(),
                         static_cast<cudaStream_t>(current_stream()));
    return C;
}

// ── Q4K Dequant GEMM ───────────────────────────────────────────────────────

torch::Tensor q4k_dequant_gemm(
    torch::Tensor x_bf16,
    torch::Tensor weight_q4k_packed
) {
    TORCH_CHECK(x_bf16.is_cuda() && x_bf16.dtype() == torch::kBFloat16,
        "x_bf16 must be BF16 on CUDA");
    TORCH_CHECK(x_bf16.dim() == 2 && x_bf16.is_contiguous(),
        "x_bf16 must be contiguous [M, K]");
    TORCH_CHECK(weight_q4k_packed.is_cuda() && weight_q4k_packed.dtype() == torch::kUInt8,
        "weight_q4k_packed must be uint8 on CUDA");
    TORCH_CHECK(weight_q4k_packed.dim() == 2 && weight_q4k_packed.is_contiguous(),
        "weight_q4k_packed must be contiguous [N, K*9/16]");

    int M = x_bf16.size(0);
    int K = x_bf16.size(1);
    int N = weight_q4k_packed.size(0);
    int expected_row_bytes = (K / 256) * 144;

    TORCH_CHECK(K % 256 == 0, "K must be divisible by 256, got K=" + std::to_string(K));
    TORCH_CHECK(weight_q4k_packed.size(1) == expected_row_bytes,
        "weight row bytes mismatch: expected " + std::to_string(expected_row_bytes) +
        ", got " + std::to_string((int)weight_q4k_packed.size(1)));

    auto output = torch::empty({M, N}, x_bf16.options());

    lc::Q4KDequantGemmParams params;
    params.M = M;
    params.N = N;
    params.K = K;
    params.A = reinterpret_cast<const __nv_bfloat16*>(x_bf16.data_ptr());
    params.B_q4k = weight_q4k_packed.data_ptr();
    params.C = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

    lc::launch_q4k_dequant_gemm(params, at::cuda::getCurrentCUDAStream());
    return output;
}

// ── Q4K Tensor-Core GEMM ───────────────────────────────────────────────────

torch::Tensor q4k_cutlass_gemm(
    torch::Tensor x_bf16,
    torch::Tensor weight_q4k_packed
) {
    TORCH_CHECK(x_bf16.is_cuda() && x_bf16.dtype() == torch::kBFloat16,
        "x_bf16 must be BF16 on CUDA");
    TORCH_CHECK(x_bf16.dim() == 2 && x_bf16.is_contiguous(),
        "x_bf16 must be contiguous [M, K]");
    TORCH_CHECK(weight_q4k_packed.is_cuda() && weight_q4k_packed.dtype() == torch::kUInt8,
        "weight_q4k_packed must be uint8 on CUDA");
    TORCH_CHECK(weight_q4k_packed.dim() == 2 && weight_q4k_packed.is_contiguous(),
        "weight_q4k_packed must be contiguous [N, K*9/16]");

    int M = x_bf16.size(0);
    int K = x_bf16.size(1);
    int N = weight_q4k_packed.size(0);
    int expected_row_bytes = (K / 256) * 144;

    TORCH_CHECK(K % 256 == 0, "K must be divisible by 256, got K=" + std::to_string(K));
    TORCH_CHECK(weight_q4k_packed.size(1) == expected_row_bytes,
        "weight row bytes mismatch: expected " + std::to_string(expected_row_bytes) +
        ", got " + std::to_string((int)weight_q4k_packed.size(1)));

    auto output = torch::empty({M, N}, x_bf16.options());

    lc::Q4KCutlassGemmParams params;
    params.M = M;
    params.N = N;
    params.K = K;
    params.A = reinterpret_cast<const __nv_bfloat16*>(x_bf16.data_ptr());
    params.B_q4k = weight_q4k_packed.data_ptr();
    params.C = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

    lc::launch_q4k_cutlass_gemm(params, at::cuda::getCurrentCUDAStream());
    return output;
}

// ── Workspace queries ───────────────────────────────────────────────────────

int64_t query_nvfp4_gemm_ws(int64_t M, int64_t N, int64_t K, const std::string& output_dtype) {
    return static_cast<int64_t>(lc::query_nvfp4_gemm_workspace_size(
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), parse_output_dtype(output_dtype)));
}
int64_t query_fp8_gemm_ws(int64_t M, int64_t N, int64_t K, const std::string& output_dtype) {
    return static_cast<int64_t>(lc::query_fp8_gemm_workspace_size(
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), parse_output_dtype(output_dtype)));
}
int64_t query_nvfp4_grouped_gemm_ws(int64_t me, int64_t N, int64_t K, const std::string& od) {
    return static_cast<int64_t>(lc::query_nvfp4_grouped_gemm_workspace_size(
        static_cast<int>(me), static_cast<int>(N), static_cast<int>(K), parse_output_dtype(od)));
}
int64_t query_fp8_grouped_gemm_ws(int64_t me, int64_t N, int64_t K, const std::string& od) {
    return static_cast<int64_t>(lc::query_fp8_grouped_gemm_workspace_size(
        static_cast<int>(me), static_cast<int>(N), static_cast<int>(K), parse_output_dtype(od)));
}
int64_t query_sf_a(int64_t M, int64_t N, int64_t K) {
    return static_cast<int64_t>(lc::query_sf_buffer_size_a(
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)));
}
int64_t query_sf_b(int64_t M, int64_t N, int64_t K) {
    return static_cast<int64_t>(lc::query_sf_buffer_size_b(
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)));
}

// ── Module definition ───────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SM120 Quantized GEMM CUDA kernels";

    // Dynamic FP8 quantization
    m.def("dynamic_fp8_quant", &dynamic_fp8_quant, py::arg("input"),
          py::arg("m_major_scales") = false,
          "Dynamic FP8 quantization: BF16 -> FP8 E4M3. Returns (output, scales). "
          "m_major_scales=True emits the fp8_gemm scale_A (SFA) layout "
          "[m + k_blk*M]; the default row-major layout is WRONG for fp8_gemm "
          "at M > 1.");

    // Grouped BF16 -> NVFP4 activation quantization
    m.def("bf16_to_nvfp4_grouped", &bf16_to_nvfp4_grouped,
          py::arg("input"), py::arg("expert_offsets"),
          py::arg("input_scales") = py::none(),
          "Grouped BF16 -> NVFP4 activation quantization with Sm1xx interleaved "
          "scales. Optional per-expert input_scales (TRT-LLM scheme: group scale "
          "= amax/(6*is), values quantize as x/(scale*is); GEMM alpha must "
          "include the same is). Returns (packed, scales, sf_offsets).");

    // Fused SiLU-mul + grouped BF16 -> NVFP4 activation quantization
    m.def("silu_mul_to_nvfp4_grouped", &silu_mul_to_nvfp4_grouped,
          py::arg("gate"), py::arg("up"), py::arg("expert_offsets"),
          py::arg("input_scales") = py::none(),
          "Fused SwiGLU (silu(gate)*up) + grouped BF16 -> NVFP4 activation "
          "quantization. Reads the gate and up grouped-GEMM outputs separately "
          "and writes FP4 + Sm1xx-interleaved UE4M3 scales BYTE-IDENTICAL to "
          "the unfused interleave->SwiGLU->bf16_to_nvfp4_grouped sequence. "
          "Optional per-expert input_scales (same TRT-LLM scheme as "
          "bf16_to_nvfp4_grouped). Returns (packed, scales, sf_offsets).");

    // Offline BF16 -> FP8 weight quantization
    m.def("weight_fp8_quant", &weight_fp8_quant, py::arg("weight"),
          "Offline BF16 -> FP8 E4M3 weight quantization with 128x128 tile "
          "scales in the K-major scale_B layout fp8_gemm consumes. "
          "Returns (output, scales).");

    // NVFP4 GEMM (low-level: pre-quantized FP4)
    m.def("nvfp4_gemm", &nvfp4_gemm,
          py::arg("A"), py::arg("B"), py::arg("scale_A"), py::arg("scale_B"),
          py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha"), py::arg("output_dtype"),
          "NVFP4 single GEMM on SM120 (pre-quantized FP4 inputs).");

    // NVFP4 high-level (BF16 input → quantize → GEMM)
    m.def("nvfp4_gemm_preprocess", &nvfp4_gemm_preprocess,
          py::arg("weight_scales"), py::arg("N"), py::arg("K"),
          "Preprocess NVFP4 weight scales to Sm1xx interleaved format.");
    m.def("nvfp4_gemm_from_bf16", &nvfp4_gemm_from_bf16,
          py::arg("x_bf16"), py::arg("weight_packed_uint8"),
          py::arg("weight_scales"), py::arg("weight_scale_global"),
          "NVFP4 GEMM from BF16 input (quantize activation + reformat scales + GEMM).");

    // NVFP4 grouped GEMM
    m.def("nvfp4_grouped_gemm", &nvfp4_grouped_gemm,
          py::arg("A_base"), py::arg("B_base"), py::arg("D_base"),
          py::arg("scale_A_base"), py::arg("scale_B_base"), py::arg("alphas"),
          py::arg("expert_offsets"), py::arg("sf_offsets"), py::arg("problem_sizes"),
          py::arg("N"), py::arg("K"), py::arg("output_dtype"),
          "NVFP4 grouped GEMM on SM120.");

    // FP8 GEMM
    m.def("fp8_gemm", &fp8_gemm,
          py::arg("A"), py::arg("B"), py::arg("scale_A"), py::arg("scale_B"),
          py::arg("M"), py::arg("N"), py::arg("K"), py::arg("output_dtype"),
          "FP8 blockwise-scaled single GEMM on SM120.");
    m.def("fp8_grouped_gemm", &fp8_grouped_gemm,
          py::arg("A_base"), py::arg("B_base"), py::arg("D_base"),
          py::arg("scale_A_base"), py::arg("scale_B_base"),
          py::arg("expert_offsets"), py::arg("problem_sizes"),
          py::arg("N"), py::arg("K"), py::arg("output_dtype"),
          "FP8 grouped GEMM on SM120.");

    // GGUF grouped GEMM (MoE expert FFN): device-routed Int (mmvq/mmq,
    // CUDA-graph capturable) + host-loop Dequant golden.
    m.def("gguf_grouped_gemm", &gguf_grouped_gemm,
          py::arg("A_base"), py::arg("expert_offsets"), py::arg("b_ptrs"),
          py::arg("type"), py::arg("strategy"), py::arg("N"), py::arg("K"),
          py::arg("force") = 0,
          "GGUF grouped GEMM on SM120. type: GgufType int (Q2_K=0..Q8_0=5, MXFP4=6); "
          "strategy: 0=Dequant,1=Int; force: 0=auto,1=mmvq,2=mmq,3=hostloop.");

    // GGUF single-weight mmvq (decode GEMV; the engine projection-graph path).
    m.def("gguf_mmvq", &gguf_mmvq,
          py::arg("A"), py::arg("B_packed"), py::arg("type"),
          py::arg("N"), py::arg("K"),
          "GGUF integer mat-vec on SM120: BF16 A[M,K] @ dequant(B)^T -> BF16 "
          "[M,N]. type: GgufType int (Q2_K=0..Q8_0=5, MXFP4=6).");

    // Q4K GEMM
    m.def("q4k_dequant_gemm", &q4k_dequant_gemm,
          py::arg("x_bf16"), py::arg("weight_q4k_packed"),
          "Q4_K dequant-GEMM: x_bf16 @ dequant(W_q4k).T -> BF16 output.");
    m.def("q4k_cutlass_gemm", &q4k_cutlass_gemm,
          py::arg("x_bf16"), py::arg("weight_q4k_packed"),
          "Q4_K tensor-core GEMM: x_bf16 @ dequant(W_q4k).T -> BF16 output (wmma TC path).");

    // Workspace queries
    m.def("query_nvfp4_gemm_workspace_size", &query_nvfp4_gemm_ws,
          py::arg("M"), py::arg("N"), py::arg("K"), py::arg("output_dtype"));
    m.def("query_fp8_gemm_workspace_size", &query_fp8_gemm_ws,
          py::arg("M"), py::arg("N"), py::arg("K"), py::arg("output_dtype"));
    m.def("query_nvfp4_grouped_gemm_workspace_size", &query_nvfp4_grouped_gemm_ws,
          py::arg("max_experts"), py::arg("N"), py::arg("K"), py::arg("output_dtype"));
    m.def("query_fp8_grouped_gemm_workspace_size", &query_fp8_grouped_gemm_ws,
          py::arg("max_experts"), py::arg("N"), py::arg("K"), py::arg("output_dtype"));
    m.def("query_sf_buffer_size_a", &query_sf_a, py::arg("M"), py::arg("N"), py::arg("K"));
    m.def("query_sf_buffer_size_b", &query_sf_b, py::arg("M"), py::arg("N"), py::arg("K"));
}
