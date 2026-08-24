"""
Kernel latency benchmarking for SM120 GEMM kernels.
Uses CUDA events for timing (not wall clock).

Benchmark patterns borrowed from sibling projects:
  - LayerStoRmKernels/benchmarks/benchmark_nvfp4_gemm.py
  - LayerStoRmKernels/benchmarks/benchmark_q4k_gemm.py
  - LayerStoRmExpertKernels/benchmarks/benchmark_speed.py

Usage:
    python benchmarks/benchmark_speed.py -v [-o results.json]
"""

import argparse
import json
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sm120_gemm_kernels as GK
from tests.test_reference import generate_nvfp4_weight, quantize_to_nvfp4, generate_q4k_weight

WARMUP_ITERS = 50
TIMED_ITERS = 500

# V3.2 model dimensions
HIDDEN_DIM = 7168
INTER_DIM = 2048


def benchmark_kernel(name, fn, warmup=WARMUP_ITERS, iters=TIMED_ITERS, verbose=False):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times_us = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
    times = np.array(times_us)

    result = {
        "name": name,
        "median_us": float(np.median(times)),
        "min_us": float(np.min(times)),
        "p95_us": float(np.percentile(times, 95)),
        "mean_us": float(np.mean(times)),
        "std_us": float(np.std(times)),
    }

    if verbose:
        print(f"  {name}: median={result['median_us']:.1f} us, "
              f"min={result['min_us']:.1f}, p95={result['p95_us']:.1f}")

    return result


def has_sm120():
    cap = torch.cuda.get_device_capability()
    return cap[0] > 12 or (cap[0] == 12 and cap[1] >= 0)


def _round_up(x, y):
    return (x + y - 1) // y * y


def main():
    parser = argparse.ArgumentParser(description="Benchmark SM120 GEMM Kernels")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o", "--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA GPU available")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"SM capability: {torch.cuda.get_device_capability()}")
    print()

    results = {"gpu": gpu_name, "benchmarks": []}

    # ── Dynamic FP8 Quant ────────────────────────────────────────────────
    print("=== Dynamic FP8 Quant ===")
    for M in [128, 256, 512, 1024]:
        inp = torch.randn(M, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")
        results["benchmarks"].append(
            benchmark_kernel(
                f"dynamic_fp8_quant_M{M}_K{HIDDEN_DIM}",
                lambda: GK.dynamic_fp8_quant(inp),
                verbose=args.verbose))

    if not has_sm120():
        print("Skipping SM120 GEMM benchmarks (no SM120 GPU)")
        return

    # ── FP8 Single GEMM ─────────────────────────────────────────────────
    print("=== FP8 Single GEMM ===")
    N, K = INTER_DIM, HIDDEN_DIM
    for M in [128, 256, 512, 1024]:
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        k_blocks = (K + 127) // 128
        n_blocks = (N + 127) // 128
        sA = torch.ones(M, k_blocks, dtype=torch.float32, device="cuda")
        sB = torch.ones(k_blocks, n_blocks, dtype=torch.float32, device="cuda")

        r = benchmark_kernel(
            f"fp8_gemm_M{M}_N{N}_K{K}",
            lambda: GK.fp8_gemm(A, B, sA, sB, M, N, K, "bf16"),
            verbose=args.verbose)
        r["tflops"] = round(2.0 * M * N * K / (r["median_us"] * 1e-6) / 1e12, 2)
        results["benchmarks"].append(r)

    # ── FP8 Grouped GEMM ────────────────────────────────────────────────
    print("=== FP8 Grouped GEMM (4 experts) ===")
    ne = 4
    N, K = INTER_DIM, HIDDEN_DIM
    for M_total in [128 * ne, 256 * ne, 512 * ne, 1024 * ne]:
        M_per = M_total // ne
        offsets_list = [i * M_per for i in range(ne + 1)]
        expert_offsets = torch.tensor(offsets_list, dtype=torch.int32, device="cuda")
        problem_sizes = torch.tensor(
            [[M_per, N, K]] * ne, dtype=torch.int32, device="cuda")
        A = torch.randn(M_total, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        B = torch.randn(ne, N, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        k_blocks = (K + 127) // 128
        n_blocks = (N + 127) // 128
        sA = torch.ones(M_total, k_blocks, dtype=torch.float32, device="cuda")
        sB = torch.ones(ne, k_blocks, n_blocks, dtype=torch.float32, device="cuda")
        D = torch.zeros(M_total, N, dtype=torch.bfloat16, device="cuda")

        r = benchmark_kernel(
            f"fp8_grouped_{ne}exp_M{M_per}_N{N}_K{K}",
            lambda: GK.fp8_grouped_gemm(
                A, B, D, sA, sB, expert_offsets, problem_sizes, N, K, "bf16"),
            verbose=args.verbose)
        r["tflops"] = round(2.0 * M_total * N * K / (r["median_us"] * 1e-6) / 1e12, 2)
        results["benchmarks"].append(r)

    # ── FP8 M=1 decode GEMV (attention projections) ─────────────────────
    # At M=1, GK.fp8_gemm dispatches to the hand-written bandwidth-bound GEMV.
    # Bandwidth-bound: weight bytes = N*K (FP8, 1 byte/elem). 5090 peak ~1790 GB/s.
    print("=== FP8 M=1 decode GEMV (q_a / q_b / kv_a) ===")
    for name, N, K in [("q_a", 1536, 7168), ("q_b", 2048, 1536), ("kv_a", 640, 7168)]:
        A = torch.randn(1, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        k_blocks = (K + 127) // 128
        n_blocks = (N + 127) // 128
        sA = torch.ones(k_blocks, 1, dtype=torch.float32, device="cuda")
        sB = torch.ones(n_blocks, k_blocks, dtype=torch.float32, device="cuda")
        r = benchmark_kernel(
            f"fp8_gemv_M1_{name}_N{N}_K{K}",
            lambda: GK.fp8_gemm(A, B, sA, sB, 1, N, K, "bf16"),
            verbose=args.verbose)
        weight_bytes = N * K  # FP8 = 1 byte/elem
        r["weight_gbps"] = round(weight_bytes / (r["median_us"] * 1e-6) / 1e9, 1)
        if args.verbose:
            print(f"    weight BW: {r['weight_gbps']} GB/s (of ~1790 peak)")
        results["benchmarks"].append(r)

    # ── NVFP4 Single GEMM (from BF16) ───────────────────────────────────
    print("=== NVFP4 GEMM (from BF16, includes quant+reformat) ===")
    N, K = INTER_DIM, HIDDEN_DIM
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16)
    w_uint8, w_scale, w_scale2 = quantize_to_nvfp4(w_bf16)
    w_uint8_gpu = w_uint8.cuda()
    w_scale_gpu = w_scale.cuda()
    scale2_val = w_scale2.item()

    for M in [128, 256, 512, 1024]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        r = benchmark_kernel(
            f"nvfp4_from_bf16_M{M}_N{N}_K{K}",
            lambda: GK.nvfp4_gemm_from_bf16(x, w_uint8_gpu, w_scale_gpu, scale2_val),
            verbose=args.verbose)
        r["tflops"] = round(2.0 * M * N * K / (r["median_us"] * 1e-6) / 1e12, 2)
        results["benchmarks"].append(r)

    # ── Q4K Dequant GEMM (CUDA core) ────────────────────────────────────
    print("=== Q4K Dequant GEMM (CUDA core) ===")
    N, K = INTER_DIM, HIDDEN_DIM
    w_q4k = generate_q4k_weight(N, K)
    w_q4k_gpu = torch.from_numpy(w_q4k).cuda()

    for M in [1, 4, 16, 64]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        r = benchmark_kernel(
            f"q4k_dequant_M{M}_N{N}_K{K}",
            lambda: GK.q4k_dequant_gemm(x, w_q4k_gpu),
            verbose=args.verbose)
        r["tflops"] = round(2.0 * M * N * K / (r["median_us"] * 1e-6) / 1e12, 2)
        results["benchmarks"].append(r)

    # ── Q4K Tensor-Core GEMM (wmma) ─────────────────────────────────────
    print("=== Q4K Cutlass GEMM (wmma tensor-core) ===")
    for M in [1, 4, 16, 64]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        r = benchmark_kernel(
            f"q4k_cutlass_M{M}_N{N}_K{K}",
            lambda: GK.q4k_cutlass_gemm(x, w_q4k_gpu),
            verbose=args.verbose)
        r["tflops"] = round(2.0 * M * N * K / (r["median_us"] * 1e-6) / 1e12, 2)
        results["benchmarks"].append(r)

    # ── Fused SiLU-mul → NVFP4 grouped vs unfused 4-op sequence ─────────
    # Engine MoE chain after gate+up grouped GEMMs:
    #   unfused: interleave gate+up → [T,2I] (2x cudaMemcpy2DAsync)
    #            + SwiGLU [T,2I]→[T,I] + bf16_to_nvfp4_grouped [T,I]→FP4
    #          = 4 kernel/launches + a [T,2I] scratch buffer.
    #   fused:   silu_mul_to_nvfp4_grouped (gate,up → FP4) = 1 launch.
    print("=== Fused SiLU-mul → NVFP4 grouped (decode MoE shapes) ===")
    I = INTER_DIM  # 2048
    ne = 8
    for T in [8, 16, 64]:
        M_per = max(T // ne, 1)
        T_real = M_per * ne
        offsets_list = [i * M_per for i in range(ne + 1)]
        expert_offsets = torch.tensor(offsets_list, dtype=torch.int32, device="cuda")
        gate = torch.randn(T_real, I, dtype=torch.bfloat16, device="cuda")
        up = torch.randn(T_real, I, dtype=torch.bfloat16, device="cuda")
        # Unfused scratch: interleaved [T, 2I] buffer the SwiGLU consumes.
        interleaved = torch.empty(T_real, 2 * I, dtype=torch.bfloat16, device="cuda")

        def unfused():
            # 2x cudaMemcpy2DAsync interleave gate→[:,0:I], up→[:,I:2I]
            interleaved[:, 0:I].copy_(gate)
            interleaved[:, I:2 * I].copy_(up)
            # SwiGLU [T,2I]→[T,I] (silu(gate)*up)
            g = interleaved[:, 0:I].float()
            u = interleaved[:, I:2 * I].float()
            swig = (g * (0.5 * torch.tanh(0.5 * g) + 0.5) * u).to(torch.bfloat16)
            # bf16→nvfp4 grouped quant [T,I]→FP4+scales
            return GK.bf16_to_nvfp4_grouped(swig.contiguous(), expert_offsets)

        def fused():
            return GK.silu_mul_to_nvfp4_grouped(gate, up, expert_offsets)

        r_un = benchmark_kernel(
            f"moe_swiglu_quant_UNFUSED_T{T_real}_I{I}_E{ne}",
            unfused, verbose=args.verbose)
        r_fu = benchmark_kernel(
            f"moe_swiglu_quant_FUSED_T{T_real}_I{I}_E{ne}",
            fused, verbose=args.verbose)
        speedup = round(r_un["median_us"] / r_fu["median_us"], 2)
        r_fu["speedup_vs_unfused"] = speedup
        r_fu["launches_saved"] = 3  # 4 ops (2 memcpy + swiglu + quant) → 1
        if args.verbose:
            print(f"    T={T_real}: unfused={r_un['median_us']:.1f}us "
                  f"fused={r_fu['median_us']:.1f}us speedup={speedup}x "
                  f"(4 launches → 1, saved 3)")
        results["benchmarks"].append(r_un)
        results["benchmarks"].append(r_fu)

    print()
    print(f"Total benchmarks: {len(results['benchmarks'])}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
