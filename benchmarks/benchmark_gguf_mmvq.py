"""
GGUF mmvq decode-GEMV benchmark (TD-GGUF-MMVQ-DECODE-EFFICIENCY).

Benchmarks the single-weight integer mat-vec (`gguf_mmvq`: Q8_1 activation
quant + dp4a dot against the packed GGUF weight) on the keeper52 GLM-5.2
decode shapes — the attention/indexer projections the engine's projection
graph launches every token (354 launches/token/rank measured) — plus a
Q8_0 N=6144 K-sweep and the grouped decode path (`gguf_grouped_gemm`,
force=mmvq) for the routed-FFN M_e=1 regime.

Reports wall (median CUDA-event) time, CUDA-graph replay time, and the
effective weight-bandwidth (packed weight bytes / kernel time) so the
%-of-peak is visible (RTX 5090 ~1.79 TB/s).

Usage:
    python benchmarks/benchmark_gguf_mmvq.py [-o results.json] [--name Q8_0]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sm120_gemm_kernels as GK
from tests.test_gguf_grouped_gemm import (
    GGUF_SPEC, make_gguf_weight, build_group, S_INT, F_MMVQ,
)

WARMUP = 30
ITERS = 200

# keeper52 GLM-5.2 Q8_0 decode projections (M=1), from the nsys profile:
#   q_a  6144->2048, q_b 2048->8192 (64 heads x 256 / TP2), kv_a 6144->576,
#   o_proj 8192->6144 (the dominant 180 us launch), indexer 6144->{4096,128}.
KEEPER52_SHAPES = [
    ("q_a    ", 2048, 6144),
    ("q_b    ", 8192, 2048),
    ("kv_a   ", 576, 6144),
    ("o_proj ", 6144, 8192),
    ("idx_qk ", 4096, 6144),
    ("idx_w  ", 128, 6144),
]

# Ticket sweep: Q8_0 N=6144, K=1536..8192, M=1.
SWEEP_KS = [1536, 2048, 4096, 6144, 8192]


def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    t = np.array([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
    return float(np.median(t))


def bench_graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            fn()
    except Exception:
        return None
    for _ in range(WARMUP):
        g.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        g.replay()
        ends[i].record()
    torch.cuda.synchronize()
    t = np.array([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
    return float(np.median(t))


def weight_bytes(name, N, K):
    spec = GGUF_SPEC[name]
    return N * (K // spec["qk"]) * spec["bytes"]


def run_single(name, label, N, K, M, results):
    t = GGUF_SPEC[name]["t"]
    A = (torch.randn(M, K, dtype=torch.float32) * 0.4).to(torch.bfloat16).cuda()
    W = make_gguf_weight(name, N, K, seed=N + K)
    fn = lambda: GK.gguf_mmvq(A, W, t, N, K)
    us = bench(fn)
    gus = bench_graph(fn)
    wb = weight_bytes(name, N, K)
    bw = wb / (gus if gus else us) / 1e3  # GB/s
    print(f"  {label} {name} M={M} N={N:5d} K={K:5d}: "
          f"{us:8.1f} us  graph {gus if gus else float('nan'):8.1f} us  "
          f"{bw:7.0f} GB/s")
    results.append(dict(kind="single", name=name, label=label.strip(),
                        M=M, N=N, K=K, us=us, graph_us=gus, gbps=bw))


def run_grouped_decode(name, E, N, K, results):
    """Routed-FFN decode regime: M_e=1 per expert, device mmvq forced."""
    t = GGUF_SPEC[name]["t"]
    counts = [1] * E
    A, off, bptr, _w = build_group(name, counts, N, K, seed=E)
    fn = lambda: GK.gguf_grouped_gemm(A, off, bptr, t, S_INT, N, K, F_MMVQ)
    us = bench(fn)
    gus = bench_graph(fn)
    wb = E * weight_bytes(name, N, K)
    bw = wb / (gus if gus else us) / 1e3
    print(f"  grouped {name} E={E:3d} M_e=1 N={N:5d} K={K:5d}: "
          f"{us:8.1f} us  graph {gus if gus else float('nan'):8.1f} us  "
          f"{bw:7.0f} GB/s")
    results.append(dict(kind="grouped", name=name, E=E, N=N, K=K,
                        us=us, graph_us=gus, gbps=bw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    results = []

    print(f"GPU: {torch.cuda.get_device_name()}")

    print("=== keeper52 GLM-5.2 Q8_0 decode projections (M=1) ===")
    for label, N, K in KEEPER52_SHAPES:
        run_single("Q8_0", label, N, K, 1, results)

    print("=== Q8_0 N=6144 K-sweep (M=1) ===")
    for K in SWEEP_KS:
        run_single("Q8_0", "sweep  ", 6144, K, 1, results)

    print("=== M sweep, o_proj shape (Q8_0 N=6144 K=8192) ===")
    for M in [2, 4, 8]:
        run_single("Q8_0", "msweep ", 6144, 8192, M, results)

    print("=== k-quant types, o_proj shape (M=1) ===")
    for name in ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]:
        run_single(name, "kquant ", 6144, 8192, 1, results)

    print("=== grouped decode (routed FFN, M_e=1, forced mmvq) ===")
    # E=1 rows: the keeper52 FETCH_AND_RUN_MOE regime (one grouped launch per
    # expert; grids (256|768|1536,1) in the nsys profile). Larger E: batched
    # multi-expert decode.
    for name, E, N, K in [("Q4_K", 1, 2048, 6144), ("Q4_K", 1, 6144, 2048),
                          ("Q4_K", 1, 12288, 6144), ("Q8_0", 1, 2048, 6144),
                          ("Q4_K", 8, 2048, 6144), ("Q4_K", 32, 2048, 6144),
                          ("Q8_0", 8, 2048, 6144)]:
        run_grouped_decode(name, E, N, K, results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"results -> {args.output}")


if __name__ == "__main__":
    main()
