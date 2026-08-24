"""
GGUF grouped GEMM benchmark (TD-GG5): device-routed int path vs host-loop.

Compares the CUDA-graph-capturable device-routed integer grouped GGUF GEMM
(mmvq decode / mmq prefill, no host sync) against the original host per-expert
dispatch loop (one D2H sync per call), across active-expert counts and token
distributions (decode small-M, prefill large-M). Also reports the CUDA-graph
replay latency for the device path (the regime the decode engine uses) and
confirms the host-loop path is NOT graph-capturable.

Usage:
    python benchmarks/benchmark_gguf_grouped_gemm.py [-o results.json]
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
    GGUF_SPEC, build_group, S_INT, F_AUTO, F_HOSTLOOP,
)

WARMUP = 30
ITERS = 200


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
    """Median replay latency after capturing fn() once. Returns None if capture
    fails (e.g. a host sync makes the path non-capturable)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--name", default="Q4_K")
    args = ap.parse_args()

    name = args.name
    K, N = 2048, 2048
    results = []

    for E in [8, 32, 128]:
        for dist, mk in [("decode", lambda E: [1 + (e % 2) for e in range(E)]),
                         ("prefill", lambda E: [64 for _ in range(E)])]:
            counts = mk(E)
            T = int(sum(counts))
            A, off, bptr, _w = build_group(name, counts, N, K, seed=E * 10)
            t = GGUF_SPEC[name]["t"]

            host = lambda: GK.gguf_grouped_gemm(A, off, bptr, t, S_INT, N, K, F_HOSTLOOP)
            dev = lambda: GK.gguf_grouped_gemm(A, off, bptr, t, S_INT, N, K, F_AUTO)

            host_us = bench(host)
            dev_us = bench(dev)
            dev_graph_us = bench_graph(dev)
            host_graph_us = bench_graph(host)  # expected None (host sync)

            row = {
                "type": name, "experts": E, "dist": dist, "T": T, "N": N, "K": K,
                "host_loop_us": round(host_us, 2),
                "device_us": round(dev_us, 2),
                "device_graph_us": None if dev_graph_us is None else round(dev_graph_us, 2),
                "host_graph_capturable": host_graph_us is not None,
                "speedup_percall": round(host_us / dev_us, 2),
                "speedup_graph": (None if dev_graph_us is None
                                  else round(host_us / dev_graph_us, 2)),
            }
            results.append(row)
            print(f"E={E:3d} {dist:7s} T={T:4d}  host-loop {host_us:8.2f}us  "
                  f"device {dev_us:8.2f}us  device-graph "
                  f"{'  n/a' if dev_graph_us is None else f'{dev_graph_us:8.2f}us'}  "
                  f"host-capturable={row['host_graph_capturable']}  "
                  f"speedup(call)={row['speedup_percall']}x")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
