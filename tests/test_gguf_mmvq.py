"""
GGUF single-weight mmvq (decode GEMV) tests.

Validates the `gguf_mmvq` binding — the integer mat-vec the engine's
projection graph launches for attention/indexer decode projections — against:
  * the dequant golden (lossless activation: dequant(W) then BF16 GEMM),
    measuring the Q8_1 activation-quant error (cosine >= 0.999), and
  * the device-routed grouped mmvq with a single expert (same int math,
    different routing) — an exact-math cross-check (cosine >= 0.9995).

Covers all 6 GGUF weight types, M in {1, 3, 8, 12, 17} (M > 8 exercises the
multi-pass M register tile), non-multiple-of-warp N, and graph capture
(the kernel sits inside the engine's decode CUDA graph).

Run: CUDA_VISIBLE_DEVICES=0 pytest tests/test_gguf_mmvq.py -v
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_reference import compute_metrics
from tests.test_gguf_grouped_gemm import (
    GGUF_SPEC, make_gguf_weight, S_DEQUANT, S_INT, F_HOSTLOOP, F_MMVQ,
)

import sm120_gemm_kernels as GK


def has_sm120():
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 12


pytestmark = pytest.mark.skipif(not has_sm120(), reason="requires SM120 GPU")


def goldens(name, A, W, N, K):
    """(dequant float golden, grouped-int cross-check) via the E=1 grouped op."""
    M = A.size(0)
    off = torch.tensor([0, M], dtype=torch.int32, device="cuda")
    bptr = torch.tensor([W.data_ptr()], dtype=torch.int64, device="cuda")
    t = GGUF_SPEC[name]["t"]
    dq = GK.gguf_grouped_gemm(A, off, bptr, t, S_DEQUANT, N, K, F_HOSTLOOP)
    gi = GK.gguf_grouped_gemm(A, off, bptr, t, S_INT, N, K, F_MMVQ)
    return dq, gi


@pytest.mark.parametrize("name", list(GGUF_SPEC.keys()))
@pytest.mark.parametrize("M", [1, 3, 8, 12, 17])
def test_mmvq_vs_goldens(name, M):
    torch.manual_seed(0)
    N, K = 320, 512
    A = (torch.randn(M, K, dtype=torch.float32) * 0.4).to(torch.bfloat16).cuda()
    W = make_gguf_weight(name, N, K, seed=42)

    out = GK.gguf_mmvq(A, W, GGUF_SPEC[name]["t"], N, K)
    dq, gi = goldens(name, A, W, N, K)
    torch.cuda.synchronize()

    m = compute_metrics(dq, out)
    assert m["cosine"] >= 0.999, \
        f"{name}/M={M} vs dequant cosine={m['cosine']:.6f}"
    m = compute_metrics(gi, out)
    assert m["cosine"] >= 0.9995, \
        f"{name}/M={M} vs grouped-int cosine={m['cosine']:.6f}"


@pytest.mark.parametrize("name", ["Q8_0", "Q4_K"])
def test_mmvq_decode_shape(name):
    """Realistic keeper52 o_proj-like shape (M=1, wide N, K > 1 superblock)."""
    torch.manual_seed(1)
    N, K = 1536, 2048
    A = (torch.randn(1, K, dtype=torch.float32) * 0.4).to(torch.bfloat16).cuda()
    W = make_gguf_weight(name, N, K, seed=7)

    out = GK.gguf_mmvq(A, W, GGUF_SPEC[name]["t"], N, K)
    dq, _ = goldens(name, A, W, N, K)
    torch.cuda.synchronize()

    m = compute_metrics(dq, out)
    assert m["cosine"] >= 0.999, f"{name} decode shape cosine={m['cosine']:.6f}"


def test_mmvq_determinism():
    """Same inputs -> bit-identical output across runs (fixed reduce order)."""
    torch.manual_seed(2)
    name = "Q8_0"
    N, K = 640, 1024
    A = (torch.randn(3, K, dtype=torch.float32) * 0.4).to(torch.bfloat16).cuda()
    W = make_gguf_weight(name, N, K, seed=9)
    t = GGUF_SPEC[name]["t"]

    ref = GK.gguf_mmvq(A, W, t, N, K)
    for _ in range(5):
        out = GK.gguf_mmvq(A, W, t, N, K)
        assert torch.equal(ref, out), "mmvq output not bit-deterministic"


def test_mmvq_graph_capture():
    """The kernel must capture + replay (it sits in the decode CUDA graph)."""
    torch.manual_seed(3)
    name = "Q8_0"
    N, K = 512, 1024
    A = (torch.randn(1, K, dtype=torch.float32) * 0.4).to(torch.bfloat16).cuda()
    W = make_gguf_weight(name, N, K, seed=11)
    t = GGUF_SPEC[name]["t"]

    eager = GK.gguf_mmvq(A, W, t, N, K)
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = GK.gguf_mmvq(A, W, t, N, K)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        captured = GK.gguf_mmvq(A, W, t, N, K)

    captured.zero_()
    g.replay()
    torch.cuda.synchronize()

    assert torch.equal(eager, captured), "graph replay != eager"
