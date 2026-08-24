"""
GGUF grouped GEMM tests (TD-GG5: device-routed int path, no host sync).

Validates the CUDA-graph-capturable device-routed integer grouped GGUF GEMM
(mmvq decode + mmq prefill) against:
  * the host-loop Dequant golden (the float reference: dequant(W) then BF16 GEMM)
    — measures the Q8_1 activation-quant error (cosine >= 0.999), AND
  * the host-loop Int golden (same int math, per-expert host dispatch) — a
    cross-check that the device routing matches (cosine >= 0.999).

Covers all 6 GGUF weight types (Q8_0 QK=32 + 5 k-quants), per-expert M
distributions (decode all M_e<=8, prefill M_e>>8, mixed, empty experts), and a
cudaGraph capture+replay (the point of the ticket: host sync removed).

Weight bytes are RANDOM but valid-layout (controlled finite half-precision
scales placed at the right offsets); both the int path and the dequant golden
consume the SAME bytes, so the cross-check is exact-math and the cosine measures
only the activation quant. (Q4_K additionally has a faithful packer in
test_reference; here we exercise all types uniformly via random bytes.)

Run: CUDA_VISIBLE_DEVICES=0 pytest tests/test_gguf_grouped_gemm.py -v
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_reference import compute_metrics

import sm120_gemm_kernels as GK


def has_sm120():
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 12


pytestmark = pytest.mark.skipif(not has_sm120(), reason="requires SM120 GPU")

# GgufType ints (gguf_dequant_gemm.h): Q2_K=0..Q8_0=5. block bytes / values per
# type, plus the (offset, value) of each finite half-precision scale field to
# stamp into every block so random byte fill never produces NaN/Inf scales.
GGUF_SPEC = {
    "Q2_K": dict(t=0, bytes=84,  qk=256, halves=[(80, 0.04), (82, 0.02)]),
    "Q3_K": dict(t=1, bytes=110, qk=256, halves=[(108, 0.012)]),
    "Q4_K": dict(t=2, bytes=144, qk=256, halves=[(0, 0.02), (2, 0.01)]),
    "Q5_K": dict(t=3, bytes=176, qk=256, halves=[(0, 0.015), (2, 0.008)]),
    "Q6_K": dict(t=4, bytes=210, qk=256, halves=[(208, 0.004)]),
    "Q8_0": dict(t=5, bytes=34,  qk=32,  halves=[(0, 0.02)]),
    # MXFP4: 1-byte E8M0 scale at offset 0 (no fp16 halves). e=121 -> d =
    # 2^(121-127)/2 = 2^-7; |w| <= 12*d = 0.094 (comparable to the others).
    "MXFP4": dict(t=6, bytes=17,  qk=32,  halves=[], e8m0=[(0, 121)]),
}

# Force selector for the binding (matches gguf_grouped_gemm.h):
F_AUTO, F_MMVQ, F_MMQ, F_HOSTLOOP = 0, 1, 2, 3
S_DEQUANT, S_INT = 0, 1


def make_gguf_weight(name, N, K, seed):
    """Random valid-layout packed GGUF weight [N, (K/QK)*bytes] uint8 (CUDA)."""
    spec = GGUF_SPEC[name]
    bb, qk = spec["bytes"], spec["qk"]
    nblk = K // qk
    row = nblk * bb
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 256, size=(N, row), dtype=np.uint8)
    for off, val in spec["halves"]:
        hb = np.frombuffer(np.float16(val).tobytes(), dtype=np.uint8)  # 2 bytes LE
        for b in range(nblk):
            base = b * bb + off
            w[:, base] = hb[0]
            w[:, base + 1] = hb[1]
    for off, e in spec.get("e8m0", []):   # 1-byte E8M0 scale (MXFP4)
        for b in range(nblk):
            w[:, b * bb + off] = e
    return torch.from_numpy(w).contiguous().cuda()


def build_group(name, counts, N, K, seed=0):
    """Build (A_base, expert_offsets, b_ptrs, weight_list) for per-expert token
    counts. Returns also the list of weight tensors to keep them alive."""
    E = len(counts)
    T = int(sum(counts))
    offsets = np.zeros(E + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts)
    A = (torch.randn(T, K, generator=torch.Generator().manual_seed(seed),
                     dtype=torch.float32) * 0.4).to(torch.bfloat16).cuda()
    weights = [make_gguf_weight(name, N, K, seed + 1 + e) for e in range(E)]
    b_ptrs = torch.tensor([w.data_ptr() for w in weights],
                          dtype=torch.int64, device="cuda")
    expert_offsets = torch.from_numpy(offsets).cuda()
    return A, expert_offsets, b_ptrs, weights


def run(name, A, off, bptr, N, K, strategy, force):
    return GK.gguf_grouped_gemm(A, off, bptr, GGUF_SPEC[name]["t"],
                                strategy, N, K, force)


# ── M distributions ────────────────────────────────────────────────────────
def dist_decode(E):                       # all M_e <= 8 (mmvq regime)
    return [1 + (e % 8) for e in range(E)]


def dist_prefill(E):                      # M_e >> 8 (mmq regime)
    return [64 + 40 * (e % 4) for e in range(E)]


def dist_mixed(E):                        # some small, some large
    return [(2 if e % 2 == 0 else 96 + 20 * (e % 3)) for e in range(E)]


def dist_empty(E):                        # several empty experts interleaved
    return [0 if e % 3 == 0 else (1 + (e % 7)) for e in range(E)]


# ============================================================================
# Correctness: device int (mmvq / mmq / auto) vs dequant golden + hostloop int
# ============================================================================
@pytest.mark.parametrize("name", list(GGUF_SPEC.keys()))
@pytest.mark.parametrize("dist", ["decode", "prefill", "mixed", "empty"])
def test_grouped_int_vs_golden(name, dist):
    torch.manual_seed(0)
    E = 6
    K = 256
    N = 256
    counts = {"decode": dist_decode, "prefill": dist_prefill,
              "mixed": dist_mixed, "empty": dist_empty}[dist](E)
    A, off, bptr, _w = build_group(name, counts, N, K, seed=100)

    # Float reference: host-loop Dequant (lossless activation).
    golden = run(name, A, off, bptr, N, K, S_DEQUANT, F_HOSTLOOP)
    # Host-loop Int golden (same int math, per-expert host dispatch).
    host_int = run(name, A, off, bptr, N, K, S_INT, F_HOSTLOOP)

    # Device-routed int: auto + both forced sub-paths.
    auto = run(name, A, off, bptr, N, K, S_INT, F_AUTO)
    mmvq = run(name, A, off, bptr, N, K, S_INT, F_MMVQ)
    mmq = run(name, A, off, bptr, N, K, S_INT, F_MMQ)
    torch.cuda.synchronize()

    # Int-vs-dequant: activation-quant error budget.
    for label, out in [("auto", auto), ("mmvq", mmvq), ("mmq", mmq)]:
        m = compute_metrics(golden, out)
        assert m["cosine"] >= 0.999, \
            f"{name}/{dist}/{label} vs dequant cosine={m['cosine']:.6f}"

    # Device int must match the host-loop int golden (same math).
    for label, out in [("auto", auto), ("mmvq", mmvq), ("mmq", mmq)]:
        m = compute_metrics(host_int, out)
        assert m["cosine"] >= 0.9995, \
            f"{name}/{dist}/{label} vs hostloop-int cosine={m['cosine']:.6f}"


# ============================================================================
# E=1 GEMV fast path (LS_GGUF_E1_KSPLIT=1, default OFF): dense-FFN / shared-
# expert regime. num_experts == 1 and M <= 8 routes to the NW=4 k-split kernel
# on the (N, 1) grid (llama.cpp mmvq launch geometry); M > 8 keeps the tiled
# paths. Engine shapes: N=2048/6144, K=6144/2048 (validated small here; the
# real shapes run in LayerStoRm3 bench/gemv_chain + the engine golden).
# The env is read ONCE per process — run this file as
#   LS_GGUF_E1_KSPLIT=1 pytest tests/test_gguf_grouped_gemm.py -k single_expert
# to exercise the fast path (E=1 shapes still validate the default path when
# the env is unset, so the test is meaningful either way).
# ============================================================================
@pytest.mark.parametrize("name", list(GGUF_SPEC.keys()))
@pytest.mark.parametrize("m", [1, 3, 8, 12])
def test_grouped_int_single_expert(name, m):
    torch.manual_seed(3)
    K = 512
    N = 384
    A, off, bptr, _w = build_group(name, [m], N, K, seed=300)

    golden = run(name, A, off, bptr, N, K, S_DEQUANT, F_HOSTLOOP)
    host_int = run(name, A, off, bptr, N, K, S_INT, F_HOSTLOOP)
    mmvq = run(name, A, off, bptr, N, K, S_INT, F_MMVQ)
    torch.cuda.synchronize()
    m1 = compute_metrics(golden, mmvq)
    assert m1["cosine"] >= 0.999, \
        f"{name}/E1/M={m} vs dequant cosine={m1['cosine']:.6f}"
    m2 = compute_metrics(host_int, mmvq)
    assert m2["cosine"] >= 0.9995, \
        f"{name}/E1/M={m} vs hostloop-int cosine={m2['cosine']:.6f}"

    # Determinism: the k-split reduce order is fixed per shape.
    mmvq2 = run(name, A, off, bptr, N, K, S_INT, F_MMVQ)
    torch.cuda.synchronize()
    assert torch.equal(mmvq, mmvq2), f"{name}/E1/M={m} non-deterministic"


# ============================================================================
# Realistic top-k routing (uneven token->expert counts) on Q8_0 + Q4_K + Q6_K
# ============================================================================
@pytest.mark.parametrize("name", ["Q8_0", "Q4_K", "Q6_K"])
def test_grouped_int_topk_routing(name):
    torch.manual_seed(1)
    E = 8
    K = 512
    N = 384
    n_tokens = 40
    top_k = 2
    rng = np.random.default_rng(7)
    routed = rng.integers(0, E, size=(n_tokens * top_k,))
    counts = [int((routed == e).sum()) for e in range(E)]
    A, off, bptr, _w = build_group(name, counts, N, K, seed=200)

    golden = run(name, A, off, bptr, N, K, S_DEQUANT, F_HOSTLOOP)
    auto = run(name, A, off, bptr, N, K, S_INT, F_AUTO)
    mmq = run(name, A, off, bptr, N, K, S_INT, F_MMQ)
    torch.cuda.synchronize()
    for label, out in [("auto", auto), ("mmq", mmq)]:
        m = compute_metrics(golden, out)
        assert m["cosine"] >= 0.999, \
            f"{name}/topk/{label} cosine={m['cosine']:.6f}"


# ============================================================================
# Graph capture: the device int path must capture + replay (host sync removed).
# ============================================================================
@pytest.mark.parametrize("force,label", [(F_MMVQ, "mmvq"), (F_MMQ, "mmq")])
def test_grouped_int_graph_capture(force, label):
    torch.manual_seed(2)
    name = "Q4_K"
    E = 6
    K = 256
    N = 256
    counts = dist_decode(E) if force == F_MMVQ else dist_prefill(E)
    A, off, bptr, _w = build_group(name, counts, N, K, seed=300)

    # Eager reference.
    eager = run(name, A, off, bptr, N, K, S_INT, force)
    torch.cuda.synchronize()

    # Warm up on a side stream (also runs cudaFuncSetAttribute before capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = run(name, A, off, bptr, N, K, S_INT, force)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        captured = run(name, A, off, bptr, N, K, S_INT, force)

    captured.zero_()
    g.replay()
    torch.cuda.synchronize()

    m = compute_metrics(eager, captured)
    assert m["cosine"] >= 0.9999, \
        f"{label} graph replay vs eager cosine={m['cosine']:.6f}"
    assert captured.abs().sum().item() > 0, "captured output is all zero"


# ── MXFP4: independent numpy dequant reference (llama.cpp type-39 math) ──────
# Ported from llama.cpp/ggml (MIT, "The ggml authors"): ggml_e8m0_to_fp32_half
# (ggml/src/ggml-impl.h) and dequantize_row_mxfp4 (ggml/src/ggml-quants.c).
# See THIRD_PARTY_NOTICES.md.

_MXFP4_KVALUES = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int8)


def _e8m0_half(e):
    """ggml_e8m0_to_fp32_half: 2^(e-127) / 2 with the e<2 denormal cases."""
    b = np.where(e < 2, np.uint32(0x00200000) << e.astype(np.uint32),
                 (e.astype(np.uint32) - 1) << 23)
    return b.astype(np.uint32).view(np.float32)


def mxfp4_dequant_ref(w_packed_u8, N, K):
    """[N, (K/32)*17] packed uint8 -> [N, K] float32 (ggml dequantize_row_mxfp4)."""
    nblk = K // 32
    blocks = w_packed_u8.reshape(N, nblk, 17)
    e = blocks[:, :, 0]
    qs = blocks[:, :, 1:]                              # [N, nblk, 16]
    d = _e8m0_half(e)[:, :, None]                      # [N, nblk, 1]
    lo = _MXFP4_KVALUES[qs & 0xF].astype(np.float32)   # elements 0..15
    hi = _MXFP4_KVALUES[qs >> 4].astype(np.float32)    # elements 16..31
    out = np.concatenate([lo * d, hi * d], axis=2)     # [N, nblk, 32]
    return out.reshape(N, K)


def test_mxfp4_dequant_matches_numpy_reference():
    """The kernel's dequant strategy must match the independent llama.cpp-math
    numpy reference bit-for-bit at the FP32 level (identity activations)."""
    N, K = 64, 128
    W = make_gguf_weight("MXFP4", N, K, seed=7)
    ref_w = mxfp4_dequant_ref(W.cpu().numpy(), N, K)   # [N, K]

    # Identity-ish probe: A = I_K rows in BF16 -> D[m, n] = W_dequant[n, m].
    M = K
    A = torch.eye(M, K, dtype=torch.bfloat16, device="cuda")
    off = torch.tensor([0, M], dtype=torch.int32, device="cuda")
    bptr = torch.tensor([W.data_ptr()], dtype=torch.int64, device="cuda")
    D = GK.gguf_grouped_gemm(A, off, bptr, GGUF_SPEC["MXFP4"]["t"],
                             S_DEQUANT, N, K, F_HOSTLOOP)
    got = D.float().cpu().numpy().T                    # [N, K]
    np.testing.assert_allclose(got, ref_w, rtol=0, atol=1e-6)
