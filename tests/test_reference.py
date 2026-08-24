"""
Pure-PyTorch reference tests for SM120 GEMM kernels.
CPU only — no GPU required. Establishes error budgets.

Reference implementations borrowed from sibling projects:
  - LayerStoRmKernels/tests/test_nvfp4_gemm_reference.py
  - LayerStoRmKernels/tests/test_q4k_gemm_reference.py
  - LayerStoRmExpertKernels/tests/test_reference.py

Run: python tests/test_reference.py -v
"""

import math
import struct
import unittest

import numpy as np
import torch
import torch.nn.functional as F


# ── Constants ────────────────────────────────────────────────────────────────

# FP4 E2M1 lookup table: 4-bit index -> float value
FP4_E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

FP4_MAX_ABS = 6.0
GROUP_SIZE = 16
FP8_E4M3_MAX = 448.0

# Q4_K constants
QK_K = 256
BLOCK_Q4K_SIZE = 144
N_SUB_BLOCKS = 8
SUB_BLOCK_SIZE = 32
K_SCALE_SIZE = 12


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(ref: torch.Tensor, test: torch.Tensor) -> dict:
    ref_f = ref.float().flatten()
    test_f = test.float().flatten()
    cosine = F.cosine_similarity(ref_f.unsqueeze(0), test_f.unsqueeze(0)).item()
    mse = ((ref_f - test_f) ** 2).mean().item()
    nrmse = math.sqrt(mse) / (ref_f.norm().item() / math.sqrt(ref_f.numel()) + 1e-12)
    max_abs = (ref_f - test_f).abs().max().item()
    return {"cosine": cosine, "mse": mse, "nrmse": nrmse, "max_abs_err": max_abs}


# ── Dynamic FP8 Quantization Reference ──────────────────────────────────────

def ref_dynamic_fp8_quant(input_tensor, block_size=128):
    """Reference BF16->FP8 quantization with per-block scales."""
    FP8_MAX = 448.0
    inp = input_tensor.float()
    M, K = inp.shape
    num_blocks = (K + block_size - 1) // block_size
    scales = torch.zeros(M, num_blocks, dtype=torch.float32)
    output = torch.zeros(M, K, dtype=torch.float32)
    for row in range(M):
        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, K)
            block = inp[row, start:end]
            amax = block.abs().max().item()
            scale = amax / FP8_MAX if amax > 0 else 1.0
            scales[row, b] = scale
            output[row, start:end] = torch.clamp(block / scale, -FP8_MAX, FP8_MAX)
    return output, scales


def ref_fp8_dequant(data_fp8, scales, M, N, block_size=128):
    """Reference FP8 E4M3 dequantization with per-block scales."""
    output = torch.zeros(M, N, dtype=torch.float32)
    for row in range(M):
        for col in range(N):
            block_idx = col // block_size
            output[row, col] = data_fp8[row, col].float().item() * scales[row, block_idx].item()
    return output


# ── Grouped GEMM Reference ──────────────────────────────────────────────────

def ref_grouped_gemm(activations, weights_list, expert_offsets):
    """Reference grouped GEMM: per-expert torch.mm."""
    num_experts = len(weights_list)
    N = weights_list[0].shape[0]
    total_tokens = activations.shape[0]
    output = torch.zeros(total_tokens, N, dtype=torch.float32)
    for e in range(num_experts):
        start = expert_offsets[e].item()
        end = expert_offsets[e + 1].item()
        if start == end:
            continue
        x = activations[start:end].float()
        w = weights_list[e].float()
        output[start:end] = x @ w.T
    return output


# ── NVFP4 Reference ─────────────────────────────────────────────────────────

def ref_nvfp4_dequant(weight_uint8, scale_e4m3, scale_2):
    """Dequant NVFP4 packed weight to FP32 matrix.

    Args:
        weight_uint8: [N, K/2] uint8, packed FP4 (low nibble=even, high nibble=odd)
        scale_e4m3:   [N, K/16] per-group scales (float32)
        scale_2:      scalar float32 global scale

    Returns:
        [N, K] float32 dequantized weight matrix
    """
    out_features, packed_in = weight_uint8.shape
    in_features = packed_in * 2

    lo = (weight_uint8 & 0x0F).to(torch.int64)
    hi = (weight_uint8 >> 4).to(torch.int64)
    unpacked = torch.stack([lo, hi], dim=-1).reshape(out_features, in_features)

    fp4_float = FP4_E2M1_TABLE[unpacked]

    scale = scale_e4m3.float().repeat_interleave(GROUP_SIZE, dim=1)
    return fp4_float * scale * scale_2.float().item()


def ref_nvfp4_gemm(x, weight_uint8, scale_e4m3, scale_2):
    """NVFP4 dequant + GEMM: x @ dequant(W).T"""
    w_float = ref_nvfp4_dequant(weight_uint8, scale_e4m3, scale_2)
    return x.float() @ w_float.T


def generate_nvfp4_weight(out_features, in_features, seed=42):
    """Generate random valid NVFP4 weight for testing."""
    assert in_features % GROUP_SIZE == 0

    rng = torch.Generator()
    rng.manual_seed(seed)

    indices = torch.randint(0, 16, (out_features, in_features), generator=rng)
    pairs = indices.reshape(out_features, in_features // 2, 2)
    weight_uint8 = (pairs[:, :, 0] | (pairs[:, :, 1] << 4)).to(torch.uint8)

    num_groups = in_features // GROUP_SIZE
    raw_scales = torch.rand(out_features, num_groups, generator=rng) * 0.1 + 0.001
    if hasattr(torch, 'float8_e4m3fn'):
        scale_e4m3 = raw_scales.to(torch.float8_e4m3fn).float()
    else:
        scale_e4m3 = raw_scales

    scale_2 = torch.tensor(
        torch.rand(1, generator=rng).item() * 0.5 + 0.1, dtype=torch.float32
    )

    return weight_uint8, scale_e4m3, scale_2


def quantize_to_nvfp4(weight):
    """Quantize FP32/BF16 weight matrix to NVFP4 format.

    Two-level quantization: global scale -> per-group E4M3 scale -> nearest FP4 index.
    """
    N, K = weight.shape
    assert K % GROUP_SIZE == 0
    w = weight.float()

    global_amax = w.abs().max()
    scale_2 = global_amax / FP4_MAX_ABS
    scale_2 = torch.clamp(scale_2, min=1e-12)

    w_groups = w.reshape(N, K // GROUP_SIZE, GROUP_SIZE)
    group_amax = w_groups.abs().amax(dim=-1)
    scale_e4m3_raw = group_amax / (scale_2 * FP4_MAX_ABS)
    scale_e4m3_raw = torch.clamp(scale_e4m3_raw, min=1e-12, max=FP8_E4M3_MAX)
    if hasattr(torch, 'float8_e4m3fn'):
        scale_e4m3 = scale_e4m3_raw.to(torch.float8_e4m3fn).float()
    else:
        scale_e4m3 = scale_e4m3_raw
    scale_e4m3 = torch.clamp(scale_e4m3, min=1e-12)

    scale_expanded = scale_e4m3.repeat_interleave(GROUP_SIZE, dim=1)
    chunk_size = 256
    all_indices = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        local_vals = w[start:end] / (scale_expanded[start:end] * scale_2.item())
        distances = (local_vals.unsqueeze(-1) - FP4_E2M1_TABLE).abs()
        indices = distances.argmin(dim=-1)
        all_indices.append(indices)
    indices = torch.cat(all_indices, dim=0)

    pairs = indices.reshape(N, K // 2, 2)
    weight_uint8 = (pairs[:, :, 0] | (pairs[:, :, 1] << 4)).to(torch.uint8)

    return weight_uint8, scale_e4m3, scale_2


# ── Q4_K Reference ───────────────────────────────────────────────────────────

def _bytes_per_row(K):
    assert K % QK_K == 0
    return K * BLOCK_Q4K_SIZE // QK_K


def unpack_q4k_scales(scales_12):
    """Extract 8 scales and 8 mins from 12-byte packed array."""
    s = np.asarray(scales_12, dtype=np.uint8)
    orig_shape = s.shape[:-1]
    s = s.reshape(-1, 3, 4)

    d = s[:, 0, :]
    m = s[:, 1, :]
    m_d = s[:, 2, :]

    sc = np.concatenate([
        (d & 0x3F),
        (m_d & 0x0F) | ((d >> 2) & 0x30),
    ], axis=-1)

    mn = np.concatenate([
        (m & 0x3F),
        (m_d >> 4) | ((m >> 2) & 0x30),
    ], axis=-1)

    return sc.reshape(*orig_shape, 8), mn.reshape(*orig_shape, 8)


def _pack_q4k_scales(sc_8, mn_8):
    """Pack 8 6-bit scales and 8 6-bit mins into 12 bytes."""
    out = np.zeros(12, dtype=np.uint8)
    for j in range(8):
        ls = int(sc_8[j]) & 63
        lm = int(mn_8[j]) & 63
        if j < 4:
            out[j] = ls
            out[j + 4] = lm
        else:
            out[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4)
            out[j - 4] |= ((ls >> 4) << 6)
            out[j] |= ((lm >> 4) << 6)
    return out


def dequant_q4k_block(block_bytes):
    """Dequant one Q4_K super-block (144 bytes) to 256 float32 values."""
    b = np.asarray(block_bytes, dtype=np.uint8)
    assert b.shape == (BLOCK_Q4K_SIZE,)

    d_fp16 = np.frombuffer(b[0:2], dtype=np.float16).astype(np.float32)[0]
    dmin_fp16 = np.frombuffer(b[2:4], dtype=np.float16).astype(np.float32)[0]
    scales_packed = b[4:16]
    qs = b[16:144]

    sc, mn = unpack_q4k_scales(scales_packed)
    sc = sc.flatten().astype(np.float32)
    mn = mn.flatten().astype(np.float32)

    out = np.zeros(QK_K, dtype=np.float32)

    is_idx = 0
    q_offset = 0
    for pair in range(4):
        d1 = d_fp16 * sc[is_idx]
        m1 = dmin_fp16 * mn[is_idx]
        d2 = d_fp16 * sc[is_idx + 1]
        m2 = dmin_fp16 * mn[is_idx + 1]

        base = pair * 64
        for l in range(32):
            qbyte = qs[q_offset + l]
            out[base + l] = d1 * float(qbyte & 0xF) - m1
            out[base + l + 32] = d2 * float(qbyte >> 4) - m2

        q_offset += 32
        is_idx += 2

    return out


def ref_q4k_dequant(weight_q4k):
    """Dequant entire Q4_K weight matrix.

    Args:
        weight_q4k: numpy uint8 array of shape (N, bytes_per_row) or torch tensor

    Returns:
        torch float32 tensor of shape (N, K)
    """
    if isinstance(weight_q4k, torch.Tensor):
        w = weight_q4k.numpy()
    else:
        w = np.asarray(weight_q4k, dtype=np.uint8)

    N = w.shape[0]
    row_bytes = w.shape[1]
    n_blocks_per_row = row_bytes // BLOCK_Q4K_SIZE
    K = n_blocks_per_row * QK_K

    blocks = w.reshape(N * n_blocks_per_row, BLOCK_Q4K_SIZE)
    n_blocks = blocks.shape[0]

    d = np.frombuffer(blocks[:, 0:2].tobytes(), dtype=np.float16).reshape(n_blocks, 1).astype(np.float32)
    dmin = np.frombuffer(blocks[:, 2:4].tobytes(), dtype=np.float16).reshape(n_blocks, 1).astype(np.float32)
    scales_packed = blocks[:, 4:16]
    qs = blocks[:, 16:144]

    sc, mn = unpack_q4k_scales(scales_packed)
    sc = sc.astype(np.float32)
    mn = mn.astype(np.float32)

    d_sc = (d * sc).reshape(n_blocks, -1, 1)
    dm_mn = (dmin * mn).reshape(n_blocks, -1, 1)

    qs_r = qs.reshape(n_blocks, -1, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    qs_r = (qs_r & np.uint8(0x0F)).reshape(n_blocks, -1, 32).astype(np.float32)

    result = (d_sc * qs_r - dm_mn).reshape(n_blocks, QK_K)
    result = result.reshape(N, K)
    return torch.from_numpy(result)


def ref_q4k_gemm(x, weight_q4k):
    """Q4_K dequant + GEMM: x @ dequant(W).T"""
    w_float = ref_q4k_dequant(weight_q4k)
    return x.float() @ w_float.T


def _quantize_sub_block(values):
    """Quantize 32 float values to Q4_K sub-block."""
    v = np.asarray(values, dtype=np.float32)
    vmin = float(np.min(v))
    vmax = float(np.max(v))

    min_offset = max(0.0, -vmin)
    range_val = vmax + min_offset

    if range_val < 1e-15:
        return 0.0, min_offset, np.zeros(32, dtype=np.uint8)

    scale = range_val / 15.0
    inv_scale = 1.0 / scale
    quants = np.clip(np.round((v + min_offset) * inv_scale), 0, 15).astype(np.uint8)

    return scale, min_offset, quants


def quantize_to_q4k(weight):
    """Quantize float weight matrix to Q4_K format.

    Args:
        weight: torch tensor [N, K] (float32 or bfloat16)

    Returns:
        numpy uint8 array [N, K*9/16]
    """
    w = weight.float().numpy()
    N, K = w.shape
    assert K % QK_K == 0

    n_blocks_per_row = K // QK_K
    bpr = _bytes_per_row(K)
    out = np.zeros((N, bpr), dtype=np.uint8)

    for row in range(N):
        for blk in range(n_blocks_per_row):
            block_vals = w[row, blk * QK_K:(blk + 1) * QK_K]
            block_offset = blk * BLOCK_Q4K_SIZE

            scales = np.zeros(8, dtype=np.float32)
            mins = np.zeros(8, dtype=np.float32)
            L = np.zeros(QK_K, dtype=np.uint8)

            for j in range(N_SUB_BLOCKS):
                sub = block_vals[j * SUB_BLOCK_SIZE:(j + 1) * SUB_BLOCK_SIZE]
                sc, mn, q = _quantize_sub_block(sub)
                scales[j] = sc
                mins[j] = mn
                L[j * SUB_BLOCK_SIZE:(j + 1) * SUB_BLOCK_SIZE] = q

            max_scale = float(np.max(scales))
            max_min = float(np.max(mins))

            d = max_scale / 63.0 if max_scale > 0 else 0.0
            dmin = max_min / 63.0 if max_min > 0 else 0.0

            inv_scale = 63.0 / max_scale if max_scale > 0 else 0.0
            inv_min = 63.0 / max_min if max_min > 0 else 0.0

            sc_q = np.clip(np.round(inv_scale * scales), 0, 63).astype(np.uint8)
            mn_q = np.clip(np.round(inv_min * mins), 0, 63).astype(np.uint8)

            d_fp16 = np.float16(d).astype(np.float32)
            dmin_fp16 = np.float16(dmin).astype(np.float32)

            packed_scales = _pack_q4k_scales(sc_q, mn_q)
            sc_actual, mn_actual = unpack_q4k_scales(packed_scales)
            sc_actual = sc_actual.flatten().astype(np.float32)
            mn_actual = mn_actual.flatten().astype(np.float32)

            for j in range(N_SUB_BLOCKS):
                d_j = d_fp16 * sc_actual[j]
                dm_j = dmin_fp16 * mn_actual[j]
                if d_j == 0:
                    L[j * SUB_BLOCK_SIZE:(j + 1) * SUB_BLOCK_SIZE] = 0
                else:
                    sub = block_vals[j * SUB_BLOCK_SIZE:(j + 1) * SUB_BLOCK_SIZE]
                    L[j * SUB_BLOCK_SIZE:(j + 1) * SUB_BLOCK_SIZE] = np.clip(
                        np.round((sub + dm_j) / d_j), 0, 15
                    ).astype(np.uint8)

            d_bytes = np.float16(d).tobytes()
            dmin_bytes = np.float16(dmin).tobytes()
            out[row, block_offset:block_offset + 2] = np.frombuffer(d_bytes, dtype=np.uint8)
            out[row, block_offset + 2:block_offset + 4] = np.frombuffer(dmin_bytes, dtype=np.uint8)

            out[row, block_offset + 4:block_offset + 16] = packed_scales

            q_dst = block_offset + 16
            for pair in range(4):
                sb0 = pair * 2
                sb1 = pair * 2 + 1
                for l in range(32):
                    lo = L[sb0 * SUB_BLOCK_SIZE + l]
                    hi = L[sb1 * SUB_BLOCK_SIZE + l]
                    out[row, q_dst + l] = (lo & 0x0F) | ((hi & 0x0F) << 4)
                q_dst += 32

    return out


def generate_q4k_weight(N, K, seed=42):
    """Generate random valid Q4_K weight for testing."""
    rng = np.random.RandomState(seed)
    w = rng.randn(N, K).astype(np.float32)
    return quantize_to_q4k(torch.from_numpy(w))


# ── Tests ────────────────────────────────────────────────────────────────────

class TestGemmReference(unittest.TestCase):
    """Reference tests establishing error budgets for GEMM kernels."""

    # ── FP8 ──────────────────────────────────────────────────────────────

    def test_fp8_quant_roundtrip(self):
        """Dynamic FP8 quantization roundtrip preserves cosine > 0.999."""
        M, K = 32, 1024
        inp = torch.randn(M, K)
        quant, scales = ref_dynamic_fp8_quant(inp)
        dequant = ref_fp8_dequant(quant, scales, M, K, block_size=128)
        cosine = F.cosine_similarity(inp.flatten(), dequant.flatten(), dim=0)
        self.assertGreater(cosine.item(), 0.999,
                           f"FP8 roundtrip cosine={cosine:.6f}")

    # ── Grouped GEMM ─────────────────────────────────────────────────────

    def test_grouped_gemm_matches_single(self):
        """Grouped GEMM matches per-expert torch.mm."""
        num_experts = 4
        tokens_per = 8
        K, N = 128, 64
        total = num_experts * tokens_per
        activations = torch.randn(total, K)
        weights = [torch.randn(N, K) for _ in range(num_experts)]
        offsets = torch.tensor([i * tokens_per for i in range(num_experts + 1)])
        result = ref_grouped_gemm(activations, weights, offsets)
        for e in range(num_experts):
            s, end = e * tokens_per, (e + 1) * tokens_per
            expected = activations[s:end].float() @ weights[e].float().T
            cosine = F.cosine_similarity(result[s:end].flatten(), expected.flatten(), dim=0)
            self.assertGreater(cosine.item(), 0.9999)

    # ── NVFP4 ────────────────────────────────────────────────────────────

    def test_nvfp4_dequant_known_values(self):
        """Dequant of hand-constructed NVFP4 weight with known expected output."""
        N, K = 4, 32
        num_groups = K // GROUP_SIZE

        # First group all 0s, second group all index 2 (=1.0)
        indices = torch.zeros(N, K, dtype=torch.int64)
        indices[:, GROUP_SIZE:] = 2

        pairs = indices.reshape(N, K // 2, 2)
        weight_uint8 = (pairs[:, :, 0] | (pairs[:, :, 1] << 4)).to(torch.uint8)

        scale_e4m3 = torch.ones(N, num_groups, dtype=torch.float32)
        scale_2 = torch.tensor(2.0, dtype=torch.float32)

        w = ref_nvfp4_dequant(weight_uint8, scale_e4m3, scale_2)

        expected = torch.zeros(N, K, dtype=torch.float32)
        expected[:, GROUP_SIZE:] = 2.0  # FP4[2]=1.0 * 1.0 * 2.0

        diff = (w - expected).abs().max().item()
        self.assertLess(diff, 1e-6, f"Known-value dequant mismatch: max diff {diff}")

    def test_nvfp4_quantize_roundtrip(self):
        """dequant(quantize(W)) vs original: cosine > 0.98."""
        torch.manual_seed(42)
        N, K = 64, 128
        w_orig = torch.randn(N, K, dtype=torch.bfloat16)

        w_uint8, w_scale, w_scale2 = quantize_to_nvfp4(w_orig)
        w_deq = ref_nvfp4_dequant(w_uint8, w_scale, w_scale2)

        m = compute_metrics(w_orig, w_deq)
        self.assertGreater(m["cosine"], 0.98,
                           f"NVFP4 roundtrip cosine {m['cosine']:.6f} < 0.98")

    def test_nvfp4_gemm_matches_manual(self):
        """ref_nvfp4_gemm matches manual dequant + matmul."""
        N, K, M = 64, 128, 16
        w_uint8, w_scale, w_scale2 = generate_nvfp4_weight(N, K, seed=123)
        x = torch.randn(M, K, dtype=torch.float32)

        out_gemm = ref_nvfp4_gemm(x, w_uint8, w_scale, w_scale2)
        w_deq = ref_nvfp4_dequant(w_uint8, w_scale, w_scale2)
        out_manual = x @ w_deq.T

        diff = (out_gemm - out_manual).abs().max().item()
        self.assertLess(diff, 1e-6, f"GEMM vs manual mismatch: {diff}")

    def test_nvfp4_quantize_table_values(self):
        """Each FP4 E2M1 table value quantizes back to itself."""
        for idx in range(16):
            val = FP4_E2M1_TABLE[idx].item()
            if val == 0.0:
                continue
            # Create a tiny "weight" with just that value
            w = torch.full((1, 16), val)
            w_uint8, w_scale, w_scale2 = quantize_to_nvfp4(w)
            w_deq = ref_nvfp4_dequant(w_uint8, w_scale, w_scale2)
            # All dequantized values should match original closely
            rel_err = (w_deq - val).abs().max().item() / (abs(val) + 1e-12)
            self.assertLess(rel_err, 0.01,
                            f"FP4 idx {idx} val={val}: rel_err={rel_err:.4f}")

    # ── Q4_K ─────────────────────────────────────────────────────────────

    def test_q4k_unpack_scales_known(self):
        """Hand-crafted 12-byte scales, verify extraction."""
        sc = np.array([1, 2, 3, 4, 5, 10, 15, 20], dtype=np.uint8)
        mn = np.array([2, 4, 6, 8, 10, 20, 30, 40], dtype=np.uint8)

        packed = _pack_q4k_scales(sc, mn)
        sc_out, mn_out = unpack_q4k_scales(packed)
        sc_out = sc_out.flatten()
        mn_out = mn_out.flatten()

        self.assertTrue(np.array_equal(sc_out, sc), f"Scales mismatch")
        self.assertTrue(np.array_equal(mn_out, mn), f"Mins mismatch")

    def test_q4k_dequant_known_values(self):
        """Hand-crafted Q4_K block with known dequant output."""
        block = np.zeros(BLOCK_Q4K_SIZE, dtype=np.uint8)

        # d = 1.0, dmin = 0.5
        block[0:2] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
        block[2:4] = np.frombuffer(np.float16(0.5).tobytes(), dtype=np.uint8)

        # All scales=1, all mins=1
        sc = np.ones(8, dtype=np.uint8)
        mn = np.ones(8, dtype=np.uint8)
        block[4:16] = _pack_q4k_scales(sc, mn)

        # All quants = 7: byte = 0x77
        block[16:144] = 0x77

        out = dequant_q4k_block(block)
        expected = 1.0 * 1.0 * 7.0 - 0.5 * 1.0  # = 6.5
        self.assertTrue(np.allclose(out, expected, atol=1e-3),
                        f"Expected all {expected}, got range [{out.min()}, {out.max()}]")

    def test_q4k_pack_roundtrip(self):
        """quantize_to_q4k() then ref_q4k_dequant() vs original — cosine > 0.995."""
        torch.manual_seed(42)
        N, K = 16, 256
        w = torch.randn(N, K)
        w_q4k = quantize_to_q4k(w)
        w_deq = ref_q4k_dequant(w_q4k)

        self.assertEqual(w_deq.shape, (N, K))
        self.assertFalse(torch.isnan(w_deq).any())

        m = compute_metrics(w, w_deq)
        self.assertGreater(m["cosine"], 0.995,
                           f"Q4K roundtrip cosine {m['cosine']:.6f} < 0.995")

    def test_q4k_gemm_shapes(self):
        """ref_q4k_gemm produces correct shapes and cosine > 0.995 vs BF16."""
        N, K = 32, 256
        torch.manual_seed(42)
        w_bf16 = torch.randn(N, K)
        w_q4k = quantize_to_q4k(w_bf16)

        for M in [1, 16]:
            x = torch.randn(M, K, dtype=torch.bfloat16)
            out_q4k = ref_q4k_gemm(x, w_q4k)
            out_bf16 = x.float() @ w_bf16.float().T

            self.assertEqual(out_q4k.shape, (M, N))
            m = compute_metrics(out_bf16, out_q4k)
            self.assertGreater(m["cosine"], 0.995,
                               f"M={M} GEMM cosine {m['cosine']:.6f} < 0.995")


if __name__ == "__main__":
    unittest.main()
