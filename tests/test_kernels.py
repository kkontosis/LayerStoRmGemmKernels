"""
GPU kernel tests for SM120 GEMM kernels.
Requires: SM120 GPU + sm120_gemm_kernels built.

Test patterns borrowed from sibling projects:
  - LayerStoRmKernels/tests/test_nvfp4_gemm_kernels.py (NVFP4 cosine validation)
  - LayerStoRmKernels/tests/test_q4k_gemm_kernels.py (Q4K cosine validation)
  - LayerStoRmKernels/tests/test_q4k_cutlass_gemm_kernels.py (Q4K TC cross-compare)
  - LayerStoRmExpertKernels/tests/test_kernels.py (FP8/NVFP4 grouped patterns)

Run: CUDA_VISIBLE_DEVICES=0 pytest tests/test_kernels.py -v
"""

import sys
import os
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_reference import (
    ref_dynamic_fp8_quant, ref_fp8_dequant, ref_grouped_gemm,
    ref_nvfp4_dequant, ref_nvfp4_gemm, generate_nvfp4_weight, quantize_to_nvfp4,
    ref_q4k_dequant, ref_q4k_gemm, generate_q4k_weight, quantize_to_q4k,
    compute_metrics,
)

import sm120_gemm_kernels as GK


def has_cuda():
    return torch.cuda.is_available()

def has_sm120():
    if not has_cuda():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] > 12 or (cap[0] == 12 and cap[1] >= 0)


# ── NVFP4 helpers (from LayerStoRmExpertKernels/tests/test_kernels.py) ───────

_E2M1_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_UE4M3_ONE = 0x38


def _quantize_e2m1(val: float) -> int:
    sign = 1 if val < 0 else 0
    abs_val = abs(val)
    best = min(range(8), key=lambda i: abs(abs_val - _E2M1_MAGNITUDES[i]))
    return (sign << 3) | best


def _decode_e2m1(nibble: int) -> float:
    sign = -1.0 if (nibble >> 3) & 1 else 1.0
    mag = _E2M1_MAGNITUDES[nibble & 0x7]
    return sign * mag


def _pack_fp4(data_float: torch.Tensor) -> torch.Tensor:
    rows, cols = data_float.shape
    assert cols % 2 == 0
    packed = torch.zeros(rows, cols // 2, dtype=torch.uint8)
    for r in range(rows):
        for c in range(0, cols, 2):
            n0 = _quantize_e2m1(data_float[r, c].item())
            n1 = _quantize_e2m1(data_float[r, c + 1].item())
            packed[r, c // 2] = (n1 << 4) | (n0 & 0xF)
    return packed


def _unpack_fp4(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    out = torch.zeros(rows, cols, dtype=torch.float32)
    for r in range(rows):
        for c in range(0, cols, 2):
            byte = packed[r, c // 2].item()
            out[r, c] = _decode_e2m1(byte & 0xF)
            out[r, c + 1] = _decode_e2m1((byte >> 4) & 0xF)
    return out


def _round_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def _make_nvfp4_scales(dim: int, K: int, group_size: int = 16) -> torch.Tensor:
    sf_dim = _round_up(dim, 128)
    sf_k = _round_up(K // group_size, 4)
    return torch.full((sf_dim, sf_k), _UE4M3_ONE, dtype=torch.uint8)


requires_gpu = unittest.skipUnless(has_cuda(), "Requires CUDA GPU")
requires_sm120 = unittest.skipUnless(has_sm120(), "Requires SM120 GPU")


# ── Sm1xx interleaved scale decode (mirrors grouped_sm1xx_sf_offset) ─────────

def _sm1xx_sf_offset(row: int, g: int, padded_groups: int) -> int:
    tile = (row // 128) * ((padded_groups + 3) // 4) + g // 4
    within = (row % 32) * 16 + ((row % 128) // 32) * 4 + (g % 4)
    return tile * 512 + within


def _decode_grouped_act_scales(scales_u8: torch.Tensor,
                               sf_offsets: torch.Tensor,
                               expert_offsets: torch.Tensor,
                               groups: int) -> torch.Tensor:
    """Decode bf16_to_nvfp4_grouped's interleaved UE4M3 scales to [T, groups]
    float32 (per-row, per-group scale_repr values)."""
    offs = expert_offsets.cpu().tolist()
    sf = sf_offsets.cpu().tolist()
    total = offs[-1]
    buf = scales_u8.cpu()
    out = torch.zeros(total, groups, dtype=torch.float32)
    padded_groups = _round_up(groups, 4)
    for e in range(len(offs) - 1):
        base = sf[e] * groups
        for row in range(offs[e], offs[e + 1]):
            local = row - offs[e]
            for g in range(groups):
                byte = buf[base + _sm1xx_sf_offset(local, g, padded_groups)]
                out[row, g] = byte.view(torch.uint8).reshape(1).view(
                    torch.float8_e4m3fn).float().item()
    return out


def _unpack_fp4_flat(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Vectorized FP4 unpack (low nibble = even col, high nibble = odd col)."""
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
    p = packed.cpu().to(torch.int64)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    out = torch.stack([table[lo], table[hi]], dim=-1).reshape(rows, cols)
    return out


class TestGemmKernels(unittest.TestCase):
    """GPU kernel tests — compare CUDA output against PyTorch reference."""

    # ── Dynamic FP8 Quantization ─────────────────────────────────────────

    @requires_gpu
    def test_dynamic_fp8_quant(self):
        """BF16 -> FP8 -> dequant roundtrip: cosine > 0.999."""
        M, K = 128, 1024
        input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        output, scales = GK.dynamic_fp8_quant(input_bf16)

        self.assertEqual(output.shape, (M, K))
        self.assertEqual(output.dtype, torch.float8_e4m3fn)
        num_blocks = (K + 127) // 128
        self.assertEqual(scales.shape, (M, num_blocks))

        dequant = output.float() * scales.repeat_interleave(128, dim=1)[:, :K]
        cosine = F.cosine_similarity(
            input_bf16.cpu().float().flatten(),
            dequant.cpu().float().flatten(), dim=0)
        self.assertGreater(cosine.item(), 0.999,
                           f"FP8 quant roundtrip cosine={cosine:.6f}")

    # ── FP8 Single GEMM ─────────────────────────────────────────────────

    @requires_sm120
    def test_fp8_gemm(self):
        """FP8 GEMM: random FP8 data, scale=1.0, GPU vs CPU matmul, cosine > 0.99.
        Uses uniform scales to isolate GEMM arithmetic from scale-layout differences
        (CUTLASS uses tile-level [ceil(K/128), ceil(N/128)] B scales, not per-row)."""
        M, N, K = 128, 128, 128
        torch.manual_seed(42)

        # Random data cast to FP8 (values clipped to E4M3 range)
        A_fp8 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        B_fp8 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)

        k_blocks = (K + 127) // 128
        n_blocks = (N + 127) // 128
        scale_A = torch.ones(M, k_blocks, dtype=torch.float32, device="cuda")
        scale_B = torch.ones(k_blocks, n_blocks, dtype=torch.float32, device="cuda")

        D = GK.fp8_gemm(A_fp8, B_fp8, scale_A, scale_B, M, N, K, "bf16")

        # CPU reference: FP8->float matmul (scale=1.0, only accumulation order differs)
        ref = A_fp8.float().cpu() @ B_fp8.float().cpu().T

        m = compute_metrics(ref, D.cpu())
        self.assertGreater(m["cosine"], 0.99,
                           f"FP8 GEMM cosine={m['cosine']:.6f}")

    # ── FP8 Grouped GEMM ────────────────────────────────────────────────

    @requires_sm120
    def test_fp8_grouped_gemm(self):
        """FP8 grouped GEMM: random FP8 data, scale=1.0, per-expert cosine > 0.99."""
        num_experts = 4
        M_per, N, K = 128, 128, 128
        total_tokens = num_experts * M_per
        torch.manual_seed(42)

        offsets = [i * M_per for i in range(num_experts + 1)]
        expert_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")
        problem_sizes = torch.tensor(
            [[M_per, N, K]] * num_experts, dtype=torch.int32, device="cuda")

        # Random FP8 activations with scale=1.0
        A_fp8 = torch.randn(total_tokens, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        k_blocks = (K + 127) // 128
        n_blocks = (N + 127) // 128
        scale_A = torch.ones(total_tokens, k_blocks, dtype=torch.float32, device="cuda")

        # Per-expert random FP8 weights with scale=1.0
        B_fp8_parts = []
        for e in range(num_experts):
            B_fp8_parts.append(
                torch.randn(N, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn))
        B_fp8_all = torch.stack(B_fp8_parts)
        scale_B_all = torch.ones(num_experts, k_blocks, n_blocks,
                                  dtype=torch.float32, device="cuda")

        D_base = torch.zeros(total_tokens, N, dtype=torch.bfloat16, device="cuda")

        GK.fp8_grouped_gemm(
            A_fp8, B_fp8_all, D_base,
            scale_A, scale_B_all,
            expert_offsets, problem_sizes, N, K, "bf16")

        # Per-expert CPU reference: FP8->float matmul
        D_cpu = D_base.cpu().float()
        A_cpu = A_fp8.float().cpu()
        for e in range(num_experts):
            s, end = offsets[e], offsets[e + 1]
            ref_e = A_cpu[s:end] @ B_fp8_parts[e].float().cpu().T
            m = compute_metrics(ref_e, D_cpu[s:end])
            self.assertGreater(m["cosine"], 0.99,
                               f"FP8 grouped expert {e} cosine={m['cosine']:.6f}")

    # ── NVFP4 Single GEMM ───────────────────────────────────────────────

    @requires_sm120
    def test_nvfp4_gemm(self):
        """NVFP4 low-level GEMM: GPU kernel vs CPU dequant+matmul, cosine > 0.99.
        Uses pre-quantized FP4 data with uniform scale=1.0."""
        M, N, K = 128, 128, 128
        torch.manual_seed(42)

        A_float = torch.randn(M, K).clamp(-5.0, 5.0)
        B_float = torch.randn(N, K).clamp(-5.0, 5.0)
        A_packed = _pack_fp4(A_float).cuda()
        B_packed = _pack_fp4(B_float).cuda()

        scale_A = _make_nvfp4_scales(M, K).cuda()
        scale_B = _make_nvfp4_scales(N, K).cuda()

        D = GK.nvfp4_gemm(A_packed, B_packed, scale_A, scale_B,
                           M, N, K, 1.0, "bf16")

        # CPU reference: decode FP4 -> float, matmul
        A_deq = _unpack_fp4(A_packed.cpu(), M, K)
        B_deq = _unpack_fp4(B_packed.cpu(), N, K)
        ref = A_deq @ B_deq.T

        m = compute_metrics(ref, D.cpu())
        self.assertGreater(m["cosine"], 0.99,
                           f"NVFP4 GEMM cosine={m['cosine']:.6f}")

    # ── NVFP4 Grouped GEMM ──────────────────────────────────────────────

    @requires_sm120
    def test_nvfp4_grouped_gemm(self):
        """NVFP4 grouped GEMM: uniform data, cosine > 0.99 vs expected."""
        num_experts = 4
        M_per, N, K = 128, 128, 128
        total_M = num_experts * M_per
        group_size = 16

        # Uniform 0.5 data: nibble 0x01, packed pair = 0x11
        half_K = K // 2
        A_packed = torch.full((total_M, half_K), 0x11,
                               dtype=torch.uint8, device="cuda")
        B_packed = torch.full((num_experts * N, half_K), 0x11,
                               dtype=torch.uint8, device="cuda")

        sf_M = _round_up(total_M, 128)
        sf_K = _round_up(K // group_size, 4)
        sf_N = _round_up(N, 128)
        scale_A = torch.full((sf_M, sf_K), _UE4M3_ONE,
                              dtype=torch.uint8, device="cuda")
        scale_B = torch.full((num_experts * sf_N, sf_K), _UE4M3_ONE,
                              dtype=torch.uint8, device="cuda")

        sf_rows_per_expert = _round_up(M_per, 128)
        expert_offsets = torch.tensor(
            [i * M_per for i in range(num_experts + 1)],
            dtype=torch.int32, device="cuda")
        sf_offsets = torch.tensor(
            [i * sf_rows_per_expert for i in range(num_experts + 1)],
            dtype=torch.int32, device="cuda")
        problem_sizes = torch.tensor(
            [[M_per, N, K]] * num_experts, dtype=torch.int32, device="cuda")
        alphas = torch.ones(num_experts, dtype=torch.float32, device="cuda")

        D_base = torch.zeros(total_M, N, dtype=torch.bfloat16, device="cuda")

        GK.nvfp4_grouped_gemm(
            A_packed, B_packed, D_base,
            scale_A, scale_B, alphas,
            expert_offsets, sf_offsets, problem_sizes, N, K, "bf16")

        # Reference: K * 0.5 * 0.5 * 1.0 = 32.0
        expected = torch.full((total_M, N), K * 0.25, dtype=torch.float32)
        m = compute_metrics(expected, D_base.cpu())
        self.assertGreater(m["cosine"], 0.99,
                           f"NVFP4 grouped GEMM cosine={m['cosine']:.6f}")

    # ── NVFP4 GEMM from BF16 ────────────────────────────────────────────

    @requires_sm120
    def test_nvfp4_gemm_from_bf16(self):
        """NVFP4 end-to-end BF16 path vs BF16 ground truth: cosine > 0.99."""
        M, N, K = 128, 128, 128
        torch.manual_seed(42)

        x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w_bf16 = torch.randn(N, K, dtype=torch.bfloat16)
        w_uint8, w_scale, w_scale2 = quantize_to_nvfp4(w_bf16)

        D = GK.nvfp4_gemm_from_bf16(
            x_bf16, w_uint8.cuda(), w_scale.cuda(), w_scale2.item())

        self.assertEqual(D.shape, (M, N))
        self.assertEqual(D.dtype, torch.bfloat16)

        # Compare vs NVFP4 reference (same quantized weights, no weight quant noise)
        ref_nvfp4 = ref_nvfp4_gemm(x_bf16.cpu(), w_uint8, w_scale, w_scale2)
        m = compute_metrics(ref_nvfp4, D.cpu())
        self.assertGreater(m["cosine"], 0.99,
                           f"nvfp4_gemm_from_bf16 vs ref cosine={m['cosine']:.6f}")

    # ── Random-scale pairing tests (TD-GOLDEN-GK-SCALE-TESTS) ───────────
    #
    # The four TD-GOLDEN kernel bugs (scale_B N-major, pad-M scale_A stride,
    # UE4M3 underflow, Sm1xx SFB interleave) plus the scale_A row-major-vs-
    # M-major mismatch all survived the uniform-scale tests above. These
    # pairing tests use RANDOM per-group/per-tile scales and realistic
    # magnitudes so layout and underflow bugs change the answer.

    @requires_sm120
    def test_weight_fp8_quant_fp8_gemm_pairing(self):
        """weight_fp8_quant → fp8_gemm with per-tile-varying weight magnitudes
        and per-row-varying activations: cosine > 0.99 vs BF16 reference.
        Catches: scale_B K-major layout, scale_A M-major (SFA) layout."""
        M, N, K = 128, 576, 1024  # N=576 ragged (kv_a case), 8 K-blocks
        torch.manual_seed(7)

        # Per-128x128-tile magnitude variation → distinct tile scales.
        w = torch.randn(N, K, dtype=torch.float32)
        n_blocks, k_blocks = (N + 127) // 128, (K + 127) // 128
        tile_mag = 2.0 ** torch.randint(-3, 4, (n_blocks, k_blocks)).float()
        w = w * tile_mag.repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]
        w_bf16 = w.to(torch.bfloat16).cuda()

        # Per-row magnitude variation → distinct per-row-block act scales.
        x = torch.randn(M, K, dtype=torch.float32)
        row_mag = 2.0 ** torch.randint(-3, 4, (M, 1)).float()
        x_bf16 = (x * row_mag).to(torch.bfloat16).cuda()

        B_fp8, scale_B = GK.weight_fp8_quant(w_bf16)
        A_fp8, scale_A = GK.dynamic_fp8_quant(x_bf16, m_major_scales=True)

        D = GK.fp8_gemm(A_fp8, B_fp8, scale_A, scale_B, M, N, K, "bf16")

        ref = x_bf16.cpu().float() @ w_bf16.cpu().float().T
        m = compute_metrics(ref, D.cpu())
        self.assertGreater(m["cosine"], 0.99,
                           f"fp8 pairing cosine={m['cosine']:.6f}")

    @requires_sm120
    def test_weight_fp8_quant_fp8_gemm_m1_pad(self):
        """M=1 pad path (kv_a decode shape) with multi-K-block random scales:
        catches the pad-M scale_A column-stride bug (zero scales for k_blk>0)."""
        M, N, K = 1, 576, 1024
        torch.manual_seed(11)

        w = torch.randn(N, K, dtype=torch.float32)
        n_blocks, k_blocks = (N + 127) // 128, (K + 127) // 128
        tile_mag = 2.0 ** torch.randint(-2, 3, (n_blocks, k_blocks)).float()
        w = w * tile_mag.repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]
        w_bf16 = w.to(torch.bfloat16).cuda()
        x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        B_fp8, scale_B = GK.weight_fp8_quant(w_bf16)
        A_fp8, scale_A = GK.dynamic_fp8_quant(x_bf16, m_major_scales=True)

        D = GK.fp8_gemm(A_fp8, B_fp8, scale_A, scale_B, M, N, K, "bf16")

        ref = x_bf16.cpu().float() @ w_bf16.cpu().float().T
        m = compute_metrics(ref, D.cpu())
        self.assertGreater(m["cosine"], 0.99,
                           f"fp8 M=1 pad cosine={m['cosine']:.6f}")

    @requires_sm120
    def test_fp8_blockscaled_gemv_m1_pairing(self):
        """Phase 2b: M=1 FP8 blockwise GEMV (decode) cross-checked against the
        exact dequant-GEMM math at the real attention-projection decode shapes
        (q_a N=1536 K=7168, q_b N=2048 K=1536, kv_a N=640 K=7168).

        At M=1 GK.fp8_gemm dispatches to the hand-written GEMV. The reference is
        the identical FP8 math the CUTLASS path computes:
          D[0,n] = sum_kb sA[kb]*sB[kb,nb] * sum_k A_f8[k]*B_f8[n,k].
        RANDOM per-128-block scales with realistic (incl. low ~1e-3) magnitudes
        exercise the M-major scale_A / K-major scale_B layouts, the tail K-block
        (K=1536 -> 12 blocks; K=7168 -> 56 blocks), and ragged/non-128 N (640,
        1536). Tight tolerance: cosine > 0.999, max abs-rel small."""
        shapes = [
            ("q_a", 1536, 7168),
            ("q_b", 2048, 1536),
            ("kv_a", 640, 7168),
        ]
        for name, N, K in shapes:
            with self.subTest(proj=name):
                torch.manual_seed(hash((name, N, K)) & 0xffff)
                M = 1
                k_blocks = (K + 127) // 128
                n_blocks = (N + 127) // 128

                A_fp8 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
                B_fp8 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)

                # Random per-128-block scales: mix of normal (2^[-2..3]) and the
                # low ~1e-3 underflow regime, in the produced buffer layouts.
                def rand_scales(shape):
                    e = torch.randint(-2, 4, shape).float()
                    s = (2.0 ** e)
                    low = torch.rand(shape) < 0.25
                    s[low] = (torch.rand(shape)[low] * 9e-4 + 1e-4)  # ~1e-4..1e-3
                    return s
                # scale_A: M-major [k_blocks, M] (M=1 -> one per K-block).
                # scale_B: K-major; the produced buffer (weight_fp8_quant) is a
                # [n_blocks, k_blocks] row-major tensor == scales[k_blk + n_blk*k_blocks].
                scale_A = rand_scales((k_blocks, M)).to(torch.float32).cuda().contiguous()
                scale_B = rand_scales((n_blocks, k_blocks)).to(torch.float32).cuda().contiguous()

                D = GK.fp8_gemm(A_fp8, B_fp8, scale_A, scale_B, M, N, K, "bf16")

                # Exact dequant-GEMM reference (block scale is a per-128 factor).
                Af = A_fp8.float().cpu()          # [1, K]
                Bf = B_fp8.float().cpu()          # [N, K]
                sA = scale_A.cpu().view(k_blocks)  # M=1
                sB = scale_B.cpu()                 # [n_blocks, k_blocks]
                # Expand block scales to per-element along K, and per-N for B.
                sA_k = sA.repeat_interleave(128)[:K]            # [K]
                ref = torch.zeros(M, N, dtype=torch.float32)
                for nb in range(n_blocks):
                    n0, n1 = nb * 128, min((nb + 1) * 128, N)
                    sB_k = sB[nb, :].repeat_interleave(128)[:K]  # [K]
                    Bn = Bf[n0:n1, :] * sB_k.unsqueeze(0)        # [n_rows, K]
                    An = Af * sA_k.unsqueeze(0)                  # [1, K]
                    ref[:, n0:n1] = An @ Bn.T

                m = compute_metrics(ref, D.cpu())
                self.assertGreater(m["cosine"], 0.999,
                                   f"{name} GEMV cosine={m['cosine']:.6f}")
                # Tight relative agreement (FP8 round to BF16 output).
                denom = ref.abs().clamp_min(1e-3)
                max_rel = ((D.cpu().float() - ref).abs() / denom).max().item()
                self.assertLess(max_rel, 0.05,
                                f"{name} GEMV max abs-rel={max_rel:.4f}")

    @requires_sm120
    def test_bf16_to_nvfp4_grouped_roundtrip(self):
        """Quantizer roundtrip with ragged experts (incl. empty): dequantized
        activations match the BF16 input, cosine > 0.99 per non-empty expert."""
        K = 256
        groups = K // 16
        offs = [0, 3, 3, 131, 148]  # E=4, ragged, expert 1 empty
        total = offs[-1]
        torch.manual_seed(21)
        x = (torch.randn(total, K) * 0.5).to(torch.bfloat16).cuda()
        expert_offsets = torch.tensor(offs, dtype=torch.int32, device="cuda")

        packed, scales, sf_offsets = GK.bf16_to_nvfp4_grouped(x, expert_offsets)

        fp4 = _unpack_fp4_flat(packed, total, K)
        sc = _decode_grouped_act_scales(scales, sf_offsets, expert_offsets, groups)
        x_hat = fp4 * sc.repeat_interleave(16, dim=1)
        for e in range(4):
            s, end = offs[e], offs[e + 1]
            if s == end:
                continue
            cos = F.cosine_similarity(x.cpu().float()[s:end].flatten(),
                                      x_hat[s:end].flatten(), dim=0)
            self.assertGreater(cos.item(), 0.99,
                               f"expert {e} roundtrip cosine={cos:.6f}")

    @requires_sm120
    def test_bf16_to_nvfp4_grouped_input_scales_tiny(self):
        """Tiny activations (group amax ~1e-3, below the 2^-9 UE4M3 floor):
        calibrated input_scales recover roundtrip accuracy that pure dynamic
        scaling loses to the clamp."""
        K = 256
        groups = K // 16
        offs = [0, 64, 128]
        total = offs[-1]
        torch.manual_seed(33)
        x = (torch.randn(total, K) * 1e-3).to(torch.bfloat16).cuda()
        expert_offsets = torch.tensor(offs, dtype=torch.int32, device="cuda")

        def roundtrip(input_scales):
            packed, scales, sf_offsets = GK.bf16_to_nvfp4_grouped(
                x, expert_offsets, input_scales)
            fp4 = _unpack_fp4_flat(packed, total, K)
            sc = _decode_grouped_act_scales(scales, sf_offsets,
                                            expert_offsets, groups)
            x_hat = fp4 * sc.repeat_interleave(16, dim=1)
            if input_scales is not None:
                is_cpu = input_scales.cpu()
                for e in range(len(offs) - 1):
                    x_hat[offs[e]:offs[e + 1]] *= is_cpu[e].item()
            return F.cosine_similarity(x.cpu().float().flatten(),
                                       x_hat.flatten(), dim=0).item()

        cos_without = roundtrip(None)

        # Realistic calibration: is = amax_tensor / (448 * 6) per expert.
        is_vals = []
        for e in range(len(offs) - 1):
            amax = x.cpu().float()[offs[e]:offs[e + 1]].abs().max().item()
            is_vals.append(amax / (448.0 * 6.0))
        input_scales = torch.tensor(is_vals, dtype=torch.float32, device="cuda")
        cos_with = roundtrip(input_scales)

        self.assertGreater(cos_with, 0.99,
                           f"with input_scales cosine={cos_with:.6f}")
        self.assertGreater(cos_with, cos_without,
                           f"input_scales must improve tiny-magnitude accuracy "
                           f"(with={cos_with:.6f} <= without={cos_without:.6f})")

    @requires_sm120
    def test_bf16_to_nvfp4_grouped_gemm_pairing(self):
        """bf16_to_nvfp4_grouped → nvfp4_grouped_gemm with random per-group
        weight scales and ragged experts (incl. empty): per-expert cosine > 0.99
        vs BF16-activation × dequantized-weight reference.
        Catches: Sm1xx SFB interleave (raw layout would scramble), alpha wiring."""
        N, K = 128, 256
        offs = [0, 5, 5, 133, 163]
        total = offs[-1]
        num_experts = 4
        torch.manual_seed(42)

        x = torch.randn(total, K, dtype=torch.bfloat16, device="cuda")
        expert_offsets = torch.tensor(offs, dtype=torch.int32, device="cuda")

        # Per-expert weights with varying magnitudes → random per-group scales.
        w_packed, w_scale_buf, w_deq, alphas = [], [], [], []
        for e in range(num_experts):
            w = torch.randn(N, K) * (0.05 if e % 2 else 1.0)
            wu8, w_sc, w_s2 = quantize_to_nvfp4(w)
            w_packed.append(wu8.cuda())
            w_scale_buf.append(GK.nvfp4_gemm_preprocess(w_sc.cuda(), N, K))
            w_deq.append(ref_nvfp4_dequant(wu8, w_sc, w_s2))
            alphas.append(w_s2.item())

        A_packed, A_scales, sf_offsets = GK.bf16_to_nvfp4_grouped(
            x, expert_offsets)

        B_packed = torch.cat(w_packed, dim=0)
        scale_B = torch.cat(w_scale_buf, dim=0)
        alphas_t = torch.tensor(alphas, dtype=torch.float32, device="cuda")
        problem_sizes = torch.tensor(
            [[offs[e + 1] - offs[e], N, K] for e in range(num_experts)],
            dtype=torch.int32, device="cuda")
        D_base = torch.zeros(total, N, dtype=torch.bfloat16, device="cuda")

        GK.nvfp4_grouped_gemm(
            A_packed, B_packed, D_base,
            A_scales, scale_B, alphas_t,
            expert_offsets, sf_offsets, problem_sizes, N, K, "bf16")

        D_cpu = D_base.cpu().float()
        x_cpu = x.cpu().float()
        for e in range(num_experts):
            s, end = offs[e], offs[e + 1]
            if s == end:
                continue
            ref_e = x_cpu[s:end] @ w_deq[e].T
            m = compute_metrics(ref_e, D_cpu[s:end])
            self.assertGreater(m["cosine"], 0.99,
                               f"nvfp4 pairing expert {e} cosine={m['cosine']:.6f}")

    @requires_sm120
    def test_bf16_to_nvfp4_grouped_gemm_input_scales(self):
        """End-to-end TRT-LLM scheme on tiny activations: quantize with
        per-expert input_scales, GEMM with alpha = ws2 * is. Accuracy must
        beat the no-input_scales run and clear 0.99."""
        N, K = 128, 256
        offs = [0, 64, 128]
        total = offs[-1]
        num_experts = 2
        torch.manual_seed(55)

        x = (torch.randn(total, K) * 1e-3).to(torch.bfloat16).cuda()
        expert_offsets = torch.tensor(offs, dtype=torch.int32, device="cuda")

        w_packed, w_scale_buf, w_deq, ws2 = [], [], [], []
        for e in range(num_experts):
            w = torch.randn(N, K)
            wu8, w_sc, w_s2 = quantize_to_nvfp4(w)
            w_packed.append(wu8.cuda())
            w_scale_buf.append(GK.nvfp4_gemm_preprocess(w_sc.cuda(), N, K))
            w_deq.append(ref_nvfp4_dequant(wu8, w_sc, w_s2))
            ws2.append(w_s2.item())

        B_packed = torch.cat(w_packed, dim=0)
        scale_B = torch.cat(w_scale_buf, dim=0)
        problem_sizes = torch.tensor(
            [[offs[e + 1] - offs[e], N, K] for e in range(num_experts)],
            dtype=torch.int32, device="cuda")

        def run(input_scales):
            A_packed, A_scales, sf_offsets = GK.bf16_to_nvfp4_grouped(
                x, expert_offsets, input_scales)
            is_cpu = (input_scales.cpu() if input_scales is not None
                      else torch.ones(num_experts))
            alphas = torch.tensor(
                [ws2[e] * is_cpu[e].item() for e in range(num_experts)],
                dtype=torch.float32, device="cuda")
            D = torch.zeros(total, N, dtype=torch.bfloat16, device="cuda")
            GK.nvfp4_grouped_gemm(
                A_packed, B_packed, D,
                A_scales, scale_B, alphas,
                expert_offsets, sf_offsets, problem_sizes, N, K, "bf16")
            cos = []
            for e in range(num_experts):
                s, end = offs[e], offs[e + 1]
                ref_e = x.cpu().float()[s:end] @ w_deq[e].T
                cos.append(compute_metrics(ref_e, D.cpu().float()[s:end])["cosine"])
            return min(cos)

        cos_without = run(None)
        is_vals = [x.cpu().float()[offs[e]:offs[e + 1]].abs().max().item()
                   / (448.0 * 6.0) for e in range(num_experts)]
        cos_with = run(torch.tensor(is_vals, dtype=torch.float32, device="cuda"))

        self.assertGreater(cos_with, 0.99,
                           f"with input_scales min cosine={cos_with:.6f}")
        self.assertGreater(cos_with, cos_without,
                           f"input_scales must improve tiny-magnitude GEMM "
                           f"(with={cos_with:.6f} <= without={cos_without:.6f})")

    # ── Fused SiLU-mul → NVFP4 grouped (byte-identity to unfused path) ───

    @staticmethod
    def _ref_swiglu_bf16(gate, up):
        """Reproduce the unfused SwiGLU kernel exactly: silu(x) =
        x*(0.5*tanh(0.5*x)+0.5) in fp32, multiply by up, round to BF16
        (LayerStoRmExpertKernels fused_swiglu.cu vectorized path)."""
        gf = gate.float()
        uf = up.float()
        silu_g = gf * (0.5 * torch.tanh(0.5 * gf) + 0.5)
        return (silu_g * uf).to(torch.bfloat16)

    @requires_sm120
    def test_silu_mul_to_nvfp4_grouped_byte_identical(self):
        """Phase 2b: the fused silu_mul_to_nvfp4_grouped kernel must be
        BYTE-IDENTICAL (packed FP4 bytes AND Sm1xx scale bytes) to the unfused
        reference sequence: r = SwiGLU(gate, up) in BF16, then the existing
        bf16_to_nvfp4_grouped on r. Identity (not cosine) proves the fused
        kernel reuses the same quant math. Covered: E in {1, 4}, ragged M_e
        incl. empty experts, RANDOM per-group magnitudes, realistic + ~1e-3
        magnitudes, with and without input_scales."""
        configs = [
            # (offsets, K, seed, magnitude_scale)
            ([0, 8], 256, 1, 1.0),            # E=1, single expert, realistic
            ([0, 8], 256, 2, 1e-3),           # E=1, tiny magnitude (clamp regime)
            ([0, 5, 5, 133, 163], 256, 3, 1.0),    # E=4 ragged, empty expert 1
            ([0, 5, 5, 133, 163], 320, 4, 1e-3),   # E=4 ragged, tiny, K=320
            ([0, 1, 65, 65, 200], 192, 5, 1.0),    # E=4, M_e=1, empty expert 2
        ]
        for offs, K, seed, mag in configs:
            num_experts = len(offs) - 1
            total = offs[-1]
            torch.manual_seed(seed)
            # Random per-row/per-group magnitudes: scale each row by a random
            # factor so per-16-group amax varies widely (hits clamp + non-clamp).
            row_scale = (torch.rand(total, 1) * 4.0 + 0.05) * mag
            gate = (torch.randn(total, K) * row_scale).to(torch.bfloat16).cuda()
            up = (torch.randn(total, K) * row_scale).to(torch.bfloat16).cuda()
            expert_offsets = torch.tensor(offs, dtype=torch.int32, device="cuda")

            for use_is in (False, True):
                if use_is:
                    # Per-expert calibrated input_scales (TRT-LLM); tiny floors
                    # would underflow without it, so this exercises the is path.
                    r_ref = self._ref_swiglu_bf16(gate, up)
                    is_vals = []
                    for e in range(num_experts):
                        s, end = offs[e], offs[e + 1]
                        if s == end:
                            is_vals.append(1.0)
                            continue
                        amax = r_ref.float()[s:end].abs().max().item()
                        is_vals.append(max(amax / (448.0 * 6.0), 1e-9))
                    input_scales = torch.tensor(
                        is_vals, dtype=torch.float32, device="cuda")
                else:
                    input_scales = None

                # Reference: unfused SwiGLU (BF16) → existing grouped quantizer.
                r = self._ref_swiglu_bf16(gate, up).contiguous()
                ref_packed, ref_scales, ref_sf = GK.bf16_to_nvfp4_grouped(
                    r, expert_offsets, input_scales)

                # Fused: read gate + up separately, silu-mul + quant in one go.
                fused_packed, fused_scales, fused_sf = \
                    GK.silu_mul_to_nvfp4_grouped(
                        gate, up, expert_offsets, input_scales)

                self.assertTrue(torch.equal(fused_sf, ref_sf),
                    f"sf_offsets differ (offs={offs}, K={K}, is={use_is})")
                self.assertTrue(torch.equal(fused_packed, ref_packed),
                    f"packed FP4 bytes differ (offs={offs}, K={K}, "
                    f"is={use_is}): "
                    f"{(fused_packed != ref_packed).sum().item()} mismatches")
                self.assertTrue(torch.equal(fused_scales, ref_scales),
                    f"Sm1xx scale bytes differ (offs={offs}, K={K}, "
                    f"is={use_is}): "
                    f"{(fused_scales != ref_scales).sum().item()} mismatches")

    # ── Q4K Dequant GEMM ────────────────────────────────────────────────

    @requires_sm120
    def test_q4k_dequant_gemm_gemv(self):
        """Q4K GEMV (M=1) vs Python reference: cosine > 0.99."""
        M, N, K = 1, 64, 256
        w_q4k = generate_q4k_weight(N, K)
        w_q4k_gpu = torch.from_numpy(w_q4k).cuda()

        torch.manual_seed(55)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        out_gpu = GK.q4k_dequant_gemm(x, w_q4k_gpu)
        out_ref = ref_q4k_gemm(x.cpu(), w_q4k)

        m = compute_metrics(out_ref, out_gpu.cpu())
        self.assertGreater(m["cosine"], 0.99,
                           f"Q4K GEMV M=1 cosine={m['cosine']:.6f}")

    @requires_sm120
    def test_q4k_dequant_gemm_tiled(self):
        """Q4K tiled GEMM (M=16) vs Python reference: cosine > 0.99."""
        M, N, K = 16, 64, 256
        w_q4k = generate_q4k_weight(N, K)
        w_q4k_gpu = torch.from_numpy(w_q4k).cuda()

        torch.manual_seed(55)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        out_gpu = GK.q4k_dequant_gemm(x, w_q4k_gpu)
        out_ref = ref_q4k_gemm(x.cpu(), w_q4k)

        m = compute_metrics(out_ref, out_gpu.cpu())
        self.assertGreater(m["cosine"], 0.99,
                           f"Q4K tiled M=16 cosine={m['cosine']:.6f}")

    # ── Q4K Tensor-Core GEMM ────────────────────────────────────────────

    @requires_sm120
    def test_q4k_cutlass_gemm(self):
        """Q4K wmma GEMM (M=16) vs Python reference: cosine > 0.99."""
        M, N, K = 16, 64, 256
        w_q4k = generate_q4k_weight(N, K)
        w_q4k_gpu = torch.from_numpy(w_q4k).cuda()

        torch.manual_seed(55)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        out_tc = GK.q4k_cutlass_gemm(x, w_q4k_gpu)
        out_ref = ref_q4k_gemm(x.cpu(), w_q4k)

        m = compute_metrics(out_ref, out_tc.cpu())
        self.assertGreater(m["cosine"], 0.99,
                           f"Q4K TC M=16 cosine={m['cosine']:.6f}")

    @requires_sm120
    def test_q4k_cutlass_vs_dequant(self):
        """Q4K TC vs CUDA-core: cosine > 0.999."""
        M, N, K = 16, 64, 256
        w_q4k = generate_q4k_weight(N, K)
        w_q4k_gpu = torch.from_numpy(w_q4k).cuda()

        torch.manual_seed(55)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        out_tc = GK.q4k_cutlass_gemm(x, w_q4k_gpu)
        out_p2 = GK.q4k_dequant_gemm(x, w_q4k_gpu)

        m = compute_metrics(out_p2.cpu(), out_tc.cpu())
        self.assertGreater(m["cosine"], 0.999,
                           f"Q4K TC vs CUDA-core cosine={m['cosine']:.6f}")

    @requires_sm120
    def test_q4k_cutlass_gemm_m1_fallback(self):
        """Q4K TC M=1 delegates to CUDA-core GEMV: cosine > 0.99."""
        M, N, K = 1, 64, 256
        w_q4k = generate_q4k_weight(N, K)
        w_q4k_gpu = torch.from_numpy(w_q4k).cuda()

        torch.manual_seed(55)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        out_tc = GK.q4k_cutlass_gemm(x, w_q4k_gpu)
        out_ref = ref_q4k_gemm(x.cpu(), w_q4k)

        m = compute_metrics(out_ref, out_tc.cpu())
        self.assertGreater(m["cosine"], 0.99,
                           f"Q4K TC M=1 fallback cosine={m['cosine']:.6f}")


if __name__ == "__main__":
    unittest.main()
