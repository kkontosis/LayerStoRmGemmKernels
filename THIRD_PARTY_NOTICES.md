# Third-Party Notices — LayerStoRmGemmKernels

LayerStoRmGemmKernels is licensed under the Apache License 2.0 (see
`LICENSE.md`). Portions of this repository are derived from, adapted from, or
reference the third-party projects listed below. Where a section says "see MIT
License text below", the full license text in Appendix A applies together with
that section's copyright line(s).

---

## llama.cpp / ggml

- Upstream: https://github.com/ggerganov/llama.cpp
  (source revision: commit `1191758c`, tag `b9780`)
- License: MIT — Copyright (c) 2023-2026 The ggml authors (full text in
  `licenses/llama.cpp-LICENSE` and Appendix A below)
- What was derived:
  - The GGUF quantized block formats and their dequant/quantize arithmetic —
    `csrc/formats/{q2k,q3k,q4k,q5k,q6k,q8_0,q8_1,mxfp4}_format.h`,
    `csrc/formats/kquant_common.h`, `csrc/formats/gguf_dequant_one.h` — are
    ports of `ggml/src/ggml-common.h`,
    `ggml/src/ggml-cuda/{convert.cu,quantize.cu,common.cuh,vecdotq.cuh}`,
    `ggml/src/ggml-quants.c`, and `ggml/src/ggml-impl.h` (block layouts,
    per-element dequant/quant math, `get_scale_min_k4`,
    `get_int_from_table_16`, `ggml_e8m0_to_fp32_half`, MXFP4 `kvalues_fp4`
    table).
  - The mat-vec (mmvq) structure — `csrc/sm120/gemm/gguf/gguf_mmvq_impl.cuh`
    adapts the thread mapping of `ggml/src/ggml-cuda/mmvq.cu`
    (`mul_mat_vec_q`), and the Q8_1 activation quantization convention follows
    `ggml/src/ggml-cuda/quantize.cu`.
  - The on-device expert/row routing pattern of the grouped integer path —
    `csrc/sm120/gemm/gguf/gguf_grouped_int.{h,cu}` adapts the `mul_mat_id`
    "ids"/tile→expert map pattern of
    `ggml/src/ggml-cuda/{mmid.cu,mmq.cu,mmvq.cu,quantize.cu}`.
  - The mmq int8 tensor-core techniques (k=32 MMA chunking, ldmatrix operand
    loads, cp.async K-pipeline, keeping the scale-rescale off the hot MMA
    path) referenced by `csrc/sm120/gemm/gguf/mmq_mma/` were reimplemented in
    CuTe following `ggml/src/ggml-cuda/{mmq.cuh,mma.cuh,quantize.cu}`
    (technique reference; see the per-file headers).
  - Test reference implementations in Python/numpy of the same ggml dequant
    math (e.g. the MXFP4 type-39 reference in
    `tests/test_gguf_grouped_gemm.py`).
- See the per-file headers for the specific upstream file each was ported or
  adapted from.

## SGLang

- Upstream: https://github.com/sgl-project/sglang
- License: Apache-2.0 — Copyright 2023-2024 SGLang Team
- What was derived: the SM120 blockwise-scaled FP8 GEMM
  (`csrc/sm120/gemm/fp8/fp8_gemm.{h,cu}`, adapted from
  `fp8_blockwise_gemm_kernel.cu`), the FP8 grouped per-expert dispatch
  (`csrc/sm120/gemm/fp8/fp8_grouped_gemm.cu`, adapted from
  `csrc/moe/fp8_blockwise_moe_kernel.cu`), and the NVFP4 grouped GEMM
  structure (`csrc/sm120/gemm/nvfp4/nvfp4_grouped_gemm.cu` and
  `csrc/sm120/gemm/grouped_gemm.h`, adapted from
  `jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh`).

## vLLM

- Upstream: https://github.com/vllm-project/vllm
- License: Apache-2.0 — Copyright contributors to the vLLM project
- What was derived: the SM120 NVFP4 GEMM
  (`csrc/sm120/gemm/nvfp4/nvfp4_gemm.{h,cu}`, adapted from
  `nvfp4_scaled_mm_sm120_kernels.cu`) and the dynamic per-block FP8
  activation quantization kernel (`csrc/smxx/quant/dynamic_fp8_quant.cu`,
  adapted from `csrc/quantization/fp8/scaled_fp8_quant_kernel.cu`).

## NVIDIA CUTLASS

- Upstream: https://github.com/NVIDIA/cutlass (consumed as the
  `3rd-party/cutlass` git submodule, v4.4.2; not vendored in this tree)
- License: BSD-3-Clause — Copyright (c) 2017 - 2026 NVIDIA CORPORATION &
  AFFILIATES. All rights reserved. (full text in Appendix B)
- What was derived / is used:
  - The FP8 and NVFP4 GEMM/grouped-GEMM kernel configurations were adapted
    from CUTLASS examples 87a
    (`blackwell_geforce_fp8_bf16_gemm_blockwise`) and 79a
    (`blackwell_geforce_nvfp4_bf16_gemm`) — see
    `csrc/sm120/gemm/{fp8,nvfp4}/` and `csrc/sm120/gemm/grouped_gemm.h`.
  - CUTLASS/CuTe headers are a build dependency of the CUDA kernels in
    `csrc/` (MMA atoms, tensor layouts, blockscaled configs). Binaries built
    from this repository incorporate CUTLASS header code; the BSD-3-Clause
    notice applies to such binaries. The `python/CuTeDSL` directory of
    upstream CUTLASS is under a separate NVIDIA EULA; this project does not
    use it.

## FlashMLA

- Upstream: https://github.com/deepseek-ai/FlashMLA (and the SM120 fork
  https://github.com/IISuperluminaLII/FlashMLA_Windows_Linux_sm120)
- License: MIT — Copyright (c) 2025 DeepSeek (see MIT License text below)
- What was derived: the `CHECK_CUDA` / `FLASH_ASSERT` utility macros in
  `csrc/smxx/utils.h` (adapted from upstream `csrc/utils.h`).

## NVIDIA TensorRT-LLM

- Upstream: https://github.com/NVIDIA/TensorRT-LLM
- License: Apache-2.0 — Copyright (c) 2011-2025 NVIDIA CORPORATION &
  AFFILIATES. All rights reserved.
- Relationship: no TensorRT-LLM code is included. The NVFP4 activation
  quantizers (`csrc/smxx/quant/bf16_to_nvfp4_grouped.*`,
  `silu_mul_to_nvfp4_grouped.cu`) follow TensorRT-LLM's per-tensor
  `input_scale` calibration convention (group scale = amax/(6·is), GEMM alpha
  carries the same is) so that TRT-LLM-calibrated checkpoints load
  unmodified. Listed here for provenance transparency.
- TensorRT-LLM ships no Apache-2.0 NOTICE file at its repository root, so
  there are no NOTICE contents to reproduce under Apache-2.0 §4(d).

---

## Appendix A — MIT License text

The following license text applies to the MIT-licensed material identified
above (llama.cpp/ggml, FlashMLA), together with the copyright lines given in
each section:

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Appendix B — BSD-3-Clause (NVIDIA CUTLASS)

```
Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Appendix C — Apache License 2.0

The Apache License 2.0 text is reproduced in this repository as `LICENSE.md`.
It applies both to LayerStoRmGemmKernels itself (Copyright 2026 Kimon
Kontosis) and to the Apache-2.0-licensed upstream material identified above
(SGLang, vLLM, NVIDIA TensorRT-LLM).
