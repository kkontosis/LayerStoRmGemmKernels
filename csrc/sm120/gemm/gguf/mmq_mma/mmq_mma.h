#pragma once

//==============================================================================
// FAST GGUF integer mat-mat (mmq) — CuTe int8 tensor-core "v3" variant.
//
// Same math + Q8_1 activation workspace as the dp4a `gguf_mmq.cu` and the v2
// CuTe `gguf_mmq_cute.cu`, but the per-tile compute uses the *m16n8k32* int8
// MMA atom (SM80_16x8x32_S32S8S8S32_TN). For the 32-subblock-aligned types
// (Q4_K/Q5_K/Q8_0) each k=32 chunk carries exactly one (d_x, A, B, sumqx), so
// the affine rescale runs once per 32-K chunk — half the rescale frequency of
// the v2 k=16 kernel, on a 2x-bigger atom.
//
// Distinct entry point from v2 (`launch_gguf_mmq_cute`). Currently wires only
// Q4_K; later agents add Q5_K/Q8_0 (32-aligned) and the 16-subblock types.
//
// Reuses GgufMmvqParams + the Q8_1 activation workspace (gguf_mmvq.h).
//==============================================================================

#include <cuda_runtime.h>
#include "sm120/gemm/gguf/gguf_mmvq.h"   // GgufMmvqParams, workspace + quantizer

namespace layerstorm::compute {

void launch_gguf_mmq_mma(const GgufMmvqParams& params, void* q8_1_workspace,
                         cudaStream_t stream);

}  // namespace layerstorm::compute
