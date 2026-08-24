#pragma once

//==============================================================================
// GGUF integer mat-mat (mmq) — CuTe int8 tensor-core (MMA) variant.
//
// Same math and the same Q8_1 activation workspace as the dp4a `gguf_mmq.cu`,
// but the per-tile compute uses the SM80 int8 tensor-core MMA atom
// (SM80_16x8x16_S32S8S8S32_TN, m16n8k16) instead of dp4a, for prefill
// throughput. 64x64 output tile, 128 threads (4 warps, 2x2 atom layout).
//
// Per 16-K MMA chunk we get an int32 partial P[m,n] = Σ Q[n,k]*qx[m,k] and
// rescale into a persistent float accumulator:
//   Acc[m,n] += d_x[m,chunk] * ( A_j[n,chunk]*P[m,n] + B_j[n,chunk]*sumqx[m,chunk] )
//
// Reuses GgufMmvqParams + the Q8_1 activation workspace (gguf_mmvq.h).
//==============================================================================

#include <cuda_runtime.h>
#include "sm120/gemm/gguf/gguf_mmvq.h"  // GgufMmvqParams, workspace + quantizer

namespace layerstorm::compute {

void launch_gguf_mmq_cute(const GgufMmvqParams& params, void* q8_1_workspace,
                          cudaStream_t stream);

}  // namespace layerstorm::compute
