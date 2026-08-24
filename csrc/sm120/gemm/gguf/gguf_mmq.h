#pragma once

//==============================================================================
// GGUF integer mat-mat (mmq), strategy "int" for large M (prefill).
//
// Tiled int8 GEMM: the BF16 activation is quantized to Q8_1, then a 64x64-tiled
// kernel reuses shared-memory tiles of the (centered) int8 weight quants and the
// int8 activation quants across the tile, accumulating via dp4a with per-block
// scaling. This is the dp4a variant of mmq (faithful to ggml's mmq dp4a path);
// the int8 tensor-core (CuTe MMA) variant is a throughput follow-up.
//
// Reuses GgufMmvqParams and the Q8_1 activation workspace (gguf_mmvq.h).
//==============================================================================

#include <cuda_runtime.h>
#include "sm120/gemm/gguf/gguf_mmvq.h"  // GgufMmvqParams, workspace + quantizer

namespace layerstorm::compute {

void launch_gguf_mmq(const GgufMmvqParams& params, void* q8_1_workspace, cudaStream_t stream);

}  // namespace layerstorm::compute
