// Public dispatch for the FAST GGUF mmq int8 tensor-core path (mmq_mma).
//
// Quantizes the BF16 activation to Q8_1 (reusing gguf_quantize_q8_1_cuda) and
// dispatches on GgufType. Currently only Q4_K is wired; later agents add the
// other 32-aligned types (Q5_K/Q8_0) and the 16-subblock types on top of the
// shared scaffolding in mmq_mma_common.cuh.

#include "sm120/gemm/gguf/mmq_mma/mmq_mma.h"
#include "sm120/gemm/gguf/gguf_dequant_gemm.h"   // GgufType, gguf_block_values
#include "formats/q8_1_format.h"

#include <stdexcept>
#include <string>

namespace layerstorm::compute {

// Defined in instances/mmq_q4_k.cu
void gguf_mmq_mma_launch_q4k(const GgufMmvqParams& params,
                             layerstorm::formats::block_q8_1* xq,
                             cudaStream_t stream);

// Defined in instances/mmq_q5_k.cu
void gguf_mmq_mma_launch_q5k(const GgufMmvqParams& params,
                             layerstorm::formats::block_q8_1* xq,
                             cudaStream_t stream);

// Defined in instances/mmq_q8_0.cu
void gguf_mmq_mma_launch_q8_0(const GgufMmvqParams& params,
                              layerstorm::formats::block_q8_1* xq,
                              cudaStream_t stream);

// Defined in instances/mmq_mxfp4.cu
void gguf_mmq_mma_launch_mxfp4(const GgufMmvqParams& params,
                               layerstorm::formats::block_q8_1* xq,
                               cudaStream_t stream);

// Defined in instances/mmq_q6_k.cu (16-subblock k=16 kernel body)
void gguf_mmq_mma_launch_q6k(const GgufMmvqParams& params,
                             layerstorm::formats::block_q8_1* xq,
                             cudaStream_t stream);

// Defined in instances/mmq_q2_k.cu (16-subblock k=16 kernel body)
void gguf_mmq_mma_launch_q2k(const GgufMmvqParams& params,
                             layerstorm::formats::block_q8_1* xq,
                             cudaStream_t stream);

// Defined in instances/mmq_q3_k.cu (16-subblock k=16 kernel body)
void gguf_mmq_mma_launch_q3k(const GgufMmvqParams& params,
                             layerstorm::formats::block_q8_1* xq,
                             cudaStream_t stream);

void launch_gguf_mmq_mma(const GgufMmvqParams& params, void* q8_1_workspace,
                         cudaStream_t stream) {
    const int qk = gguf_block_values(params.type);
    if (params.K % qk != 0) {
        throw std::invalid_argument(
            "gguf_mmq_mma: K must be divisible by " + std::to_string(qk) +
            ", got K=" + std::to_string(params.K));
    }

    auto* xq = reinterpret_cast<layerstorm::formats::block_q8_1*>(q8_1_workspace);
    gguf_quantize_q8_1_cuda(params.A, xq, params.M, params.K, stream);

    switch (params.type) {
        case GgufType::Q4_K:
            gguf_mmq_mma_launch_q4k(params, xq, stream);
            break;
        case GgufType::Q5_K:
            gguf_mmq_mma_launch_q5k(params, xq, stream);
            break;
        case GgufType::Q8_0:
            gguf_mmq_mma_launch_q8_0(params, xq, stream);
            break;
        case GgufType::MXFP4:
            gguf_mmq_mma_launch_mxfp4(params, xq, stream);
            break;
        case GgufType::Q6_K:
            gguf_mmq_mma_launch_q6k(params, xq, stream);
            break;
        case GgufType::Q2_K:
            gguf_mmq_mma_launch_q2k(params, xq, stream);
            break;
        case GgufType::Q3_K:
            gguf_mmq_mma_launch_q3k(params, xq, stream);
            break;
        default:
            throw std::invalid_argument(
                "gguf_mmq_mma: type not yet supported (only Q4_K wired); "
                "GgufType=" + std::to_string((int)params.type));
    }
}

}  // namespace layerstorm::compute
