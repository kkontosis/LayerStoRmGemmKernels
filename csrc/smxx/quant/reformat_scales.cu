#ifndef REFORMAT_SCALES_CU_INCLUDED
#define REFORMAT_SCALES_CU_INCLUDED
#include "smxx/quant/reformat_scales.h"
#include <cuda_fp8.h>

namespace layerstorm::compute {

// Sm1xx SfAtom offset computation — same formula as in bf16_to_nvfp4.cu
// Duplicated here to avoid cross-file device function linkage issues in single-TU builds.
__device__ __forceinline__
int reformat_sm1xx_sf_offset(int row, int k_group, int padded_rows, int padded_groups) {
    constexpr int BLK_MN = 128;
    constexpr int BLK_SF = 4;
    constexpr int ATOM_SIZE = 512;

    int tile_row = row / BLK_MN;
    int tile_grp = k_group / BLK_SF;
    int row_in_tile = row % BLK_MN;
    int grp_in_tile = k_group % BLK_SF;

    int num_tile_grps = (padded_groups + BLK_SF - 1) / BLK_SF;
    int tile_offset = (tile_row * num_tile_grps + tile_grp) * ATOM_SIZE;

    int r0 = row_in_tile % 32;
    int r1 = row_in_tile / 32;
    int within_atom = r0 * 16 + r1 * 4 + grp_in_tile;

    return tile_offset + within_atom;
}

// Convert float32 scale to UE4M3 byte
__device__ __forceinline__
uint8_t reformat_float_to_ue4m3(float val) {
    val = fmaxf(val, 0.0f);
    val = fminf(val, 480.0f);
    __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(val);
    return *reinterpret_cast<uint8_t*>(&fp8);
}

__global__ void __launch_bounds__(256)
reformat_scales_kernel(const ReformatScalesParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = params.rows * params.groups_per_row;
    if (idx >= total) return;

    int row = idx / params.groups_per_row;
    int grp = idx % params.groups_per_row;

    // Read from simple row-major float32 source
    float scale_val = params.src_scales[idx];

    // Determine padded dims for Sm1xx layout
    int dim = params.is_scale_a ? params.M : params.N;
    int padded_dim = ((dim + 127) / 128) * 128;
    int padded_groups = ((params.groups_per_row + 3) / 4) * 4;

    // Convert to UE4M3 and write to Sm1xx interleaved position
    uint8_t ue4m3_bits = reformat_float_to_ue4m3(scale_val);
    int sf_offset = reformat_sm1xx_sf_offset(row, grp, padded_dim, padded_groups);
    params.dst_scales[sf_offset] = ue4m3_bits;
}

void launch_reformat_scales(const ReformatScalesParams& params, cudaStream_t stream) {
    int total = params.rows * params.groups_per_row;
    int block = 256;
    int grid = (total + block - 1) / block;
    reformat_scales_kernel<<<grid, block, 0, stream>>>(params);
}

}  // namespace layerstorm::compute

#endif  // REFORMAT_SCALES_CU_INCLUDED
