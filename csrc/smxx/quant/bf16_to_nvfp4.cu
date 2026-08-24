#ifndef BF16_TO_NVFP4_CU_INCLUDED
#define BF16_TO_NVFP4_CU_INCLUDED
#include "smxx/quant/bf16_to_nvfp4.h"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cfloat>

namespace layerstorm::compute {

static constexpr float FP4_MAX_ABS = 6.0f;
static constexpr int GROUP_SIZE = 16;

// Sm1xx SfAtom offset computation for SFVecSize=16, K-major:
//   SfKMajorAtom: Shape((32,4), (16,4)), Stride((16,4), (0,1))
//   offset_in_atom = (row%32)*16 + (row/32)*4 + (group%4)
//   Atoms tiled: 128 rows x 4 groups per atom.
__device__ __forceinline__
int sm1xx_sf_offset(int row, int k_group, int padded_rows, int padded_groups) {
    constexpr int BLK_MN = 128;
    constexpr int BLK_SF = 4;
    constexpr int ATOM_SIZE = 512;  // cosize of SfKMajorAtom with SFVecSize=16

    int tile_row = row / BLK_MN;
    int tile_grp = k_group / BLK_SF;
    int row_in_tile = row % BLK_MN;
    int grp_in_tile = k_group % BLK_SF;

    int num_tile_grps = (padded_groups + BLK_SF - 1) / BLK_SF;

    // Tile offset: row-major tiling of atoms
    int tile_offset = (tile_row * num_tile_grps + tile_grp) * ATOM_SIZE;

    // Within-atom offset: row = (r0, r1) where r0 = row%32, r1 = row/32
    int r0 = row_in_tile % 32;
    int r1 = row_in_tile / 32;
    int within_atom = r0 * 16 + r1 * 4 + grp_in_tile;

    return tile_offset + within_atom;
}

// Quantize a single float to nearest FP4 E2M1 index (0-15)
// Direct encoding: the 8 positive FP4 E2M1 values are
//   idx: 0    1    2    3    4    5    6    7
//   val: 0.0  0.5  1.0  1.5  2.0  3.0  4.0  6.0
// Boundaries (midpoints): 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
// We use a branchless comparison chain (7 compares vs 16 before).
__device__ __forceinline__
uint8_t quantize_to_fp4(float val) {
    uint8_t sign = (val < 0.0f) ? 8 : 0;
    float a = fabsf(val);
    // Branchless index via ordered comparisons against midpoints
    uint8_t idx = (a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) +
                  (a >= 1.75f) + (a >= 2.5f) + (a >= 3.5f) + (a >= 5.0f);
    return idx | sign;
}

// Convert float32 scale to UE4M3 (unsigned E4M3, range [0, 480])
// UE4M3: 4 exponent bits, 3 mantissa bits, no sign, bias=7
// We approximate by: clamp to [0, 480], then cast via FP8 E4M3
__device__ __forceinline__
uint8_t float_to_ue4m3(float val) {
    val = fmaxf(val, 0.0f);
    val = fminf(val, 480.0f);
    // Use E4M3 encoding: treat as unsigned (bit pattern same for positive values)
    __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(val);
    return *reinterpret_cast<uint8_t*>(&fp8);
}

// Convert UE4M3 byte back to float32
__device__ __forceinline__
float ue4m3_to_float(uint8_t bits) {
    __nv_fp8_e4m3 fp8 = *reinterpret_cast<__nv_fp8_e4m3*>(&bits);
    return float(fp8);
}

__global__ void __launch_bounds__(256)
bf16_to_nvfp4_kernel(const Bf16ToNvfp4Params params) {
    const int row = blockIdx.x;
    if (row >= params.M) return;

    const int K = params.K;
    const int groups_per_row = K / GROUP_SIZE;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    const __nv_bfloat16* row_in = params.input + row * K;
    uint8_t* row_packed = params.output_packed + row * (K / 2);

    // Padded dimensions for Sm1xx layout
    int padded_rows = ((params.M + 127) / 128) * 128;
    int padded_groups = ((groups_per_row + 3) / 4) * 4;

    // Each thread processes one or more groups of 16 elements
    for (int g = tid; g < groups_per_row; g += num_threads) {
        int base = g * GROUP_SIZE;

        // Load 16 BF16 values and compute amax
        float vals[GROUP_SIZE];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < GROUP_SIZE; i++) {
            vals[i] = __bfloat162float(row_in[base + i]);
            amax = fmaxf(amax, fabsf(vals[i]));
        }

        // Compute per-group scale
        float scale = amax / FP4_MAX_ABS;
        // Clamp to the UE4M3 minimum positive subnormal (2^-9): smaller
        // scales round to 0 in float_to_ue4m3 and the group dequants to
        // ZERO downstream (see bf16_to_nvfp4_grouped.cu).
        scale = fmaxf(scale, 0.001953125f);  // 2^-9

        // Convert to UE4M3 and back for representable value
        uint8_t scale_bits = float_to_ue4m3(scale);
        float scale_repr = ue4m3_to_float(scale_bits);
        scale_repr = fmaxf(scale_repr, 1e-12f);

        // Write scale to Sm1xx interleaved position
        int sf_offset = sm1xx_sf_offset(row, g, padded_rows, padded_groups);
        params.output_scales[sf_offset] = scale_bits;

        // Quantize each value to FP4 and pack pairs
        float inv_scale = 1.0f / scale_repr;
        #pragma unroll
        for (int i = 0; i < GROUP_SIZE; i += 2) {
            float v0 = vals[i] * inv_scale;
            float v1 = vals[i + 1] * inv_scale;
            uint8_t idx0 = quantize_to_fp4(v0);
            uint8_t idx1 = quantize_to_fp4(v1);
            // Pack: low nibble = even, high nibble = odd
            row_packed[(base + i) / 2] = (idx0 & 0x0F) | ((idx1 & 0x0F) << 4);
        }
    }
}

void launch_bf16_to_nvfp4(const Bf16ToNvfp4Params& params, cudaStream_t stream) {
    if (params.K % GROUP_SIZE != 0) return;  // K must be divisible by 16
    int grid = params.M;
    bf16_to_nvfp4_kernel<<<grid, 256, 0, stream>>>(params);
}

}  // namespace layerstorm::compute

#endif  // BF16_TO_NVFP4_CU_INCLUDED
