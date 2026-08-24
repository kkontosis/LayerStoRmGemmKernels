#ifndef BF16_TO_NVFP4_GROUPED_DETAIL_CUH
#define BF16_TO_NVFP4_GROUPED_DETAIL_CUH

// Shared device-side helpers for the grouped NVFP4 quantizers
// (bf16_to_nvfp4_grouped.cu and silu_mul_to_nvfp4_grouped.cu).
//
// All symbols here are internal-linkage (static constexpr / __device__
// __forceinline__), so this header may be included by multiple translation
// units without ODR collisions — which is exactly why the fused
// silu_mul_to_nvfp4_grouped kernel includes THIS header instead of the
// bf16_to_nvfp4_grouped.cu translation unit (whose __global__ kernel and
// launch_* host function have external linkage and would duplicate at link
// time in LayerStoRm3's separate-TU CMake build).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cfloat>
#include <cstdint>

namespace layerstorm::compute {

// ── Constants ───────────────────────────────────────────────────────────────

static constexpr float GROUPED_FP4_MAX_ABS = 6.0f;
static constexpr int   GROUPED_GROUP_SIZE  = 16;

// ── Sm1xx SfAtom offset ─────────────────────────────────────────────────────

__device__ __forceinline__
int grouped_sm1xx_sf_offset(int row, int k_group, int padded_rows, int padded_groups) {
    constexpr int BLK_MN    = 128;
    constexpr int BLK_SF    = 4;
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

// ── Quantize scalar to nearest FP4 E2M1 index (0-15) ────────────────────────

__device__ __forceinline__
uint8_t grouped_quantize_to_fp4(float val) {
    uint8_t sign = (val < 0.0f) ? 8 : 0;
    float a = fabsf(val);
    uint8_t idx = (a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) +
                  (a >= 1.75f) + (a >= 2.5f) + (a >= 3.5f) + (a >= 5.0f);
    return idx | sign;
}

// ── Float to UE4M3 conversion ────────────────────────────────────────────────

__device__ __forceinline__
uint8_t grouped_float_to_ue4m3(float val) {
    val = fmaxf(val, 0.0f);
    val = fminf(val, 480.0f);
    __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(val);
    return *reinterpret_cast<uint8_t*>(&fp8);
}

__device__ __forceinline__
float grouped_ue4m3_to_float(uint8_t bits) {
    __nv_fp8_e4m3 fp8 = *reinterpret_cast<__nv_fp8_e4m3*>(&bits);
    return float(fp8);
}

// ── Binary search: find expert_id s.t. offsets[eid] <= row < offsets[eid+1] ──

__device__ __forceinline__
int find_expert(const int32_t* __restrict__ expert_offsets, int num_experts, int row) {
    int lo = 0, hi = num_experts;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (expert_offsets[mid + 1] <= row)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

}  // namespace layerstorm::compute

#endif  // BF16_TO_NVFP4_GROUPED_DETAIL_CUH
