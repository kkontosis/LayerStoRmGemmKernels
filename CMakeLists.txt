# CMakeLists.txt — LayerStoRmGemmKernels
#
# Provides gemm_kernels OBJECT library for consumption by LayerStoRm3.
#
# Usage from parent project (e.g., LayerStoRm3/src/CMakeLists.txt):
#
#   set(GEMM_DIR "${CMAKE_SOURCE_DIR}/deps/LayerStoRmGemmKernels")
#   add_subdirectory(${GEMM_DIR} ${CMAKE_BINARY_DIR}/gemm_kernels)
#
#   target_include_directories(my_target PRIVATE ${GEMM_DIR}/csrc)
#   target_link_libraries(my_target PUBLIC $<TARGET_OBJECTS:gemm_kernels>)
#
# Or manually list the sources (like SnapMLA pattern):
#
#   set(GEMM_SOURCES
#     ${GEMM_DIR}/csrc/sm120/gemm/nvfp4/nvfp4_gemm.cu
#     ...
#   )
#   add_library(gemm_kernels OBJECT ${GEMM_SOURCES})
#
# Headers are self-contained: include "sm120/gemm/nvfp4/nvfp4_gemm.h" etc.
# No aggregate header needed — consumers include only what they use.

cmake_minimum_required(VERSION 3.18)
project(LayerStoRmGemmKernels LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# ── Source files ─────────────────────────────────────────────────────────────
# SM120 GEMM kernels — compiled as separate TUs (heavy CUTLASS templates).
# smxx quant kernels are lightweight and can be compiled in a single TU
# or individually.

set(GEMM_CSRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")

set(GEMM_SM120_SOURCES
  ${GEMM_CSRC_DIR}/sm120/gemm/nvfp4/nvfp4_gemm.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/nvfp4/nvfp4_grouped_gemm.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/fp8/fp8_gemm.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/fp8/fp8_blockscaled_gemv.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/fp8/fp8_grouped_gemm.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/q4k/q4k_dequant_gemm.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/q4k/q4k_cutlass_gemm.cu
  # GGUF k-quant + Q8_0 weight-quant GEMMs (all 6 types): dequant / mmvq / mmq
  ${GEMM_CSRC_DIR}/sm120/gemm/gguf/gguf_dequant_gemm.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/gguf/gguf_mmvq.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/gguf/gguf_mmq.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/gguf/gguf_mmq_cute.cu
  # Fast per-type int8 tensor-core mmq (v3)
  ${GEMM_CSRC_DIR}/sm120/gemm/gguf/mmq_mma/mmq_mma_dispatch.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/gguf/mmq_mma/instances/mmq_q4_k.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/gguf/mmq_mma/instances/mmq_q5_k.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/gguf/mmq_mma/instances/mmq_q8_0.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/gguf/mmq_mma/instances/mmq_q6_k.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/gguf/mmq_mma/instances/mmq_q2_k.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/gguf/mmq_mma/instances/mmq_q3_k.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/gguf/mmq_mma/instances/mmq_mxfp4.cu
  # GGUF grouped GEMM (MoE expert FFN): device-routed Int (mmvq/mmq, CUDA-graph
  # capturable, TD-GG5) + host-loop Dequant golden
  ${GEMM_CSRC_DIR}/sm120/gemm/gguf/gguf_grouped_int.cu
  ${GEMM_CSRC_DIR}/sm120/gemm/gguf/gguf_grouped_gemm.cu
)

set(GEMM_SMXX_SOURCES
  ${GEMM_CSRC_DIR}/smxx/quant/bf16_to_nvfp4.cu
  ${GEMM_CSRC_DIR}/smxx/quant/bf16_to_nvfp4_grouped.cu
  ${GEMM_CSRC_DIR}/smxx/quant/silu_mul_to_nvfp4_grouped.cu
  ${GEMM_CSRC_DIR}/smxx/quant/reformat_scales.cu
  ${GEMM_CSRC_DIR}/smxx/quant/dynamic_fp8_quant.cu
  ${GEMM_CSRC_DIR}/smxx/quant/weight_fp8_quant.cu
  ${GEMM_CSRC_DIR}/smxx/quant/nvfp4_dequant_bf16.cu
)

set(GEMM_ALL_SOURCES ${GEMM_SM120_SOURCES} ${GEMM_SMXX_SOURCES})

# ── OBJECT library ───────────────────────────────────────────────────────────

add_library(gemm_kernels OBJECT ${GEMM_ALL_SOURCES})

# SM120 with feature flag (required for blockwise-scaled FP8 and NVFP4)
set_target_properties(gemm_kernels PROPERTIES CUDA_ARCHITECTURES "120f")

target_compile_options(gemm_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -std=c++17
    -O2
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
  >
)

# Include paths: csrc/ for kernel headers, CUTLASS for template library.
# CUTLASS path: prefer parent project's cutlass_headers target, fall back to
# local 3rd-party/cutlass submodule.
target_include_directories(gemm_kernels PUBLIC
  ${GEMM_CSRC_DIR}
)

if(TARGET cutlass_headers)
  # Parent project provides cutlass_headers (preferred)
  target_link_libraries(gemm_kernels PRIVATE cutlass_headers)
else()
  # Use local 3rd-party/cutlass submodule
  set(GEMM_CUTLASS_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/cutlass/include")
  set(GEMM_CUTLASS_UTIL "${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/cutlass/tools/util/include")
  if(EXISTS ${GEMM_CUTLASS_INCLUDE})
    target_include_directories(gemm_kernels PRIVATE
      ${GEMM_CUTLASS_INCLUDE}
      ${GEMM_CUTLASS_UTIL}
    )
  else()
    message(FATAL_ERROR "CUTLASS headers not found. Run: git submodule update --init --recursive")
  endif()
endif()

target_link_libraries(gemm_kernels PRIVATE CUDA::cudart)

# ── Export variables for parent projects ─────────────────────────────────────
# These let the parent CMakeLists.txt reference our paths without hardcoding.

set(GEMM_KERNELS_INCLUDE_DIR ${GEMM_CSRC_DIR} PARENT_SCOPE)
set(GEMM_KERNELS_SM120_SOURCES ${GEMM_SM120_SOURCES} PARENT_SCOPE)
set(GEMM_KERNELS_SMXX_SOURCES ${GEMM_SMXX_SOURCES} PARENT_SCOPE)
set(GEMM_KERNELS_ALL_SOURCES ${GEMM_ALL_SOURCES} PARENT_SCOPE)
