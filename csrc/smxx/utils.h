#pragma once

// Utility macros (CHECK_CUDA / FLASH_ASSERT) adapted from FlashMLA
// (https://github.com/deepseek-ai/FlashMLA, csrc/utils.h).
// License: MIT, Copyright (c) 2025 DeepSeek — see THIRD_PARTY_NOTICES.md

#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        abort(); \
    } \
} while (0)

#define FLASH_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "Assertion failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        abort(); \
    } \
} while (0)
