#pragma once
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <zstd.h>
#include <type_traits>
#include <limits>
#include <random>


using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

// --- [NEW] Float Conversion Helpers ---
// BF16 to Float32: Shift left 16 bits
inline float bf16_to_float(u16 x) {
    u32 y = (u32)x << 16;
    float f;
    std::memcpy(&f, &y, sizeof(float));
    return f;
}

// FP16 to Float32: Standard IEEE 754 conversion
inline float fp16_to_float(u16 h) {
    u32 s = (h >> 15) & 0x00000001;
    u32 e = (h >> 10) & 0x0000001f;
    u32 m = h & 0x000003ff;
    u32 v = 0;

    if (e == 0) {
        if (m == 0) {
            // Signed Zero
            v = s << 31;
        } else {
            // Subnormal
            while (!(m & 0x00000400)) {
                m <<= 1;
                v -= 1;
            }
            v += 1;
            m &= 0x000003ff;
            v = (s << 31) | ((v + 112) << 23) | (m << 13);
        }
    } else if (e == 31) {
        // Inf or NaN
        v = (s << 31) | 0x7f800000 | (m << 13);
    } else {
        // Normalized
        v = (s << 31) | ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &v, sizeof(float));
    return f;
}