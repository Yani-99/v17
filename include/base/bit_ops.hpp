#pragma once
#include "types.hpp"

// --- Helper Functions ---
template <typename T>
inline int bits_for(T x) {
    if (x == 0) return 0;
    using U = std::make_unsigned_t<T>;
    U ux = static_cast<U>(x);
    if constexpr (sizeof(T) <= 4) return 32 - __builtin_clz(ux);
    else return 64 - __builtin_clzll(ux);
}

// Zigzag Encoding
template <typename T>
inline T zigzag(T x) {
    using SignedT = std::make_signed_t<T>;
    SignedT v = static_cast<SignedT>(x);
    return (T)((v << 1) ^ (v >> (sizeof(T) * 8 - 1)));
}

template <typename T>
inline T zigzag_decode(T z) {
    using SignedT = std::make_signed_t<T>;
    SignedT v = (z >> 1) ^ (-(z & 1));
    return (T)v;
}

// --- FM-Delta Mapping Logic ---
template <typename T>
inline T fm_map(T val) {
    constexpr T msb_mask = (T(1) << (sizeof(T) * 8 - 1));
    if (val & msb_mask) return ~val;
    else return val | msb_mask;
}

template <typename T>
inline T fm_unmap(T val) {
    constexpr T msb_mask = (T(1) << (sizeof(T) * 8 - 1));
    if (val & msb_mask) return val & (~msb_mask);
    else return ~val;
}