#pragma once
#include "../base/types.hpp"

// --- Byte Shuffle (Bit-Plane Slicing) ---
template <typename T>
void byte_shuffle(const std::vector<T>& src, std::vector<u8>& dest) {
    size_t n = src.size();
    size_t type_size = sizeof(T);
    dest.resize(n * type_size);
    
    const u8* src_ptr = reinterpret_cast<const u8*>(src.data());
    u8* dest_ptr = dest.data();
    
    for (size_t k = 0; k < type_size; ++k) {
        for (size_t i = 0; i < n; ++i) {
            dest_ptr[k * n + i] = src_ptr[i * type_size + k];
        }
    }
}

template <typename T>
void byte_unshuffle(const std::vector<u8>& src, std::vector<T>& dest) {
    size_t total_bytes = src.size();
    size_t type_size = sizeof(T);
    size_t n = total_bytes / type_size;
    dest.resize(n);
    
    const u8* src_ptr = src.data();
    u8* dest_ptr = reinterpret_cast<u8*>(dest.data());
    
    for (size_t k = 0; k < type_size; ++k) {
        for (size_t i = 0; i < n; ++i) {
            dest_ptr[i * type_size + k] = src_ptr[k * n + i];
        }
    }
}