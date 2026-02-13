#pragma once
#include "../base/types.hpp"
#include "../base/bit_ops.hpp"

// --- PForDelta Core ---
struct SplitStreams {
    std::vector<u8> meta;
    std::vector<u8> packed;
    std::vector<u8> exc;
    void clear() { meta.clear(); packed.clear(); exc.clear(); }
};

template <typename T>
void pfor_encode_block_split(const std::vector<T>& block, SplitStreams& out) {
    size_t n = block.size();
    size_t needed_packed = out.packed.size() + n * sizeof(T);
    if (out.packed.capacity() < needed_packed) {
        out.packed.reserve(std::max(needed_packed, out.packed.capacity() * 2));
    }

    size_t needed_exc = out.exc.size() + n * (sizeof(T) + 1);
    if (out.exc.capacity() < needed_exc) {
        out.exc.reserve(std::max(needed_exc, out.exc.capacity() * 2));
    }
    
    size_t needed_meta = out.meta.size() + 2;
    if (out.meta.capacity() < needed_meta) {
        out.meta.reserve(std::max(needed_meta, out.meta.capacity() * 2));
    }

    T max_val = 0;
    for(T x : block) if(x > max_val) max_val = x;
    int max_b = bits_for(max_val);
    
    int best_w = max_b;
    u32 min_cost = n * max_b; 
    
    for(int w = 0; w < max_b; ++w) {
        u32 exc_count = 0;
        for(T x : block) if (bits_for(x) > w) exc_count++;
        u32 current_cost = (n * w) + (exc_count * (8 + sizeof(T)*8));
        if (current_cost <= min_cost) { min_cost = current_cost; best_w = w; }
    }
    
    out.meta.push_back((u8)best_w);
    out.meta.push_back((u8)0);
    
    u32 exc_count = 0;
    u64 bitbuf = 0;
    int bitlen = 0;
    u64 mask = (best_w == sizeof(T)*8) ? (u64)-1 : ((1ULL << best_w) - 1);
    
    for(size_t i = 0; i < n; ++i) {
        T val = block[i];
        if (bits_for(val) > best_w) {
            out.exc.push_back((u8)i); 
            for(size_t b=0; b<sizeof(T); ++b) out.exc.push_back((val >> (b*8)) & 0xFF); 
            val = 0; 
            exc_count++;
        }
        if (best_w > 0) {
            bitbuf |= ((u64(val) & mask) << bitlen);
            bitlen += best_w;
            while(bitlen >= 8) {
                out.packed.push_back((u8)(bitbuf & 0xFF));
                bitbuf >>= 8;
                bitlen -= 8;
            }
        }
    }
    if (bitlen > 0) out.packed.push_back((u8)(bitbuf & 0xFF));
    out.meta.back() = (u8)exc_count; 
}

template <typename T>
void pfor_decode_block_split(const u8* &meta_ptr, const u8* &packed_ptr, const u8* &exc_ptr, 
                             size_t count, std::vector<T>& out) {
    int w = *meta_ptr++;
    int exc_count = *meta_ptr++;
    size_t start_idx = out.size();
    
    if (w == 0) {
        for(size_t i=0; i<count; ++i) out.push_back(0);
    } else {
        u64 bitbuf = 0;
        int bitlen = 0;
        u64 mask = (w == sizeof(T)*8) ? (u64)-1 : ((1ULL << w) - 1);
        for(size_t i=0; i<count; ++i) {
            while(bitlen < w) {
                bitbuf |= (u64(*packed_ptr++) << bitlen);
                bitlen += 8;
            }
            out.push_back((T)(bitbuf & mask));
            bitbuf >>= w;
            bitlen -= w;
        }
    }
    
    for(int i=0; i<exc_count; ++i) {
        u8 idx = *exc_ptr++;
        T val = 0;
        for(size_t b=0; b<sizeof(T); ++b) val |= ((T)(*exc_ptr++) << (b*8));
        if (start_idx + idx < out.size()) out[start_idx + idx] = val;
    }
}
