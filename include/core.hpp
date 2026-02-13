#pragma once
#include "base/types.hpp"
#include "base/bit_ops.hpp"
#include "kernels/pfor.hpp"
#include "kernels/shuffle.hpp"
#include "layers/lossy.hpp"


template <typename T>
size_t measure_compressed_size(const std::vector<T>& data, bool use_shuffle) {
    if (data.empty()) return 0;
    
    thread_local std::vector<u8> tl_raw;
    thread_local std::vector<u8> tl_comp_buf;
    thread_local ZSTD_CCtx* tl_cctx = ZSTD_createCCtx(); 
    
    tl_raw.clear(); 
    
    if (use_shuffle) {
        byte_shuffle(data, tl_raw);
    } else {
        SplitStreams streams;
        streams.meta.reserve(data.size() / 64 + 512);
        streams.packed.reserve(data.size() * sizeof(T));
        streams.exc.reserve(data.size() / 4);

        size_t block_size = 128;
        for(size_t i=0; i<data.size(); i+=block_size) {
            size_t end = std::min(i+block_size, data.size());
            std::vector<T> block(data.begin()+i, data.begin()+end);
            while(block.size() < 128) block.push_back(0);
            pfor_encode_block_split(block, streams);
        }
        
        size_t needed = streams.meta.size() + streams.packed.size() + streams.exc.size();
        if (tl_raw.capacity() < needed) tl_raw.reserve(needed * 2); 
        
        tl_raw.insert(tl_raw.end(), streams.meta.begin(), streams.meta.end());
        tl_raw.insert(tl_raw.end(), streams.packed.begin(), streams.packed.end());
        tl_raw.insert(tl_raw.end(), streams.exc.begin(), streams.exc.end());
    }

    size_t bound = ZSTD_compressBound(tl_raw.size());
    if (tl_comp_buf.capacity() < bound) tl_comp_buf.resize(bound); 

    size_t c_size = ZSTD_compressCCtx(tl_cctx, tl_comp_buf.data(), bound, tl_raw.data(), tl_raw.size(), 1);
    
    if (ZSTD_isError(c_size)) return tl_raw.size();
    return c_size;
}

template <typename T>
struct Sampler {
    std::vector<T> src_samples;
    std::vector<T> ft_samples;
    
    Sampler(const T* src, const T* ft, size_t n) {
        const size_t BLOCK_SIZE = 128;
        const size_t NUM_SAMPLE_BLOCKS = 16;
        
        if (n <= BLOCK_SIZE * NUM_SAMPLE_BLOCKS) {
            src_samples.assign(src, src + n);
            ft_samples.assign(ft, ft + n);
        } else {
            size_t step = (n - BLOCK_SIZE) / (NUM_SAMPLE_BLOCKS - 1);
            if (step < BLOCK_SIZE) step = BLOCK_SIZE;
            for (size_t i = 0; i < NUM_SAMPLE_BLOCKS; ++i) {
                size_t offset = i * step;
                if (offset + BLOCK_SIZE <= n) {
                    src_samples.insert(src_samples.end(), src + offset, src + offset + BLOCK_SIZE);
                    ft_samples.insert(ft_samples.end(), ft + offset, ft + offset + BLOCK_SIZE);
                }
            }
        }
    }
};

template <typename T>
struct CompressStrategy {
    std::string method;
    std::vector<T> data;
    bool use_shuffle = false;
};

template <typename T>
CompressStrategy<T> pick_strategy_and_transform(const T* src, const T* cleaned_ft, size_t n) {
    Sampler<T> sampler(src, cleaned_ft, n);
    size_t sample_n = sampler.src_samples.size();
    
    std::vector<T> s_xor(sample_n);
    for(size_t i=0; i<sample_n; ++i) s_xor[i] = sampler.src_samples[i] ^ sampler.ft_samples[i];
    
    std::vector<T> s_delta(sample_n);
    for(size_t i=0; i<sample_n; ++i) s_delta[i] = zigzag(sampler.ft_samples[i] - sampler.src_samples[i]);
        
    std::vector<T> s_fm(sample_n);
    for(size_t i=0; i<sample_n; ++i) {
        T u_src = fm_map(sampler.src_samples[i]);
        T u_ft  = fm_map(sampler.ft_samples[i]);
        s_fm[i] = zigzag(u_ft - u_src);
    }

    size_t sz_xor   = measure_compressed_size(s_xor, false);
    size_t sz_delta = measure_compressed_size(s_delta, false);
    size_t sz_fm    = measure_compressed_size(s_fm, false);
    size_t sz_fm_shuffle = measure_compressed_size(s_fm, true);
    
    size_t min_sz = sz_xor;
    std::string method = "raw_xor";
    bool use_shuffle = false;

    if (sz_delta < min_sz) { min_sz = sz_delta; method = "signed_zig"; }
    if (sz_fm < min_sz)    { min_sz = sz_fm;    method = "fm_delta"; }
    
    if (sz_fm_shuffle < min_sz) { 
        min_sz = sz_fm_shuffle; 
        method = "fm_shuffle"; 
        use_shuffle = true; 
    }
    
    std::vector<T> final_data(n);
    if (method == "raw_xor") {
        for(size_t i=0; i<n; ++i) final_data[i] = src[i] ^ cleaned_ft[i];
    }
    else if (method == "signed_zig") {
        for(size_t i=0; i<n; ++i) final_data[i] = zigzag(cleaned_ft[i] - src[i]);
    }
    else if (method == "fm_delta" || method == "fm_shuffle") {
        for(size_t i=0; i<n; ++i) {
            T u_src = fm_map(src[i]);
            T u_ft  = fm_map(cleaned_ft[i]);
            final_data[i] = zigzag(u_ft - u_src);
        }
    }
    
    return {method, std::move(final_data), use_shuffle};
}

// --- Main Entry Points ---
template <typename T>
std::vector<u8> compress_core(const T* src, const T* ft, size_t n, float lossy_rate, bool is_bf16 = false) { // [UPDATED] Added is_bf16
    // 1. [Lossy Filter] (UPDATED with is_bf16)
    std::vector<T> cleaned_ft = apply_lossy_preprocessing(src, ft, n, lossy_rate, is_bf16);
    
    // 2. [Strategy Selection] (Untouched)
    auto strat = pick_strategy_and_transform(src, cleaned_ft.data(), n);
    
    // 3. [Compression] (Untouched optimized logic)
    thread_local std::vector<u8> tl_final_buf;
    thread_local ZSTD_CCtx* tl_cctx_final = ZSTD_createCCtx(); 
    tl_final_buf.clear(); 

    if (strat.use_shuffle) {
        std::vector<u8> raw;
        byte_shuffle(strat.data, raw);

        std::string header = strat.method + "|";
        u32 head_len = header.size();
        u32 packed_len = raw.size();
        u32 meta_len = 0, exc_len = 0; 
        
        size_t total = 16 + head_len + packed_len;
        
        if (tl_final_buf.capacity() < total) tl_final_buf.reserve(total * 2);
        tl_final_buf.resize(total);
        
        u8* ptr = tl_final_buf.data();
        memcpy(ptr, &head_len, 4); ptr+=4; memcpy(ptr, header.data(), head_len); ptr+=head_len;
        memcpy(ptr, &meta_len, 4); ptr+=4; 
        memcpy(ptr, &packed_len, 4); ptr+=4; memcpy(ptr, raw.data(), packed_len); ptr+=packed_len;
        memcpy(ptr, &exc_len, 4); ptr+=4; 
    } else {
        SplitStreams streams;
        streams.meta.reserve(n / 64 + 1024); 
        streams.packed.reserve(n * sizeof(T)); 
        streams.exc.reserve(n / 2);

        size_t block_size = 128;
        for(size_t i=0; i<n; i+=block_size) {
            size_t end = std::min(i+block_size, n);
            std::vector<T> block(strat.data.begin()+i, strat.data.begin()+end);
            while(block.size() < 128) block.push_back(0);
            pfor_encode_block_split(block, streams);
        }
        
        u32 meta_len = streams.meta.size();
        u32 packed_len = streams.packed.size();
        u32 exc_len = streams.exc.size();
        
        std::string header = strat.method + "|";
        u32 head_len = header.size();
        size_t total = 16 + head_len + meta_len + packed_len + exc_len;
        
        if (tl_final_buf.capacity() < total) tl_final_buf.reserve(total * 2);
        tl_final_buf.resize(total);

        u8* ptr = tl_final_buf.data();
        memcpy(ptr, &head_len, 4); ptr+=4; memcpy(ptr, header.data(), head_len); ptr+=head_len;
        memcpy(ptr, &meta_len, 4); ptr+=4; memcpy(ptr, streams.meta.data(), meta_len); ptr+=meta_len;
        memcpy(ptr, &packed_len, 4); ptr+=4; memcpy(ptr, streams.packed.data(), packed_len); ptr+=packed_len;
        memcpy(ptr, &exc_len, 4); ptr+=4; memcpy(ptr, streams.exc.data(), exc_len); ptr+=exc_len;
    }

    size_t bound = ZSTD_compressBound(tl_final_buf.size());
    std::vector<u8> comp(bound); 
    size_t c_size = ZSTD_compressCCtx(tl_cctx_final, comp.data(), bound, tl_final_buf.data(), tl_final_buf.size(), 1);
    comp.resize(c_size);
    return comp;
}

template <typename T>
std::vector<T> decompress_core(const u8* comp_data, size_t comp_size, const T* src, size_t n) {
    unsigned long long rSize = ZSTD_getFrameContentSize(comp_data, comp_size);
    if (rSize == ZSTD_CONTENTSIZE_UNKNOWN || rSize == ZSTD_CONTENTSIZE_ERROR) throw std::runtime_error("Zstd Error");
        
    std::vector<u8> raw(rSize);
    ZSTD_decompress(raw.data(), rSize, comp_data, comp_size);
    
    const u8* ptr = raw.data();
    u32 head_len; memcpy(&head_len, ptr, 4); ptr+=4;
    std::string header((char*)ptr, head_len); ptr+=head_len;
    
    u32 meta_len; memcpy(&meta_len, ptr, 4); ptr+=4;
    const u8* meta_ptr = ptr; ptr += meta_len;
    u32 packed_len; memcpy(&packed_len, ptr, 4); ptr+=4;
    const u8* packed_ptr = ptr; ptr += packed_len;
    u32 exc_len; memcpy(&exc_len, ptr, 4); ptr+=4;
    const u8* exc_ptr = ptr; ptr += exc_len;
    
    std::string method = header.substr(0, header.find('|'));
    std::vector<T> temp_data;
    temp_data.reserve(n);
    
    if (method == "fm_shuffle") {
        std::vector<u8> shuffled_bytes(packed_ptr, packed_ptr + packed_len);
        byte_unshuffle(shuffled_bytes, temp_data);
        if(temp_data.size() > n) temp_data.resize(n);
        else if (temp_data.size() < n) temp_data.resize(n, 0); 
    } else {
        const u8* meta_end = meta_ptr + meta_len;
        while(meta_ptr < meta_end) { pfor_decode_block_split(meta_ptr, packed_ptr, exc_ptr, 128, temp_data); }
        if(temp_data.size() > n) temp_data.resize(n);
    }
    
    std::vector<T> recovered(n);
    
    if(method == "raw_xor") {
        for(size_t i=0; i<n; ++i) recovered[i] = src[i] ^ temp_data[i];
    } else if(method == "signed_zig") {
        for(size_t i=0; i<n; ++i) recovered[i] = src[i] + zigzag_decode(temp_data[i]);
    } else if (method == "fm_delta" || method == "fm_shuffle") {
        for(size_t i=0; i<n; ++i) {
            T delta = zigzag_decode(temp_data[i]);
            T u_src = fm_map(src[i]);
            T u_ft  = u_src + delta;
            recovered[i] = fm_unmap(u_ft);
        }
    } else {
        memcpy(recovered.data(), temp_data.data(), n * sizeof(T));
    }
    return recovered;
}