#pragma once
#include "../base/types.hpp"

// --- [Fixed] Adaptive Lossy Filter with Float Support ---
template <typename T>
double estimate_layer_scale(const T* data, size_t n, bool is_bf16 = false) { // [UPDATED] Added is_bf16
    size_t samples = std::min(n, (size_t)4096);
    double sum = 0;
    size_t step = n / samples;
    if (step == 0) step = 1;
    
    for(size_t i=0; i<samples; ++i) {
        size_t idx = i * step;
        float val = 0.0f;
        if constexpr (sizeof(T) == 4) {
             const float* f_ptr = reinterpret_cast<const float*>(data);
             val = std::abs(f_ptr[idx]);
        } else {
            // [FIXED] Correct conversion based on type
            if (is_bf16) {
                val = std::abs(bf16_to_float(data[idx]));
            } else {
                val = std::abs(fp16_to_float(data[idx]));
            }
        }
        sum += val;
    }
    return sum / samples;
}

template <typename T>
std::vector<T> apply_lossy_preprocessing(const T* src, const T* ft, size_t n, float lossy_rate, bool is_bf16 = false) { // [UPDATED] Added is_bf16
    if (lossy_rate <= 1e-9) {
        return std::vector<T>(ft, ft + n);
    }

    // [UPDATED] Pass is_bf16 to scale estimator
    double scale = estimate_layer_scale(ft, n, is_bf16);
    double threshold_val = scale * lossy_rate; 

    // [Mantissa Masking Heuristic]
    // BF16 has 7 mantissa bits, FP16 has 10, FP32 has 23.
    int total_mantissa = (sizeof(T) == 4) ? 23 : (is_bf16 ? 7 : 10);
    int bits_to_drop = 0;
    
    if (lossy_rate > 0) {
        float precision_needed = -std::log2(lossy_rate); 
        int keep_bits = (int)precision_needed + 4;
        if (keep_bits > total_mantissa) keep_bits = total_mantissa;
        if (keep_bits < 0) keep_bits = 0;
        bits_to_drop = total_mantissa - keep_bits;
    }
    
    T mask = (T)-1;
    if (bits_to_drop > 0) {
        T drop_mask = (1ULL << bits_to_drop) - 1;
        mask = ~drop_mask;
    }

    const float* f_src_ptr = reinterpret_cast<const float*>(src);
    const float* f_ft_ptr  = reinterpret_cast<const float*>(ft);
    
    // [FIXED] Lambda to calculate physical difference correctly
    auto get_diff = [&](size_t i) -> double {
        if constexpr (sizeof(T) == 4) {
            return std::abs(f_ft_ptr[i] - f_src_ptr[i]);
        } else {
             // [FIXED] Use float conversion instead of integer sub or cast
             float v_src, v_ft;
             if (is_bf16) {
                 v_src = bf16_to_float(src[i]);
                 v_ft  = bf16_to_float(ft[i]);
             } else {
                 v_src = fp16_to_float(src[i]);
                 v_ft  = fp16_to_float(ft[i]);
             }
             return std::abs(v_ft - v_src);
        }
    };

    std::vector<T> processed_ft(n);
    for(size_t i = 0; i < n; ++i) {
        double diff = get_diff(i);
        
        // [Logic 1] Hard Thresholding
        if (diff < threshold_val) {
            processed_ft[i] = src[i];
        } 
        // [Logic 2] Mantissa Masking
        else {
            processed_ft[i] = ft[i] & mask;
        }
    }

    return processed_ft;
}
