#include "../include/core.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
// #include "pforex_core.hpp"

namespace py = pybind11;

// Wrapper for Compression
// [UPDATED] Added is_bf16 flag to handle bfloat16 correctly in C++
py::bytes compress_wrapper(py::array src_np, py::array ft_np, float lossy_rate, bool is_bf16 = false) {
    py::buffer_info buf_src = src_np.request();
    py::buffer_info buf_ft = ft_np.request();
    
    if (buf_src.size != buf_ft.size) throw std::runtime_error("Size mismatch between Base and FT");
    
    // Release GIL for parallelism
    py::gil_scoped_release release;
    
    std::vector<u8> result;
    
    // Robust dispatch based on itemsize (bytes per element)
    if (buf_src.itemsize == 4) { 
        // Handles float32, uint32, int32 (treated as bits)
        const u32* s = static_cast<const u32*>(buf_src.ptr);
        const u32* f = static_cast<const u32*>(buf_ft.ptr);
        result = compress_core<u32>(s, f, buf_src.size, lossy_rate, false); // float32 doesn't need bf16 flag
    } 
    else if (buf_src.itemsize == 2) {
        // Handles float16, bfloat16, uint16, int16
        const u16* s = static_cast<const u16*>(buf_src.ptr);
        const u16* f = static_cast<const u16*>(buf_ft.ptr);
        // [UPDATED] Pass is_bf16 flag to core
        result = compress_core<u16>(s, f, buf_src.size, lossy_rate, is_bf16);
    } 
    else {
        // Re-acquire GIL to throw exception safely
        py::gil_scoped_acquire acquire;
        std::string err_msg = "Unsupported itemsize: " + std::to_string(buf_src.itemsize) + 
                              ". Only 4 (float32) or 2 (float16/bf16) supported.";
        throw std::runtime_error(err_msg);
    }
    
    py::gil_scoped_acquire acquire;
    return py::bytes(reinterpret_cast<const char*>(result.data()), result.size());
}

// Wrapper for Decompression
py::array decompress_wrapper(std::string comp_data, py::array src_np) {
    py::buffer_info buf_src = src_np.request();
    
    py::gil_scoped_release release;
    
    if (buf_src.itemsize == 4) {
        const u32* s = static_cast<const u32*>(buf_src.ptr);
        auto res_vec = decompress_core<u32>((const u8*)comp_data.data(), comp_data.size(), s, buf_src.size);
        
        py::gil_scoped_acquire acquire;
        return py::array_t<u32>(res_vec.size(), res_vec.data());
    } 
    else if (buf_src.itemsize == 2) {
        const u16* s = static_cast<const u16*>(buf_src.ptr);
        auto res_vec = decompress_core<u16>((const u8*)comp_data.data(), comp_data.size(), s, buf_src.size);
        
        py::gil_scoped_acquire acquire;
        return py::array_t<u16>(res_vec.size(), res_vec.data());
    }
    
    py::gil_scoped_acquire acquire;
    throw std::runtime_error("Unsupported itemsize during decompression.");
}

PYBIND11_MODULE(pforex_cpp, m) {
    m.doc() = "FM-Delta-Pro: High-Performance Model Compression Extension";
    // [UPDATED] Updated signature explanation
    m.def("compress_layer", &compress_wrapper, "Compress layer (src, ft, rate, is_bf16)", 
          py::arg("src"), py::arg("ft"), py::arg("rate"), py::arg("is_bf16") = false);
    m.def("decompress_layer", &decompress_wrapper, "Decompress layer (releases GIL)");
}