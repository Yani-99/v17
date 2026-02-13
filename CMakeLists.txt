cmake_minimum_required(VERSION 3.10)
project(pforex_extension)

set(CMAKE_CXX_STANDARD 20)
# 保持高性能优化参数
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fPIC")

execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")

# 1. 寻找必要依赖
find_package(pybind11 REQUIRED)
find_library(ZSTD_LIB zstd REQUIRED)

# 2. 包含目录：指向您新建立的 include 文件夹
# 这样代码中可以使用 #include "base/types.hpp" 这种清晰的路径
include_directories(include)
include_directories(src)

# 3. 定义模块：指定 pybind 源代码位置
pybind11_add_module(pforex_cpp src/bind.cpp)

# 4. 链接库
target_link_libraries(pforex_cpp PRIVATE ${ZSTD_LIB})