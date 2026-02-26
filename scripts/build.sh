#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# 创建并进入构建目录
mkdir -p build && cd build

# 让 CMake 生成构建文件并执行编译
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 回到根目录并运行测试
cd ..
echo "[RUN] Running benchmarks..."
CUDA_VISIBLE_DEVICES=0,2 python bench/main.py  --eval