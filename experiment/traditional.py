# import os
# import time
# import zlib
# import gzip
# import zstandard as zstd
# import torch
# import numpy as np
# import logging
# from glob import glob
# from huggingface_hub import snapshot_download

# # ==========================================
# # 1. 复制您的配置和逻辑
# # ==========================================
# TEST_CONFIGS = {
#     f"bert_ner{i}": {
#         "task_type": "",
#         "base_model": "bert-large-uncased",
#         "ft_model": model_id
#     } for i, model_id in enumerate([
#         "assemblyai/bert-large-uncased-sst2",
#         "samrawal/bert-large-uncased_med-ner",
#         "yoshitomo-matsubara/bert-large-uncased-mnli",
#         "princeton-nlp/sup-simcse-bert-large-uncased",
#         "SarielSinLuo/bert-large-uncased-finetuned-cola",
#         "princeton-nlp/unsup-simcse-bert-large-uncased",
#         "yoshitomo-matsubara/bert-large-uncased-mrpc",
#         "yoshitomo-matsubara/bert-large-uncased-qnli",
#         "StevenLimcorn/bert-large-uncased-semeval2016-restaurants",
#         "Jorgeutd/bert-large-uncased-finetuned-ner"
#     ], 1)
# }

# def get_model_files(model_id):
#     """ 沿用您的代码逻辑获取模型权重文件 """
#     try:
#         folder = snapshot_download(model_id, allow_patterns=["*.safetensors"])
#         files = glob(os.path.join(folder, "*.safetensors"))
#         if files: return sorted(files), "safetensors"
#         folder = snapshot_download(model_id, allow_patterns=["*.bin"])
#         files = [f for f in glob(os.path.join(folder, "*.bin")) if "training_args" not in f]
#         return sorted(files), "bin"
#     except:
#         return [], None

# # ==========================================
# # 2. 对比实验核心逻辑
# # ==========================================
# def run_comparison_benchmark():
#     target_models = [f"bert_ner{i}" for i in range(1, 11)]
    
#     print("\n" + "="*85)
#     print(f"{'Model Name':<15} | {'Method':<8} | {'Orig(MB)':<10} | {'Comp%':<10} | {'Throughput(MB/s)':<15}")
#     print("-" * 85)

#     for name in target_models:
#         cfg = TEST_CONFIGS[name]
#         ft_id = cfg['ft_model']
        
#         # 获取并读取原始二进制数据
#         files, _ = get_model_files(ft_id)
#         if not files: continue
        
#         raw_data = b""
#         for f in files:
#             with open(f, 'rb') as rb:
#                 raw_data += rb.read()
        
#         orig_size = len(raw_data)
#         orig_mb = orig_size / (1024 * 1024)
        
#         # 定义压缩方法
#         methods = [
#             ("zlib", lambda d: zlib.compress(d, level=6)),
#             ("gzip", lambda d: gzip.compress(d, compresslevel=6)),
#             ("zstd", lambda d: zstd.ZstdCompressor(level=3, threads=1).compress(d))
#         ]
        
#         for meth_name, compress_fn in methods:
#             # 计算吞吐量 (Throughput)
#             start_t = time.perf_counter()
#             compressed = compress_fn(raw_data)
#             duration = time.perf_counter() - start_t
            
#             comp_size = len(compressed)
#             ratio = (comp_size / orig_size) * 100
#             throughput = orig_mb / duration if duration > 0 else 0
            
#             print(f"{name:<15} | {meth_name:<8} | {orig_mb:>8.1f} | {ratio:>8.2f}% | {throughput:>12.2f}")
            
#             # 及时释放内存
#             del compressed
#         # del raw_data
#         # shutil.rmtree(folder) # 删除下载的临时模型文件
#         # gc.collect()          # 强制进行垃圾回收

# if __name__ == "__main__":
#     run_comparison_benchmark()


import os
import time
import zlib
import gzip
import zstandard as zstd
import shutil
import gc
import json
from glob import glob
from huggingface_hub import snapshot_download

# ==========================================
# 配置与初始化
# ==========================================
MODELS_TO_TEST = [
    "assemblyai/bert-large-uncased-sst2",
    "samrawal/bert-large-uncased_med-ner",
    "yoshitomo-matsubara/bert-large-uncased-mnli",
    "princeton-nlp/sup-simcse-bert-large-uncased",
    "SarielSinLuo/bert-large-uncased-finetuned-cola",
    "princeton-nlp/unsup-simcse-bert-large-uncased",
    "yoshitomo-matsubara/bert-large-uncased-mrpc",
    "yoshitomo-matsubara/bert-large-uncased-qnli",
    "StevenLimcorn/bert-large-uncased-semeval2016-restaurants",
    "Jorgeutd/bert-large-uncased-finetuned-ner"
]

# 用于存放最终聚合结果
aggregated_results = {
    "zstd": {},
    "zlib": {},
    "gzip": {}
}

def get_and_load_model(model_id):
    try:
        # 尝试下载 safetensors 或 bin
        folder = snapshot_download(model_id, allow_patterns=["*.safetensors"], local_files_only=False)
        files = glob(os.path.join(folder, "*.safetensors"))
        if not files:
            folder = snapshot_download(model_id, allow_patterns=["*.bin"], local_files_only=False)
            files = [f for f in glob(os.path.join(folder, "*.bin")) if "training_args" not in f]
        
        if not files: return None, None
            
        raw_data = bytearray()
        for f in sorted(files):
            with open(f, 'rb') as rb:
                raw_data.extend(rb.read())
        return raw_data, folder
    except Exception as e:
        print(f"Skipping {model_id} due to error: {e}")
        return None, None

# ==========================================
# 核心测试逻辑
# ==========================================
def run_benchmark():
    print("\n" + "="*95)
    print(f"{'Model Name':<15} | {'Method':<8} | {'Orig(MB)':<10} | {'Comp%':<10} | {'C-Speed(MB/s)':<15}")
    print("-" * 95)

    for i, model_id in enumerate(MODELS_TO_TEST, 1):
        model_label = f"bert_ner{i}"
        raw_data, folder_path = get_and_load_model(model_id)
        if raw_data is None: continue
        
        orig_size = len(raw_data)
        orig_mb = orig_size / (1024 * 1024)

        # 定义算法及其对应的 (压缩函数, 解压函数)
        methods = [
            ("zlib", lambda d: zlib.compress(d, level=6), zlib.decompress),
            ("gzip", lambda d: gzip.compress(d, compresslevel=6), gzip.decompress),
            ("zstd", lambda d: zstd.ZstdCompressor(level=3).compress(d), zstd.ZstdDecompressor().decompress)
        ]
        
        for meth_name, comp_fn, decomp_fn in methods:
            # 测试压缩
            t0 = time.perf_counter()
            compressed_data = comp_fn(raw_data)
            t1 = time.perf_counter()
            comp_time = t1 - t0
            
            # 测试解压
            t2 = time.perf_counter()
            _ = decomp_fn(compressed_data)
            t3 = time.perf_counter()
            decomp_time = t3 - t2
            
            # 计算指标
            ratio = (len(compressed_data) / orig_size) * 100
            comp_speed = orig_mb / comp_time if comp_time > 0 else 0
            decomp_speed = orig_mb / decomp_time if decomp_time > 0 else 0
            
            # 实时打印行
            print(f"{model_label:<15} | {meth_name:<8} | {orig_mb:>8.1f} | {ratio:>8.2f}% | {comp_speed:>12.2f}")
            
            # 存入聚合字典
            aggregated_results[meth_name][model_label] = {
                "Comp%": round(ratio, 2),
                "CompSpeed": round(comp_speed, 2),
                "DecompSpeed": round(decomp_speed, 2)
            }
            
            del compressed_data, _
            gc.collect()

        # 模型级清理
        del raw_data
        if folder_path and os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        gc.collect()

    # ==========================================
    # 输出聚合结果
    # ==========================================
    print("\n" + "="*30)
    print("AGGREGATED RESULTS BY METHOD")
    print("="*30)
    for method, results in aggregated_results.items():
        print(f"\n{method}:")
        # 使用 json.dumps 格式化输出字典，使其整齐漂亮
        print(json.dumps(results, indent=4))

if __name__ == "__main__":
    run_benchmark()