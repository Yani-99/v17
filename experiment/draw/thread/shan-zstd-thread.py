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


# 当前测试的目标模型
MODELS_TO_TEST = [
    "meta-llama/Llama-3.1-8B-Instruct"
]

# 设置要测试的 zstd 线程数
THREADS_TO_TEST = [1, 2, 4, 8, 16, 32, 48]

# 用于存放最终聚合结果
aggregated_results = {f"zstd_threads_{t}": {} for t in THREADS_TO_TEST}

# 每次读取的块大小设为 256 MB，最大化 CPU 吞吐量并对齐 v17 的系统级内存表现
CHUNK_SIZE = 256 * 1024 * 1024 

def get_model_files(model_id):
    base_path = "/home/newdrive2/liu4441/temp_traditional/"
    model_folder_name = model_id.replace("/", "_")
    target_dir = os.path.join(base_path, model_folder_name)

    try:
        folder = snapshot_download(
            model_id, 
            allow_patterns=["*.safetensors"], 
            local_files_only=False,
            local_dir=target_dir
        )
        files = glob(os.path.join(folder, "*.safetensors"))
        
        if not files:
            folder = snapshot_download(
                model_id, 
                allow_patterns=["*.bin"], 
                local_files_only=False,
                local_dir=target_dir
            )
            files = [f for f in glob(os.path.join(folder, "*.bin")) if "training_args" not in f]
        
        if not files: 
            print(f"No suitable model files found for {model_id}")
            return [], 0
            
        total_size = sum(os.path.getsize(f) for f in files)
        return sorted(files), total_size
    except Exception as e:
        print(f"Skipping {model_id} due to error: {e}")
        return [], 0

def stream_test_method(files, method_name, threads=1):
    comp_time = 0.0
    decomp_time = 0.0
    comp_size = 0

    # 原有的 zlib / gzip 逻辑保留，以备后续切换复用
    if method_name == "zlib":
        c_obj = zlib.compressobj(level=6)
        d_obj = zlib.decompressobj()
    elif method_name == "gzip":
        c_obj = zlib.compressobj(level=6, wbits=31)
        d_obj = zlib.decompressobj(wbits=31)
    elif method_name == "zstd":
        # 增加 threads 参数传递
        c_obj = zstd.ZstdCompressor(level=3, threads=threads).compressobj()
        d_obj = zstd.ZstdDecompressor().decompressobj()

    for f in files:
        with open(f, 'rb') as rb:
            while True:
                chunk = rb.read(CHUNK_SIZE)
                if not chunk:
                    break
                
                t0 = time.perf_counter()
                c_chunk = c_obj.compress(chunk)
                t1 = time.perf_counter()
                comp_time += (t1 - t0)
                
                comp_size += len(c_chunk)
                
                t2 = time.perf_counter()
                d_chunk = d_obj.decompress(c_chunk)
                t3 = time.perf_counter()
                decomp_time += (t3 - t2)
                
                del chunk, c_chunk, d_chunk

    t0 = time.perf_counter()
    if method_name == "zstd":
        c_chunk = c_obj.flush(zstd.COMPRESSOBJ_FLUSH_FINISH)
    else:
        c_chunk = c_obj.flush()
    t1 = time.perf_counter()
    comp_time += (t1 - t0)
    
    comp_size += len(c_chunk)
    
    t2 = time.perf_counter()
    d_chunk = d_obj.decompress(c_chunk)
    if method_name != "zstd":
        d_chunk += d_obj.flush()
    t3 = time.perf_counter()
    decomp_time += (t3 - t2)

    del c_chunk, d_chunk
    return comp_size, comp_time, decomp_time

def run_benchmark():
    # 调整表头格式加入线程数 (Threads)
    header = (
        f"{'Model Name':<35} | {'Method':<8} | {'Threads':<7} | {'Orig(MB)':<10} | "
        f"{'Comp%':<10} | {'C-Speed(MB/s)':<15} | {'D-Speed(MB/s)':<15}"
    )
    
    print("\n" + "="*len(header))
    print(header)
    print("-" * len(header))

    for model_id in MODELS_TO_TEST:
        model_label = model_id
        
        files, orig_size = get_model_files(model_id)
        if not files or orig_size == 0: 
            continue
        
        orig_mb = orig_size / (1024 * 1024)
        meth_name = "zstd"
        
        # 遍历指定的线程数进行测试
        for threads in THREADS_TO_TEST:
            comp_size, comp_time, decomp_time = stream_test_method(files, meth_name, threads=threads)
            
            ratio = (comp_size / orig_size) * 100
            comp_speed = orig_mb / comp_time if comp_time > 0 else 0.0
            decomp_speed = orig_mb / decomp_time if decomp_time > 0 else 0.0
            
            print(f"{model_label:<35} | {meth_name:<8} | {threads:<7} | {orig_mb:>8.1f} | {ratio:>8.2f}% | {comp_speed:>13.2f} | {decomp_speed:>13.2f}")
            
            res_key = f"zstd_threads_{threads}"
            aggregated_results[res_key][model_label] = {
                "OrigMB": round(orig_mb, 2),
                "CompRatio": round(ratio, 2),
                "CompSpeed": round(comp_speed, 2),
                "DecompSpeed": round(decomp_speed, 2)
            }
            
            gc.collect()

    print("\n" + "="*40)
    print("AGGREGATED RESULTS BY THREAD COUNT")
    print("="*40)
    
    for config, results in aggregated_results.items():
        print(f"\nConfiguration: {config}")
        print(json.dumps(results, indent=4))

if __name__ == "__main__":
    run_benchmark()