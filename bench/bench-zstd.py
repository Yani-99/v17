import os
import sys
import time
import torch
import numpy as np
import logging
import gc
import zstandard as zstd
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# 禁用进度条等
os.environ["TQDM_DISABLE"] = "1"              
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 配置路径以导入你的模块
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "build"))
sys.path.append(os.path.join(base_path, "bench/modules"))

from configs import TEST_CONFIGS
from loader import ModelManager
from engine import CompressionEngine, get_parent_module_and_name
from utils import get_np_view, force_cleanup

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ======================================================================
# 核心：继承你的 CompressionEngine，仅替换压缩/解压内核为 zstd
# ======================================================================
class ZstdCompressionEngine(CompressionEngine):
    def __init__(self, cfg, native_dtype, num_workers=16, zstd_level=3):
        super().__init__(cfg, native_dtype, num_workers)
        self.zstd_level = zstd_level

    def _compress_worker(self, args):
        key, ft_model, base_model, rate, native_dtype = args
        
        # --- 1. 完全保留原有的 FT 数据加载逻辑 ---
        p_ft, n_ft = get_parent_module_and_name(ft_model, key)
        if p_ft is None: return None
        
        self._safe_load_hook(ft_model, p_ft, n_ft)
        t_ft = getattr(p_ft, n_ft)
        
        if t_ft.device.type == 'meta':
            ft_cpu = torch.zeros(t_ft.shape, dtype=torch.float32)
        elif t_ft.device.type != 'cpu':
            ft_cpu = t_ft.detach().cpu().contiguous()
        else:
            ft_cpu = t_ft.detach().contiguous()
            
        ft_np = get_np_view(ft_cpu)
        orig_bytes = ft_np.nbytes
        current_shape = t_ft.shape
        self._safe_unload_hook(p_ft)

        # --- 2. 完全保留原有的 Base 数据加载逻辑 (保证 I/O 开销与 pforex 完全一致) ---
        p_base, n_base = get_parent_module_and_name(base_model, key)
        if p_base is not None:
            self._safe_load_hook(base_model, p_base, n_base)
            t_base = getattr(p_base, n_base)
            if t_base.device.type == 'meta':
                base_cpu = torch.zeros(t_base.shape, dtype=ft_cpu.dtype)
            elif t_base.device.type != 'cpu':
                base_cpu = t_base.detach().cpu().contiguous()
            else:
                base_cpu = t_base.detach().contiguous()
            base_np = get_np_view(base_cpu) if base_cpu.shape == ft_cpu.shape else np.zeros_like(ft_np)
            self._safe_unload_hook(p_base)
        else:
            base_np = np.zeros_like(ft_np)

        # --- 3. 替换为 zstd 压缩内核 ---
        t_start = time.perf_counter()
        
        # 为了公平，将 FT 张量的字节流直接喂给 zstd。
        # (ZstdCompressor 是单线程的，多线程由外部 ThreadPoolExecutor 控制，与你的 pforex 机制完全相同)
        ft_bytes = ft_np.tobytes()
        compressor = zstd.ZstdCompressor(level=self.zstd_level, threads=1)
        c_bytes = compressor.compress(ft_bytes)
        
        duration = time.perf_counter() - t_start

        return key, c_bytes, orig_bytes, len(c_bytes), duration, current_shape

    def _decompress_worker(self, args):
        key, shape, c_bytes, base_model, is_llm_task, target_dtype = args
        
        # --- 1. 完全保留原有的 Base 加载逻辑 (保持 I/O 负担一致) ---
        p_base, n_base = get_parent_module_and_name(base_model, key)
        if p_base is None and "." in key:
            p_base, n_base = get_parent_module_and_name(base_model, key.split(".", 1)[1])
        
        used_real_base = False
        if p_base is not None:
            self._safe_load_hook(base_model, p_base, n_base)
            t_base = getattr(p_base, n_base)
            if tuple(t_base.shape) == tuple(shape):
                if t_base.device.type != 'meta':
                    if t_base.device.type != 'cpu':
                        base_cpu = t_base.detach().cpu().contiguous()
                    else:
                        base_cpu = t_base.detach().contiguous()
                    used_real_base = True
            self._safe_unload_hook(p_base)

        # --- 2. 替换为 zstd 解压内核 ---
        t_start = time.perf_counter()
        
        decompressor = zstd.ZstdDecompressor()
        d_bytes = decompressor.decompress(c_bytes)
        
        # 将字节流安全转回 Tensor
        view_dtype = np.uint32 if target_dtype == torch.float32 else np.uint16
        rec_np = np.frombuffer(d_bytes, dtype=view_dtype)
        tens = torch.from_numpy(rec_np).reshape(shape).clone()
        
        duration = time.perf_counter() - t_start
            
        if target_dtype == torch.bfloat16: tens = tens.view(torch.bfloat16)
        elif target_dtype == torch.float16: tens = tens.view(torch.float16)
        elif target_dtype == torch.float32: tens = tens.view(torch.float32)
        
        return key, tens, duration


# ======================================================================
# 运行扩展性对比主流程
# ======================================================================
def main():
    THREADS_TO_TEST = [1, 2, 4, 8, 16, 32, 48]
    ACTIVE_CONFIG = "llama3.1-8b-1"
    cfg = TEST_CONFIGS[ACTIVE_CONFIG]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"=== ZSTD MULTI-THREADING SCALABILITY BENCHMARK (Project I/O) ===")
    logger.info(f"Model: {ACTIVE_CONFIG} | Task: {cfg['task_type']}")
    logger.info(f"{'='*60}\n")

    native_dtype = ModelManager.detect_native_dtype(cfg)
    ModelClass, _, _ = ModelManager.get_classes_and_kwargs(cfg)
    
    # 【重点】我们移除了 offload_folder 避免触发引擎底层的 min(num_workers, 4) 锁
    cpu_kwargs = {
        "low_cpu_mem_usage": True,
        "device_map": "cpu"
    }
    if "LLM" in cfg['task_type']: 
        cpu_kwargs["torch_dtype"] = native_dtype

    results_table = []

    for threads in THREADS_TO_TEST:
        logger.info(f"\n>>> Running with {threads} threads...")
        
        # 实例化我们定制的 Zstd 引擎
        engine = ZstdCompressionEngine(cfg, native_dtype, num_workers=threads, zstd_level=3)
        
        try:
            # 执行管线 (lossy rate 在纯速度测试中给 0.0 即可)
            recovered_gen, comp_pct, c_speed, stats_tracker = engine.run_pipeline(0.0, ModelClass, cpu_kwargs)
            
            # 极速消耗解压生成器，驱动所有多线程执行完毕
            deque(recovered_gen, maxlen=0)
            
            # 计算解压速度
            t_decomp_total = stats_tracker["time"] / threads
            d_speed = (engine.stats_orig_bytes / 1024**2 / t_decomp_total) if t_decomp_total > 0 else 0.0
            
            # 打印当前日志
            print(f"  -> Threads: {threads:<2} | Size: {comp_pct:.2f}% | Comp: {c_speed:.1f} MB/s | Decomp: {d_speed:.1f} MB/s")
            
            results_table.append({
                "Threads": threads,
                "CompRatio": comp_pct,
                "CompSpeed": c_speed,
                "DecompSpeed": d_speed
            })
            
        except Exception as e:
            logger.error(f"Error at {threads} threads: {e}")
            import traceback
            traceback.print_exc()

        force_cleanup()
        gc.collect()

    # 打印最终聚合表格
    print("\n" + "="*80)
    print(f" FINAL ZSTD SCALABILITY RESULTS (Using Engine I/O Logic) ")
    print("="*80)
    header = f"{'Threads':<8} | {'Comp %':<10} | {'C-Speed(MB/s)':<15} | {'D-Speed(MB/s)':<15}"
    print(header)
    print("-" * len(header))
    for res in results_table:
        print(f"{res['Threads']:<8} | {res['CompRatio']:<9.2f}% | {res['CompSpeed']:<15.1f} | {res['DecompSpeed']:<15.1f}")
    print("="*80)

if __name__ == "__main__":
    main()