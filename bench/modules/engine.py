import time
import torch
import numpy as np
import pforex_cpp
import logging
import gc
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoModel
from utils import get_np_view, force_cleanup
from accelerate.utils import set_module_tensor_to_device

logger = logging.getLogger(__name__)

def get_nested_tensor(model, name):
    parts = name.split('.')
    curr = model
    try:
        for part in parts:
            curr = getattr(curr, part)
        return curr
    except AttributeError:
        return None
    
def get_parent_module_and_name(model, full_name):
    parts = full_name.split('.')
    curr = model
    for i in range(len(parts) - 1):
        part = parts[i]
        if not hasattr(curr, part):
            continue 
        curr = getattr(curr, part)
    
    if hasattr(curr, parts[-1]):
        return curr, parts[-1]
    return None, None   

class CompressionEngine:
    def __init__(self, cfg, native_dtype, num_workers=16):
        self.cfg = cfg
        self.native_dtype = native_dtype
        self.num_workers = num_workers
        self.BaseLoader = AutoModelForCausalLM if "LLM" in cfg['task_type'] else AutoModel
        # 新增：用于记录原始模型大小，方便计算 MB/s
        self.stats_orig_bytes = 0 

    def _compress_worker(self, args):
        key, ft_model, base_model, rate, native_dtype = args
        
        # 1. 动态加载 FT 数据
        p_ft, n_ft = get_parent_module_and_name(ft_model, key)
        if p_ft is None: return None
        
        if hasattr(p_ft, "_hf_hook"): p_ft._hf_hook.pre_forward(p_ft)
        t_ft = getattr(p_ft, n_ft)
        
        # 优化：立即转移到 CPU 并获取 numpy 视图，断开与模型的联系
        if t_ft.device.type != 'cpu':
            ft_cpu = t_ft.detach().cpu().contiguous()
        else:
            ft_cpu = t_ft.detach().contiguous()
        
            
        is_bf16 = "bfloat16" in str(t_ft.dtype)
        ft_np = get_np_view(ft_cpu)
        orig_bytes = ft_np.nbytes
        current_shape = t_ft.shape

        if hasattr(p_ft, "_hf_hook"): p_ft._hf_hook.post_forward(p_ft, None)

        # 2. 动态加载 Base 数据
        p_base, n_base = get_parent_module_and_name(base_model, key)
        if p_base is not None:
            if hasattr(p_base, "_hf_hook"): p_base._hf_hook.pre_forward(p_base)
            t_base = getattr(p_base, n_base)
            # 优化：Base 也立即转 CPU
            if t_base.device.type != 'cpu':
                base_cpu = t_base.detach().cpu().contiguous()
            else:
                base_cpu = t_base.detach().contiguous()
                
            base_np = get_np_view(base_cpu) if base_cpu.shape == ft_cpu.shape else np.zeros_like(ft_np)
            if hasattr(p_base, "_hf_hook"): p_base._hf_hook.post_forward(p_base, None)
        else:
            base_np = np.zeros_like(ft_np)

        # 3. 压缩内核
        t_start = time.perf_counter()
        try:
            c_bytes = pforex_cpp.compress_layer(base_np, ft_np, rate, is_bf16)
        except:
            c_bytes = pforex_cpp.compress_layer(np.zeros_like(ft_np), ft_np, rate, is_bf16)
        duration = time.perf_counter() - t_start

        return key, c_bytes, orig_bytes, len(c_bytes), duration, current_shape

    def _decompress_worker(self, args):
        key, shape, c_bytes, base_model, is_llm_task, target_dtype = args
        
        # 1. Worker 内部触发 Base 加载
        # p_base, n_base = get_parent_module_and_name(base_model, key)
        # if p_base is None and "." in key:
        #     p_base, n_base = get_parent_module_and_name(base_model, key.split(".", 1)[1])
        
        # if p_base is not None:
        #     if hasattr(p_base, "_hf_hook"):
        #         p_base._hf_hook.pre_forward(p_base)
            
        #     t_base = getattr(p_base, n_base)
        #     # 优化：立即转 CPU numpy 视图
        #     if t_base.device.type != 'cpu':
        #         base_cpu = t_base.detach().cpu().contiguous()
        #     else:
        #         base_cpu = t_base.detach().contiguous()
        #     base_np = get_np_view(base_cpu)
        # else:
        #     is_half = (target_dtype == torch.float16 or target_dtype == torch.bfloat16)
        #     dtype = np.uint16 if is_half else np.uint32
        #     base_np = np.zeros(shape, dtype=dtype)

        # 1. Worker 内部触发 Base 加载
        p_base, n_base = get_parent_module_and_name(base_model, key)
        if p_base is None and "." in key:
            p_base, n_base = get_parent_module_and_name(base_model, key.split(".", 1)[1])
        
        # 定义一个 helper 变量来标记是否使用了真实的 base
        used_real_base = False

        if p_base is not None:
            if hasattr(p_base, "_hf_hook"):
                p_base._hf_hook.pre_forward(p_base)
            
            t_base = getattr(p_base, n_base)
            
            # [FIX] 增加形状检查！必须与压缩时的逻辑（mismatch -> zeros）保持一致
            if tuple(t_base.shape) == tuple(shape):
                # 优化：立即转 CPU numpy 视图
                if t_base.device.type != 'cpu':
                    base_cpu = t_base.detach().cpu().contiguous()
                else:
                    base_cpu = t_base.detach().contiguous()
                base_np = get_np_view(base_cpu)
                used_real_base = True
            else:
                # 如果形状不匹配（如 Llama3 128256 vs 128258），这里的 Base 不能用
                # 压缩时这种情况使用了全0，所以解压也要用全0
                pass 
        
        if not used_real_base:
            # Fallback: 如果没有找到 Base 或者形状不匹配，使用全 0
            is_half = (target_dtype == torch.float16 or target_dtype == torch.bfloat16)
            dtype = np.uint16 if is_half else np.uint32
            base_np = np.zeros(shape, dtype=dtype)

        # 2. 核心算法计时
        t_start = time.perf_counter()
        rec_uint = pforex_cpp.decompress_layer(c_bytes, base_np)
        
        view_dtype = np.float32 if base_np.itemsize == 4 else np.float16
        rec_np = rec_uint.view(view_dtype)
        tens = torch.from_numpy(rec_np).reshape(shape).clone() # clone 是必须的，因为 rec_np 即将释放
        duration = time.perf_counter() - t_start
        
        # 3. 释放 Base 引用
        if p_base is not None and hasattr(p_base, "_hf_hook"):
            p_base._hf_hook.post_forward(p_base, None)
            
        if target_dtype == torch.bfloat16: tens = tens.view(torch.bfloat16)
        elif target_dtype == torch.float16: tens = tens.view(torch.float16)
        
        return key, tens, duration
    
    def _write_chunk(self, f, key, shape, c_bytes):
        # Format:
        # Key Len (4 bytes) | Key Bytes | N_Dim (4 bytes) | Shape (N_Dim * 4 bytes) | Data Len (8 bytes) | Data Bytes
        key_bytes = key.encode('utf-8')
        f.write(struct.pack('I', len(key_bytes)))
        f.write(key_bytes)
        
        f.write(struct.pack('I', len(shape)))
        for dim in shape:
            f.write(struct.pack('I', dim))
            
        f.write(struct.pack('Q', len(c_bytes))) # Use Q for unsigned long long (64bit)
        f.write(c_bytes)

    def _read_stream(self, filename):
        with open(filename, 'rb') as f:
            while True:
                # Read Key Len
                buf = f.read(4)
                if not buf: break
                key_len = struct.unpack('I', buf)[0]
                
                # Read Key
                key = f.read(key_len).decode('utf-8')
                
                # Read Shape
                ndim = struct.unpack('I', f.read(4))[0]
                shape = []
                for _ in range(ndim):
                    shape.append(struct.unpack('I', f.read(4))[0])
                shape = tuple(shape)
                
                # Read Data Len
                data_len = struct.unpack('Q', f.read(8))[0]
                
                # Read Data
                c_bytes = f.read(data_len)
                
                yield key, shape, c_bytes

    def run_pipeline(self, rate, ModelClass, cpu_kwargs):
        comp_kwargs = cpu_kwargs.copy()
        comp_kwargs["device_map"] = "cpu"
        # --- Prepare ---
        # base_native = self.BaseLoader.from_pretrained(self.cfg['base_model'], **comp_kwargs)
        # ft_model = ModelClass.from_pretrained(self.cfg['ft_model'], **comp_kwargs)
        base_native = self.BaseLoader.from_pretrained(self.cfg['base_model'], **comp_kwargs)
        ft_model = ModelClass.from_pretrained(self.cfg['ft_model'], **comp_kwargs)
        
        # [核心修复] 主动扫描：检查是否有静默残留的 meta tensor
        def has_meta(m):
            for p in m.parameters():
                if p.device.type == "meta": return True
            return False

        if has_meta(base_native) or has_meta(ft_model):
            logger.warning("[WORKAROUND] 检测到残留的 meta tensor (GPT-2/BERT常见Bug)。使用安全模式重新加载...")
            del base_native
            del ft_model
            force_cleanup()
            
            # 剔除引发 Bug 的 accelerate 加速参数，使用原生 PyTorch 加载
            safe_kwargs = {k: v for k, v in comp_kwargs.items() if k not in ["device_map", "low_cpu_mem_usage", "offload_folder"]}
            base_native = self.BaseLoader.from_pretrained(self.cfg['base_model'], **safe_kwargs)
            ft_model = ModelClass.from_pretrained(self.cfg['ft_model'], **safe_kwargs)
        
        ft_keys = []
        for n, _ in ft_model.named_parameters(): ft_keys.append(n)
        for name, module in ft_model.named_modules():
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if buffer_name not in module._non_persistent_buffers_set:
                    full_name = f"{name}.{buffer_name}" if name else buffer_name
                    ft_keys.append(full_name)
        
        # ft_shapes = {}
        
        # --- Compression ---
        stats = {"orig": 0, "comp": 0}
        total_kernel_time = 0.0
        
        output_file = "model_compressed.bin"
        with open(output_file, "wb") as f_out:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_key = {}
                
                for key in ft_keys:
                    # 使用 comp_kwargs 加载的模型都在 CPU/Disk 上，此处安全
                    future = executor.submit(self._compress_worker, (key, ft_model, base_native, rate, self.native_dtype))
                    future_to_key[future] = key
                    
                    if len(future_to_key) > self.num_workers * 2:
                        for future in as_completed(list(future_to_key.keys())):
                            k, cb, ob, cmb, dur, shape = future.result()
                            stats["orig"] += ob
                            stats["comp"] += cmb
                            total_kernel_time += dur
                            
                            self._write_chunk(f_out, k, shape, cb)
                            del cb 
                            
                            del future_to_key[future]
                            if len(future_to_key) <= self.num_workers: break

                for future in as_completed(future_to_key):
                    k, cb, ob, cmb, dur, shape = future.result()
                    stats["orig"] += ob
                    stats["comp"] += cmb
                    total_kernel_time += dur
                    self._write_chunk(f_out, k, shape, cb)
                    del cb

        t_comp = total_kernel_time / self.num_workers
        self.stats_orig_bytes = stats["orig"]
        
        del ft_model
        force_cleanup()
        gc.collect()

        # --- Decompression (Generator) ---
        # loaded_store = torch.load("model_compressed.pforex")
        stats_tracker = {"time": 0.0} # 必须是一个引用对象，以便在生成器内修改

        def result_generator():
            # [Fix 1] nonlocal 必须放在第一行！
            nonlocal base_native 
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_key = {}
                # keys_list = list(ft_shapes.keys())
                
                stream_reader = self._read_stream(output_file)
                
                for key, shape, c_bytes in stream_reader:
                    # 提交任务
                    future = executor.submit(
                        self._decompress_worker, 
                        (key, shape, c_bytes, base_native, "LLM" in self.cfg['task_type'], self.native_dtype)
                    )
                    future_to_key[future] = key

                    # [关键] 流控：防止读取速度过快撑爆内存
                    # 只有当正在处理的任务少于 2*workers 时才继续读取
                    if len(future_to_key) >= self.num_workers * 2:
                        # 等待至少一个任务完成
                        done_future = next(iter(as_completed(future_to_key)))
                        k, tens, dur = done_future.result()
                        stats_tracker["time"] += dur
                        del future_to_key[done_future]
                        yield k, tens
                        
                        # 额外检查：如果还有很多完成的，也一并yield出去
                        # (简单起见，这里依赖下一次循环或最后的 drain)

                # 处理剩余所有任务
                for future in as_completed(future_to_key):
                    k, tens, dur = future.result()
                    stats_tracker["time"] += dur
                    yield k, tens
            
            del base_native
            force_cleanup()

        return result_generator(), (stats["comp"]/stats["orig"]*100) if stats["orig"] > 0 else 0, (stats["orig"]/1024**2/t_comp) if t_comp > 0 else 0, stats_tracker


    # def run_pipeline(self, rate, ModelClass, cpu_kwargs):
    #     comp_kwargs = cpu_kwargs.copy()
    #     comp_kwargs["device_map"] = "cpu"
    #     # --- Prepare ---
    #     # base_native = self.BaseLoader.from_pretrained(self.cfg['base_model'], **comp_kwargs)
    #     # ft_model = ModelClass.from_pretrained(self.cfg['ft_model'], **comp_kwargs)
    #     base_native = self.BaseLoader.from_pretrained(self.cfg['base_model'], **comp_kwargs)
    #     ft_model = ModelClass.from_pretrained(self.cfg['ft_model'], **comp_kwargs)
        
    #     # [核心修复] 主动扫描：检查是否有静默残留的 meta tensor
    #     def has_meta(m):
    #         for p in m.parameters():
    #             if p.device.type == "meta": return True
    #         return False

    #     if has_meta(base_native) or has_meta(ft_model):
    #         logger.warning("[WORKAROUND] 检测到残留的 meta tensor (GPT-2/BERT常见Bug)。使用安全模式重新加载...")
    #         del base_native
    #         del ft_model
    #         force_cleanup()
            
    #         # 剔除引发 Bug 的 accelerate 加速参数，使用原生 PyTorch 加载
    #         safe_kwargs = {k: v for k, v in comp_kwargs.items() if k not in ["device_map", "low_cpu_mem_usage", "offload_folder"]}
    #         base_native = self.BaseLoader.from_pretrained(self.cfg['base_model'], **safe_kwargs)
    #         ft_model = ModelClass.from_pretrained(self.cfg['ft_model'], **safe_kwargs)
        
    #     ft_keys = []
    #     for n, _ in ft_model.named_parameters(): ft_keys.append(n)
    #     for name, module in ft_model.named_modules():
    #         for buffer_name, buffer in module.named_buffers(recurse=False):
    #             if buffer_name not in module._non_persistent_buffers_set:
    #                 full_name = f"{name}.{buffer_name}" if name else buffer_name
    #                 ft_keys.append(full_name)
        
    #     # ft_shapes = {}
        
    #     # --- Compression ---
    #     stats = {"orig": 0, "comp": 0}
    #     total_kernel_time = 0.0
        
    #     output_file = "model_compressed.bin"
    #     with open(output_file, "wb") as f_out:
    #         with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
    #             future_to_key = {}
                
    #             for key in ft_keys:
    #                 # 使用 comp_kwargs 加载的模型都在 CPU/Disk 上，此处安全
    #                 future = executor.submit(self._compress_worker, (key, ft_model, base_native, rate, self.native_dtype))
    #                 future_to_key[future] = key
                    
    #                 if len(future_to_key) > self.num_workers * 2:
    #                     for future in as_completed(list(future_to_key.keys())):
    #                         k, cb, ob, cmb, dur, shape = future.result()
    #                         stats["orig"] += ob
    #                         stats["comp"] += cmb
    #                         total_kernel_time += dur
                            
    #                         self._write_chunk(f_out, k, shape, cb)
    #                         del cb 
                            
    #                         del future_to_key[future]
    #                         if len(future_to_key) <= self.num_workers: break

    #             for future in as_completed(future_to_key):
    #                 k, cb, ob, cmb, dur, shape = future.result()
    #                 stats["orig"] += ob
    #                 stats["comp"] += cmb
    #                 total_kernel_time += dur
    #                 self._write_chunk(f_out, k, shape, cb)
    #                 del cb

    #     t_comp = total_kernel_time / self.num_workers
    #     self.stats_orig_bytes = stats["orig"]
        
    #     del ft_model
    #     force_cleanup()
    #     gc.collect()

    #     # --- Decompression (Generator) ---
    #     # loaded_store = torch.load("model_compressed.pforex")
    #     stats_tracker = {"time": 0.0} # 必须是一个引用对象，以便在生成器内修改

    #     def result_generator():
    #         # [Fix 1] nonlocal 必须放在第一行！
    #         nonlocal base_native 
            
    #         with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
    #             future_to_key = {}
    #             # keys_list = list(ft_shapes.keys())
                
    #             stream_reader = self._read_stream(output_file)
                
    #             for key, shape, c_bytes in stream_reader:
    #                 # 提交任务
    #                 future = executor.submit(
    #                     self._decompress_worker, 
    #                     (key, shape, c_bytes, base_native, "LLM" in self.cfg['task_type'], self.native_dtype)
    #                 )
    #                 future_to_key[future] = key

    #                 # [关键] 流控：防止读取速度过快撑爆内存
    #                 # 只有当正在处理的任务少于 2*workers 时才继续读取
    #                 if len(future_to_key) >= self.num_workers * 2:
    #                     # 等待至少一个任务完成
    #                     done_future = next(iter(as_completed(future_to_key)))
    #                     k, tens, dur = done_future.result()
    #                     stats_tracker["time"] += dur
    #                     del future_to_key[done_future]
    #                     yield k, tens
                        
    #                     # 额外检查：如果还有很多完成的，也一并yield出去
    #                     # (简单起见，这里依赖下一次循环或最后的 drain)

    #             # 处理剩余所有任务
    #             for future in as_completed(future_to_key):
    #                 k, tens, dur = future.result()
    #                 stats_tracker["time"] += dur
    #                 yield k, tens
            
    #         del base_native
    #         force_cleanup()

    #     return result_generator(), (stats["comp"]/stats["orig"]*100) if stats["orig"] > 0 else 0, (stats["orig"]/1024**2/t_comp) if t_comp > 0 else 0, stats_tracker

