import os
import sys
import argparse
import time
import torch
import numpy as np
import logging
import warnings
import gc
import json
from concurrent.futures import ThreadPoolExecutor, as_completed




os.environ["TQDM_DISABLE"] = "1"              # Completely disable progress bars
os.environ["GIT_PYTHON_REFRESH"] = "quiet"    # Silence git errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    all_phys_cores = list(range(os.cpu_count())) 
    os.sched_setaffinity(0, all_phys_cores)
except Exception as e:
    print(f"[WARNING] 重置 Affinity 失败: {e}")


num_workers = 1
print(f"[CONFIG] 手动设定: 启动 {num_workers} 个并行线程")

from transformers import (
    AutoConfig, AutoModel, AutoModelForCausalLM, 
    AutoModelForSequenceClassification, AutoModelForTokenClassification,
    AutoModelForImageClassification, 
    AutoTokenizer, AutoImageProcessor,
    Trainer, TrainingArguments,
    DataCollatorForTokenClassification,
    logging as hf_logging
)
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "build"))
sys.path.append(os.path.join(base_path, "bench/modules"))
from datasets import load_dataset, logging as ds_logging
import evaluate 
import pforex_cpp

from configs import TEST_CONFIGS
from evaluator import UniversalEvaluator
from utils import get_np_view, force_cleanup
from loader import ModelManager
from engine import CompressionEngine





hf_logging.set_verbosity_error()
ds_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Configure root logger
logging.basicConfig(
    level=logging.INFO, 
    format='%(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

COMPRESSED_FILE = "model_compressed.pforex"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# LOSSY_RATES = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2]
# LOSSY_RATES = [0.0,0.01,0.05,0.1,0.15]
# LOSSY_RATES = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2,0.1,0.12,0.15]
# LOSSY_RATES = [0.18,0.2,0.25,0.3]
LOSSY_RATES = [0.0]

# LOSSY_RATES = [0.0,0.01,0.05,0.1,0.15]



# ACTIVE_CONFIG = "roberta_sst2" 
# ACTIVE_CONFIG = "roberta_mnli"
# ACTIVE_CONFIG = "bert_ner"
# ACTIVE_CONFIG = "vit_cifar10"
# ACTIVE_CONFIG = "gpt2_imdb" 
# ACTIVE_CONFIG = "vicuna"
# ACTIVE_CONFIG = "llama3_8b_instruct"
# ACTIVE_CONFIG = "mistral_7b_instruct"
# ACTIVE_CONFIG = "vicuna_13b"
# ACTIVE_CONFIG = "bert_ner1"
# ACTIVE_CONFIG = "llama3.1-1"
# ACTIVE_CONFIG = "llama3_70b"

# bert-ner
# CONFIG_LIST = ["bert_ner1","bert_ner2","bert_ner3","bert_ner4","bert_ner5","bert_ner6","bert_ner7","bert_ner8","bert_ner9","bert_ner10"]
# llama2-7b
# CONFIG_LIST = ["llama2-7b-1","llama2-7b-2","llama2-7b-3","llama2-7b-4","llama2-7b-5","llama2-7b-6","llama2-7b-7","llama2-7b-8","llama2-7b-9","llama2-7b-10"]
# llama3.1-8b
# CONFIG_LIST = ["llama3.1-8b-1","llama3.1-8b-2","llama3.1-8b-3","llama3.1-8b-4","llama3.1-8b-5","llama3.1-8b-6","llama3.1-8b-7","llama3.1-8b-8","llama3.1-8b-9","llama3.1-8b-10"]
# llama2-13b
# CONFIG_LIST = ["llama2-13b-1","llama2-13b-2","llama2-13b-3","llama2-13b-4", "llama2-13b-5","llama2-13b-6","llama2-13b-7","llama2-13b-8","llama2-13b-9","llama2-13b-10"]
# gpt2
# CONFIG_LIST = ["gpt2-1", "gpt2-2", "gpt2-3", "gpt2-4", "gpt2-5", "gpt2-6", "gpt2-7", "gpt2-8", "gpt2-9", "gpt2-10"]
# roberta-large
# CONFIG_LIST = ["roberta-large-1", "roberta-large-2", "roberta-large-3", "roberta-large-4", "roberta-large-5", "roberta-large-6", "roberta-large-7", "roberta-large-8", "roberta-large-9", "roberta-large-10"]
# roberta-base
# CONFIG_LIST = ["roberta-base-1", "roberta-base-2", "roberta-base-3", "roberta-base-4", "roberta-base-5", "roberta-base-6", "roberta-base-7", "roberta-base-8", "roberta-base-9", "roberta-base-10"]
# mistral-7b
# CONFIG_LIST = ["mistral-7b-1", "mistral-7b-2", "mistral-7b-3", "mistral-7b-4", "mistral-7b-5", "mistral-7b-6", "mistral-7b-7", "mistral-7b-8", "mistral-7b-9", "mistral-7b-10"]
# meta-llama/Llama-3.1-70B
# CONFIG_LIST = ["llama-3.1-70b-1", "llama-3.1-70b-2", "llama-3.1-70b-3", "llama-3.1-70b-4", "llama-3.1-70b-5", "llama-3.1-70b-6", "llama-3.1-70b-7", "llama-3.1-70b-8", "llama-3.1-70b-9", "llama-3.1-70b-10"]
CONFIG_LIST = ["llama3_70b"]


from engine import CompressionEngine
from loader import ModelManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    DO_EVAL = args.eval

    for ACTIVE_CONFIG in CONFIG_LIST:
        # print(f"\n{'#'*60}")
        # print(f"### STARTING BATCH FOR: {ACTIVE_CONFIG}")
        # print(f"{'#'*60}\n")

        try:
            cfg = TEST_CONFIGS[ACTIVE_CONFIG]
            logger.info(f"\n>>> Running Config: {ACTIVE_CONFIG} <<<")

            # 1. 初始化
            ft_config_obj = ModelManager.load_ft_config(cfg)
            native_dtype = ModelManager.detect_native_dtype(cfg)
            cpu_kwargs = {
                "low_cpu_mem_usage": True,
                "device_map": "auto", 
                "offload_folder": "tmp_offload"
            }
            if "LLM" in cfg['task_type']: cpu_kwargs["torch_dtype"] = native_dtype

            # # -------------------------------------------------------
            # # Phase -1: Source & Phase 0: Baseline (使用统一加载函数)
            # # -------------------------------------------------------
            # source_metrics, baseline_metrics = {}, {}
            # if DO_EVAL:
            #     # Phase -1
            #     logger.info("\n[Phase -1] Source Evaluation...")
            #     m, p = ModelManager.prepare_model_for_eval(cfg['base_model'], cfg, native_dtype, DEVICE, config_obj=ft_config_obj)
            #     source_metrics = UniversalEvaluator(m, p, cfg, "source", DEVICE).run()
            #     del m, p; force_cleanup()

            #     # Phase 0
            #     logger.info("\n[Phase 0] Baseline Evaluation...")
            #     m, p = ModelManager.prepare_model_for_eval(cfg['ft_model'], cfg, native_dtype, DEVICE)
            #     baseline_metrics = UniversalEvaluator(m, p, cfg, "baseline", DEVICE).run()
            #     del m, p; force_cleanup()

            # -------------------------------------------------------
            # Loop over Lossy Rates
            # -------------------------------------------------------
            results_table = []
            engine = CompressionEngine(cfg, native_dtype, num_workers=num_workers)
            ModelClass, _, _ = ModelManager.get_classes_and_kwargs(cfg)

            for rate in LOSSY_RATES:
                logger.info(f"\n[{'='*10} Rate: {rate} = Model: {ACTIVE_CONFIG}{'='*10}]")
                
                # 1. [FIX] 接收 stats_tracker 字典，而不是直接接收 d_speed
                recovered_gen, comp_pct, c_speed, stats_tracker = engine.run_pipeline(rate, ModelClass, cpu_kwargs)
                
                rec_metrics = {}
                d_speed = 0.0 # [FIX] 必须在此处初始化，防止 UnboundLocalError
                
                if DO_EVAL:
                    try:
                        # 评估模式：在模型加载过程中消耗生成器
                        m, p = ModelManager.prepare_model_for_eval(cfg['ft_model'], cfg, native_dtype, DEVICE, 
                                                            state_dict=recovered_gen, config_obj=ft_config_obj)
                        
                        # 此时生成器已跑完，统计时间
                        t_decomp_total = stats_tracker["time"] / num_workers 
                        d_speed = (engine.stats_orig_bytes / 1024**2 / t_decomp_total) if t_decomp_total > 0 else 0.0
                        
                        rec_metrics = UniversalEvaluator(m, p, cfg, str(rate), DEVICE).run()
                        del m, p; force_cleanup()
                    except Exception as e:
                        logger.error(f"Eval Error: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # 非评估模式：使用 deque 极速消耗生成器，仅为了触发解压并测量速度
                    try:
                        from collections import deque
                        # maxlen=0 的 deque 是消耗迭代器最高效的方法，它会驱动 engine 中的 workers 跑完
                        deque(recovered_gen, maxlen=0)
                        
                        # 此时生成器已跑完，统计时间
                        t_decomp_total = stats_tracker["time"] / num_workers 
                        d_speed = (engine.stats_orig_bytes / 1024**2 / t_decomp_total) if t_decomp_total > 0 else 0.0
                        
                    except Exception as e:
                        logger.error(f"Decompression Error: {e}")
                        import traceback
                        traceback.print_exc()

                results_table.append({"Rate": rate, "Comp%": comp_pct, "CompSpeed": c_speed, "DecompSpeed": d_speed, "Metrics": rec_metrics})
                # [FIX] 打印日志增加 Decomp Speed
                print(f"  -> Size: {comp_pct:.2f}% | Comp: {c_speed:.1f} MB/s | Decomp: {d_speed:.1f} MB/s | Metrics: {rec_metrics}")


            # # -------------------------------------------------------
            # # Final Summary Table
            # # -------------------------------------------------------
            # print("\n" + "="*100)
            # print(f" FINAL SUMMARY: {ACTIVE_CONFIG} ({cfg['task_type']}) ")
            # print("="*100)
            
            # # 定义表头
            # header = f"{'Rate':<8} | {'Comp %':<10} | {'C-Speed':<12} | {'D-Speed':<12} | {'Main Metric'}"
            # print(header)
            # print("-" * len(header))

            # for res in results_table:
            #     rate = res["Rate"]
            #     comp_pct = f"{res['Comp%']:.2f}%"
            #     c_speed = f"{res['CompSpeed']:.1f} MB/s"
            #     d_speed = f"{res['DecompSpeed']:.1f} MB/s"
                
            #     # 自动提取 Metrics 里的核心指标 (例如 f1, accuracy 或 loss)
            #     metrics = res["Metrics"]
            #     metric_str = ""
            #     if metrics:
            #         # 过滤掉 runtime 相关指标，只保留业务指标
            #         important_keys = [k for k in metrics.keys() if 'runtime' not in k and 'per_second' not in k and 'loss' not in k]
            #         # 如果有 f1 或 accuracy 优先显示，否则显示第一个指标
            #         display_keys = sorted(important_keys, key=lambda x: ('f1' in x or 'acc' in x), reverse=True)
            #         metric_str = ", ".join([f"{k}: {metrics[k]:.4f}" for k in display_keys[:2]]) # 取前两个重要指标
            #     else:
            #         metric_str = "N/A (No Eval)"

            #     print(f"{rate:<8} | {comp_pct:<10} | {c_speed:<12} | {d_speed:<12} | {metric_str}")
            
            # print("="*100)

            force_cleanup() 
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"!!! CRITICAL ERROR processing {ACTIVE_CONFIG} !!!")
            logger.error(e)
            import traceback
            traceback.print_exc()
            continue # 继续跑下一个配置

        
if __name__ == "__main__":
    main()