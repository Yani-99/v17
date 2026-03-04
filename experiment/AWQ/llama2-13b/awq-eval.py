import os
import sys
import json
import time
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# 动态添加项目中的 bench/modules 到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../bench/modules")))
from evaluator import UniversalEvaluator

# 配置基础的日志输出，方便在终端查看评测进度
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def evaluate_awq_model():
    model_path = "/home/newdrive2/liu4441/meta-llama/Llama-2-13b-chat-hf-awq"
    
    print(f"[*] 开始加载 Tokenizer: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 强制左侧填充以支持 Batching，这对提升评测速度至关重要
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    print(f"[*] 开始加载 AWQ 模型 (强制锁死单卡和 FP16，开启 Flash Attention 2)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )
    
    cfg = {
        "task_type": "LLM_HARNESS",
        "ft_model": "meta-llama/Llama-2-13b-chat-hf", 
        "llm_tasks": ["mmlu", "gsm8k", "ifeval"], 
        "run_mt_bench": True,
        "eval_limit": None
        # gen_kwargs 已移除，因为 UniversalEvaluator 内部已有完善的截断与限长机制
    }
    
    print(f"[*] 开始使用项目的 UniversalEvaluator 进行对齐评测...")
    print(f"[*] 评测任务: {cfg['llm_tasks']}")
    
    # 直接使用 UniversalEvaluator 即可完美兼容 AWQ
    evaluator = UniversalEvaluator(
        model=model, 
        processor=tokenizer, 
        config=cfg, 
        run_tag="awq-4bit-baseline"
    )
    
    metrics = evaluator.run()
    
    print("\n" + "="*60)
    print(" 📊 AWQ 4-bit 模型评测结果 (对齐项目标准)")
    print("="*60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f" {k:<20}: {v:.4f}")
        else:
            print(f" {k:<20}: {v}")
    print("="*60)

if __name__ == "__main__":
    evaluate_awq_model()