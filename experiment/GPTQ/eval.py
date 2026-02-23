import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 动态添加你项目中的 bench/modules 到系统路径，以便直接复用你的评估代码
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bench/modules")))
from evaluator import UniversalEvaluator

def evaluate_gptq_model():
    # 你的 GPTQ 模型绝对路径
    model_path = "/home/newdrive2/liu4441/Llama-2-7b-chat-hf-gptq-4bit"
    
    print(f"[*] 开始加载 Tokenizer: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"[*] 开始加载 GPTQ 模型 (自动分配显存)...")
    # GPTQ 模型的加载只需要 device_map="auto" 即可，无需传 torch_dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto"
    )
    
    # ==========================================
    # 构造与你的 configs.py 完全一致的配置字典
    # ==========================================
    cfg = {
        "task_type": "LLM_HARNESS",
        "ft_model": "meta-llama/Llama-2-7b-chat-hf", # 保持你的原模型名称，以便可能存在的模板匹配
        "llm_tasks": ["mmlu", "gsm8k"],  # 这里写上你需要对比的任务，比如 ["mmlu"] 或 ["wikitext"]
        "run_mt_bench": True,           # 是否顺带跑你代码里的 MT-Bench
        "eval_limit": None               # 设为具体数字(如100)可用于快速 debug 测试，跑全量设为 None
    }
    
    print(f"[*] 开始使用项目的 UniversalEvaluator 进行对齐评测...")
    print(f"[*] 评测任务: {cfg['llm_tasks']}")
    
    # 实例化你自己的评测器
    evaluator = UniversalEvaluator(
        model=model, 
        processor=tokenizer, 
        config=cfg, 
        run_tag="gptq-4bit-baseline", 
        device="cuda"
    )
    
    # 运行评测，它会走你 _run_lm_harness() 里面的全部逻辑
    metrics = evaluator.run()
    
    # 打印最终你需要的、可以直接填到你论文表格里的结果
    print("\n" + "="*60)
    print(" 📊 GPTQ 4-bit 模型评测结果 (对齐项目标准)")
    print("="*60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f" {k:<20}: {v:.4f}")
        else:
            print(f" {k:<20}: {v}")
    print("="*60)

if __name__ == "__main__":
    evaluate_gptq_model()