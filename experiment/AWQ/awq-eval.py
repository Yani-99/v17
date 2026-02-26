import os
import sys
import json
import time
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# 动态添加你项目中的 bench/modules 到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bench/modules")))
from evaluator import UniversalEvaluator

logger = logging.getLogger(__name__)

# ======================================================================
# [核心魔法] 通过继承重写，不修改 v17 原有代码，只在这里修复 batch_size 和 截断问题
# ======================================================================
class AWQEvaluator(UniversalEvaluator):
    def _run_lm_harness(self):
        logger.info(f"Preparing LM-Eval Harness: {self.cfg['llm_tasks']}...")
        
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        
        # 1. Detect Multi-GPU Sharding
        eval_device = self.device
        is_multi_gpu = False
        if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
            logger.info("Detected multi-GPU model sharding (Accelerate).")
            eval_device = None 
            is_multi_gpu = True

        eval_batch_size = self.cfg.get("batch_size", 1)
        
        # 只要我们在这里给 processor 设置了 pad_token，lm_eval 底层自己就会正确读取，
        # 我们不需要（也不能）在 generate 时重复传入！
        if self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token
            self.processor.pad_token_id = self.processor.eos_token_id

        lm_obj = HFLM(
            pretrained=self.model, 
            tokenizer=self.processor, 
            batch_size=eval_batch_size, 
            device=eval_device 
        )

        first_device = self.model.device 
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
             first_device = self.model.model.embed_tokens.weight.device
             
        original_model_generate = lm_obj._model_generate

        # 【核心修复】：移除重复的 pad_token_id，并通过计算绝对长度来安全截断
        def patched_model_generate(context, max_length, stop, **generation_kwargs):
            if isinstance(context, torch.Tensor):
                context = context.to(first_device)
            
            # 读取我们配置的最大生成长度
            forced_max_new_tokens = self.cfg.get("gen_kwargs", {}).get("max_new_tokens", 512)
            
            # 计算当前 prompt 的输入长度
            input_seq_len = context.shape[1] if isinstance(context, torch.Tensor) else len(context[0])
            
            # 修改 max_length: 取 (模型原生上限) 和 (输入长度 + 我们想要生成的长度) 的较小值
            safe_max_length = min(max_length, input_seq_len + forced_max_new_tokens)
            
            # 确保我们没有把冲突参数传进去
            generation_kwargs.pop("max_new_tokens", None)
            generation_kwargs.pop("pad_token_id", None)
            
            res = original_model_generate(context, safe_max_length, stop, **generation_kwargs)
            
            if is_multi_gpu and isinstance(res, torch.Tensor):
                return res.to(first_device)
            return res

        lm_obj._model_generate = patched_model_generate

        limit = self.limit if self.limit else None
        max_retries = 3
        
        raw_results = {}
        for attempt in range(max_retries):
            try:
                logger.info(f"Running LM-Eval (Attempt {attempt+1}/{max_retries})...")
                if is_multi_gpu: torch.cuda.empty_cache()
                
                raw_results = lm_eval.simple_evaluate(
                    model=lm_obj, 
                    tasks=self.cfg['llm_tasks'], 
                    device=eval_device, 
                    limit=limit, 
                    log_samples=False
                )
                break 
            except Exception as e:
                logger.warning(f"LM-Eval Error: {e}")
                import traceback
                traceback.print_exc()
                
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 10
                    logger.info(f"Retry in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error("LM-Eval Failed after max retries.")
                    return {}
        
        metrics = {}
        if "groups" in raw_results:
            for group_name, res in raw_results["groups"].items():
                if "acc" in res: metrics[f"{group_name}/acc"] = res["acc"]
                if "acc,none" in res: metrics[f"{group_name}/acc"] = res["acc,none"]

        if "results" in raw_results:
            mmlu_scores = []
            gsm8k_candidates = []

            for task, res in raw_results["results"].items():
                if task.startswith("mmlu"):
                    val = res.get("acc") or res.get("acc,none")
                    if val is not None: mmlu_scores.append(val)
                
                if "gsm8k" in task.lower():
                    val = None
                    priorities = ["strict_match,none", "exact_match,none", "acc,none", 
                                  "strict_match", "exact_match", "acc"]
                    for p in priorities:
                        if p in res:
                            val = res[p]
                            break
                    if val is None:
                        for k, v in res.items():
                            if isinstance(v, (int, float)) and ("acc" in k or "match" in k):
                                val = v
                                break
                    if val is not None: gsm8k_candidates.append(val)

            if mmlu_scores and "mmlu/acc" not in metrics:
                metrics["mmlu/acc"] = sum(mmlu_scores) / len(mmlu_scores)
            if gsm8k_candidates and "gsm8k/acc" not in metrics:
                metrics["gsm8k/acc"] = gsm8k_candidates[0]

        return metrics


def evaluate_awq_model():
    model_path = "/home/newdrive2/liu4441/Llama-3.1-8B-Instruct-awq-4bit"
    
    print(f"[*] 开始加载 Tokenizer: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"[*] 开始加载 AWQ 模型 (强制锁死单卡和 FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )
    
    cfg = {
        "task_type": "LLM_HARNESS",
        "ft_model": "meta-llama/Llama-3.1-8B-Instruct", 
        "llm_tasks": ["mmlu", "gsm8k", "ifeval"], 
        "run_mt_bench": True,
        "eval_limit": None,
        "batch_size": 32, # 现在这里的 batch_size 能够真正生效了
        "gen_kwargs": {"max_new_tokens": 512} # 防止输出死循环，截断长输出
    }
    
    print(f"[*] 开始使用项目的 AWQEvaluator 进行对齐评测...")
    print(f"[*] 评测任务: {cfg['llm_tasks']}")
    
    # 【注意看这里】实例化我们刚刚写的重写子类 AWQEvaluator，而不是 UniversalEvaluator
    evaluator = AWQEvaluator(
        model=model, 
        processor=tokenizer, 
        config=cfg, 
        run_tag="awq-4bit-baseline", 
        device="cuda"
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