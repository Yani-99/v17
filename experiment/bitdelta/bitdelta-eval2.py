import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bench/modules")))
from evaluator import UniversalEvaluator

@torch.no_grad()
def load_diff(model, diff_file):
    print(f"[*] 正在读取全新生成的增量文件: {diff_file}")
    
    diff_dict = torch.load(diff_file, map_location="cpu", weights_only=False)

    for name, module in model.named_modules():
        if name + ".mask" in diff_dict:
            device = module.weight.device 
            coeff = diff_dict[name + ".coeff"].to(device)
            packed_mask = diff_dict[name + ".mask"].to(device)

            shape = packed_mask.shape
            packed_flat = packed_mask.reshape(-1, 1)
            powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8)
            unpacked_bool = (packed_flat & powers) > 0
            unpacked_bool = unpacked_bool.reshape(shape[0], shape[1] * 8)
            
            mask_float = unpacked_bool.to(module.weight.dtype) * 2.0 - 1.0
            weight = mask_float * coeff

            module.weight.add_(weight.T)
            
        elif name + ".weight" in diff_dict:
            device = module.weight.device
            module.weight = nn.Parameter(diff_dict[name + ".weight"].to(device).to(module.weight.dtype))

    model.config.vocab_size = model.lm_head.weight.size(0)
    print("[*] 动态注入完成，模型已准备就绪。")

def evaluate_bitdelta_model():
    diff_file = "/home/newdrive2/liu4441/bitdelta2/diff.pt"
    
    print(f"[*] 开始加载官方分词器。")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] 开始加载官方纯净基础模型。")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa"
    )

    load_diff(model, diff_file)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    if hasattr(model, "generation_config"):
        model.generation_config.eos_token_id = terminators
    
    cfg = {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-3.1-8B",
        "ft_model": "meta-llama/Llama-3.1-8B-Instruct", 
        "llm_tasks": ["mmlu", "gsm8k"],
        "run_mt_bench": True,
        "eval_limit": None,
        "batch_size": "32"
    }

    print(f"[*] 开始执行严谨的对齐评测。")
    
    evaluator = UniversalEvaluator(
        model=model, 
        processor=tokenizer, 
        config=cfg, 
        run_tag="bitdelta_honest_retest", 
        device="cuda"
    )
    
    metrics = evaluator.run()
    
    print("\n============================================================")
    print(" 模型客观评测真实结果")
    print("============================================================")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f" {k:<20}: {v:.4f}")
        else:
            print(f" {k:<20}: {v}")
    print("============================================================")

if __name__ == "__main__":
    evaluate_bitdelta_model()