import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# 动态添加项目中的 bench/modules 到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bench/modules")))
from evaluator import UniversalEvaluator

# ==========================================
# 核心魔法：直接在评测脚本中进行 1-bit 动态解包与合并
# ==========================================
@torch.no_grad()
def load_diff(model, diff_file):
    print(f"[*] 正在读取 1-bit 增量文件: {diff_file}")
    # 先加载到 CPU 内存，防止 GPU 显存尖峰
    diff_dict = torch.load(diff_file, map_location="cpu")

    # 遍历 Base 模型的每一层，将 1-bit 增量动态贴合上去
    for name, module in model.named_modules():
        if name + ".mask" in diff_dict:
            device = module.weight.device  # 获取当前层所在的 GPU 编号
            coeff = diff_dict[name + ".coeff"].to(device)
            packed_mask = diff_dict[name + ".mask"].to(device)

            # 动态解包 (Unpack)
            shape = packed_mask.shape
            packed_flat = packed_mask.reshape(-1, 1)
            powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8)
            unpacked_bool = (packed_flat & powers) > 0
            unpacked_bool = unpacked_bool.reshape(shape[0], shape[1] * 8)
            
            # 还原为浮点数并乘以缩放因子
            mask_float = unpacked_bool.to(module.weight.dtype) * 2.0 - 1.0
            weight = mask_float * coeff

            # 与 Base 权重矩阵合并
            module.weight.add_(weight.T)
            
        elif name + ".weight" in diff_dict:
            device = module.weight.device
            module.weight = nn.Parameter(diff_dict[name + ".weight"].to(device).to(module.weight.dtype))

    model.config.vocab_size = model.lm_head.weight.size(0)
    print("[*] 1-bit 权重动态注入完成！模型已蜕变为 Instruct 版本！")


def evaluate_bitdelta_model():
    # 核心修改点：精准指向目录下的 diff.pt 文件！
    diff_file = "/home/newdrive2/liu4441/bitdelta/diff.pt"
    
    print(f"[*] 开始加载官方 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] 开始加载官方 Base 模型 (Llama-3.1-8B)...")
    # 先把纯净的 Base 模型加载到显存里
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa"  # PyTorch 原生的高效加速，免去安装编译
    )

    # 实施降维打击：在显存里直接把 Base 变身！
    load_diff(model, diff_file)

    # 适配 Llama-3.1 的终止符
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    if hasattr(model, "generation_config"):
        model.generation_config.eos_token_id = terminators
    
    # 构造与你的 gptq-eval.py 完全一致的配置字典
    cfg = {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-3.1-8B",
        "ft_model": "meta-llama/Llama-3.1-8B-Instruct", 
        "llm_tasks": ["mmlu", "gsm8k"],
        "run_mt_bench": True,
        "eval_limit": None,
        "batch_size": "32"
    }

    print(f"[*] 开始使用 UniversalEvaluator 进行公平对齐评测...")
    
    evaluator = UniversalEvaluator(
        model=model, 
        processor=tokenizer, 
        config=cfg, 
        run_tag="bitdelta-baseline",  # 确保 MT-Bench 答案不会和其他实验串台
        device="cuda"
    )
    
    metrics = evaluator.run()
    
    print("\n" + "="*60)
    print(" 📊 BitDelta 模型客观评测结果")
    print("="*60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f" {k:<20}: {v:.4f}")
        else:
            print(f" {k:<20}: {v}")
    print("="*60)


if __name__ == "__main__":
    evaluate_bitdelta_model()