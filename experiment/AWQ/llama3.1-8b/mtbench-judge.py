# 运行命令:
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python mtbench-judge.py > mtjudge.log

import argparse
import json
import os
import re
import glob
import torch
import numpy as np
import gc
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from huggingface_hub import login

# ==========================================
# [HOTFIX] 终极拦截魔法：修补 Accelerate 与 BNB 的底层参数冲突
# ==========================================
import bitsandbytes as bnb
if hasattr(bnb.nn, "Int8Params"):
    _orig_int8_new = bnb.nn.Int8Params.__new__
    def _patched_int8_new(cls, *args, **kwargs):
        keys_to_remove = [k for k in kwargs.keys() if k.startswith("_")]
        for k in keys_to_remove:
            kwargs.pop(k, None)
        return _orig_int8_new(cls, *args, **kwargs)
    bnb.nn.Int8Params.__new__ = staticmethod(_patched_int8_new)
# ==========================================

# ==========================================
# [Settings]
# ==========================================
DEFAULT_JUDGE_MODEL = "meta-llama/Llama-3.1-70B-Instruct"

JUDGE_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful and precise assistant for checking the quality of the answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

JUDGE_PROMPT_TEMPLATE_CTX = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful and precise assistant for checking the quality of the answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question_1}

[The Start of Assistant's Answer]
{answer_1}
[The End of Assistant's Answer]

[Question]
{question_2}

[The Start of Assistant's Answer]
{answer_2}
[The End of Assistant's Answer]<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def parse_score(review):
    try:
        match = re.search(r"\[\[(\d+\.?\d*)\]\]", review)
        if match:
            return float(match.group(1))
        match = re.search(r"Rating:\s*(\d+)", review)
        if match:
            return float(match.group(1))
        return -1.0
    except:
        return -1.0

def get_sort_key(filename):
    """
    辅助函数：尝试从文件名提取数字进行排序。
    """
    base = os.path.basename(filename)
    match = re.search(r"(\d+\.\d+|\d+)", base)
    if "baseline" in base:
        return -1.0
    if match:
        return float(match.group(1))
    return 999.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="mt_bench_answers_*.jsonl", help="File pattern to match")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    files = glob.glob(args.pattern)
    files.sort(key=get_sort_key)
    
    if not files:
        print(f"Error: No files found matching pattern: {args.pattern}")
        return

    print(f"[Init] Found {len(files)} files to judge.")
    for f in files:
        print(f"  - {f}")

    # ==========================================
    # 核心修改点：极致退让显存到 4GB，为动态碎片留出 20GB  runway
    # ==========================================
    print(f"\n[Init] Loading Model with Single-GPU BitsAndBytes 8-bit + CPU Offloading: {args.judge_model}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True, 
    )

    memory_limits = {
        0: "4GiB",      # 生死线：只放 4G 静态权重，强制空出 20G 处理 FP32 的层级拉取和计算！
        "cpu": "150GiB" 
    }
    
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=memory_limits,
        trust_remote_code=True
    )

    input_device = next(model.parameters()).device

    print("[Data] Loading MT-Bench questions...")
    dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    question_map = {}
    for item in dataset:
        qid = item.get('question_id') or item.get('prompt_id')
        question_map[qid] = item

    final_report = []

    print("\n" + "="*60)
    print("STARTING BATCH EVALUATION (BitsAndBytes 8-bit + CPU Offload)")
    print("="*60)

    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"\n>>> Processing: {file_name}")
        
        output_file = file_path.replace("answers", "judgments")
        
        answers = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    answers.append(json.loads(line))
        
        judgments = []
        scores_t1 = []
        scores_t2 = []

        for item in tqdm(answers, desc=f"Judging {file_name}", leave=False):
            qid = item['question_id']
            model_id = item['model_id']
            turns_answers = item['choices'][0]['turns']
            
            if qid not in question_map: continue
            ref_item = question_map[qid]
            turns_questions = ref_item['prompt']
            category = ref_item['category']

            # ==========================================
            # Turn 1 (加入严格的内存清理机制)
            # ==========================================
            prompt_t1 = JUDGE_PROMPT_TEMPLATE.format(question=turns_questions[0], answer=turns_answers[0])
            inputs = tokenizer(prompt_t1, return_tensors="pt").to(input_device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0, do_sample=False)
            
            review_t1 = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            score_t1 = parse_score(review_t1)
            if score_t1 != -1: scores_t1.append(score_t1)

            # 强行释放 Turn 1 的计算缓存
            del inputs, outputs
            gc.collect()
            torch.cuda.empty_cache()

            # ==========================================
            # Turn 2 (加入严格的内存清理机制)
            # ==========================================
            score_t2 = -1.0
            review_t2 = ""
            if len(turns_questions) > 1 and len(turns_answers) > 1:
                prompt_t2 = JUDGE_PROMPT_TEMPLATE_CTX.format(
                    question_1=turns_questions[0], answer_1=turns_answers[0],
                    question_2=turns_questions[1], answer_2=turns_answers[1]
                )
                inputs = tokenizer(prompt_t2, return_tensors="pt").to(input_device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0, do_sample=False)
                review_t2 = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
                score_t2 = parse_score(review_t2)
                if score_t2 != -1: scores_t2.append(score_t2)

                # 强行释放 Turn 2 的计算缓存
                del inputs, outputs
                gc.collect()
                torch.cuda.empty_cache()

            judgments.append({
                "question_id": qid, "category": category, "model_id": model_id,
                "turn_1_score": score_t1, "turn_1_review": review_t1,
                "turn_2_score": score_t2, "turn_2_review": review_t2
            })

        with open(output_file, "w", encoding="utf-8") as f:
            for j in judgments:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")

        avg_t1 = np.mean(scores_t1) if scores_t1 else 0
        avg_t2 = np.mean(scores_t2) if scores_t2 else 0
        final_score = (avg_t1 + avg_t2) / 2 if (avg_t1 and avg_t2) else avg_t1
        
        print(f"    Finished. Score: {final_score:.4f}")
        
        final_report.append({
            "File": file_name,
            "Turn 1": avg_t1,
            "Turn 2": avg_t2,
            "Average": final_score
        })

    # ==========================================
    # [Final Report]
    # ==========================================
    print("\n\n")
    print("="*80)
    print(f"FINAL BATCH REPORT (Judge: {args.judge_model} via BNB 8-bit Single-GPU Offload)")
    print("="*80)
    
    header = f"| {'Filename':<35} | {'Turn 1':<10} | {'Turn 2':<10} | {'Average':<10} |"
    print(header)
    print("|" + "-"*37 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|")
    
    for item in final_report:
        print(f"| {item['File']:<35} | {item['Turn 1']:<10.4f} | {item['Turn 2']:<10.4f} | {item['Average']:<10.4f} |")
    print("="*80)

if __name__ == "__main__":
    main()



# import argparse
# import json
# import os
# import re
# import glob
# import numpy as np
# from tqdm import tqdm
# from datasets import load_dataset
# from huggingface_hub import login
# from vllm import LLM, SamplingParams

# # ==========================================
# # [Settings]
# # ==========================================
# # 官方稳定版 AWQ 量化模型，完美契合 vLLM 和双 3090 显存
# DEFAULT_JUDGE_MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

# JUDGE_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# You are a helpful and precise assistant for checking the quality of the answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

# [Instruction]
# Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

# [Question]
# {question}

# [The Start of Assistant's Answer]
# {answer}
# [The End of Assistant's Answer]<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """

# JUDGE_PROMPT_TEMPLATE_CTX = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# You are a helpful and precise assistant for checking the quality of the answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

# [Instruction]
# Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

# [Question]
# {question_1}

# [The Start of Assistant's Answer]
# {answer_1}
# [The End of Assistant's Answer]

# [Question]
# {question_2}

# [The Start of Assistant's Answer]
# {answer_2}
# [The End of Assistant's Answer]<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """

# def parse_score(review):
#     try:
#         match = re.search(r"\[\[(\d+\.?\d*)\]\]", review)
#         if match:
#             return float(match.group(1))
#         match = re.search(r"Rating:\s*(\d+)", review)
#         if match:
#             return float(match.group(1))
#         return -1.0
#     except:
#         return -1.0

# def get_sort_key(filename):
#     """
#     辅助函数：尝试从文件名提取数字进行排序。
#     """
#     base = os.path.basename(filename)
#     match = re.search(r"(\d+\.\d+|\d+)", base)
#     if "baseline" in base:
#         return -1.0 
#     if match:
#         return float(match.group(1))
#     return 999.0 

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--pattern", type=str, default="mt_bench_answers_*.jsonl", help="File pattern to match")
#     parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
#     parser.add_argument("--hf-token", type=str, default=None)
    
#     # 双卡并发核心参数
#     parser.add_argument("--tensor-parallel-size", type=int, default=2, help="Number of GPUs to use for vLLM")
#     parser.add_argument("--quantization", type=str, default="awq", help="Quantization method")
#     args = parser.parse_args()

#     if args.hf_token:
#         login(token=args.hf_token)

#     # 1. 查找所有文件并排序
#     files = glob.glob(args.pattern)
#     files.sort(key=get_sort_key)
    
#     if not files:
#         print(f"Error: No files found matching pattern: {args.pattern}")
#         return

#     print(f"[Init] Found {len(files)} files to judge.")
#     for f in files:
#         print(f"  - {f}")

#     # 2. 加载 vLLM 模型 (利用张量并行将模型精准切分到两张卡)
#     print(f"\n[Init] Loading Judge Model via vLLM: {args.judge_model}")
#     print(f"[Init] Configuration: TP_Size={args.tensor_parallel_size}, Quantization={args.quantization}")
    
#     llm = LLM(
#         model=args.judge_model,
#         tensor_parallel_size=args.tensor_parallel_size,
#         quantization=args.quantization,
#         trust_remote_code=True,
#         # 预留 10% 的显存作为防爆安全气囊
#         gpu_memory_utilization=0.9
#     )
    
#     # 评测任务关闭采样，确保打分可复现
#     sampling_params = SamplingParams(
#         temperature=0.0, 
#         max_tokens=512,
#         skip_special_tokens=True
#     )

#     # 3. 加载问题集
#     print("[Data] Loading MT-Bench questions...")
#     dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
#     question_map = {}
#     for item in dataset:
#         qid = item.get('question_id') or item.get('prompt_id')
#         question_map[qid] = item

#     # 4. 循环处理每个文件
#     final_report = []

#     print("\n" + "="*60)
#     print("STARTING BATCH EVALUATION (vLLM Accelerated - AWQ 4-bit Dual GPU)")
#     print("="*60)

#     for file_path in files:
#         file_name = os.path.basename(file_path)
#         print(f"\n>>> Processing: {file_name}")
        
#         output_file = file_path.replace("answers", "judgments")
        
#         answers = []
#         with open(file_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 if line.strip():
#                     answers.append(json.loads(line))
        
#         # 收集所有的 Prompts，一次性推入 vLLM 引擎，吃满双卡算力
#         t1_prompts = []
#         t2_prompts = []
#         valid_items_t1 = []
#         valid_items_t2 = []

#         for item in answers:
#             qid = item['question_id']
#             if qid not in question_map: continue
            
#             ref_item = question_map[qid]
#             turns_questions = ref_item['prompt']
#             turns_answers = item['choices'][0]['turns']

#             # 准备 Turn 1 的 Prompt
#             prompt_t1 = JUDGE_PROMPT_TEMPLATE.format(question=turns_questions[0], answer=turns_answers[0])
#             t1_prompts.append(prompt_t1)
#             valid_items_t1.append((item, ref_item))

#             # 准备 Turn 2 的 Prompt
#             if len(turns_questions) > 1 and len(turns_answers) > 1:
#                 prompt_t2 = JUDGE_PROMPT_TEMPLATE_CTX.format(
#                     question_1=turns_questions[0], answer_1=turns_answers[0],
#                     question_2=turns_questions[1], answer_2=turns_answers[1]
#                 )
#                 t2_prompts.append(prompt_t2)
#                 valid_items_t2.append((item, ref_item))

#         print(f"    -> Running Batch Inference for Turn 1 ({len(t1_prompts)} prompts)...")
#         outputs_t1 = llm.generate(t1_prompts, sampling_params, use_tqdm=True)
        
#         print(f"    -> Running Batch Inference for Turn 2 ({len(t2_prompts)} prompts)...")
#         outputs_t2 = llm.generate(t2_prompts, sampling_params, use_tqdm=True)

#         # 解析分数并重新组装结果
#         judgments = []
#         scores_t1_list = []
#         scores_t2_list = []

#         # 处理 Turn 1 结果
#         t1_results_map = {} 
#         for i, output in enumerate(outputs_t1):
#             review_t1 = output.outputs[0].text.strip()
#             score_t1 = parse_score(review_t1)
#             qid = valid_items_t1[i][0]['question_id']
#             t1_results_map[qid] = (score_t1, review_t1)
#             if score_t1 != -1: scores_t1_list.append(score_t1)

#         # 处理 Turn 2 结果
#         t2_results_map = {} 
#         for i, output in enumerate(outputs_t2):
#             review_t2 = output.outputs[0].text.strip()
#             score_t2 = parse_score(review_t2)
#             qid = valid_items_t2[i][0]['question_id']
#             t2_results_map[qid] = (score_t2, review_t2)
#             if score_t2 != -1: scores_t2_list.append(score_t2)

#         # 合并写入 jsonl
#         for item, ref_item in valid_items_t1:
#             qid = item['question_id']
#             score_t1, review_t1 = t1_results_map.get(qid, (-1.0, ""))
#             score_t2, review_t2 = t2_results_map.get(qid, (-1.0, ""))

#             judgments.append({
#                 "question_id": qid, 
#                 "category": ref_item['category'], 
#                 "model_id": item['model_id'],
#                 "turn_1_score": score_t1, 
#                 "turn_1_review": review_t1,
#                 "turn_2_score": score_t2, 
#                 "turn_2_review": review_t2
#             })

#         with open(output_file, "w", encoding="utf-8") as f:
#             for j in judgments:
#                 f.write(json.dumps(j, ensure_ascii=False) + "\n")

#         avg_t1 = np.mean(scores_t1_list) if scores_t1_list else 0
#         avg_t2 = np.mean(scores_t2_list) if scores_t2_list else 0
#         final_score = (avg_t1 + avg_t2) / 2 if (avg_t1 and avg_t2) else avg_t1
        
#         print(f"    Finished. Score: {final_score:.4f}")
        
#         final_report.append({
#             "File": file_name,
#             "Turn 1": avg_t1,
#             "Turn 2": avg_t2,
#             "Average": final_score
#         })

#     # ==========================================
#     # [Final Report]
#     # ==========================================
#     print("\n\n")
#     print("="*80)
#     print(f"FINAL BATCH REPORT (Judge: {args.judge_model} via vLLM TP=2)")
#     print("="*80)
    
#     header = f"| {'Filename':<35} | {'Turn 1':<10} | {'Turn 2':<10} | {'Average':<10} |"
#     print(header)
#     print("|" + "-"*37 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|")
    
#     for item in final_report:
#         print(f"| {item['File']:<35} | {item['Turn 1']:<10.4f} | {item['Turn 2']:<10.4f} | {item['Average']:<10.4f} |")
#     print("="*80)

# if __name__ == "__main__":
#     main()