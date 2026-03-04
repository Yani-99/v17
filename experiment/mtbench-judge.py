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
# DEFAULT_JUDGE_MODEL = "meta-llama/Llama-3.1-70B-Instruct"

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
#     # vLLM 专属参数
#     parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use for vLLM")
#     parser.add_argument("--quantization", type=str, default="None", help="Quantization method (e.g., bitsandbytes, awq, gptq, or None)")
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

#     # 2. 加载 vLLM 模型 (只加载一次!)
#     print(f"\n[Init] Loading Judge Model via vLLM: {args.judge_model}")
    
#     # 初始化 vLLM 引擎
#     # note: 如果你有多张显卡，请调整 --tensor-parallel-size，70B模型通常需要 2 到 4 张卡并行。
#     llm = LLM(
#         model=args.judge_model,
#         tensor_parallel_size=args.tensor_parallel_size,
#         quantization=args.quantization if args.quantization.lower() != "none" else None,
#         load_format="bitsandbytes" if args.quantization == "bitsandbytes" else "auto",
#         trust_remote_code=True,
#         enforce_eager=True # 对于某些量化格式可能需要开启
#     )
    
#     # 配置采样参数 (等同于原代码的 temperature=0.0, max_new_tokens=512)
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
#     print("STARTING BATCH EVALUATION (vLLM Accelerated)")
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
        
#         # 为了使用 vLLM 的并发能力，我们需要先提取所有的 Prompts
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
#         t1_results_map = {} # qid -> (score, review)
#         for i, output in enumerate(outputs_t1):
#             review_t1 = output.outputs[0].text.strip()
#             score_t1 = parse_score(review_t1)
#             qid = valid_items_t1[i][0]['question_id']
#             t1_results_map[qid] = (score_t1, review_t1)
#             if score_t1 != -1: scores_t1_list.append(score_t1)

#         # 处理 Turn 2 结果
#         t2_results_map = {} # qid -> (score, review)
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
#     print(f"FINAL BATCH REPORT (Judge: {args.judge_model} via vLLM)")
#     print("="*80)
    
#     header = f"| {'Filename':<35} | {'Turn 1':<10} | {'Turn 2':<10} | {'Average':<10} |"
#     print(header)
#     print("|" + "-"*37 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|")
    
#     for item in final_report:
#         print(f"| {item['File']:<35} | {item['Turn 1']:<10.4f} | {item['Turn 2']:<10.4f} | {item['Average']:<10.4f} |")
#     print("="*80)

# if __name__ == "__main__":
#     main()


# 运行命令示例:
# CUDA_VISIBLE_DEVICES=0,1 python vllm_awq_judge.py --pattern "mt_bench_answers_*.jsonl" > mtjudge.log

import argparse
import json
import os
import re
import glob
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from vllm import LLM, SamplingParams

# ==========================================
# [Settings]
# ==========================================
# 默认直接使用高质量的 4-bit AWQ 量化版 Llama 3.1 70B，体积约 35GB，双 3090 完美运行
DEFAULT_JUDGE_MODEL = "casperhansen/llama-3.1-70b-instruct-awq"

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
    
    # 针对双卡 3090 优化的默认参数
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="Number of GPUs to use for vLLM")
    parser.add_argument("--quantization", type=str, default="awq", help="Quantization method (e.g., bitsandbytes, awq, gptq, or None)")
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    # 1. 查找所有文件并排序
    files = glob.glob(args.pattern)
    files.sort(key=get_sort_key)
    
    if not files:
        print(f"Error: No files found matching pattern: {args.pattern}")
        return

    print(f"[Init] Found {len(files)} files to judge.")
    for f in files:
        print(f"  - {f}")

    # 2. 加载 vLLM 模型 (利用并发和张量并行)
    print(f"\n[Init] Loading Judge Model via vLLM: {args.judge_model}")
    print(f"[Init] Configuration: TP_Size={args.tensor_parallel_size}, Quantization={args.quantization}")
    
    llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization if args.quantization.lower() != "none" else None,
        trust_remote_code=True,
        # 预留部分显存防止极端长文本时 OOM，0.9 是稳妥的甜点值
        gpu_memory_utilization=0.9
    )
    
    # 评测任务必须关闭采样，确保打分的一致性和可复现性
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=512,
        skip_special_tokens=True
    )

    # 3. 加载问题集
    print("[Data] Loading MT-Bench questions...")
    dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    question_map = {}
    for item in dataset:
        qid = item.get('question_id') or item.get('prompt_id')
        question_map[qid] = item

    # 4. 循环处理每个文件
    final_report = []

    print("\n" + "="*60)
    print("STARTING BATCH EVALUATION (vLLM Accelerated - AWQ 4-bit)")
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
        
        # 收集所有的 Prompts，一次性推入 vLLM 引擎吃满算力
        t1_prompts = []
        t2_prompts = []
        valid_items_t1 = []
        valid_items_t2 = []

        for item in answers:
            qid = item['question_id']
            if qid not in question_map: continue
            
            ref_item = question_map[qid]
            turns_questions = ref_item['prompt']
            turns_answers = item['choices'][0]['turns']

            # 准备 Turn 1 的 Prompt
            prompt_t1 = JUDGE_PROMPT_TEMPLATE.format(question=turns_questions[0], answer=turns_answers[0])
            t1_prompts.append(prompt_t1)
            valid_items_t1.append((item, ref_item))

            # 准备 Turn 2 的 Prompt
            if len(turns_questions) > 1 and len(turns_answers) > 1:
                prompt_t2 = JUDGE_PROMPT_TEMPLATE_CTX.format(
                    question_1=turns_questions[0], answer_1=turns_answers[0],
                    question_2=turns_questions[1], answer_2=turns_answers[1]
                )
                t2_prompts.append(prompt_t2)
                valid_items_t2.append((item, ref_item))

        print(f"    -> Running Batch Inference for Turn 1 ({len(t1_prompts)} prompts)...")
        outputs_t1 = llm.generate(t1_prompts, sampling_params, use_tqdm=True)
        
        print(f"    -> Running Batch Inference for Turn 2 ({len(t2_prompts)} prompts)...")
        outputs_t2 = llm.generate(t2_prompts, sampling_params, use_tqdm=True)

        # 解析分数并重新组装结果
        judgments = []
        scores_t1_list = []
        scores_t2_list = []

        # 处理 Turn 1 结果
        t1_results_map = {} 
        for i, output in enumerate(outputs_t1):
            review_t1 = output.outputs[0].text.strip()
            score_t1 = parse_score(review_t1)
            qid = valid_items_t1[i][0]['question_id']
            t1_results_map[qid] = (score_t1, review_t1)
            if score_t1 != -1: scores_t1_list.append(score_t1)

        # 处理 Turn 2 结果
        t2_results_map = {} 
        for i, output in enumerate(outputs_t2):
            review_t2 = output.outputs[0].text.strip()
            score_t2 = parse_score(review_t2)
            qid = valid_items_t2[i][0]['question_id']
            t2_results_map[qid] = (score_t2, review_t2)
            if score_t2 != -1: scores_t2_list.append(score_t2)

        # 合并写入 jsonl
        for item, ref_item in valid_items_t1:
            qid = item['question_id']
            score_t1, review_t1 = t1_results_map.get(qid, (-1.0, ""))
            score_t2, review_t2 = t2_results_map.get(qid, (-1.0, ""))

            judgments.append({
                "question_id": qid, 
                "category": ref_item['category'], 
                "model_id": item['model_id'],
                "turn_1_score": score_t1, 
                "turn_1_review": review_t1,
                "turn_2_score": score_t2, 
                "turn_2_review": review_t2
            })

        with open(output_file, "w", encoding="utf-8") as f:
            for j in judgments:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")

        avg_t1 = np.mean(scores_t1_list) if scores_t1_list else 0
        avg_t2 = np.mean(scores_t2_list) if scores_t2_list else 0
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
    print(f"FINAL BATCH REPORT (Judge: {args.judge_model} via vLLM TP=2)")
    print("="*80)
    
    header = f"| {'Filename':<35} | {'Turn 1':<10} | {'Turn 2':<10} | {'Average':<10} |"
    print(header)
    print("|" + "-"*37 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|")
    
    for item in final_report:
        print(f"| {item['File']:<35} | {item['Turn 1']:<10.4f} | {item['Turn 2']:<10.4f} | {item['Average']:<10.4f} |")
    print("="*80)

if __name__ == "__main__":
    main()