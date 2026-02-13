import argparse
import json
import os
import re
import glob
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from huggingface_hub import login

# ==========================================
# [Settings]
# ==========================================
DEFAULT_JUDGE_MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"

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
    例如: mt_bench_answers_0.01.jsonl -> 0.01
    mt_bench_answers_baseline.jsonl -> 999 (放最后) 或 -1 (放最前)
    """
    # 提取文件名中的数字部分
    base = os.path.basename(filename)
    # 匹配浮点数
    match = re.search(r"(\d+\.\d+|\d+)", base)
    if "baseline" in base:
        return -1.0 # 让 baseline 排在最前面，或者改成 999 排最后
    if match:
        return float(match.group(1))
    return 999.0 # 未知格式排最后

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="mt_bench_answers_*.jsonl", help="File pattern to match")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    # 1. 查找所有文件并排序
    files = glob.glob(args.pattern)
    # 按提取出的数字排序，方便观察参数变化趋势
    files.sort(key=get_sort_key)
    
    if not files:
        print(f"Error: No files found matching pattern: {args.pattern}")
        return

    print(f"[Init] Found {len(files)} files to judge.")
    for f in files:
        print(f"  - {f}")

    # 2. 加载模型 (只加载一次!)
    print(f"\n[Init] Loading Judge Model: {args.judge_model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
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
    print("STARTING BATCH EVALUATION")
    print("="*60)

    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"\n>>> Processing: {file_name}")
        
        # 准备输出文件名
        output_file = file_path.replace("answers", "judgments")
        
        # 加载答案
        answers = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                answers.append(json.loads(line))
        
        judgments = []
        scores_t1 = []
        scores_t2 = []

        # 进度条
        for item in tqdm(answers, desc=f"Judging {file_name}", leave=False):
            qid = item['question_id']
            model_id = item['model_id']
            turns_answers = item['choices'][0]['turns']
            
            if qid not in question_map: continue
            ref_item = question_map[qid]
            turns_questions = ref_item['prompt']
            category = ref_item['category']

            # Turn 1
            prompt_t1 = JUDGE_PROMPT_TEMPLATE.format(question=turns_questions[0], answer=turns_answers[0])
            inputs = tokenizer(prompt_t1, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0, do_sample=False)
            review_t1 = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            score_t1 = parse_score(review_t1)
            if score_t1 != -1: scores_t1.append(score_t1)

            # Turn 2
            score_t2 = -1.0
            review_t2 = ""
            if len(turns_questions) > 1 and len(turns_answers) > 1:
                prompt_t2 = JUDGE_PROMPT_TEMPLATE_CTX.format(
                    question_1=turns_questions[0], answer_1=turns_answers[0],
                    question_2=turns_questions[1], answer_2=turns_answers[1]
                )
                inputs = tokenizer(prompt_t2, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0, do_sample=False)
                review_t2 = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                score_t2 = parse_score(review_t2)
                if score_t2 != -1: scores_t2.append(score_t2)

            judgments.append({
                "question_id": qid, "category": category, "model_id": model_id,
                "turn_1_score": score_t1, "turn_1_review": review_t1,
                "turn_2_score": score_t2, "turn_2_review": review_t2
            })

        # 保存该文件的详细 Judgments
        with open(output_file, "w", encoding="utf-8") as f:
            for j in judgments:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")

        # 计算平均分
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
    print(f"FINAL BATCH REPORT (Judge: {args.judge_model})")
    print("="*80)
    
    # 打印表头
    header = f"| {'Filename':<35} | {'Turn 1':<10} | {'Turn 2':<10} | {'Average':<10} |"
    print(header)
    print("|" + "-"*37 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|")
    
    # 打印每一行
    for item in final_report:
        print(f"| {item['File']:<35} | {item['Turn 1']:<10.4f} | {item['Turn 2']:<10.4f} | {item['Average']:<10.4f} |")
    print("="*80)

if __name__ == "__main__":
    main()