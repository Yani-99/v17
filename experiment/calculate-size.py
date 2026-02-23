import re
import os
from huggingface_hub import HfApi

# 完整保留您提供的 7 组实验配置
EXPERIMENT_GROUPS = [
    # # 组 1: BERT
    # [
    #     "--base-model bert-large-uncased --finetuned-model assemblyai/bert-large-uncased-sst2 --dtype fp32 --compressor fmd",
    #     "--base-model bert-large-uncased --finetuned-model samrawal/bert-large-uncased_med-ner --dtype fp32 --compressor fmd",
    #     "--base-model bert-large-uncased --finetuned-model yoshitomo-matsubara/bert-large-uncased-mnli --dtype fp32 --compressor fmd",
    #     "--base-model bert-large-uncased --finetuned-model princeton-nlp/sup-simcse-bert-large-uncased --dtype fp32 --compressor fmd",
    #     "--base-model bert-large-uncased --finetuned-model SarielSinLuo/bert-large-uncased-finetuned-cola --dtype fp32 --compressor fmd",
    #     "--base-model bert-large-uncased --finetuned-model princeton-nlp/unsup-simcse-bert-large-uncased --dtype fp32 --compressor fmd",
    #     "--base-model bert-large-uncased --finetuned-model yoshitomo-matsubara/bert-large-uncased-mrpc --dtype fp32 --compressor fmd",
    #     "--base-model bert-large-uncased --finetuned-model yoshitomo-matsubara/bert-large-uncased-qnli --dtype fp32 --compressor fmd",
    #     "--base-model bert-large-uncased --finetuned-model StevenLimcorn/bert-large-uncased-semeval2016-restaurants --dtype fp32 --compressor fmd",
    #     "--base-model bert-large-uncased --finetuned-model Jorgeutd/bert-large-uncased-finetuned-ner --dtype fp32 --compressor fmd"
    # ],
    # # 组 2: Llama-3.1-8B
    # [
    #     "--base-model meta-llama/Llama-3.1-8B --finetuned-model meta-llama/Llama-3.1-8B-Instruct  --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-3.1-8B --finetuned-model NousResearch/Hermes-3-Llama-3.1-8B --dtype fp16 --compressor fmd", 
    #     "--base-model meta-llama/Llama-3.1-8B --finetuned-model meta-llama/Llama-Guard-3-8B --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-3.1-8B --finetuned-model dphn/Dolphin3.0-Llama3.1-8B --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-3.1-8B --finetuned-model mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-3.1-8B --finetuned-model OpenSciLM/Llama-3.1_OpenScholar-8B --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-3.1-8B --finetuned-model Sao10K/Llama-3.1-8B-Stheno-v3.4 --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-3.1-8B --finetuned-model cognitivecomputations/dolphin-2.9.4-llama3.1-8b --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-3.1-8B --finetuned-model akjindal53244/Llama-3.1-Storm-8B --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-3.1-8B --finetuned-model Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.2 --dtype fp16 --compressor fmd"
    # ],
    # # 组 3: Llama-2-7B
    # [
    #     "--base-model meta-llama/Llama-2-7b-hf --finetuned-model meta-llama/Llama-2-7b-chat-hf --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-7b-hf --finetuned-model lmsys/vicuna-7b-v1.5 --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-7b-hf --finetuned-model NousResearch/Nous-Hermes-llama-2-7b --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-7b-hf --finetuned-model garage-bAInd/Platypus2-7B --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-7b-hf --finetuned-model WizardLM/WizardMath-7B-V1.0 --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-7b-hf --finetuned-model georgesung/llama2_7b_chat_uncensored --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-7b-hf --finetuned-model allenai/tulu-2-7b --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-7b-hf --finetuned-model PygmalionAI/pygmalion-2-7b --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-7b-hf --finetuned-model h2oai/h2ogpt-4096-llama2-7b-chat --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-7b-hf --finetuned-model stabilityai/StableBeluga-7B --dtype fp16 --compressor fmd"
    # ],
    # # 组 4: Llama-2-13B
    # [
    #     "--base-model meta-llama/Llama-2-13b-hf --finetuned-model meta-llama/Llama-2-13b-chat-hf --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-13b-hf --finetuned-model lmsys/vicuna-13b-v1.5 --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-13b-hf --finetuned-model NousResearch/Nous-Hermes-Llama2-13b --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-13b-hf --finetuned-model WizardLM/WizardLM-13B-V1.2 --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-13b-hf --finetuned-model garage-bAInd/Platypus2-13B --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-13b-hf --finetuned-model stabilityai/StableBeluga-13B --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-13b-hf --finetuned-model allenai/tulu-2-dpo-13b --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-13b-hf --finetuned-model Open-Orca/OpenOrca-Platypus2-13B --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-13b-hf --finetuned-model Riiid/sheep-duck-llama-2-13b --dtype fp16 --compressor fmd",
    #     "--base-model meta-llama/Llama-2-13b-hf --finetuned-model Xwin-LM/Xwin-LM-13B-V0.1 --dtype fp16 --compressor fmd"
    # ],
    # # 组 5: GPT2
    # [
    #     "--base-model gpt2 --finetuned-model lvwerra/gpt2-imdb --dtype fp32 --compressor fmd",
    #     "--base-model gpt2 --finetuned-model Gustavosta/MagicPrompt-Stable-Diffusion --dtype fp32 --compressor fmd",
    #     "--base-model gpt2 --finetuned-model mrm8488/GPT-2-finetuned-common_gen --dtype fp32 --compressor fmd",
    #     "--base-model gpt2 --finetuned-model succinctly/text2image-prompt-generator --dtype fp32 --compressor fmd",
    #     "--base-model gpt2 --finetuned-model shibing624/code-autocomplete-gpt2-base --dtype fp32 --compressor fmd",
    #     "--base-model gpt2 --finetuned-model rhysjones/gpt2-124M-edu-fineweb-10B --dtype fp32 --compressor fmd",
    #     "--base-model gpt2 --finetuned-model huggingtweets/elonmusk --dtype fp32 --compressor fmd",
    #     "--base-model gpt2 --finetuned-model neulab/gpt2-finetuned-wikitext103 --dtype fp32 --compressor fmd",
    #     "--base-model gpt2 --finetuned-model vicgalle/gpt2-open-instruct-v1 --dtype fp32 --compressor fmd",
    #     "--base-model gpt2 --finetuned-model Ssarion/gpt2-multi-news --dtype fp32 --compressor fmd"
    # ],
    # # 组 6: RoBERTa
    # [
    #     "--base-model roberta-base --finetuned-model deepset/roberta-base-squad2 --dtype fp32 --compressor fmd",
    #     "--base-model roberta-base --finetuned-model cross-encoder/stsb-roberta-base --dtype fp32 --compressor fmd",
    #     "--base-model roberta-base --finetuned-model cross-encoder/nli-roberta-base --dtype fp32 --compressor fmd",
    #     "--base-model roberta-base --finetuned-model textattack/roberta-base-SST-2 --dtype fp32 --compressor fmd",
    #     "--base-model roberta-base --finetuned-model openai-community/roberta-base-openai-detector --dtype fp32 --compressor fmd",
    #     "--base-model roberta-base --finetuned-model textattack/roberta-base-MNLI --dtype fp32 --compressor fmd",
    #     "--base-model roberta-base --finetuned-model cardiffnlp/twitter-roberta-base-sentiment-latest --dtype fp32 --compressor fmd",
    #     "--base-model roberta-base --finetuned-model SamLowe/roberta-base-go_emotions --dtype fp32 --compressor fmd",
    #     "--base-model roberta-base --finetuned-model textattack/roberta-base-ag-news --dtype fp32 --compressor fmd",
    #     "--base-model roberta-base --finetuned-model cardiffnlp/twitter-roberta-base-hate-latest --dtype fp32 --compressor fmd"
    # ],
    # # 组 7: Mistral
    # [
    #     "--base-model mistralai/Mistral-7B-v0.1 --finetuned-model mistralai/Mistral-7B-Instruct-v0.1 --dtype fp16 --compressor fmd",
    #     "--base-model mistralai/Mistral-7B-v0.1 --finetuned-model HuggingFaceH4/zephyr-7b-beta --dtype fp16 --compressor fmd",
    #     "--base-model mistralai/Mistral-7B-v0.1 --finetuned-model HuggingFaceH4/zephyr-7b-alpha --dtype fp16 --compressor fmd",
    #     "--base-model mistralai/Mistral-7B-v0.1 --finetuned-model Intel/neural-chat-7b-v3-1 --dtype fp16 --compressor fmd",
    #     "--base-model mistralai/Mistral-7B-v0.1 --finetuned-model berkeley-nest/Starling-LM-7B-alpha --dtype fp16 --compressor fmd",
    #     "--base-model mistralai/Mistral-7B-v0.1 --finetuned-model argilla/notus-7b-v1 --dtype fp16 --compressor fmd",
    #     "--base-model mistralai/Mistral-7B-v0.1 --finetuned-model jondurbin/airoboros-m-7b-3.1.2 --dtype fp16 --compressor fmd",
    #     "--base-model mistralai/Mistral-7B-v0.1 --finetuned-model ehartford/samantha-1.2-mistral-7b --dtype fp16 --compressor fmd",
    #     "--base-model mistralai/Mistral-7B-v0.1 --finetuned-model migtissera/SynthIA-7B-v1.3 --dtype fp16 --compressor fmd",
    #     "--base-model mistralai/Mistral-7B-v0.1 --finetuned-model TIGER-Lab/MAmmoTH-7B-Mistral --dtype fp16 --compressor fmd"
    # ]
    [
        "--base-model roberta-large --finetuned-model roberta-large-mnli --dtype fp32 --compressor fmd",
        "--base-model roberta-large --finetuned-model deepset/roberta-large-squad2 --dtype fp32 --compressor fmd",
        "--base-model roberta-large --finetuned-model cross-encoder/stsb-roberta-large --dtype fp32 --compressor fmd",
        "--base-model roberta-large --finetuned-model siebert/sentiment-roberta-large-english --dtype fp32 --compressor fmd",
        "--base-model roberta-large --finetuned-model sentence-transformers/all-roberta-large-v1 --dtype fp32 --compressor fmd",
        "--base-model roberta-large --finetuned-model ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli --dtype fp32 --compressor fmd",
        "--base-model roberta-large --finetuned-model openai-community/roberta-large-openai-detector --dtype fp32 --compressor fmd",
        "--base-model roberta-large --finetuned-model navteca/roberta-large-squad2 --dtype fp32 --compressor fmd",
        "--base-model roberta-large --finetuned-model cross-encoder/quora-roberta-large --dtype fp32 --compressor fmd",
        "--base-model roberta-large --finetuned-model jean-baptiste/roberta-large-ner-english --dtype fp32 --compressor fmd"
    ],
    [
        "--base-model meta-llama/Llama-3.1-70B --finetuned-model meta-llama/Llama-3.1-70B-Instruct --dtype fp32 --compressor fmd",
        "--base-model meta-llama/Llama-3.1-70B --finetuned-model NousResearch/Hermes-3-Llama-3.1-70B --dtype fp32 --compressor fmd",
        "--base-model meta-llama/Llama-3.1-70B --finetuned-model nvidia/Llama-3.1-Nemotron-70B-Instruct-HF --dtype fp32 --compressor fmd",
        "--base-model meta-llama/Llama-3.1-70B --finetuned-model unsloth/Meta-Llama-3.1-70B-Instruct --dtype fp32 --compressor fmd",
        "--base-model meta-llama/Llama-3.1-70B --finetuned-model mattshumer/Reflection-Llama-3.1-70B --dtype fp32 --compressor fmd",
        "--base-model meta-llama/Llama-3.1-70B --finetuned-model VAGOsolutions/Llama-3.1-SauerkrautLM-70b-Instruct --dtype fp32 --compressor fmd",
        "--base-model meta-llama/Llama-3.1-70B --finetuned-model allenai/Llama-3.1-Tulu-3-70B-SFT --dtype fp32 --compressor fmd",
        "--base-model meta-llama/Llama-3.1-70B --finetuned-model nvidia/OpenMath2-Llama3.1-70B --dtype fp32 --compressor fmd",
        "--base-model meta-llama/Llama-3.1-70B --finetuned-model migtissera/Tess-3-Llama-3.1-70B --dtype fp32 --compressor fmd",
        "--base-model meta-llama/Llama-3.1-70B --finetuned-model mylesgoose/Llama-3.1-70B-Instruct-abliterated --dtype fp32 --compressor fmd"
    ]

]

def get_pure_weight_size(api, model_id, token):
    """
    通过 API 获取仓库中纯模型权重文件的大小（单位：GB）。
    严格排除了优化器状态、训练参数文件，以及避免 bin 和 safetensors 双重计算。
    """
    try:
        info = api.model_info(model_id, files_metadata=True, token=token)
        
        # 提取出带有文件大小信息的有效文件
        all_files = [f for f in info.siblings if f.size is not None]
        
        safetensors_files = []
        bin_files = []
        
        for f in all_files:
            fname = f.rfilename.lower()
            
            # 1. 严格排除非权重文件
            if "optimizer" in fname or "training_args" in fname or "arguments" in fname:
                continue
                
            # 2. 收集 .safetensors 格式的纯权重文件
            if fname.endswith(".safetensors"):
                safetensors_files.append(f)
                
            # 3. 收集 .bin / .pt 格式的纯权重文件（作为备选）
            elif fname.endswith(".bin") or fname.endswith(".pt"):
                if "pytorch_model" in fname or "adapter_model" in fname:
                    bin_files.append(f)

        # 4. 计算大小：优先使用 safetensors，如果开发者没传 safetensors 才用 bin
        if len(safetensors_files) > 0:
            total_bytes = sum(f.size for f in safetensors_files)
        elif len(bin_files) > 0:
            total_bytes = sum(f.size for f in bin_files)
        else:
            print(f"      [警告] 在 {model_id} 中没有找到常见的权重文件格式。")
            return 0.0
            
        return total_bytes / (1024 ** 3) # 转换为 GB
        
    except Exception as e:
        print(f"      [错误] 获取 {model_id} 失败 (网络或授权问题): {e}")
        return 0.0

def process_experiments():
    api = HfApi()
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("\n⚠️ 警告: 未检测到 HF_TOKEN 环境变量。")
        print("建议在终端运行: export HF_TOKEN='your_hf_token'\n")

    for i, group in enumerate(EXPERIMENT_GROUPS, 1):
        if not group:
            continue
            
        base_match = re.search(r'--base-model\s+([^\s]+)', group[0])
        base_model_id = base_match.group(1) if base_match else "Unknown"
        
        print(f"==================================================")
        print(f"实验组 {i} - 正在处理基座模型: {base_model_id}")
        
        finetuned_models = []
        for line in group:
            ft_match = re.search(r'--finetuned-model\s+([^\s]+)', line)
            if ft_match:
                finetuned_models.append(ft_match.group(1))
                
        # 计算纯权重基座模型大小 (仅用于估算乘数)
        print(f"  > 正在查询基座模型纯权重大小...")
        base_size_gb = get_pure_weight_size(api, base_model_id, hf_token)
        
        # 收集所有 10 个微调模型的大小
        finetuned_sizes_gb = []
        for ft_model in finetuned_models:
            print(f"  > 正在查询微调模型: {ft_model} ...")
            size = get_pure_weight_size(api, ft_model, hf_token)
            finetuned_sizes_gb.append(size)
            
        # ---------------------------------------------------------
        # 场景 A：仅前 5 个微调模型
        # ---------------------------------------------------------
        actual_total_size_5 = sum(finetuned_sizes_gb[:5])
        estimated_total_size_5 = base_size_gb * 5  # 基座 * 5

        # ---------------------------------------------------------
        # 场景 B：仅全部 10 个微调模型
        # ---------------------------------------------------------
        actual_total_size_10 = sum(finetuned_sizes_gb)
        estimated_total_size_10 = base_size_gb * 10 # 基座 * 10

        # ---------------------------------------------------------
        # 打印输出对比结果
        # ---------------------------------------------------------
        print(f"\n【纯权重统计结果 - 组 {i}】")
        print(f"  基准参考: 基座模型单体纯权重大小 = {base_size_gb:.2f} GB\n")

        print(f"  ▶ 场景 A: 仅计算前 5 个微调模型")
        print(f"    - 粗略估计大小 (基座*5) : {estimated_total_size_5:.2f} GB")
        print(f"    - 真实总占用纯权重大小  : {actual_total_size_5:.2f} GB")
        diff_5 = actual_total_size_5 - estimated_total_size_5
        if diff_5 > 0:
            print(f"    - 结论: 粗略估计比实际【少算了】 {diff_5:.2f} GB\n")
        elif diff_5 < 0:
            print(f"    - 结论: 粗略估计比实际【多算了】 {abs(diff_5):.2f} GB\n")
        else:
            print(f"    - 结论: 粗略估计与实际完全一致\n")

        print(f"  ▶ 场景 B: 仅计算全部 10 个微调模型")
        print(f"    - 粗略估计大小 (基座*10): {estimated_total_size_10:.2f} GB")
        print(f"    - 真实总占用纯权重大小  : {actual_total_size_10:.2f} GB")
        diff_10 = actual_total_size_10 - estimated_total_size_10
        if diff_10 > 0:
            print(f"    - 结论: 粗略估计比实际【少算了】 {diff_10:.2f} GB\n")
        elif diff_10 < 0:
            print(f"    - 结论: 粗略估计比实际【多算了】 {abs(diff_10):.2f} GB\n")
        else:
            print(f"    - 结论: 粗略估计与实际完全一致\n")

if __name__ == "__main__":
    process_experiments()