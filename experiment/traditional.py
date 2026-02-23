# import os
# import time
# import zlib
# import gzip
# import zstandard as zstd
# import shutil
# import gc
# import json
# from glob import glob
# from huggingface_hub import snapshot_download

# # ==========================================
# # 配置与初始化
# # ==========================================

# # 示例模型列表
# # MODELS_TO_TEST = [
# #     # bert-large-uncased
# #     "assemblyai/bert-large-uncased-sst2",
# #     "samrawal/bert-large-uncased_med-ner",
# #     "yoshitomo-matsubara/bert-large-uncased-mnli",
# #     "princeton-nlp/sup-simcse-bert-large-uncased",
# #     "SarielSinLuo/bert-large-uncased-finetuned-cola",
# #     "princeton-nlp/unsup-simcse-bert-large-uncased",
# #     "yoshitomo-matsubara/bert-large-uncased-mrpc",
# #     "yoshitomo-matsubara/bert-large-uncased-qnli",
# #     "StevenLimcorn/bert-large-uncased-semeval2016-restaurants",
# #     "Jorgeutd/bert-large-uncased-finetuned-ner"
# # ]

# # MODELS_TO_TEST = [
# #     "meta-llama/Llama-3.1-8B-Instruct",
# #     "NousResearch/Hermes-3-Llama-3.1-8B",
# #     "meta-llama/Llama-Guard-3-8B",
# #     "dphn/Dolphin3.0-Llama3.1-8B",
# #     "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
# #     "OpenSciLM/Llama-3.1_OpenScholar-8B",
# #     "Sao10K/Llama-3.1-8B-Stheno-v3.4",
# #     "cognitivecomputations/dolphin-2.9.4-llama3.1-8b",
# #     "akjindal53244/Llama-3.1-Storm-8B",
# #     "Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.2"
# # ]

# # MODELS_TO_TEST = [
# #     "meta-llama/Llama-2-7b-chat-hf",
# #     "lmsys/vicuna-7b-v1.5",
# #     "NousResearch/Nous-Hermes-llama-2-7b",
# #     "garage-bAInd/Platypus2-7B",
# #     "WizardLM/WizardMath-7B-V1.0",
# #     "georgesung/llama2_7b_chat_uncensored",
# #     "allenai/tulu-2-7b",
# #     "PygmalionAI/pygmalion-2-7b",
# #     "h2oai/h2ogpt-4096-llama2-7b-chat",
# #     "stabilityai/StableBeluga-7B"
# # ]

# # MODELS_TO_TEST = [
# #     "meta-llama/Llama-2-13b-chat-hf",
# #     "lmsys/vicuna-13b-v1.5",
# #     "NousResearch/Nous-Hermes-Llama2-13b",
# #     "WizardLM/WizardLM-13B-V1.2",
# #     "garage-bAInd/Platypus2-13B",
# #     "stabilityai/StableBeluga-13B",
# #     "allenai/tulu-2-dpo-13b",
# #     "Open-Orca/OpenOrca-Platypus2-13B",
# #     "Riiid/sheep-duck-llama-2-13b",
# #     "Xwin-LM/Xwin-LM-13B-V0.1"
# # ]

# # MODELS_TO_TEST = [
# #     "lvwerra/gpt2-imdb",
# #     "Gustavosta/MagicPrompt-Stable-Diffusion",
# #     "mrm8488/GPT-2-finetuned-common_gen",
# #     "succinctly/text2image-prompt-generator",
# #     "shibing624/code-autocomplete-gpt2-base",
# #     "rhysjones/gpt2-124M-edu-fineweb-10B",
# #     "huggingtweets/elonmusk",
# #     "neulab/gpt2-finetuned-wikitext103",
# # 	"vicgalle/gpt2-open-instruct-v1",
# # 	"Ssarion/gpt2-multi-news"
# # ]

# # MODELS_TO_TEST = [
# #     "mistralai/Mistral-7B-Instruct-v0.1",
# #     "HuggingFaceH4/zephyr-7b-beta",
# #     "HuggingFaceH4/zephyr-7b-alpha",
# #     "Intel/neural-chat-7b-v3-1",
# #     "berkeley-nest/Starling-LM-7B-alpha",
# #     "argilla/notus-7b-v1",
# #     "jondurbin/airoboros-m-7b-3.1.2",
# #     "ehartford/samantha-1.2-mistral-7b",
# #     "migtissera/SynthIA-7B-v1.3",
# #     "TIGER-Lab/MAmmoTH-7B-Mistral"
# # ]

# # MODELS_TO_TEST = [
# #     "roberta-large-mnli",
# #     "deepset/roberta-large-squad2",
# #     "cross-encoder/stsb-roberta-large",
# #     "siebert/sentiment-roberta-large-english",
# #     "sentence-transformers/all-roberta-large-v1",
# #     "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
# #     "openai-community/roberta-large-openai-detector",
# #     "navteca/roberta-large-squad2",
# #     "cross-encoder/quora-roberta-large",
# #     "jean-baptiste/roberta-large-ner-english"
# # ]

# # MODELS_TO_TEST = [
# #     "deepset/roberta-base-squad2",
# #     "cross-encoder/stsb-roberta-base",
# #     "cross-encoder/nli-roberta-base",
# #     "textattack/roberta-base-SST-2",
# #     "openai-community/roberta-base-openai-detector",
# #     "textattack/roberta-base-MNLI",
# #     "cardiffnlp/twitter-roberta-base-sentiment-latest",
# #     "SamLowe/roberta-base-go_emotions",
# #     "textattack/roberta-base-ag-news",
# #     "cardiffnlp/twitter-roberta-base-hate-latest"
# # ]

# MODELS_TO_TEST = [
#     "meta-llama/Llama-3.1-70B-Instruct",
#     "NousResearch/Hermes-3-Llama-3.1-70B",
#     "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
#     "unsloth/Meta-Llama-3.1-70B-Instruct",
#     "mattshumer/Reflection-Llama-3.1-70B",
#     "VAGOsolutions/Llama-3.1-SauerkrautLM-70b-Instruct",
#     "allenai/Llama-3.1-Tulu-3-70B-SFT",
#     "nvidia/OpenMath2-Llama3.1-70B",
#     "migtissera/Tess-3-Llama-3.1-70B",
#     "mylesgoose/Llama-3.1-70B-Instruct-abliterated"
# ]
# # 用于存放最终聚合结果
# aggregated_results = {
#     "zstd": {},
#     "zlib": {},
#     "gzip": {}
# }

# # def get_and_load_model(model_id):
# #     """
# #     从 HuggingFace 下载模型并加载为字节流。
# #     优先寻找 safetensors，其次寻找 bin 文件。
# #     """
# #     try:
# #         # 尝试下载 safetensors
# #         # print(f"Downloading/Loading {model_id} (safetensors)...")
# #         folder = snapshot_download(model_id, allow_patterns=["*.safetensors"], local_files_only=False)
# #         files = glob(os.path.join(folder, "*.safetensors"))
        
# #         # 如果没有 safetensors，尝试下载 bin
# #         if not files:
# #             # print(f"Safetensors not found for {model_id}, trying pytorch_model.bin...")
# #             folder = snapshot_download(model_id, allow_patterns=["*.bin"], local_files_only=False)
# #             files = [f for f in glob(os.path.join(folder, "*.bin")) if "training_args" not in f]
        
# #         if not files: 
# #             print(f"No suitable model files found for {model_id}")
# #             return None, None
            
# #         raw_data = bytearray()
# #         for f in sorted(files):
# #             with open(f, 'rb') as rb:
# #                 raw_data.extend(rb.read())
# #         return raw_data, folder
# #     except Exception as e:
# #         print(f"Skipping {model_id} due to error: {e}")
# #         return None, None

# def get_and_load_model(model_id):
#     # 定义基础自定义路径
#     base_path = "/home/newdrive2/liu4441/temp_traditional/"
#     # 将模型 ID 中的 "/" 替换为下划线，作为子目录名
#     model_folder_name = model_id.replace("/", "_")
#     target_dir = os.path.join(base_path, model_folder_name)

#     try:
#         # 添加 local_dir 参数，HuggingFace 会自动检查该路径下是否已有文件
#         # print(f"Downloading/Loading {model_id} to {target_dir}...")
#         folder = snapshot_download(
#             model_id, 
#             allow_patterns=["*.safetensors"], 
#             local_files_only=False,
#             local_dir=target_dir  # 指定自定义路径
#         )
#         files = glob(os.path.join(folder, "*.safetensors"))
        
#         if not files:
#             folder = snapshot_download(
#                 model_id, 
#                 allow_patterns=["*.bin"], 
#                 local_files_only=False,
#                 local_dir=target_dir  # 指定自定义路径
#             )
#             files = [f for f in glob(os.path.join(folder, "*.bin")) if "training_args" not in f]
        
#         if not files: 
#             print(f"No suitable model files found for {model_id}")
#             return None, None
            
#         raw_data = bytearray()
#         for f in sorted(files):
#             with open(f, 'rb') as rb:
#                 raw_data.extend(rb.read())
#         return raw_data, folder
#     except Exception as e:
#         print(f"Skipping {model_id} due to error: {e}")
#         return None, None

# # ==========================================
# # 核心测试逻辑
# # ==========================================
# def run_benchmark():
#     # 调整表头宽度：
#     # Model Name 增加到 60 字符宽，以容纳较长的 huggingface ID
#     header = (
#         f"{'Model Name':<60} | {'Method':<8} | {'Orig(MB)':<10} | "
#         f"{'Comp%':<10} | {'C-Speed(MB/s)':<15} | {'D-Speed(MB/s)':<15}"
#     )
    
#     print("\n" + "="*len(header))
#     print(header)
#     print("-" * len(header))

#     # 直接遍历模型列表，不再使用索引别名
#     for model_id in MODELS_TO_TEST:
#         # 直接使用原始 ID
#         model_label = model_id
        
#         # 获取模型数据
#         raw_data, folder_path = get_and_load_model(model_id)
#         if raw_data is None: 
#             continue
        
#         orig_size = len(raw_data)
#         orig_mb = orig_size / (1024 * 1024)

#         # 定义算法及其对应的 (压缩函数, 解压函数)
#         methods = [
#             ("zlib", lambda d: zlib.compress(d, level=6), zlib.decompress),
#             ("gzip", lambda d: gzip.compress(d, compresslevel=6), gzip.decompress),
#             ("zstd", lambda d: zstd.ZstdCompressor(level=3).compress(d), zstd.ZstdDecompressor().decompress)
#         ]
        
#         for meth_name, comp_fn, decomp_fn in methods:
#             # 1. 测试压缩
#             t0 = time.perf_counter()
#             compressed_data = comp_fn(raw_data)
#             t1 = time.perf_counter()
#             comp_time = t1 - t0
            
#             # 2. 测试解压
#             t2 = time.perf_counter()
#             _ = decomp_fn(compressed_data)
#             t3 = time.perf_counter()
#             decomp_time = t3 - t2
            
#             # 3. 计算指标
#             ratio = (len(compressed_data) / orig_size) * 100
#             # 防止除以0
#             comp_speed = orig_mb / comp_time if comp_time > 0 else 0.0
#             decomp_speed = orig_mb / decomp_time if decomp_time > 0 else 0.0
            
#             # 4. 实时打印行 (Model Name列宽调整为60)
#             print(f"{model_label:<60} | {meth_name:<8} | {orig_mb:>8.1f} | {ratio:>8.2f}% | {comp_speed:>13.2f} | {decomp_speed:>13.2f}")
            
#             # 5. 存入聚合字典
#             aggregated_results[meth_name][model_label] = {
#                 "OrigMB": round(orig_mb, 2),
#                 "CompRatio": round(ratio, 2),
#                 "CompSpeed": round(comp_speed, 2),
#                 "DecompSpeed": round(decomp_speed, 2)
#             }
            
#             # 内存清理
#             del compressed_data, _
#             gc.collect()

#         # 模型级清理：删除内存中的原始数据和下载的临时文件夹
#         del raw_data
#         if folder_path and os.path.exists(folder_path):
#             try:
#                 shutil.rmtree(folder_path) 
#             except Exception as e:
#                 print(f"Warning: Failed to delete {folder_path}: {e}")
                
#         gc.collect()

#     # ==========================================
#     # 输出聚合结果
#     # ==========================================
#     print("\n" + "="*30)
#     print("AGGREGATED RESULTS BY METHOD")
#     print("="*30)
    
#     # 直接打印整个字典，或者按需遍历
#     for method, results in aggregated_results.items():
#         print(f"\nMethod: {method}")
#         print(json.dumps(results, indent=4))

# if __name__ == "__main__":
#     run_benchmark()



import os
import time
import zlib
import gzip
import zstandard as zstd
import shutil
import gc
import json
from glob import glob
from huggingface_hub import snapshot_download

# ==========================================
# 配置与初始化
# ==========================================

# 示例模型列表
# MODELS_TO_TEST = [
#     # bert-large-uncased
#     "assemblyai/bert-large-uncased-sst2",
#     "samrawal/bert-large-uncased_med-ner",
#     "yoshitomo-matsubara/bert-large-uncased-mnli",
#     "princeton-nlp/sup-simcse-bert-large-uncased",
#     "SarielSinLuo/bert-large-uncased-finetuned-cola",
#     "princeton-nlp/unsup-simcse-bert-large-uncased",
#     "yoshitomo-matsubara/bert-large-uncased-mrpc",
#     "yoshitomo-matsubara/bert-large-uncased-qnli",
#     "StevenLimcorn/bert-large-uncased-semeval2016-restaurants",
#     "Jorgeutd/bert-large-uncased-finetuned-ner"
# ]

# MODELS_TO_TEST = [
#     "meta-llama/Llama-3.1-8B-Instruct",
#     "NousResearch/Hermes-3-Llama-3.1-8B",
#     "meta-llama/Llama-Guard-3-8B",
#     "dphn/Dolphin3.0-Llama3.1-8B",
#     "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
#     "OpenSciLM/Llama-3.1_OpenScholar-8B",
#     "Sao10K/Llama-3.1-8B-Stheno-v3.4",
#     "cognitivecomputations/dolphin-2.9.4-llama3.1-8b",
#     "akjindal53244/Llama-3.1-Storm-8B",
#     "Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.2"
# ]

# MODELS_TO_TEST = [
#     "meta-llama/Llama-2-7b-chat-hf",
#     "lmsys/vicuna-7b-v1.5",
#     "NousResearch/Nous-Hermes-llama-2-7b",
#     "garage-bAInd/Platypus2-7B",
#     "WizardLM/WizardMath-7B-V1.0",
#     "georgesung/llama2_7b_chat_uncensored",
#     "allenai/tulu-2-7b",
#     "PygmalionAI/pygmalion-2-7b",
#     "h2oai/h2ogpt-4096-llama2-7b-chat",
#     "stabilityai/StableBeluga-7B"
# ]

# MODELS_TO_TEST = [
#     "meta-llama/Llama-2-13b-chat-hf",
#     "lmsys/vicuna-13b-v1.5",
#     "NousResearch/Nous-Hermes-Llama2-13b",
#     "WizardLM/WizardLM-13B-V1.2",
#     "garage-bAInd/Platypus2-13B",
#     "stabilityai/StableBeluga-13B",
#     "allenai/tulu-2-dpo-13b",
#     "Open-Orca/OpenOrca-Platypus2-13B",
#     "Riiid/sheep-duck-llama-2-13b",
#     "Xwin-LM/Xwin-LM-13B-V0.1"
# ]

# MODELS_TO_TEST = [
#     "lvwerra/gpt2-imdb",
#     "Gustavosta/MagicPrompt-Stable-Diffusion",
#     "mrm8488/GPT-2-finetuned-common_gen",
#     "succinctly/text2image-prompt-generator",
#     "shibing624/code-autocomplete-gpt2-base",
#     "rhysjones/gpt2-124M-edu-fineweb-10B",
#     "huggingtweets/elonmusk",
#     "neulab/gpt2-finetuned-wikitext103",
# 	"vicgalle/gpt2-open-instruct-v1",
# 	"Ssarion/gpt2-multi-news"
# ]

# MODELS_TO_TEST = [
#     "mistralai/Mistral-7B-Instruct-v0.1",
#     "HuggingFaceH4/zephyr-7b-beta",
#     "HuggingFaceH4/zephyr-7b-alpha",
#     "Intel/neural-chat-7b-v3-1",
#     "berkeley-nest/Starling-LM-7B-alpha",
#     "argilla/notus-7b-v1",
#     "jondurbin/airoboros-m-7b-3.1.2",
#     "ehartford/samantha-1.2-mistral-7b",
#     "migtissera/SynthIA-7B-v1.3",
#     "TIGER-Lab/MAmmoTH-7B-Mistral"
# ]

# MODELS_TO_TEST = [
#     "roberta-large-mnli",
#     "deepset/roberta-large-squad2",
#     "cross-encoder/stsb-roberta-large",
#     "siebert/sentiment-roberta-large-english",
#     "sentence-transformers/all-roberta-large-v1",
#     "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
#     "openai-community/roberta-large-openai-detector",
#     "navteca/roberta-large-squad2",
#     "cross-encoder/quora-roberta-large",
#     "jean-baptiste/roberta-large-ner-english"
# ]

MODELS_TO_TEST = [
    # "deepset/roberta-base-squad2",
    # "cross-encoder/stsb-roberta-base",
    # "cross-encoder/nli-roberta-base",
    # "textattack/roberta-base-SST-2",
    # "openai-community/roberta-base-openai-detector",
    # "textattack/roberta-base-MNLI",
    # "cardiffnlp/twitter-roberta-base-sentiment-latest",
    # "SamLowe/roberta-base-go_emotions",
    # "textattack/roberta-base-ag-news",
    # "cardiffnlp/twitter-roberta-base-hate-latest"
]

MODELS_TO_TEST = [
    "meta-llama/Llama-3.1-70B-Instruct",
    "NousResearch/Hermes-3-Llama-3.1-70B",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "unsloth/Meta-Llama-3.1-70B-Instruct",
    "mattshumer/Reflection-Llama-3.1-70B",
    "VAGOsolutions/Llama-3.1-SauerkrautLM-70b-Instruct",
    "allenai/Llama-3.1-Tulu-3-70B-SFT",
    "nvidia/OpenMath2-Llama3.1-70B",
    "migtissera/Tess-3-Llama-3.1-70B",
    "mylesgoose/Llama-3.1-70B-Instruct-abliterated"
]

# 用于存放最终聚合结果
aggregated_results = {
    "zstd": {},
    # "zlib": {},
    "gzip": {}
}

# 每次读取的块大小设为 256 MB，最大化 CPU 吞吐量并对齐 v17 的系统级内存表现
CHUNK_SIZE = 256 * 1024 * 1024 

def get_model_files(model_id):
    base_path = "/home/newdrive2/liu4441/temp_traditional/"
    model_folder_name = model_id.replace("/", "_")
    target_dir = os.path.join(base_path, model_folder_name)

    try:
        folder = snapshot_download(
            model_id, 
            allow_patterns=["*.safetensors"], 
            local_files_only=False,
            local_dir=target_dir
        )
        files = glob(os.path.join(folder, "*.safetensors"))
        
        if not files:
            folder = snapshot_download(
                model_id, 
                allow_patterns=["*.bin"], 
                local_files_only=False,
                local_dir=target_dir
            )
            files = [f for f in glob(os.path.join(folder, "*.bin")) if "training_args" not in f]
        
        if not files: 
            print(f"No suitable model files found for {model_id}")
            return [], 0
            
        total_size = sum(os.path.getsize(f) for f in files)
        return sorted(files), total_size
    except Exception as e:
        print(f"Skipping {model_id} due to error: {e}")
        return [], 0

def stream_test_method(files, method_name):
    comp_time = 0.0
    decomp_time = 0.0
    comp_size = 0

    if method_name == "zlib":
        c_obj = zlib.compressobj(level=6)
        d_obj = zlib.decompressobj()
    elif method_name == "gzip":
        c_obj = zlib.compressobj(level=6, wbits=31)
        d_obj = zlib.decompressobj(wbits=31)
    elif method_name == "zstd":
        c_obj = zstd.ZstdCompressor(level=3).compressobj()
        d_obj = zstd.ZstdDecompressor().decompressobj()

    for f in files:
        with open(f, 'rb') as rb:
            while True:
                chunk = rb.read(CHUNK_SIZE)
                if not chunk:
                    break
                
                t0 = time.perf_counter()
                c_chunk = c_obj.compress(chunk)
                t1 = time.perf_counter()
                comp_time += (t1 - t0)
                
                comp_size += len(c_chunk)
                
                t2 = time.perf_counter()
                d_chunk = d_obj.decompress(c_chunk)
                t3 = time.perf_counter()
                decomp_time += (t3 - t2)
                
                del chunk, c_chunk, d_chunk

    t0 = time.perf_counter()
    if method_name == "zstd":
        c_chunk = c_obj.flush(zstd.COMPRESSOBJ_FLUSH_FINISH)
    else:
        c_chunk = c_obj.flush()
    t1 = time.perf_counter()
    comp_time += (t1 - t0)
    
    comp_size += len(c_chunk)
    
    t2 = time.perf_counter()
    d_chunk = d_obj.decompress(c_chunk)
    if method_name != "zstd":
        d_chunk += d_obj.flush()
    t3 = time.perf_counter()
    decomp_time += (t3 - t2)

    del c_chunk, d_chunk
    return comp_size, comp_time, decomp_time

def run_benchmark():
    header = (
        f"{'Model Name':<60} | {'Method':<8} | {'Orig(MB)':<10} | "
        f"{'Comp%':<10} | {'C-Speed(MB/s)':<15} | {'D-Speed(MB/s)':<15}"
    )
    
    print("\n" + "="*len(header))
    print(header)
    print("-" * len(header))

    for model_id in MODELS_TO_TEST:
        model_label = model_id
        
        files, orig_size = get_model_files(model_id)
        if not files or orig_size == 0: 
            continue
        
        orig_mb = orig_size / (1024 * 1024)
        # methods = ["zlib", "gzip", "zstd"]
        methods = ["gzip", "zstd"]
        
        for meth_name in methods:
            comp_size, comp_time, decomp_time = stream_test_method(files, meth_name)
            
            ratio = (comp_size / orig_size) * 100
            comp_speed = orig_mb / comp_time if comp_time > 0 else 0.0
            decomp_speed = orig_mb / decomp_time if decomp_time > 0 else 0.0
            
            print(f"{model_label:<60} | {meth_name:<8} | {orig_mb:>8.1f} | {ratio:>8.2f}% | {comp_speed:>13.2f} | {decomp_speed:>13.2f}")
            
            aggregated_results[meth_name][model_label] = {
                "OrigMB": round(orig_mb, 2),
                "CompRatio": round(ratio, 2),
                "CompSpeed": round(comp_speed, 2),
                "DecompSpeed": round(decomp_speed, 2)
            }
            
            gc.collect()

    print("\n" + "="*30)
    print("AGGREGATED RESULTS BY METHOD")
    print("="*30)
    
    for method, results in aggregated_results.items():
        print(f"\nMethod: {method}")
        print(json.dumps(results, indent=4))

if __name__ == "__main__":
    run_benchmark()