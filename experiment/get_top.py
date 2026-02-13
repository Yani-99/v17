import json
import shutil
import tempfile
import os
from huggingface_hub import HfApi, hf_hub_download

def get_structural_fingerprint(config):
    """提取结构指纹 (与之前逻辑一致)"""
    fingerprint = {}
    fingerprint['vocab_size'] = config.get('vocab_size')
    fingerprint['hidden_size'] = config.get('hidden_size') or config.get('d_model') or config.get('n_embd')
    fingerprint['num_layers'] = config.get('num_hidden_layers') or config.get('n_layer') or config.get('num_layers')
    fingerprint['num_heads'] = config.get('num_attention_heads') or config.get('n_head')
    return fingerprint

def find_structurally_compatible_models_clean(base_model_name, top_k=5):
    api = HfApi()
    
    # 1. 创建一个临时目录用于存放下载的 config.json
    # tempfile.mkdtemp() 会在系统临时区创建一个随机命名的文件夹
    temp_cache_dir = tempfile.mkdtemp(prefix="hf_config_check_")
    print(f"📁 创建临时缓存目录: {temp_cache_dir}")
    print(f"   (任务结束后将自动删除此目录)")

    try:
        print(f"📊 正在获取基础模型 [{base_model_name}] 的结构指纹...")
        
        # 下载基础模型 config 到临时目录
        base_config_path = hf_hub_download(
            base_model_name, 
            "config.json", 
            cache_dir=temp_cache_dir  # <--- 关键点：指定缓存路径
        )
        with open(base_config_path, 'r') as f:
            base_config = json.load(f)
        base_fp = get_structural_fingerprint(base_config)
        print(f"   基础模型指纹: {base_fp}")

        print(f"🔍 正在搜索并比对...")
        candidates = api.list_models(
            search=base_model_name.split("/")[-1],
            sort="downloads",
            direction=-1,
            limit=100
        )
        
        valid_models = []
        
        for model in candidates:
            if model.id == base_model_name: continue
            if "whole-word-masking" in model.id.lower(): continue
            invalid_keywords = ["openai-community", "xenova", "onnx", "quantized"]
            
            is_invalid = False
            for kw in invalid_keywords:
                if kw in model.id.lower():
                    is_invalid = True
                    break
            if is_invalid:
                continue
            try:
                # 下载候选模型 config 到临时目录
                config_path = hf_hub_download(
                    model.id, 
                    "config.json", 
                    cache_dir=temp_cache_dir # <--- 关键点：指定缓存路径
                )
                
                with open(config_path, 'r') as f:
                    cand_config = json.load(f)
                
                cand_fp = get_structural_fingerprint(cand_config)
                
                if cand_fp == base_fp:
                    valid_models.append(model)
                    print(f"✅ [匹配] {model.id}")
            except:
                continue
                
            if len(valid_models) >= top_k:
                break
                
        return valid_models

    finally:
        # 2. 清理工作：无论程序是否出错，都在最后删除临时目录
        if os.path.exists(temp_cache_dir):
            shutil.rmtree(temp_cache_dir)
            print(f"🧹 已删除临时缓存目录: {temp_cache_dir}")

# --- 使用示例 ---
if __name__ == "__main__":
    # target = "gpt2"
    # target = "bert-large-uncased"
    # target = "meta-llama/Llama-2-7b-hf"
    # target = "meta-llama/Meta-Llama-3.1-8B"
    target = "meta-llama/Llama-2-7b-hf"
    
    final_list = find_structurally_compatible_models_clean(target, top_k=25)
    
    print(f"\n🎉 最终结果 (无残留文件):")
    for i, m in enumerate(final_list, 1):
        print(f"{i}. {m.id}")