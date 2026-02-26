import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import time

def quantize_and_save_gptq(model_id, output_dir, bits=4):
    print(f"[*] 开始加载Tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    dataset = [
        "The capital of France is Paris.", 
        "Machine learning is fascinating.", 
        "Language models are powerful."
    ]
    
    gptq_config = GPTQConfig(
        bits=bits, 
        dataset=dataset,
        tokenizer=tokenizer,
        desc_act=False,
    )
    
    print(f"[*] 开始加载并量化模型 (比特数: {bits})... 这可能需要一些时间。")
    start_time = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=gptq_config,
        device_map="auto",
        torch_dtype=torch.float16, 
    )
    
    end_time = time.time()
    print(f"[*] 量化完成！耗时: {end_time - start_time:.2f} 秒")
    
    print(f"[*] 将量化后的模型保存至: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[*] 保存完毕。")

if __name__ == "__main__":
    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" 
    OUTPUT_DIR = "/home/newdrive2/liu4441/Llama-3.1-8B-Instruct-gptq-4bit"
    
    quantize_and_save_gptq(MODEL_ID, OUTPUT_DIR, bits=4)



# import time
# import torch
# from transformers import AutoTokenizer
# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# def quantize_and_save_gptq(model_id, output_dir, bits=4):
#     print(f"[*] 开始加载Tokenizer: {model_id}")
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
    
#     # 1. 直接使用 auto_gptq 原生的配置类
#     quantize_config = BaseQuantizeConfig(
#         bits=bits, 
#         group_size=128,  # GPTQ 标准组大小
#         desc_act=False   # 设为 False 以提高后续的推理速度
#     )
    
#     print(f"[*] 开始加载模型并准备量化 (比特数: {bits})... 这可能需要一些时间。")
#     # 2. 直接使用 AutoGPTQForCausalLM 加载模型
#     model = AutoGPTQForCausalLM.from_pretrained(
#         model_id,
#         quantize_config=quantize_config,
#         low_cpu_mem_usage=True,
#         device_map="auto"
#     )
    
#     # 3. 准备格式化的校准数据集 (核心修复：必须强制等长)
#     texts = [
#         "The capital of France is Paris. " * 10, 
#         "Machine learning is fascinating. " * 10, 
#         "Language models are powerful. " * 10
#     ]
#     examples = []
    
#     # Llama 默认没有 pad_token，需要为其指定一个，防止 padding 报错
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
        
#     for text in texts:
#         # 强制将所有样本处理为完全相同的长度 (例如统一为 64 个 Token)
#         # 这样 auto_gptq 缓存的位置编码就能完美契合每一条数据
#         encodings = tokenizer(
#             text, 
#             return_tensors="pt", 
#             padding="max_length", 
#             max_length=64, 
#             truncation=True
#         )
#         examples.append({
#             "input_ids": encodings.input_ids,
#             "attention_mask": encodings.attention_mask
#         })
    
#     print("[*] 开始执行底层 GPTQ 校准与量化 (预计耗时几分钟)...")
#     start_time = time.time()
    
#     # 4. 执行核心量化
#     model.quantize(examples)
    
#     end_time = time.time()
#     print(f"[*] 量化完成！耗时: {end_time - start_time:.2f} 秒")
    
#     print(f"[*] 将量化后的模型保存至: {output_dir}")
#     # 5. 原生保存为 safetensors 格式
#     model.save_quantized(output_dir, use_safetensors=True)
#     tokenizer.save_pretrained(output_dir)
#     print("[*] 保存完毕。")

# if __name__ == "__main__":
#     MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" 
#     OUTPUT_DIR = "/home/newdrive2/liu4441/Llama-3.1-8B-Instruct-gptq-4bit"
    
#     quantize_and_save_gptq(MODEL_ID, OUTPUT_DIR, bits=4)




# # import torch
# # from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
# # import time

# # def quantize_and_save_gptq(model_id, output_dir, bits=4):
# #     print(f"开始加载Tokenizer: {model_id}")
# #     tokenizer = AutoTokenizer.from_pretrained(model_id)
    
# #     # 准备校准数据集，GPTQ需要使用一小部分数据来校准量化误差
# #     # 这里使用 wikitext 作为示例校准数据
# #     dataset = ["The capital of France is Paris.", "Machine learning is fascinating.", "Language models are powerful."]
    
# #     # 配置 GPTQ 量化参数
# #     gptq_config = GPTQConfig(
# #         bits=bits, 
# #         dataset=dataset, # 可以传入数据集名称如 "c4" 或自定义列表
# #         tokenizer=tokenizer,
# #         desc_act=False,  # 设为 True 可能会提高精度，但减慢推理速度
# #     )
    
# #     print(f"开始加载并量化模型 (比特数: {bits})... 这可能需要一些时间。")
# #     start_time = time.time()
    
# #     # 加载模型时即触发量化
# #     # 已经将 torch_dtype 替换为部分新版本库推荐的参数形式
# #     model = AutoModelForCausalLM.from_pretrained(
# #         model_id,
# #         quantization_config=gptq_config,
# #         device_map="auto",
# #         torch_dtype=torch.float16, # 升级 transformers 后，标准的 torch_dtype 会恢复正常工作
# #     )
    
# #     end_time = time.time()
# #     print(f"量化完成！耗时: {end_time - start_time:.2f} 秒")
    
# #     print(f"将量化后的模型保存至: {output_dir}")
# #     model.save_pretrained(output_dir)
# #     tokenizer.save_pretrained(output_dir)
# #     print("保存完毕。")

# # if __name__ == "__main__":
# #     # 以你实验中用到的 Llama-2-7b-hf 为例
# #     MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" 
# #     OUTPUT_DIR = "/home/newdrive2/liu4441/Llama-3.1-8B-Instruct-gptq-4bit"
    
# #     quantize_and_save_gptq(MODEL_ID, OUTPUT_DIR, bits=4)