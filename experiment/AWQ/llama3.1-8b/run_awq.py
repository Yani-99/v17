import torch
import time
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

def quantize_and_save_awq(model_id, output_dir, bits=4):
    print(f"开始加载Tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # 配置 AWQ 量化参数
    # zero_point: True 表示使用非对称量化 (Asymmetric quantization)，通常能获得更好的精度
    # q_group_size: 量化组大小，128 是 AWQ 推荐的默认值
    # w_bit: 量化比特数
    # version: "GEMM" 或 "GEMV"，通常使用 GEMM 以获得更好的推理吞吐量
    quant_config = {
        "zero_point": True, 
        "q_group_size": 128, 
        "w_bit": bits, 
        "version": "GEMM"
    }
    
    print(f"开始加载原始模型 (准备进行 AWQ 量化)...")
    # 使用 AutoAWQ 加载未量化的原始模型
    # model = AutoAWQForCausalLM.from_pretrained(
    #     model_id, 
    #     **{"low_cpu_mem_usage": True, "use_cache": False}
    # )
    model = AutoAWQForCausalLM.from_pretrained(
        model_id, 
        **{
            "low_cpu_mem_usage": True, 
            "use_cache": False, 
            "device_map": "auto",  # 核心修复：强制一次性将所有权重和缓冲矩阵全部塞进显卡！
            "torch_dtype": torch.float16  # 顺手锁死半精度，防止内存溢出
        }
    )
    
    print(f"开始量化模型 (比特数: {bits})... 这需要基于激活值校准并搜索最优缩放因子，可能会消耗一些时间。")
    start_time = time.time()
    
    # 触发量化过程
    # AWQ 的 quantize 方法默认会从 HuggingFace 自动下载一小部分维基百科/C4数据作为校准数据集。
    # 如果你想使用和 GPTQ 完全一样的一句话数据，也可以在此处构造一个列表并通过 calib_data=... 传入。
    model.quantize(tokenizer, quant_config=quant_config)
    
    end_time = time.time()
    print(f"量化完成！耗时: {end_time - start_time:.2f} 秒")
    
    print(f"将量化后的模型保存至: {output_dir}")
    # 注意：AWQ 库保存量化模型的方法名为 save_quantized，而不是 save_pretrained
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("保存完毕。")

if __name__ == "__main__":
    # 以你实验中用到的 Llama-2-7b-chat-hf 为例
    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" 
    OUTPUT_DIR = "/home/newdrive2/liu4441/Llama-3.1-8B-Instruct-awq-4bit"
    
    quantize_and_save_awq(MODEL_ID, OUTPUT_DIR, bits=4)