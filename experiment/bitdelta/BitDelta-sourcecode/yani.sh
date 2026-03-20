# CUDA_VISIBLE_DEVICES=0,1,2 python bitdelta/train.py \
#     --base_model meta-llama/Llama-2-7b-hf \
#     --finetuned_model meta-llama/Llama-2-7b-chat-hf \
#     --save_dir /home/newdrive2/liu4441/bitdelta \
#     --batch_size 4 \
#     --num_steps 200 \
#     --save_full_model True



#!/bin/bash

# 定义显卡空闲的显存阈值（MB），低于 1000MB 认为该卡处于空闲状态
IDLE_THRESHOLD=1000
# 定义日志输出路径
LOG_FILE="bitdelta_train.log"

echo "开始监控 3x RTX 3090 状态，等待两张显卡同时空闲..." | tee -a $LOG_FILE

while true; do
    # 获取所有 GPU 的已用显存大小（纯数字格式）
    mem_usages=($(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits))
    
    free_gpus=()
    
    # 遍历检测空闲显卡
    for i in "${!mem_usages[@]}"; do
        if [ "${mem_usages[$i]}" -lt "$IDLE_THRESHOLD" ]; then
            free_gpus+=($i)
        fi
    done

    # 判断是否凑齐了 2 张空闲显卡来避免 OOM
    if [ "${#free_gpus[@]}" -ge 2 ]; then
        GPU1=${free_gpus[0]}
        GPU2=${free_gpus[1]}
        
        echo "$(date '+%Y-%m-%d %H:%M:%S') - 发现空闲显卡 GPU $GPU1 和 GPU $GPU2，开始执行 BitDelta 压缩..." | tee -a $LOG_FILE
        
        # 自动配置 CUDA_VISIBLE_DEVICES，并分配模型加载的设备 ID
        CUDA_VISIBLE_DEVICES=$GPU1,$GPU2 python bitdelta/train.py \
            --base_model meta-llama/Llama-2-7b-hf \
            --finetuned_model meta-llama/Llama-2-7b-chat-hf \
            --base_model_device cuda:0 \
            --finetuned_model_device cuda:1 \
            --save_dir /home/newdrive2/liu4441/bitdelta \
            --batch_size 4 \
            --num_steps 200 \
            --save_full_model True >> $LOG_FILE 2>&1
        
        echo "$(date '+%Y-%m-%d %H:%M:%S') - 任务执行完毕，详情请查看 $LOG_FILE。" | tee -a $LOG_FILE
        break
    else
        # 等待 60 秒后再次轮询检测
        sleep 60
    fi
done

# CUDA_VISIBLE_DEVICES=0 python bitdelta/train.py     --base_model gpt2     --finetuned_model lvwerra/gpt2-imdb     --base_model_device 0     --finetuned_model_device 0     --finetuned_compressed_model_device 0     --save_dir /home/newdrive2/liu4441/bitdelta     --batch_size 4     --num_steps 200     --save_full_model True > gpt2.log


# CUDA_VISIBLE_DEVICES=0 python bitdelta/train.py     --base_model bert-large-uncased     --finetuned_model assemblyai/bert-large-uncased-sst2     --base_model_device 0     --finetuned_model_device 0     --finetuned_compressed_model_device 0     --save_dir /home/newdrive2/liu4441/bitdelta     --batch_size 4     --num_steps 200     --save_full_model True > gpt2.log

# CUDA_VISIBLE_DEVICES=0 python bitdelta/train.py     --base_model roberta-large     --finetuned_model roberta-large-mnli     --base_model_device 0     --finetuned_model_device 0     --finetuned_compressed_model_device 0     --save_dir /home/newdrive2/liu4441/bitdelta     --batch_size 4     --num_steps 200     --save_full_model True > gpt2.log

# CUDA_VISIBLE_DEVICES=0 python bitdelta/train.py     --base_model roberta-base     --finetuned_model textattack/roberta-base-SST-2     --base_model_device 0     --finetuned_model_device 0     --finetuned_compressed_model_device 0     --save_dir /home/newdrive2/liu4441/bitdelta     --batch_size 4     --num_steps 200     --save_full_model True > gpt2.log



# CUDA_VISIBLE_DEVICES=1 python bitdelta/train.py     --base_model meta-llama/Llama-3.1-8B     --finetuned_model meta-llama/Llama-3.1-8B-Instruct     --base_model_device cpu     --finetuned_model_device cpu     --finetuned_compressed_model_device 0     --save_dir /home/newdrive2/liu4441/bitdelta     --batch_size 4     --num_steps 200     --save_full_model True > LLama3.1.log


 CUDA_VISIBLE_DEVICES=0,1,2 python bitdelta/train.py      --base_model meta-llama/Llama-3.1-8B      --finetuned_model meta-llama/Llama-3.1-8B-Instruct      --base_model_device 0      --finetuned_model_device 1      --finetuned_compressed_model_device 2      --save_dir /home/newdrive2/liu4441/bitdelta      --batch_size 4      --num_steps 200      --save_full_model True > LLama3.1.log


 运行这个：
 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2 python bitdelta/train.py \
     --base_model meta-llama/Llama-3.1-8B \
     --finetuned_model meta-llama/Llama-3.1-8B-Instruct \
     --base_model_device 0 \
     --finetuned_model_device 1 \
     --finetuned_compressed_model_device 2 \
     --save_dir /home/newdrive2/liu4441/bitdelta2 \
     --batch_size 4 \
     --num_steps 200 \
     --save_full_model True > LLama3.1.log