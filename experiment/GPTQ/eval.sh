# 测试原模型 (FP16)
python -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,dtype=float16 \
    --tasks wikitext \
    --device cuda:0 \
    --batch_size auto

# 测试刚才生成的 GPTQ 4-bit 模型
python -m lm_eval --model hf \
    --model_args pretrained=/home/newdrive2/liu4441/Llama-2-7b-chat-hf-gptq-4bit \
    --tasks wikitext \
    --device cuda:0 \
    --batch_size auto