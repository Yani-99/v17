import torch
import numpy as np
from transformers import (
    AutoConfig, AutoModel, AutoModelForCausalLM, 
    AutoModelForSequenceClassification, AutoModelForTokenClassification,
    AutoModelForImageClassification, 
    AutoTokenizer, AutoImageProcessor,
    Trainer, TrainingArguments,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    logging as hf_logging
)
from datasets import load_dataset, logging as ds_logging
import evaluate 
import logging
import time
import json
from utils import smart_column_filter
logger = logging.getLogger(__name__)


class UniversalEvaluator:
    def __init__(self, model, processor, config, run_tag="baseline", device="cuda"):
        self.model = model
        self.processor = processor 
        self.cfg = config
        self.limit = config.get("eval_limit", None)
        self.run_tag = run_tag 
        self.device = device

    def run(self):
        torch.cuda.empty_cache()
        tt = self.cfg['task_type']
        
        metrics = {}
        
        # --- Dispatcher ---
        if tt == "LLM_HARNESS": 
            # 1. Run MMLU/GSM8K
            if self.cfg.get("llm_tasks") and len(self.cfg["llm_tasks"]) > 0:
                try:
                    metrics.update(self._run_lm_harness())
                except Exception as e:
                    logger.error(f"[ERROR] LM-Harness tasks failed: {e}")
            else:
                logger.info("[CONFIG] Skipping LM-Harness tasks (List is empty).")

            # 2. Run MT-Bench
            if self.cfg.get("run_mt_bench", False):
                try:
                    metrics.update(self._run_mt_bench_gen())
                except Exception as e:
                    logger.error(f"[ERROR] MT-Bench Generation failed: {e}")

        elif tt == "LLM_PPL": 
            metrics.update(self._run_domain_ppl())
        elif tt == "NER": 
            metrics.update(self._run_ner_eval())
        elif tt == "GLUE": 
            metrics.update(self._run_glue_eval())
        elif tt == "CV": 
            metrics.update(self._run_cv_eval())
            
        return metrics


    # def _run_lm_harness(self):
    #     logger.info(f"Preparing LM-Eval Harness: {self.cfg['llm_tasks']}...")
        
    #     import lm_eval
    #     from lm_eval.models.huggingface import HFLM
        
    #     # 1. Detect Multi-GPU Sharding
    #     eval_device = self.device
    #     is_multi_gpu = False
    #     if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
    #         logger.info("Detected multi-GPU model sharding (Accelerate).")
    #         eval_device = None # Let Accelerate handle placement
    #         is_multi_gpu = True

    #     # 2. Initialize HFLM Wrapper
    #     # We explicitly pass the model instance we already loaded
    #     lm_obj = HFLM(
    #         pretrained=self.model, 
    #         tokenizer=self.processor, 
    #         batch_size=1, 
    #         device=eval_device 
    #     )

    #     # 3. [CRITICAL FIX] Multi-GPU Patch for "generate_until" (GSM8K etc)
    #     # The previous patch on model.generate was not enough. 
    #     # We need to ensure inputs match the device of the first layer.
    #     if is_multi_gpu:
    #         logger.info("Applying robust multi-GPU generation patch...")
            
    #         # Find the device of the input embeddings (first layer)
    #         first_device = self.model.device 
    #         if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
    #              first_device = self.model.model.embed_tokens.weight.device
            
    #         # Hook into HFLM's internal tokenization/generation flow is hard.
    #         # Instead, we monkey-patch the HFLM._model_generate method which lm_eval uses.
            
    #         original_model_generate = lm_obj._model_generate

    #         def patched_model_generate(context, max_length, stop, **generation_kwargs):
    #             # Ensure context (input_ids) is on the correct start device
    #             if isinstance(context, torch.Tensor):
    #                 context = context.to(first_device)
                
    #             # Call original generation
    #             # Accelerate will handle the movement across layers (cuda:0 -> cuda:1)
    #             res = original_model_generate(context, max_length, stop, **generation_kwargs)
                
    #             # Accelerate leaves the output on the last device (e.g., cuda:1).
    #             # lm_eval expects it on the same device as input (cuda:0) or CPU.
    #             # We move it back to the input device to prevent "Expected all tensors..." error.
    #             if isinstance(res, torch.Tensor):
    #                 return res.to(first_device)
    #             return res

    #         # Apply the patch to the wrapper, not the model itself
    #         lm_obj._model_generate = patched_model_generate

    #     limit = self.limit if self.limit else None
    #     max_retries = 3
        
    #     # Retry loop
    #     raw_results = {}
    #     for attempt in range(max_retries):
    #         try:
    #             logger.info(f"Running LM-Eval (Attempt {attempt+1}/{max_retries})...")
    #             # Force empty cache before start
    #             if is_multi_gpu: torch.cuda.empty_cache()
                
    #             raw_results = lm_eval.simple_evaluate(
    #                 model=lm_obj, 
    #                 tasks=self.cfg['llm_tasks'], 
    #                 device=eval_device, 
    #                 limit=limit, 
    #                 log_samples=False
    #             )
    #             break 
    #         except Exception as e:
    #             logger.warning(f"LM-Eval Error: {e}")
    #             import traceback
    #             traceback.print_exc() # Print full stack trace for debugging
                
    #             if attempt < max_retries - 1:
    #                 wait = (attempt + 1) * 10
    #                 logger.info(f"Retry in {wait}s...")
    #                 time.sleep(wait)
    #             else:
    #                 logger.error("LM-Eval Failed after max retries.")
    #                 return {}
        
    #     # --- ROBUST METRIC EXTRACTION ---
    #     # (This part remains exactly the same as your previous code)
    #     metrics = {}
        
    #     if "groups" in raw_results:
    #         for group_name, res in raw_results["groups"].items():
    #             if "acc" in res:
    #                 metrics[f"{group_name}/acc"] = res["acc"]
    #             if "acc,none" in res:
    #                 metrics[f"{group_name}/acc"] = res["acc,none"]

    #     if "results" in raw_results:
    #         mmlu_scores = []
    #         gsm8k_candidates = []

    #         for task, res in raw_results["results"].items():
    #             if task.startswith("mmlu"):
    #                 val = res.get("acc") or res.get("acc,none")
    #                 if val is not None:
    #                     mmlu_scores.append(val)
                
    #             if "gsm8k" in task.lower():
    #                 val = None
    #                 priorities = ["strict_match,none", "exact_match,none", "acc,none", 
    #                               "strict_match", "exact_match", "acc"]
    #                 for p in priorities:
    #                     if p in res:
    #                         val = res[p]
    #                         break
    #                 if val is None:
    #                     for k, v in res.items():
    #                         if isinstance(v, (int, float)) and ("acc" in k or "match" in k):
    #                             val = v
    #                             break
    #                 if val is not None:
    #                     gsm8k_candidates.append(val)

    #         if mmlu_scores and "mmlu/acc" not in metrics:
    #             metrics["mmlu/acc"] = sum(mmlu_scores) / len(mmlu_scores)
            
    #         if gsm8k_candidates and "gsm8k/acc" not in metrics:
    #             metrics["gsm8k/acc"] = gsm8k_candidates[0]

    #     return metrics

    def _run_lm_harness(self):
        logger.info(f"Preparing LM-Eval Harness: {self.cfg['llm_tasks']}...")
        
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        
        # 1. Detect Multi-GPU Sharding
        eval_device = self.device
        is_multi_gpu = False
        if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
            logger.info("Detected multi-GPU model sharding (Accelerate).")
            eval_device = None # Let Accelerate handle placement
            is_multi_gpu = True

        # 2. Initialize HFLM Wrapper
        # We explicitly pass the model instance we already loaded
        # lm_obj = HFLM(
        #     pretrained=self.model, 
        #     tokenizer=self.processor, 
        #     batch_size=1, 
        #     device=eval_device 
        # )

        eval_batch_size = self.cfg.get("batch_size", 1)
        
        lm_obj = HFLM(
            pretrained=self.model, 
            tokenizer=self.processor, 
            batch_size=eval_batch_size, 
            device=eval_device 
        )

        # 3. [CRITICAL FIX] Multi-GPU Patch for "generate_until" (GSM8K etc)
        # The previous patch on model.generate was not enough. 
        # We need to ensure inputs match the device of the first layer.
        if is_multi_gpu:
            logger.info("Applying robust multi-GPU generation patch...")
            
            # Find the device of the input embeddings (first layer)
            first_device = self.model.device 
            if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
                 first_device = self.model.model.embed_tokens.weight.device
            
            # Hook into HFLM's internal tokenization/generation flow is hard.
            # Instead, we monkey-patch the HFLM._model_generate method which lm_eval uses.
            
            original_model_generate = lm_obj._model_generate

            def patched_model_generate(context, max_length, stop, **generation_kwargs):
                # Ensure context (input_ids) is on the correct start device
                if isinstance(context, torch.Tensor):
                    context = context.to(first_device)
                
                # Call original generation
                # Accelerate will handle the movement across layers (cuda:0 -> cuda:1)
                res = original_model_generate(context, max_length, stop, **generation_kwargs)
                
                # Accelerate leaves the output on the last device (e.g., cuda:1).
                # lm_eval expects it on the same device as input (cuda:0) or CPU.
                # We move it back to the input device to prevent "Expected all tensors..." error.
                if isinstance(res, torch.Tensor):
                    return res.to(first_device)
                return res

            # Apply the patch to the wrapper, not the model itself
            lm_obj._model_generate = patched_model_generate

        limit = self.limit if self.limit else None
        max_retries = 3
        
        # Retry loop
        raw_results = {}
        for attempt in range(max_retries):
            try:
                logger.info(f"Running LM-Eval (Attempt {attempt+1}/{max_retries})...")
                # Force empty cache before start
                if is_multi_gpu: torch.cuda.empty_cache()
                
                raw_results = lm_eval.simple_evaluate(
                    model=lm_obj, 
                    tasks=self.cfg['llm_tasks'], 
                    device=eval_device, 
                    limit=limit, 
                    log_samples=False
                )
                break 
            except Exception as e:
                logger.warning(f"LM-Eval Error: {e}")
                import traceback
                traceback.print_exc() # Print full stack trace for debugging
                
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 10
                    logger.info(f"Retry in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error("LM-Eval Failed after max retries.")
                    return {}
        
        # --- ROBUST METRIC EXTRACTION ---
        metrics = {}
        
        if "groups" in raw_results:
            for group_name, res in raw_results["groups"].items():
                if "acc" in res:
                    metrics[f"{group_name}/acc"] = res["acc"]
                if "acc,none" in res:
                    metrics[f"{group_name}/acc"] = res["acc,none"]

        if "results" in raw_results:
            mmlu_scores = []
            gsm8k_candidates = []

            for task, res in raw_results["results"].items():
                if task.startswith("mmlu"):
                    val = res.get("acc") or res.get("acc,none")
                    if val is not None:
                        mmlu_scores.append(val)
                
                if "gsm8k" in task.lower():
                    val = None
                    priorities = ["strict_match,none", "exact_match,none", "acc,none", 
                                  "strict_match", "exact_match", "acc"]
                    for p in priorities:
                        if p in res:
                            val = res[p]
                            break
                    if val is None:
                        for k, v in res.items():
                            if isinstance(v, (int, float)) and ("acc" in k or "match" in k):
                                val = v
                                break
                    if val is not None:
                        gsm8k_candidates.append(val)
                
                # --- 新增 IFEval 指标提取逻辑 ---
                if "ifeval" in task.lower():
                    for k, v in res.items():
                        if isinstance(v, (int, float)):
                            if k in ["prompt_level_strict_acc,none", "prompt_level_strict_acc"]:
                                metrics["ifeval/prompt_strict_acc"] = v
                            elif k in ["inst_level_strict_acc,none", "inst_level_strict_acc"]:
                                metrics["ifeval/inst_strict_acc"] = v
                            elif k in ["prompt_level_loose_acc,none", "prompt_level_loose_acc"]:
                                metrics["ifeval/prompt_loose_acc"] = v
                            elif k in ["inst_level_loose_acc,none", "inst_level_loose_acc"]:
                                metrics["ifeval/inst_loose_acc"] = v

            if mmlu_scores and "mmlu/acc" not in metrics:
                metrics["mmlu/acc"] = sum(mmlu_scores) / len(mmlu_scores)
            
            if gsm8k_candidates and "gsm8k/acc" not in metrics:
                metrics["gsm8k/acc"] = gsm8k_candidates[0]

        return metrics

    def _run_mt_bench_gen(self):
        logger.info("Running MT-Bench Generation (2-Turn) [Universal Chat Template]...")
        
        try:
            dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        except Exception as e:
            logger.error(f"Failed to load MT-Bench prompts: {e}")
            return {}

        # ====================================================
        # [FIX] Manually inject chat_template if missing
        # ====================================================
        if self.processor.chat_template is None:
            model_name_lower = self.cfg['ft_model'].lower()
            logger.warning(f"Chat template is None. Attempting manual injection for {self.cfg['ft_model']}...")
            
            if "mistral" in model_name_lower:
                # Mistral Standard Template: <s>[INST] Instruction [/INST] Model Answer</s>
                self.processor.chat_template = (
                    "{{ bos_token }}"
                    "{% for message in messages %}"
                        "{% if (message['role'] == 'user') %}"
                            "[INST] {{ message['content'] }} [/INST]"
                        "{% elif (message['role'] == 'assistant') %}"
                            "{{ message['content'] + eos_token }}"
                        "{% endif %}"
                    "{% endfor %}"
                )
                logger.info("-> Injected Mistral chat template.")
            
            elif "llama-3" in model_name_lower:
                # Llama-3 Standard Template
                self.processor.chat_template = (
                    "{% set loop_messages = messages %}"
                    "{% for message in loop_messages %}"
                        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
                        "{% if loop.index0 == 0 %}"
                            "{% set content = bos_token + content %}"
                        "{% endif %}"
                        "{{ content }}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                    "{% endif %}"
                )
                logger.info("-> Injected Llama-3 chat template.")

        results = []
        total_tokens = 0
        start_time = time.time()
        
        eval_data = dataset
        if self.limit: 
            eval_data = dataset.select(range(min(len(dataset), self.limit)))
        
        count = 0
        for item in eval_data:
            count += 1
            q_id = item.get('prompt_id') or item.get('question_id')
            category = item.get('category', 'unknown')
            prompts = item.get('prompt', []) 
            
            if not prompts or q_id is None:
                continue

            # ====================================================
            # Turn 1
            # ====================================================
            messages = [{"role": "user", "content": prompts[0]}]
            
            try:
                # Now this should work because we injected the template
                prompt_t1 = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception as e:
                # Last resort fallback (Mistral/Llama3 won't like this, but better than crashing)
                logger.warning(f"Template apply failed: {e}. Using raw fallback.")
                prompt_t1 = f"[INST] {prompts[0]} [/INST]" if "mistral" in self.cfg['ft_model'].lower() else f"User: {prompts[0]}\nAssistant:"

            inputs_t1 = self.processor(prompt_t1, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out_t1 = self.model.generate(
                    **inputs_t1, 
                    max_new_tokens=1024, 
                    do_sample=True, 
                    temperature=0.7,
                    pad_token_id=self.processor.pad_token_id or self.processor.eos_token_id
                )
            
            gen_len_t1 = out_t1.shape[1] - inputs_t1.input_ids.shape[1]
            total_tokens += gen_len_t1
            ans_t1 = self.processor.decode(out_t1[0][inputs_t1.input_ids.shape[1]:], skip_special_tokens=True)

            # ====================================================
            # Turn 2
            # ====================================================
            messages.append({"role": "assistant", "content": ans_t1})
            messages.append({"role": "user", "content": prompts[1]})

            try:
                prompt_t2 = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except:
                # Fallback for manual construction if template fails
                if "mistral" in self.cfg['ft_model'].lower():
                    prompt_t2 = f"{prompt_t1}{ans_t1}</s>[INST] {prompts[1]} [/INST]"
                else:
                    prompt_t2 = f"{prompt_t1}{ans_t1}\nUser: {prompts[1]}\nAssistant:"

            inputs_t2 = self.processor(prompt_t2, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out_t2 = self.model.generate(
                    **inputs_t2, 
                    max_new_tokens=1024, 
                    do_sample=True, 
                    temperature=0.7,
                    pad_token_id=self.processor.pad_token_id or self.processor.eos_token_id
                )

            gen_len_t2 = out_t2.shape[1] - inputs_t2.input_ids.shape[1]
            total_tokens += gen_len_t2
            ans_t2 = self.processor.decode(out_t2[0][inputs_t2.input_ids.shape[1]:], skip_special_tokens=True)

            results.append({
                "question_id": q_id,
                "category": category,
                "model_id": f"compressed-{self.run_tag}",
                "choices": [{"index": 0, "turns": [ans_t1, ans_t2]}]
            })

        duration = time.time() - start_time
        tps = total_tokens / duration if duration > 0 else 0
        
        logger.info(f"Processed {count} MT-Bench items. TPS: {tps:.2f}")

        out_file = f"mt_bench_answers_{self.run_tag}.jsonl"
        with open(out_file, "w", encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        return {
            "mt_bench/speed": tps,
            "mt_bench/saved": 1.0 
        }

    def _run_domain_ppl(self):
        logger.info(f"Evaluating PPL on {self.cfg['dataset_name']}...")
        data = load_dataset(self.cfg['dataset_name'], self.cfg['dataset_config'], split=self.cfg['split'])
        if self.limit: data = data.select(range(min(len(data), self.limit)))
        
        text_col = self.cfg.get("text_col", "text")
        raw_texts = [x for x in data[text_col] if x]
        encodings = self.processor("\n\n".join(raw_texts), return_tensors="pt")
        
        max_length = self.model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc 
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return {"domain_ppl": ppl.item()}

    def _run_ner_eval(self):
        data = load_dataset(self.cfg['dataset_name'], self.cfg['dataset_config'], split=self.cfg['split'])
        if self.limit: data = data.select(range(min(len(data), self.limit)))
        metric = evaluate.load("seqeval")
        
        def tokenize(examples):
            tokenized = self.processor(examples["tokens"], truncation=True, is_split_into_words=True)
            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized.word_ids(batch_index=i)
                labels.append([-100 if w is None else label[w] for w in word_ids])
            tokenized["labels"] = labels
            return tokenized

        keep_cols = ["labels", "input_ids", "attention_mask", "ner_tags"]
        tokenized_data = data.map(tokenize, batched=True, remove_columns=smart_column_filter(data.column_names, keep_cols))
        
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir="/tmp/eval", per_device_eval_batch_size=32, report_to="none"),
            eval_dataset=tokenized_data,
            data_collator=DataCollatorForTokenClassification(self.processor),
            compute_metrics=lambda p: compute_ner_metrics(p, metric, data.features["ner_tags"].feature.names)
        )
        return trainer.evaluate()

    def _run_glue_eval(self):
        data = load_dataset(self.cfg['dataset_name'], self.cfg['dataset_config'], split=self.cfg['split'])
        if self.limit: data = data.select(range(min(len(data), self.limit)))
        metric = evaluate.load("glue", self.cfg['dataset_config'])
        cols = data.column_names
        
        key1, key2 = "sentence", None
        if "premise" in cols: key1, key2 = "premise", "hypothesis"
        elif "sentence1" in cols: key1, key2 = "sentence1", "sentence2"
        elif "question" in cols: key1, key2 = "question", "sentence"
        
        def preprocess(examples):
            args = (examples[key1],) if not key2 else (examples[key1], examples[key2])
            return self.processor(*args, truncation=True, max_length=128)

        keep_cols = ["label", "labels", "idx", "input_ids", "attention_mask", "token_type_ids"]
        tokenized_data = data.map(preprocess, batched=True, remove_columns=smart_column_filter(cols, keep_cols))
        
        label_remap = self.cfg.get("label_remap", None)

        from transformers import DataCollatorWithPadding
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir="/tmp/eval", per_device_eval_batch_size=32, report_to="none"),
            eval_dataset=tokenized_data,
            processing_class=self.processor,
            data_collator=DataCollatorWithPadding(self.processor),
            compute_metrics=lambda p: compute_glue_metrics(p, metric, label_remap)
        )
        return trainer.evaluate()

    def _run_cv_eval(self):
        data = load_dataset(self.cfg['dataset_name'], split=self.cfg['split'])
        if self.limit: data = data.select(range(min(len(data), self.limit)))
        metric = evaluate.load("accuracy")
        
        def transform(example_batch):
            images = [x.convert("RGB") for x in example_batch['img']]
            inputs = self.processor(images, return_tensors='pt')
            inputs['labels'] = example_batch['fine_label'] if 'fine_label' in example_batch else example_batch['label']
            return inputs

        tokenized_data = data.with_transform(transform)
        
        def collate_fn(batch):
            return {
                'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
                'labels': torch.tensor([x['labels'] for x in batch])
            }

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir="/tmp/eval", per_device_eval_batch_size=32, report_to="none", remove_unused_columns=False),
            eval_dataset=tokenized_data,
            data_collator=collate_fn,
            compute_metrics=lambda p: compute_accuracy(p, metric)
        )
        return trainer.evaluate()


def compute_ner_metrics(p, metric, label_list):
    preds = np.argmax(p.predictions, axis=2)
    true_preds = [[label_list[p] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(preds, p.label_ids)]
    true_labs = [[label_list[l] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(preds, p.label_ids)]
    res = metric.compute(predictions=true_preds, references=true_labs)
    return {"f1": res["overall_f1"]}

def compute_glue_metrics(p, metric, label_remap=None):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(logits, axis=1)
    if label_remap:
        new_preds = np.copy(preds)
        for old_id, new_id in label_remap.items():
            new_preds[preds == old_id] = new_id
        preds = new_preds
    res = metric.compute(predictions=preds, references=p.label_ids)
    return res 

def compute_accuracy(p, metric):
    preds = np.argmax(p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions, axis=1)
    res = metric.compute(predictions=preds, references=p.label_ids)
    return res