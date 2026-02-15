TEST_CONFIGS = {
    "roberta_sst2": {
        "task_type": "GLUE",
        "base_model": "roberta-base",
        "ft_model": "textattack/roberta-base-SST-2",
        "dataset_name": "glue",
        "dataset_config": "sst2",
        "split": "validation",
        "eval_limit": 1000, 
    },
    "roberta_mnli": {
        "task_type": "GLUE",
        "base_model": "roberta-large",
        "ft_model": "roberta-large-mnli",
        "dataset_name": "glue",
        "dataset_config": "mnli",
        "split": "validation_matched",
        "eval_limit": 1000,
        "label_remap": {0: 2, 2: 0} 
    },
    "gpt2_imdb": {
        "task_type": "LLM_PPL", 
        "base_model": "gpt2",
        "ft_model": "lvwerra/gpt2-imdb", 
        "dataset_name": "imdb",
        "dataset_config": None,
        "split": "test",
        "text_col": "text",
        "eval_limit": 500, 
    },
    "bert_ner": {
        "task_type": "NER",
        "base_model": "bert-large-uncased",
        "ft_model": "Jorgeutd/bert-large-uncased-finetuned-ner",
        "dataset_name": "conll2003",
        "dataset_config": None,
        "split": "test",
        "eval_limit": 500,
    },
    "vit_cifar10": {
        "task_type": "CV",
        "base_model": "google/vit-base-patch16-224-in21k",
        "ft_model": "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
        "dataset_name": "cifar10",
        "dataset_config": None,
        "split": "test",
        "eval_limit": 1000, 
    },
    "vicuna": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-7b-hf",
        "ft_model": "lmsys/vicuna-7b-v1.5",
        # "llm_tasks": ["mmlu", "gsm8k"], 
        "llm_tasks": ["mmlu"], 
        "run_mt_bench": False,
        "eval_limit": 100, 
    },
    "llama3_8b_instruct": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "ft_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llm_tasks": ["mmlu", "gsm8k"],
        "run_mt_bench": False,
        "eval_limit": None,
    },
    "mistral_7b_instruct": {
        "task_type": "LLM_HARNESS",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "ft_model": "mistralai/Mistral-7B-Instruct-v0.1",
        "llm_tasks": ["mmlu", "gsm8k"],
        # "llm_tasks": [],
        "run_mt_bench": True,
        "eval_limit": None,
    },
    "vicuna_13b": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-13b-hf",
        "ft_model": "lmsys/vicuna-13b-v1.5",
        "llm_tasks": ["mmlu", "gsm8k"],
        "run_mt_bench": False,
        "eval_limit": None,
    },
    "llama3_70b": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Meta-Llama-3-70B",
        "ft_model": "meta-llama/Meta-Llama-3-70B-Instruct",
        # "llm_tasks": ["mmlu"], 
        "run_mt_bench": False,
        "eval_limit": 100, # 限制评估样本数，防止跑太久
    },
    "llama3.1-1": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-3.1-8B",
        "ft_model": "meta-llama/Llama-3.1-8B-Instruct",
        "llm_tasks": ["mmlu", "gsm8k"],
        "run_mt_bench": False,
        "eval_limit": None,
    },

    "bert_ner1": {
        "task_type":"",
        "base_model": "bert-large-uncased",
        "ft_model": "assemblyai/bert-large-uncased-sst2"
    },
    "bert_ner2": {
        "task_type":"",
        "base_model": "bert-large-uncased",
        "ft_model": "samrawal/bert-large-uncased_med-ner"
    },
    "bert_ner3": {
        "task_type":"",
        "base_model": "bert-large-uncased",
        "ft_model": "yoshitomo-matsubara/bert-large-uncased-mnli"
    },
    "bert_ner4": {
        "task_type":"",
        "base_model": "bert-large-uncased",
        "ft_model": "princeton-nlp/sup-simcse-bert-large-uncased"
    },
    "bert_ner5": {
        "task_type":"",
        "base_model": "bert-large-uncased",
        "ft_model": "SarielSinLuo/bert-large-uncased-finetuned-cola"
    },
    "bert_ner6": {
        "task_type":"",
        "base_model": "bert-large-uncased",
        "ft_model": "princeton-nlp/unsup-simcse-bert-large-uncased"
    },
    "bert_ner7": {
        "task_type":"",
        "base_model": "bert-large-uncased",
        "ft_model": "yoshitomo-matsubara/bert-large-uncased-mrpc"
    },
    "bert_ner8": {
        "task_type":"",
        "base_model": "bert-large-uncased",
        "ft_model": "yoshitomo-matsubara/bert-large-uncased-qnli"
    },
    "bert_ner9": {
        "task_type":"",
        "base_model": "bert-large-uncased",
        "ft_model": "StevenLimcorn/bert-large-uncased-semeval2016-restaurants"
    },
    "bert_ner10": {
        "task_type":"",
        "base_model": "bert-large-uncased",
        "ft_model": "Jorgeutd/bert-large-uncased-finetuned-ner"
    },
    "llama2-7b-1": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-7b-hf",
        "ft_model": "meta-llama/Llama-2-7b-chat-hf"
    },
    "llama2-7b-2": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-7b-hf",
        "ft_model": "lmsys/vicuna-7b-v1.5"
    },
    "llama2-7b-3": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-7b-hf",
        "ft_model": "NousResearch/Nous-Hermes-llama-2-7b"
    },
    "llama2-7b-4": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-7b-hf",
        "ft_model": "garage-bAInd/Platypus2-7B"
    },
    "llama2-7b-5": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-7b-hf",
        "ft_model": "WizardLM/WizardMath-7B-V1.0"
    },
    "llama2-7b-6": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-7b-hf",
        "ft_model": "georgesung/llama2_7b_chat_uncensored"
    },
    "llama2-7b-7": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-7b-hf",
        "ft_model": "allenai/tulu-2-7b"
    },
    "llama2-7b-8": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-7b-hf",
        "ft_model": "PygmalionAI/pygmalion-2-7b"
    },
    "llama2-7b-9": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-7b-hf",
        "ft_model": "h2oai/h2ogpt-4096-llama2-7b-chat"
    },
    "llama2-7b-10": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-7b-hf",
        "ft_model": "stabilityai/StableBeluga-7B"
    },
    "llama3.1-8b-1": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-3.1-8B",
        "ft_model": "meta-llama/Llama-3.1-8B-Instruct"
    },
    "llama3.1-8b-2": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-3.1-8B",
        "ft_model": "NousResearch/Hermes-3-Llama-3.1-8B"
    },
    "llama3.1-8b-3": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-3.1-8B",
        "ft_model": "meta-llama/Llama-Guard-3-8B"
    },
    "llama3.1-8b-4": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-3.1-8B",
        "ft_model": "dphn/Dolphin3.0-Llama3.1-8B"
    },
    "llama3.1-8b-5": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-3.1-8B",
        "ft_model": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
    },
    "llama3.1-8b-6": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-3.1-8B",
        "ft_model": "OpenSciLM/Llama-3.1_OpenScholar-8B"
    },
    "llama3.1-8b-7": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-3.1-8B",
        "ft_model": "Sao10K/Llama-3.1-8B-Stheno-v3.4"
    },
    "llama3.1-8b-8": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-3.1-8B",
        "ft_model": "cognitivecomputations/dolphin-2.9.4-llama3.1-8b"
    },
    "llama3.1-8b-9": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-3.1-8B",
        "ft_model": "akjindal53244/Llama-3.1-Storm-8B"
    },
    "llama3.1-8b-10": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-3.1-8B",
        "ft_model": "Magpie-Align/Llama-3.1-8B-Magpie-Align-SFT-v0.2"
    }
}

