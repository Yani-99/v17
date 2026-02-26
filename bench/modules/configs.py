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
        "ft_model": "meta-llama/Llama-3.1-8B-Instruct",
        "llm_tasks": ["mmlu", "gsm8k","ifeval"],
        "run_mt_bench": False,
        "eval_limit": None,
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
    },
    "llama2-13b-1": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-13b-hf",
        "ft_model": "meta-llama/Llama-2-13b-chat-hf"
    },
    "llama2-13b-2": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-13b-hf",
        "ft_model": "lmsys/vicuna-13b-v1.5"
    },
    "llama2-13b-3": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-13b-hf",
        "ft_model": "NousResearch/Nous-Hermes-Llama2-13b"
    },
    "llama2-13b-4": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-13b-hf",
        "ft_model": "WizardLM/WizardLM-13B-V1.2"
    },
    "llama2-13b-5": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-13b-hf",
        "ft_model": "garage-bAInd/Platypus2-13B"
    },
    "llama2-13b-6": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-13b-hf",
        "ft_model": "stabilityai/StableBeluga-13B"
    },
    "llama2-13b-7": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-13b-hf",
        "ft_model": "allenai/tulu-2-dpo-13b"
    },
    "llama2-13b-8": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-13b-hf",
        "ft_model": "Open-Orca/OpenOrca-Platypus2-13B"
    },
    "llama2-13b-9": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-13b-hf",
        "ft_model": "Riiid/sheep-duck-llama-2-13b"
    },
    "llama2-13b-10": {
        "task_type": "LLM_HARNESS",
        "base_model": "meta-llama/Llama-2-13b-hf",
        "ft_model": "Xwin-LM/Xwin-LM-13B-V0.1"
    },
    "gpt2-1": {
        "task_type": "LLM_HARNESS",
        "base_model": "gpt2",
        "ft_model": "lvwerra/gpt2-imdb"
    },
    "gpt2-2": {
        "task_type": "LLM_HARNESS",
        "base_model": "gpt2",
        "ft_model": "Gustavosta/MagicPrompt-Stable-Diffusion"
    },
    "gpt2-3": {
        "task_type": "LLM_HARNESS",
        "base_model": "gpt2",
        "ft_model": "mrm8488/GPT-2-finetuned-common_gen"
    },
    "gpt2-4": {
        "task_type": "LLM_HARNESS",
        "base_model": "gpt2",
        "ft_model": "succinctly/text2image-prompt-generator"
    },
    "gpt2-5": {
        "task_type": "LLM_HARNESS",
        "base_model": "gpt2",
        "ft_model": "shibing624/code-autocomplete-gpt2-base"
    },
    "gpt2-6": {
        "task_type": "LLM_HARNESS",
        "base_model": "gpt2",
        "ft_model": "rhysjones/gpt2-124M-edu-fineweb-10B"
    },
    "gpt2-7": {
        "task_type": "LLM_HARNESS",
        "base_model": "gpt2",
        # "ft_model": "locutusque/gpt2-conversational-retrain"
        "ft_model": "huggingtweets/elonmusk"
    },
    "gpt2-8": {
        "task_type": "LLM_HARNESS",
        "base_model": "gpt2",
        "ft_model": "neulab/gpt2-finetuned-wikitext103"
    },
    "gpt2-9": {
        "task_type": "LLM_HARNESS",
        "base_model": "gpt2",
        "ft_model": "vicgalle/gpt2-open-instruct-v1"
    },
    "gpt2-10": {
        "task_type": "LLM_HARNESS",
        "base_model": "gpt2",
        "ft_model": "Ssarion/gpt2-multi-news"
    },
    "roberta-large-1": {
        "task_type": "GLUE",
        "base_model": "roberta-large",
        "ft_model": "roberta-large-mnli"
    },
    "roberta-large-2": {
        "task_type": "QA",
        "base_model": "roberta-large",
        "ft_model": "deepset/roberta-large-squad2"
    },
    "roberta-large-3": {
        "task_type": "STS",
        "base_model": "roberta-large",
        "ft_model": "cross-encoder/stsb-roberta-large"
    },
    "roberta-large-4": {
        "task_type": "Sentiment",
        "base_model": "roberta-large",
        "ft_model": "siebert/sentiment-roberta-large-english"
    },
    "roberta-large-5": {
        "task_type": "Sentence-Embedding",
        "base_model": "roberta-large",
        "ft_model": "sentence-transformers/all-roberta-large-v1"
    },
    "roberta-large-6": {
        "task_type": "NLI",
        "base_model": "roberta-large",
        "ft_model": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    },
    "roberta-large-7": {
        "task_type": "Classification",
        "base_model": "roberta-large",
        "ft_model": "openai-community/roberta-large-openai-detector"
    },
    "roberta-large-8": {
        "task_type": "QA",
        "base_model": "roberta-large",
        "ft_model": "navteca/roberta-large-squad2"
    },
    "roberta-large-9": {
        "task_type": "Paraphrase",
        "base_model": "roberta-large",
        "ft_model": "cross-encoder/quora-roberta-large"
    },
    "roberta-large-10": {
        "task_type": "NER",
        "base_model": "roberta-large",
        "ft_model": "jean-baptiste/roberta-large-ner-english"
    },
    "roberta-base-1": {
        "task_type": "QA",
        "base_model": "roberta-base",
        "ft_model": "deepset/roberta-base-squad2"
    },
    "roberta-base-2": {
        "task_type": "STS",
        "base_model": "roberta-base",
        "ft_model": "cross-encoder/stsb-roberta-base"
    },
    "roberta-base-3": {
        "task_type": "NLI",
        "base_model": "roberta-base",
        "ft_model": "cross-encoder/nli-roberta-base"
    },
    "roberta-base-4": {
        "task_type": "Sentence-Embedding",
        "base_model": "roberta-base",
        "ft_model": "textattack/roberta-base-SST-2"
    },
    "roberta-base-5": {
        "task_type": "Classification",
        "base_model": "roberta-base",
        "ft_model": "openai-community/roberta-base-openai-detector"
    },
    "roberta-base-6": {
        "task_type": "GLUE",
        "base_model": "roberta-base",
        "ft_model": "textattack/roberta-base-MNLI"
    },
    "roberta-base-7": {
        "task_type": "Sentiment",
        "base_model": "roberta-base",
        "ft_model": "cardiffnlp/twitter-roberta-base-sentiment-latest"
    },
    "roberta-base-8": {
        "task_type": "Emotion",
        "base_model": "roberta-base",
        "ft_model": "SamLowe/roberta-base-go_emotions"
    },
    "roberta-base-9": {
        "task_type": "NER",
        "base_model": "roberta-base",
        "ft_model": "textattack/roberta-base-ag-news"
    },
    "roberta-base-10": {
        "task_type": "Hate-Speech",
        "base_model": "roberta-base",
        "ft_model": "cardiffnlp/twitter-roberta-base-hate-latest"
    },
    "mistral-7b-1": {
        "task_type": "Official-Instruct",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "ft_model": "mistralai/Mistral-7B-Instruct-v0.1"
    },
    "mistral-7b-2": {
        "task_type": "DPO-Alignment",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "ft_model": "HuggingFaceH4/zephyr-7b-beta"
    },
    "mistral-7b-3": {
        "task_type": "SFT-DPO-Alpha",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "ft_model": "HuggingFaceH4/zephyr-7b-alpha"
    },
    "mistral-7b-4": {
        "task_type": "General-Chat",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "ft_model": "Intel/neural-chat-7b-v3-1"
    },
    "mistral-7b-5": {
        "task_type": "RLAIF-Alignment",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "ft_model": "berkeley-nest/Starling-LM-7B-alpha"
    },
    "mistral-7b-6": {
        "task_type": "Math-Reasoning",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "ft_model": "argilla/notus-7b-v1"
    },
    "mistral-7b-7": {
        "task_type": "Synthetic-Data",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "ft_model": "jondurbin/airoboros-m-7b-3.1.2"
    },
    "mistral-7b-8": {
        "task_type": "Empathy-Companion",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "ft_model": "ehartford/samantha-1.2-mistral-7b"
    },
    "mistral-7b-9": {
        "task_type": "Instruction-Following",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "ft_model": "migtissera/SynthIA-7B-v1.3"
    },
    "mistral-7b-10": {
        "task_type": "Math-Instruction",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "ft_model": "TIGER-Lab/MAmmoTH-7B-Mistral"
    },
    "llama-3.1-70b-1": {
        "task_type": "Official-Instruct",
        "base_model": "meta-llama/Llama-3.1-70B",
        "ft_model": "meta-llama/Llama-3.1-70B-Instruct"
    },
    "llama-3.1-70b-2": {
        "task_type": "General-Chat",
        "base_model": "meta-llama/Llama-3.1-70B",
        "ft_model": "NousResearch/Hermes-3-Llama-3.1-70B"
    },
    "llama-3.1-70b-3": {
        "task_type": "RLHF-Reward-Tuned",
        "base_model": "meta-llama/Llama-3.1-70B",
        "ft_model": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
    },
    "llama-3.1-70b-4": {
        "task_type": "Instruct-Clone",
        "base_model": "meta-llama/Llama-3.1-70B",
        "ft_model": "unsloth/Meta-Llama-3.1-70B-Instruct"
    },
    "llama-3.1-70b-5": {
        "task_type": "Reflection-Tuning",
        "base_model": "meta-llama/Llama-3.1-70B",
        "ft_model": "mattshumer/Reflection-Llama-3.1-70B"
    },
    "llama-3.1-70b-6": {
        "task_type": "Multilingual-German",
        "base_model": "meta-llama/Llama-3.1-70B",
        "ft_model": "VAGOsolutions/Llama-3.1-SauerkrautLM-70b-Instruct"
    },
    "llama-3.1-70b-7": {
        "task_type": "SFT-Academic-Baseline",
        "base_model": "meta-llama/Llama-3.1-70B",
        "ft_model": "allenai/Llama-3.1-Tulu-3-70B-SFT"
    },
    "llama-3.1-70b-8": {
        "task_type": "Math-Reasoning",
        "base_model": "meta-llama/Llama-3.1-70B",
        "ft_model": "nvidia/OpenMath2-Llama3.1-70B"
    },
    "llama-3.1-70b-9": {
        "task_type": "Synthetic-Data",
        "base_model": "meta-llama/Llama-3.1-70B",
        "ft_model": "migtissera/Tess-3-Llama-3.1-70B"
    },
    "llama-3.1-70b-10": {
        "task_type": "Abliterated-Refusal",
        "base_model": "meta-llama/Llama-3.1-70B",
        "ft_model": "mylesgoose/Llama-3.1-70B-Instruct-abliterated"
    }
}

