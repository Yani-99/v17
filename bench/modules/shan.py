import torch
import logging
from transformers import (
    AutoConfig, AutoModel, AutoModelForCausalLM, 
    AutoModelForSequenceClassification, AutoModelForTokenClassification,
    AutoModelForImageClassification, AutoTokenizer, AutoImageProcessor
)
from utils import force_cleanup

logger = logging.getLogger(__name__)

class ModelManager:
    @staticmethod
    def get_classes_and_kwargs(cfg):
        tt = cfg['task_type']
        if "LLM" in tt:
            return AutoModelForCausalLM, AutoTokenizer, {"torch_dtype": "auto", "device_map": "auto"}
        elif tt == "CV":
            return AutoModelForImageClassification, AutoImageProcessor, {}
        elif tt == "NER":
            return AutoModelForTokenClassification, AutoTokenizer, {}
        else:
            return AutoModelForSequenceClassification, AutoTokenizer, {}

    @staticmethod
    def load_ft_config(cfg):
        try:
            return AutoConfig.from_pretrained(cfg['ft_model'])
        except:
            return None

    @staticmethod
    def detect_native_dtype(cfg):
        logger.info("Detecting FT model native dtype...")
        ModelClass, _, kwargs = ModelManager.get_classes_and_kwargs(cfg)
        temp_kwargs = kwargs.copy()
        if "device_map" in temp_kwargs: temp_kwargs["device_map"] = "cpu"
        
        try:
            m = ModelClass.from_pretrained(cfg['ft_model'], **temp_kwargs)
            dtype = m.dtype
            del m
            force_cleanup()
            return dtype
        except Exception as e:
            logger.warning(f"Dtype detection failed: {e}. Defaulting to float16.")
            return torch.float16

    @staticmethod
    def prepare_model_for_eval(model_id, cfg, native_dtype, device, state_dict=None, config_obj=None):
        """
        统一加载函数：实现逻辑复用
        - 如果提供 state_dict: 执行手动注入 (用于 Loop 内部)
        - 如果不提供: 执行标准加载 (用于 Phase -1 / 0)
        """
        ModelClass, ProcessorClass, kwargs = ModelManager.get_classes_and_kwargs(cfg)
        
        # 准备加载参数
        eval_kwargs = kwargs.copy()
        eval_kwargs["low_cpu_mem_usage"] = True
        if "LLM" in cfg['task_type']:
            eval_kwargs["torch_dtype"] = native_dtype

        # 1. 加载模型（标准或空壳）
        model = ModelClass.from_pretrained(model_id, config=config_obj, **eval_kwargs)
        
        # 2. 手动注入权重 (如果提供)
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)

        # 3. 搬运到设备 (非 LLM 任务)
        if "LLM" not in cfg['task_type']:
            model = model.to(device)
            
        # 4. 加载并处理 Processor
        processor = ProcessorClass.from_pretrained(model_id)
        if hasattr(processor, "pad_token") and processor.pad_token is None:
            processor.pad_token = processor.eos_token
            
        return model, processor