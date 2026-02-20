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
        # try:
        #     # 1. 尝试从 Config 获取
        #     config = AutoConfig.from_pretrained(cfg['ft_model'], trust_remote_code=True)
        #     if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        #         logger.info(f"-> Found dtype in config: {config.torch_dtype}")
        #         return config.torch_dtype
            
        #     # 2. 如果 Config 里没有（极少见），再 fallback 到 meta device 加载（不耗内存）
        #     logger.info("-> Dtype not in config, trying meta load...")
        #     ModelClass, _, kwargs = ModelManager.get_classes_and_kwargs(cfg)
        #     with torch.device("meta"):
        #         # 使用 meta device 只构建图，不加载权重
        #         m = ModelClass.from_pretrained(cfg['ft_model'], config=config, low_cpu_mem_usage=True, trust_remote_code=True)
        #     return m.dtype

        # except Exception as e:
        #     logger.warning(f"Dtype detection failed: {e}. Defaulting to float16.")
        #     return torch.float16
        
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(cfg['ft_model'], trust_remote_code=True)
            if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
                logger.info(f"-> Found dtype in config: {config.torch_dtype}")
                return config.torch_dtype
            
            logger.info("-> Dtype not in config. Defaulting to float32 (common for GPT-2/BERT).")
            return torch.float32

        except Exception as e:
            logger.warning(f"Dtype detection failed: {e}. Defaulting to float32.")
            return torch.float32
        
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

        # 1. 加载模型（标准或空壳）- [双重保险 1]
        try:
            model = ModelClass.from_pretrained(model_id, config=config_obj, **eval_kwargs)
        except Exception as e:
            if "meta tensor" in str(e).lower():
                logger.warning(f"[WORKAROUND] Eval 模型加载崩溃: 触发老模型 meta tensor Bug，尝试原生安全加载...")
                safe_kwargs = {k: v for k, v in eval_kwargs.items() if k not in ["device_map", "low_cpu_mem_usage", "offload_folder"]}
                model = ModelClass.from_pretrained(model_id, config=config_obj, **safe_kwargs)
            else:
                raise e
        
        # [双重保险 2]：如果是 Baseline 评估（没有 state_dict），主动扫描静默残留的 meta tensor
        if state_dict is None:
            def has_meta(m):
                for p in m.parameters():
                    if p.device.type == "meta": return True
                return False

            if has_meta(model):
                logger.warning("[WORKAROUND] Eval 中检测到 meta tensor 残留！使用原生安全模式重新加载...")
                del model
                force_cleanup()
                safe_kwargs = {k: v for k, v in eval_kwargs.items() if k not in ["device_map", "low_cpu_mem_usage", "offload_folder"]}
                model = ModelClass.from_pretrained(model_id, config=config_obj, **safe_kwargs)

        # 2. 手动注入权重 (如果提供)
        if state_dict is not None:
            import inspect
            if inspect.isgenerator(state_dict):
                from accelerate.utils import set_module_tensor_to_device
                for key, tensor in state_dict:
                    # 边拿边注入，注入后立即释放 CPU 内存
                    set_module_tensor_to_device(model, key, device=device, value=tensor)
                    del tensor 
            else:
                model.load_state_dict(state_dict, strict=True)

        # 3. 搬运到设备 (非 LLM 任务)
        if "LLM" not in cfg['task_type']:
            model = model.to(device)
            
        # 4. 加载并处理 Processor
        processor = ProcessorClass.from_pretrained(model_id)
        if hasattr(processor, "pad_token") and processor.pad_token is None:
            processor.pad_token = processor.eos_token
            
        return model, processor

    # @staticmethod
    # def prepare_model_for_eval(model_id, cfg, native_dtype, device, state_dict=None, config_obj=None):
    #     """
    #     统一加载函数：实现逻辑复用
    #     - 如果提供 state_dict: 执行手动注入 (用于 Loop 内部)
    #     - 如果不提供: 执行标准加载 (用于 Phase -1 / 0)
    #     """
    #     ModelClass, ProcessorClass, kwargs = ModelManager.get_classes_and_kwargs(cfg)
        
    #     # 准备加载参数
    #     eval_kwargs = kwargs.copy()
    #     eval_kwargs["low_cpu_mem_usage"] = True
    #     if "LLM" in cfg['task_type']:
    #         eval_kwargs["torch_dtype"] = native_dtype

    #     # 1. 加载模型（标准或空壳）
    #     model = ModelClass.from_pretrained(model_id, config=config_obj, **eval_kwargs)
        
    #     # [核心修复] 如果是 Baseline 评估（没有 state_dict），主动扫描 meta tensor
    #     if state_dict is None:
    #         def has_meta(m):
    #             for p in m.parameters():
    #                 if p.device.type == "meta": return True
    #             return False

    #         if has_meta(model):
    #             logger.warning("[WORKAROUND] Eval 中检测到 meta tensor！使用安全模式重新加载...")
    #             del model
    #             force_cleanup()
    #             safe_kwargs = {k: v for k, v in eval_kwargs.items() if k not in ["device_map", "low_cpu_mem_usage", "offload_folder"]}
    #             model = ModelClass.from_pretrained(model_id, config=config_obj, **safe_kwargs)
        
    #     # 2. 手动注入权重 (如果提供)
        
    #     # 2. 手动注入权重 (如果提供)
    #     if state_dict is not None:
    #         import inspect
    #         # if inspect.isgenerator(state_dict): # 如果传入的是生成器
    #         #     for key, tensor in state_dict:
    #         #         # 直接将解压出来的 CPU Tensor 移动到目标设备（GPU或Offload盘）
    #         #         # 这样做完这一步，tensor 的 CPU 引用就会被销毁，内存立即回收
    #         #         from accelerate.utils import set_module_tensor_to_device
    #         #         set_module_tensor_to_device(model, key, device=device, value=tensor)
    #         #         del tensor # 显式删除
    #         # else:
    #         #     model.load_state_dict(state_dict, strict=True)
    #         if inspect.isgenerator(state_dict):
    #             from accelerate.utils import set_module_tensor_to_device
    #             for key, tensor in state_dict:
    #                 # 边拿边注入，注入后立即释放 CPU 内存
    #                 set_module_tensor_to_device(model, key, device=device, value=tensor)
    #                 del tensor 
    #         else:
    #             model.load_state_dict(state_dict, strict=True)

    #     # 3. 搬运到设备 (非 LLM 任务)
    #     if "LLM" not in cfg['task_type']:
    #         model = model.to(device)
            
    #     # 4. 加载并处理 Processor
    #     processor = ProcessorClass.from_pretrained(model_id)
    #     if hasattr(processor, "pad_token") and processor.pad_token is None:
    #         processor.pad_token = processor.eos_token
            
    #     return model, processor