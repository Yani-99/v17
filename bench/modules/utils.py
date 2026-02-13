import gc      
import torch   
import numpy as np

def get_np_view(tensor):
    # [FIXED] PyTorch .numpy() crashes on BF16. Handle it explicitly.
    if tensor.dtype == torch.bfloat16:
        # View as int16 (2 bytes) -> numpy -> view as uint16
        # This preserves the bits exactly for passing to C++
        return tensor.view(torch.int16).detach().cpu().numpy().view(np.uint16)
    
    np_arr = tensor.detach().cpu().numpy()
    if np_arr.dtype == np.float32 or np_arr.dtype == np.int32:
        return np_arr.view(np.uint32)
    elif np_arr.dtype == np.float16:
        return np_arr.view(np.uint16)
    return np_arr

def smart_column_filter(dataset_cols, keep_list):
    return [c for c in dataset_cols if c not in keep_list]

def force_cleanup():
    gc.collect()
    torch.cuda.empty_cache()