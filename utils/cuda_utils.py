import torch
import gc

def print_cuda_memory():
    """打印CUDA显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        cached = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"CUDA Memory - Allocated: {allocated:.2f} MB, Cached: {cached:.2f} MB")

def clear_cuda_cache():
    """清理CUDA缓存"""
    torch.cuda.empty_cache()
    gc.collect()
