# ft_pipeline/env.py
import os
 
def apply_env(
    disable_torchdynamo: bool = True,
    disable_torch_compile: bool = True,
    cuda_alloc_conf: str = "expandable_segments:True,max_split_size_mb:64",
    cuda_module_loading: str = "LAZY",
):
    """
    Call this at the TOP of your notebook BEFORE importing torch/transformers,
    if you want these flags to take effect reliably.
    """
    if disable_torchdynamo:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
    if disable_torch_compile:
        os.environ["TORCH_COMPILE"] = "0"
 
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc_conf
    os.environ["CUDA_MODULE_LOADING"] = cuda_module_loading
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
 