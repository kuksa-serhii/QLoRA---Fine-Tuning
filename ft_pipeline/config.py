# ft_pipeline/config.py
from dataclasses import dataclass
from typing import Optional, List
import torch
 
@dataclass
class FTConfig:
    # paths
    model_id: str
    train_jsonl: str
    val_jsonl: str
    out_dir: str
 
    # seq/batch
    max_seq_len: int = 1024
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
 
    # training
    learning_rate: float = 0.00005
    weight_decay: float = 0.01   # <-- ADD
    num_train_epochs: float = 2.0
    max_steps: Optional[int] = None  # if set -> overrides epochs
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 20
    eval_steps: int = 200
    save_steps: int = 400
    save_total_limit: int = 2
 
    # precision
    use_bf16: bool = True
    use_fp16: bool = False  # fallback
 
    # qlora/bnb
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    attn_implementation: str = "sdpa"   # "sdpa" recommended, fallback: "eager"
 
    # lora
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None  # if None -> use defaults below
 
    # trainer behavior
    packing: bool = False
    optim: str = "paged_adamw_8bit"
    report_to: str = "none"
 
    # inference sanity checks
    max_new_tokens_eval: int = 512
 
    def compute_dtype(self):
        return torch.bfloat16 if self.use_bf16 else torch.float16
 
    def resolved_target_modules(self):
        if self.target_modules is not None:
            return self.target_modules
        # safe defaults for Gemma/Llama-like blocks
        return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
 
 
 
@dataclass
class DPOCfg:
    # base + adapter
    model_id: str
    sft_adapter_dir: str          # <- outputs.../lora_adapter after SFT
 
    # data
    dpo_train_jsonl: str
    dpo_val_jsonl: str
 
    # output
    out_dir: str
 
    # seq/batch
    max_seq_len: int = 2048
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
 
    # optim
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: float = 1.0
    max_steps: Optional[int] = None
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 20
    eval_steps: int = 200
    save_steps: int = 400
    save_total_limit: int = 2
 
    # DPO
    beta: float = 0.1
 
    # precision / qlora
    use_bf16: bool = True
    use_fp16: bool = False
    load_in_4bit: bool = True
    attn_implementation: str = "sdpa"
 
    # reporting
    report_to: str = "none"
    optim: str = "paged_adamw_8bit"
 
    # optional sanity
    max_new_tokens_eval: int = 512
 
    def compute_dtype(self):
        return torch.bfloat16 if self.use_bf16 else torch.float16
 
 