# ft_pipeline/model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from .logger import get_logger
 
def load_tokenizer(model_id: str, use_fast: bool = True):
    log = get_logger()
    log.info(f"Loading tokenizer: {model_id}")
   
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast)
    # decoder-only models typically use eos as pad
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    log.info("Tokenizer loaded")
    return tok
 
def load_qlora_model(
    model_id: str,
    compute_dtype,
    use_bf16: bool,
    load_in_4bit: bool = True,
    attn_implementation: str = "sdpa",   # <---
):
   
    log = get_logger()
    log.info("Loading base model (QLoRA)")
    log.info(f"  model_id: {model_id}")
    log.info(f"  dtype: {compute_dtype}")
    log.info(f"  4bit: {load_in_4bit}")
    log.info(f"  attn_implementation: {attn_implementation}")
   
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    max_memory = {0: "39GiB", "cpu": "64GiB"}  
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto", #{"": "cuda:0"},
        max_memory=max_memory,
        offload_buffers=True,
        torch_dtype=compute_dtype,
        attn_implementation=attn_implementation,  
    )
    log.info("Base model loaded")
    log.info("Enabling gradient checkpointing")
   
    # memory
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
 
    model = prepare_model_for_kbit_training(model)
    return model
 
 
def apply_lora(model, r: int, alpha: int, dropout: float, target_modules):
    log = get_logger()
    log.info("Applying LoRA")
    log.info(f"  r={r}, alpha={alpha}, dropout={dropout}")
    log.info(f"  target_modules={target_modules}")
 
    # quick sanity: ensure target modules exist
    names = [n for n, _ in model.named_modules()]
    missing = [m for m in target_modules if not any(m in n for n in names)]
    if missing:
        raise ValueError(f"Some target_modules not found in model modules: {missing}")
 
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    log.info("LoRA applied successfully")
    return model
 
 
 
def load_trainable_adapter(base_model, adapter_dir: str):
    """
    Load LoRA adapter weights and keep them trainable (continue training).
    """
    log = get_logger()
    log.info(f"Loading trainable LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
    log.info("Trainable adapter loaded")
    return model
 
 