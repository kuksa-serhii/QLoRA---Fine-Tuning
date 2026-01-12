# ft_pipeline/trainer.py
from __future__ import annotations
 
from dataclasses import dataclass
from typing import List, Optional, Callable, Any, Dict, Sequence, Tuple
import torch
 
from trl import SFTTrainer, SFTConfig
from trl import DPOTrainer, DPOConfig
from .logger import get_logger
 
 
@dataclass
class CompletionOnlyMaskingCollator:
    """
    Builds (input_ids, attention_mask, labels) from dataset fields:
      - ex["prompt"]: list[{"role","content"}, ...]
      - ex["completion"]: list[{"role":"assistant","content":...}]  (list with 1 msg)
 
    Loss is computed ONLY on completion tokens by masking prompt tokens with -100.
    Also records last batch token stats for debugging/monitoring via callback.
    """
    tokenizer: Any
    max_length: int
    pad_to_multiple_of: Optional[int] = 8
 
    # Updated each time __call__ is used
    last_stats: Dict[str, float] = None
 
    def __post_init__(self):
        self.last_stats = {}
 
    def _build_texts(self, ex: Dict[str, Any]) -> Tuple[str, str]:
        prompt_msgs = ex["prompt"]
        completion_msgs = ex["completion"]
        # prompt boundary text: ends right before assistant answer
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        # full training text: prompt + assistant completion
        full_text = self.tokenizer.apply_chat_template(
            prompt_msgs + completion_msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not full_text.endswith("\n"):
            full_text += "\n"
        return prompt_text, full_text
 
    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompt_texts: List[str] = []
        full_texts: List[str] = []
        for ex in examples:
            p, t = self._build_texts(ex)
            prompt_texts.append(p)
            full_texts.append(t)
 
        # Tokenize full texts -> model inputs
        tok_full = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
 
        input_ids = tok_full["input_ids"]
        attention_mask = tok_full.get("attention_mask", None)
 
        # Prompt lengths (in tokens) computed from prompt_texts.
        # IMPORTANT: also truncation=True to match cutoffs when seq is too long.
        tok_prompt = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_lens = [int((row != self.tokenizer.pad_token_id).sum().item()) for row in tok_prompt["input_ids"]]
 
        labels = input_ids.clone()
        for i, p_len in enumerate(prompt_lens):
            p_len = max(0, min(p_len, labels.shape[1]))
            labels[i, :p_len] = -100  # mask prompt tokens
 
        # Basic stats for monitoring
        seq_lens = attention_mask.sum(dim=1).tolist() if attention_mask is not None else [labels.shape[1]] * labels.shape[0]
        loss_tokens = (labels != -100).sum(dim=1).tolist()
 
        self.last_stats = {
            "batch_size": float(labels.shape[0]),
            "seq_len_mean": float(sum(seq_lens) / max(1, len(seq_lens))),
            "prompt_len_mean": float(sum(prompt_lens) / max(1, len(prompt_lens))),
            "loss_tokens_mean": float(sum(loss_tokens) / max(1, len(loss_tokens))),
            "seq_len_max": float(max(seq_lens) if seq_lens else 0.0),
            "loss_tokens_max": float(max(loss_tokens) if loss_tokens else 0.0),
        }
 
        out = {
            "input_ids": input_ids,
            "labels": labels,
        }
        if attention_mask is not None:
            out["attention_mask"] = attention_mask.to(torch.long)
        return out
 
 
def build_sft_config(cfg):
    """
    TRL SFTConfig.
    NOTE: We provide our own labels (prompt masked), so completion_only_loss is irrelevant.
    """
    log = get_logger()
    log.info("Building SFTConfig")
    log.info(f"  max_seq_len={cfg.max_seq_len}")
    log.info(f"  batch_size={cfg.per_device_train_batch_size}")
    log.info(f"  grad_accum={cfg.gradient_accumulation_steps}")
    log.info(f"  lr={cfg.learning_rate}")
 
    max_steps = cfg.max_steps if cfg.max_steps is not None else -1
    num_train_epochs = cfg.num_train_epochs if cfg.max_steps is None else 1.0
 
    return SFTConfig(
        output_dir=cfg.out_dir,
        max_seq_length=cfg.max_seq_len,
        packing=False,  # keep explicit for clarity
 
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
 
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,   # <-- ADD
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
 
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
 
        report_to=cfg.report_to,
        optim=cfg.optim,
 
        bf16=cfg.use_bf16,
        fp16=cfg.use_fp16,
 
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
 
        remove_unused_columns=False,
        do_train=True,
        gradient_checkpointing=True,
 
        torch_compile=False,
        logging_first_step=True,
        logging_strategy="steps",
        log_level="info",
        include_tokens_per_second=True,
    )
 
 
def build_trainer(
    model,
    tokenizer,
    train_ds,
    val_ds,
    sft_args,
    callbacks: Optional[List[Any]] = None,
    *,
    dataset_mode: str = "prompt_completion",
) -> SFTTrainer:
 
    log = get_logger()
    log.info("Building SFTTrainer")
    log.info(f"  train_samples={len(train_ds)}")
    log.info(f"  val_samples={len(val_ds)}")
    log.info(f"  dataset_mode={dataset_mode}")
    log.info(f"  max_seq_length={getattr(sft_args, 'max_seq_length', None)}")
 
    if dataset_mode != "prompt_completion":
        raise ValueError("This trainer build now expects dataset_mode='prompt_completion' (prompt+completion fields).")
 
    # NEW collator that masks prompt tokens -> completion-only loss
    collator = CompletionOnlyMaskingCollator(
        tokenizer=tokenizer,
        max_length=getattr(sft_args, "max_seq_length", 1024),
        pad_to_multiple_of=8,
    )
 
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        # IMPORTANT: no formatting_func needed; we collate from prompt/completion directly
        formatting_func=None,
    )
 
    # Attach callbacks
    if callbacks:
        for cb in callbacks:
            trainer.add_callback(cb)
        log.info(f"Attached callbacks: {[cb.__class__.__name__ for cb in callbacks]}")
 
    # Sanity check: labels must contain non -100
    try:
        batch = next(iter(trainer.get_train_dataloader()))
        labels = batch.get("labels", None)
        if labels is not None:
            non_mask = int((labels != -100).sum().item())
            log.info(f"labels shape: {tuple(labels.shape)}")
            log.info(f"non -100 labels: {non_mask}")
            if non_mask == 0:
                raise RuntimeError("All labels are masked (-100). This should not happen with masking collator.")
    except Exception as e:
        log.warning(f"Sanity batch check skipped/failed: {e}")
 
    return trainer
 
 
def build_dpo_config(cfg):
    """
    TRL DPOConfig.
    """
    log = get_logger()
    log.info("Building DPOConfig")
    log.info(f"  max_seq_len={cfg.max_seq_len}")
    log.info(f"  beta={cfg.beta}")
    log.info(f"  lr={cfg.learning_rate}")
 
    max_steps = cfg.max_steps if cfg.max_steps is not None else -1
    num_train_epochs = cfg.num_train_epochs if cfg.max_steps is None else 1.0
 
    return DPOConfig(
        output_dir=cfg.out_dir,
        max_length=cfg.max_seq_len,          # full length prompt+response
        max_prompt_length=min(3600, cfg.max_seq_len),  # safe default
        beta=cfg.beta,
 
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
 
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
 
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
 
        report_to=cfg.report_to,
        optim=cfg.optim,
 
        bf16=cfg.use_bf16,
        fp16=cfg.use_fp16,
 
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
 
        remove_unused_columns=False,
        do_train=True,
        gradient_checkpointing=True,
 
        torch_compile=False,
        logging_first_step=True,
        logging_strategy="steps",
        log_level="info",
        include_tokens_per_second=True,
    )
 
 
def build_dpo_trainer(
    model,
    tokenizer,
    train_ds,
    val_ds,
    dpo_args,
    callbacks: Optional[List[Any]] = None,
):
    """
    Expects datasets with columns: prompt(str), chosen(str), rejected(str)
    """
    log = get_logger()
    log.info("Building DPOTrainer")
    log.info(f"  train_samples={len(train_ds)}")
    log.info(f"  val_samples={len(val_ds)}")
 
    trainer = DPOTrainer(
        model=model,
        ref_model=None,                # TRL will create a frozen reference copy
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
 
    if callbacks:
        for cb in callbacks:
            trainer.add_callback(cb)
        log.info(f"Attached callbacks: {[cb.__class__.__name__ for cb in callbacks]}")
 
    return trainer
 
 
 
 
 