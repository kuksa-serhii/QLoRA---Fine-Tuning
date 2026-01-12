# ft_pipeline/run_dpo.py
from __future__ import annotations
 
import os
import gc
import torch
 
from ft_pipeline.data import load_dpo_dataset
from ft_pipeline.model import load_tokenizer, load_qlora_model, load_trainable_adapter
from ft_pipeline.trainer import build_dpo_config, build_dpo_trainer
from ft_pipeline.logger import get_logger
from ft_pipeline.callbacks import GPUMetricsCallback, ABSanityCallback, DPOMetricsCallback
from ft_pipeline.callbacks import EarlyStopCfg, EarlyStoppingOnMetricCallback,DPOOverOptimizationStopCallback
 
 
 
 
 
def run_dpo(
    cfg,
    *,
    ab_val_jsonl: str | None = None,     # optional: reuse SFT val for sanity generation
    ab_indices=(0, 1, 2, 10, 25),
    do_ab_sanity: bool = False,
    dataset_limits=(None, None),
    clean_cuda_cache_before: bool = True,
):
    log = get_logger()
    os.makedirs(cfg.out_dir, exist_ok=True)
 
    if clean_cuda_cache_before:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
    log.info("=== DPO RUN START ===")
    log.info("CUDA available=%s", torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("CUDA device=%s", torch.cuda.get_device_name(0))
 
    # ---- Tokenizer
    tokenizer = load_tokenizer(cfg.model_id)
 
    # ---- Dataset (needs tokenizer for chat template prompt)
    train_limit, val_limit = dataset_limits
    ds = load_dpo_dataset(
        cfg.dpo_train_jsonl,
        cfg.dpo_val_jsonl,
        tokenizer=tokenizer,
        train_limit=train_limit,
        val_limit=val_limit,
    )
 
    # # ---- Base model (QLoRA) + load SFT adapter as trainable
    # base = load_qlora_model(
    #     cfg.model_id,
    #     cfg.compute_dtype(),
    #     cfg.use_bf16,
    #     load_in_4bit=cfg.load_in_4bit,
    #     attn_implementation=cfg.attn_implementation,
    # )
    # model = load_trainable_adapter(base, cfg.sft_adapter_dir)
 
    base = load_qlora_model(
        cfg.model_id,
        cfg.compute_dtype(),
        cfg.use_bf16,
        load_in_4bit=cfg.load_in_4bit,
        attn_implementation=cfg.attn_implementation,
    )
 
    # If SFT adapter provided -> continue training that adapter
    if getattr(cfg, "sft_adapter_dir", None):
        model = load_trainable_adapter(base, cfg.sft_adapter_dir)
    else:
        # No SFT adapter -> train a fresh LoRA adapter for DPO
        # We reuse the same LoRA config as in SFT (r/alpha/dropout/target_modules)
        from peft import LoraConfig, get_peft_model
        from ft_pipeline.model import resolved_target_modules
   
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=resolved_target_modules(base, None),
        )
        model = get_peft_model(base, lora_cfg)
 
 
    log.info("Trainable parameters:")
    model.print_trainable_parameters()
 
    # ---- Trainer
    dpo_args = build_dpo_config(cfg)
    trainer = build_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=ds["train"],
        val_ds=ds["validation"],
        dpo_args=dpo_args,
        callbacks=[],
    )
 
    # ---- Callbacks
    cb_metrics = GPUMetricsCallback(
        tokenizer=tokenizer,
        collator=None,  # DPOTrainer handles collator internally
        every_n_steps=cfg.logging_steps,
        report_eval_token_stats=True,
    )
    trainer.add_callback(cb_metrics)
 
    cb_dpo = DPOMetricsCallback(
        out_dir=cfg.out_dir,
        every_n_steps=cfg.logging_steps,
        csv_name="dpo_metrics.csv",
    )
    trainer.add_callback(cb_dpo)
 
    # ---- Early stopping on eval_loss (DPO)
    trainer.add_callback(
        EarlyStoppingOnMetricCallback(
            EarlyStopCfg(
                metric_name="eval_loss",
                minimize=True,
                patience=2,      # DPO краще зупиняти швидше
                min_delta=0.0,
                max_steps=cfg.max_steps,
            )
        )
    )
   
    # ---- Over-optimization guard (DPO-specific)
    trainer.add_callback(
        DPOOverOptimizationStopCallback(
            acc_threshold=0.98,      # якщо ~1.0 занадто рано
            margin_threshold=8.0,    # для твого кейсу good default
            consecutive=5,           # 5 логів поспіль → стоп
        )
)
 
 
    if do_ab_sanity and ab_val_jsonl:
        cb_ab = ABSanityCallback(
            tokenizer=tokenizer,
            val_jsonl=ab_val_jsonl,
            out_dir=cfg.out_dir,
            indices=list(ab_indices),
            max_new_tokens=cfg.max_new_tokens_eval,
            use_bf16=cfg.use_bf16,
            every_eval=1,
        )
        trainer.add_callback(cb_ab)
 
    # ---- Train
    log.info("Starting DPO training…")
    trainer.train()
 
    # ---- Save adapter
    adapter_dir = os.path.join(cfg.out_dir, "lora_adapter")
    tok_dir = os.path.join(cfg.out_dir, "tokenizer")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tok_dir)
 
    artifacts = {
        "out_dir": cfg.out_dir,
        "lora_adapter_dir": adapter_dir,
        "tokenizer_dir": tok_dir,
    }
    log.info(f"DPO Artifacts: {artifacts}")
    log.info("=== DPO RUN END ===")
    return artifacts
 
 