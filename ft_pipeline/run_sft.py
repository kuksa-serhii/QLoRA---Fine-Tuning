# ft_pipeline/run_sft.py (UPDATED)
from __future__ import annotations
 
import os
import gc
import torch
 
from ft_pipeline.data import load_prompt_completion_dataset
from ft_pipeline.model import load_tokenizer, load_qlora_model, apply_lora
from ft_pipeline.trainer import build_sft_config, build_trainer
from ft_pipeline.ab_eval import run_ab_dump, make_ab_report
from ft_pipeline.logger import get_logger
from ft_pipeline.callbacks import GPUMetricsCallback, ABSanityCallback
from ft_pipeline.callbacks import EarlyStopCfg, EarlyStoppingOnMetricCallback
 
 
def _cuda_mem_snapshot(prefix: str = "") -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    return {
        "cuda": True,
        "allocated_mb": float(torch.cuda.memory_allocated() / 1024**2),
        "reserved_mb": float(torch.cuda.memory_reserved() / 1024**2),
        "max_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
        "max_reserved_mb": float(torch.cuda.max_memory_reserved() / 1024**2),
        "prefix": prefix,
    }
 
 
def run_finetune(
    cfg,
    *,
    ab_indices=None,
    do_ab_before: bool = True,
    do_ab_after: bool = True,
    dataset_limits=(None, None),
    dataset_mode: str = "prompt_completion",
    clean_cuda_cache_before: bool = True,
):
    """
    One-call runner.
    """
    log = get_logger()
    os.makedirs(cfg.out_dir, exist_ok=True)
 
    if clean_cuda_cache_before:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
    log.info("=== FT RUN START ===")
    log.info("CUDA available=%s", torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("CUDA device=%s", torch.cuda.get_device_name(0))
 
    # ---- Dataset
    train_limit, val_limit = dataset_limits
    ds = load_prompt_completion_dataset(
        cfg.train_jsonl,
        cfg.val_jsonl,
        train_limit=train_limit,
        val_limit=val_limit,
    )
    len(ds['validation'])
 
    # ---- Model + tokenizer
    tokenizer = load_tokenizer(cfg.model_id)
    model = load_qlora_model(
        cfg.model_id,
        cfg.compute_dtype(),
        cfg.use_bf16,
        load_in_4bit=cfg.load_in_4bit,
        attn_implementation=cfg.attn_implementation,
    )
    model = apply_lora(
        model,
        cfg.lora_r,
        cfg.lora_alpha,
        cfg.lora_dropout,
        cfg.resolved_target_modules(),
    )
 
    log.info("Trainable parameters:")
    model.print_trainable_parameters()
 
    # ---- A/B BEFORE
    ab_before_path = os.path.join(cfg.out_dir, "ab_before.json")
    ab_after_path = os.path.join(cfg.out_dir, "ab_after.json")
    ab_report_path = os.path.join(cfg.out_dir, "ab_report.md")
 
    if do_ab_before:
        run_ab_dump(
            model=model,
            tokenizer=tokenizer,
            val_jsonl=cfg.val_jsonl,
            # indices=list(ab_indices),
            indices = list(range(len(ds['validation']))),
            out_path=ab_before_path,
            tag="before",
            max_new_tokens=cfg.max_new_tokens_eval,
            use_bf16=cfg.use_bf16,
        )
 
    # ---- Trainer
    sft_args = build_sft_config(cfg)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=ds["train"],
        val_ds=ds["validation"],
        sft_args=sft_args,
        callbacks=[],                # callbacks додамо нижче
        dataset_mode=dataset_mode,
    )
 
    # ---- Callbacks (HERE — correct place)
    cb_metrics = GPUMetricsCallback(
        tokenizer=tokenizer,
        collator=trainer.data_collator,
        every_n_steps=cfg.logging_steps,
        report_eval_token_stats=True,
    )
    trainer.add_callback(cb_metrics)
   
    if ab_indices:
        cb_ab = ABSanityCallback(
            tokenizer=tokenizer,
            val_jsonl=cfg.val_jsonl,
            out_dir=cfg.out_dir,
            indices=list(ab_indices),
            max_new_tokens=cfg.max_new_tokens_eval,
            use_bf16=cfg.use_bf16,
            every_eval=1,
        )
        trainer.add_callback(cb_ab)
 
    # ---- Early stopping on eval_loss (SFT)
 
    cb_early_stop = EarlyStoppingOnMetricCallback(EarlyStopCfg(
            metric_name="eval_loss",
            minimize=True,
            patience=2,      
            min_delta=0.01,    
            max_steps=cfg.max_steps,  
        )
    )
 
    trainer.add_callback(cb_early_stop)
 
    # ---- Train
    log.info("Starting training…")
    trainer.train()
 
    # ---- Save adapter
    trainer.model.save_pretrained(os.path.join(cfg.out_dir, "lora_adapter"))
    tokenizer.save_pretrained(os.path.join(cfg.out_dir, "tokenizer"))
 
    # ---- A/B AFTER
    if do_ab_after:
        run_ab_dump(
            model=model,
            tokenizer=tokenizer,
            val_jsonl=cfg.val_jsonl,
            # indices=list(ab_indices),
            indices = list(range(len(ds['validation']))),
            out_path=ab_after_path,
            tag="after",
            max_new_tokens=cfg.max_new_tokens_eval,
            use_bf16=cfg.use_bf16,
        )
        make_ab_report(ab_before_path, ab_after_path, ab_report_path)
   
 
        # ---- Return artifacts (NEW)
    artifacts = {"paths":{
        "out_dir": cfg.out_dir,
        "lora_adapter_dir": os.path.join(cfg.out_dir, "lora_adapter"),
        "tokenizer_dir": os.path.join(cfg.out_dir, "tokenizer"),
        "ab_before": ab_before_path if do_ab_before else None,
        "ab_after": ab_after_path if do_ab_after else None,
        "ab_report": ab_report_path if do_ab_after else None,
    }}
    log.info(f"Artifacts: {artifacts}")
   
    # =========================
    # Summary / Report
    # =========================
    paths = artifacts.get("paths", {}) if isinstance(artifacts, dict) else {}
    out_dir = paths.get("out_dir", cfg.out_dir)
   
    print("\n" + "=" * 80)
    print("✅ FINETUNE SUMMARY")
    print("=" * 80)
    print("Output dir:   ", out_dir)
    print("Adapter dir:  ", paths.get("lora_adapter_dir"))
    print("Tokenizer dir:", paths.get("tokenizer_dir"))
    print("AB before:    ", paths.get("ab_before"))
    print("AB after:     ", paths.get("ab_after"))
    print("AB report:    ", paths.get("ab_report"))
   
    # --- Training steps & metrics (best-effort) ---
    trainer = artifacts.get("trainer", None) if isinstance(artifacts, dict) else None
    if trainer is not None:
        try:
            step = int(getattr(trainer.state, "global_step", -1))
            epoch = getattr(trainer.state, "epoch", None)
            print("\nTraining state:")
            print("  global_step:", step)
            if epoch is not None:
                print("  epoch:", epoch)
        except Exception as e:
            print("\nTraining state: (failed to read)", e)
   
    # --- GPU memory snapshot ---
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        res = torch.cuda.memory_reserved() / 1024**3
        max_alloc = torch.cuda.max_memory_allocated() / 1024**3
        max_res = torch.cuda.max_memory_reserved() / 1024**3
        print("\nGPU memory (GB):")
        print(f"  alloc_now:     {alloc:.2f}")
        print(f"  reserved_now:  {res:.2f}")
        print(f"  max_alloc:     {max_alloc:.2f}")
        print(f"  max_reserved:  {max_res:.2f}")
   
    # --- A/B JSON validity rates (best-effort) ---
    def _read_valid_json_rate(p: str):
        if not p or not os.path.exists(p):
            return None
        try:
            data = json.load(open(p, "r", encoding="utf-8"))
            return float(data.get("valid_json_rate"))
        except Exception:
            return None
   
    vj_before = _read_valid_json_rate(paths.get("ab_before"))
    vj_after = _read_valid_json_rate(paths.get("ab_after"))
   
    print("\nA/B strict-JSON validity:")
    print("  BEFORE:", f"{vj_before:.2%}" if vj_before is not None else "n/a")
    print("  AFTER: ", f"{vj_after:.2%}" if vj_after is not None else "n/a")
   
    # --- quick pointer to ab_sanity files (if produced by ABSanityCallback) ---
    ab_sanity_dir = os.path.join(out_dir, "ab_sanity")
    if os.path.isdir(ab_sanity_dir):
        files = sorted([f for f in os.listdir(ab_sanity_dir) if f.endswith(".json")])
        if files:
            print("\nAB sanity snapshots:")
            print(" ", "\n  ".join(files[-5:]))  # show last up to 5
        else:
            print("\nAB sanity snapshots: dir exists but no json files found")
    else:
        print("\nAB sanity snapshots: n/a (dir not found)")
   
    print("=" * 80)
    print("✅ Done.")
 
 
 
   
    log.info("=== FT RUN END ===")
    return artifacts
 
 
 