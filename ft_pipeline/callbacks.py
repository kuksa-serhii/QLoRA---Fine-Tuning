# ft_pipeline/callbacks.py
from __future__ import annotations
import os
import time
import torch
import csv
from transformers import TrainerCallback
from .logger import get_logger
from .ab_eval import run_ab_dump
 
from dataclasses import dataclass
from typing import Optional, Dict, Any
 
 
class GPUMetricsCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer=None,
        collator=None,          
        every_n_steps: int = 10,
        dump_first_step: bool = True,
        dump_chars: int = 1200,
        dump_loss_region_tokens: int = 200,
        reset_peak_each_log: bool = False,
        report_eval_token_stats: bool = True,  
    ):
        self.tokenizer = tokenizer
        self.collator = collator
        self.every_n_steps = every_n_steps
        self.dump_first_step = dump_first_step
        self.dump_chars = dump_chars
        self.dump_loss_region_tokens = dump_loss_region_tokens
        self.reset_peak_each_log = reset_peak_each_log
        self.report_eval_token_stats = report_eval_token_stats
 
        self.t0 = None
        self._did_first_dump = False
 
        # NEW: eval stats cache
        self._eval_last = {}         # last eval batch stats
        self._eval_running = None    # running aggregates during eval
 
    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        log = get_logger()
        log.info("GPUMetricsCallback enabled")
 
    # -------- GPU mem --------
    def _gpu_mem_str(self):
        if not torch.cuda.is_available():
            return "gpu_mem=n/a"
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev) / (1024**3)
        res = torch.cuda.memory_reserved(dev) / (1024**3)
        max_alloc = torch.cuda.max_memory_allocated(dev) / (1024**3)
        max_res = torch.cuda.max_memory_reserved(dev) / (1024**3)
        return f"gpu_mem(GB)=alloc:{alloc:.2f} res:{res:.2f} max_alloc:{max_alloc:.2f} max_res:{max_res:.2f}"
 
    # -------- TRAIN token stats from collator --------
    def _train_tokens_str(self):
        if self.collator is None:
            return None
        st = getattr(self.collator, "last_stats", None) or {}
        if not st:
            return None
        return (
            "train_tok="
            f"seq_mean:{st.get('seq_len_mean', 0):.0f} "
            f"prompt_mean:{st.get('prompt_len_mean', 0):.0f} "
            f"loss_mean:{st.get('loss_tokens_mean', 0):.0f} "
            f"loss_max:{st.get('loss_tokens_max', 0):.0f}"
        )
 
    # -------- EVAL token stats (computed on_prediction_step) --------
    def _init_eval_running(self):
        self._eval_running = {
            "batches": 0,
            "seq_sum": 0,
            "prompt_sum": 0,
            "loss_sum": 0,
            "seq_max": 0,
            "loss_max": 0,
        }
 
    def _update_eval_running(self, seq_lens, prompt_lens, loss_lens):
        r = self._eval_running
        r["batches"] += 1
        r["seq_sum"] += int(sum(seq_lens))
        r["prompt_sum"] += int(sum(prompt_lens))
        r["loss_sum"] += int(sum(loss_lens))
        r["seq_max"] = max(r["seq_max"], int(max(seq_lens) if seq_lens else 0))
        r["loss_max"] = max(r["loss_max"], int(max(loss_lens) if loss_lens else 0))
 
        # keep last batch
        self._eval_last = {
            "seq_mean": float(sum(seq_lens) / max(1, len(seq_lens))),
            "prompt_mean": float(sum(prompt_lens) / max(1, len(prompt_lens))),
            "loss_mean": float(sum(loss_lens) / max(1, len(loss_lens))),
            "seq_max": float(max(seq_lens) if seq_lens else 0),
            "loss_max": float(max(loss_lens) if loss_lens else 0),
        }
 
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # called AFTER evaluation; summarize running stats if any
        if not self.report_eval_token_stats:
            return
        if not self._eval_running or self._eval_running["batches"] == 0:
            return
 
        r = self._eval_running
        # convert sums into means across all examples seen during eval
        # NOTE: we summed per-batch sums; to be strict we'd track n_examples too.
        # With batch_size=1 it is exact; with bigger batch it’s still good approx.
        batches = max(1, r["batches"])
        eval_tok_summary = {
            "seq_mean_approx": r["seq_sum"] / batches,
            "prompt_mean_approx": r["prompt_sum"] / batches,
            "loss_mean_approx": r["loss_sum"] / batches,
            "seq_max": r["seq_max"],
            "loss_max": r["loss_max"],
        }
        self._eval_last = eval_tok_summary
        self._eval_running = None
 
    def on_prediction_step(self, args, state, control, inputs=None, **kwargs):
        """
        Called during evaluation/prediction loop.
        We compute eval token stats from (attention_mask, labels).
        """
        if not self.report_eval_token_stats:
            return
        if inputs is None:
            return
        labels = inputs.get("labels", None)
        attn = inputs.get("attention_mask", None)
 
        # DPO batches may not have labels -> compute only seq_len stats
        if labels is None:
            if attn is None:
                return
            if self._eval_running is None:
                self._init_eval_running()
            seq_lens = attn.sum(dim=1).detach().cpu().tolist()
            # prompt/loss unknown -> set 0
            self._update_eval_running(seq_lens, [0]*len(seq_lens), [0]*len(seq_lens))
            return
 
        # Initialize running aggregates when first prediction_step happens in an eval cycle
        if self._eval_running is None:
            self._init_eval_running()
 
        # Compute per-example lengths
        # seq_len: count of attended tokens
        if attn is not None:
            seq_lens = attn.sum(dim=1).detach().cpu().tolist()
        else:
            seq_lens = [labels.shape[1]] * labels.shape[0]
 
        # loss_tokens: labels != -100 within attended region
        if attn is not None:
            valid = (attn != 0)
            loss_lens = ((labels != -100) & valid).sum(dim=1).detach().cpu().tolist()
            prompt_lens = ((labels == -100) & valid).sum(dim=1).detach().cpu().tolist()
        else:
            loss_lens = (labels != -100).sum(dim=1).detach().cpu().tolist()
            prompt_lens = (labels == -100).sum(dim=1).detach().cpu().tolist()
 
        self._update_eval_running(seq_lens, prompt_lens, loss_lens)
 
    def _eval_tokens_str(self):
        if not self.report_eval_token_stats:
            return None
        st = self._eval_last or {}
        if not st:
            return None
 
        # support both formats: per-batch and eval-summary
        if "seq_mean" in st:
            return (
                "eval_tok="
                f"seq_mean:{st.get('seq_mean', 0):.0f} "
                f"prompt_mean:{st.get('prompt_mean', 0):.0f} "
                f"loss_mean:{st.get('loss_mean', 0):.0f} "
                f"loss_max:{st.get('loss_max', 0):.0f}"
            )
        else:
            return (
                "eval_tok="
                f"seq_mean~:{st.get('seq_mean_approx', 0):.0f} "
                f"prompt_mean~:{st.get('prompt_mean_approx', 0):.0f} "
                f"loss_mean~:{st.get('loss_mean_approx', 0):.0f} "
                f"loss_max:{st.get('loss_max', 0):.0f}"
            )
    def _loss_tokens_per_sec(self, logs):
        tps = None
        for k in ["tokens_per_second", "train_tokens_per_second", "tokens/s"]:
            if k in logs and logs[k] is not None:
                tps = float(logs[k])
                break
        if tps is None:
            return None
 
        if self.collator is None:
            return None
        st = getattr(self.collator, "last_stats", None) or {}
        seq = float(st.get("seq_len_mean", 0.0))
        loss = float(st.get("loss_tokens_mean", 0.0))
        if seq <= 1e-6:
            return None
 
        return tps * (loss / seq)
 
    # -------- main logging --------
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
 
        step = int(state.global_step)
        if step != 1 and (step % self.every_n_steps != 0):
            return
 
        log = get_logger()
        elapsed = (time.time() - self.t0) if self.t0 else 0.0
 
        tps = None
        for k in ["tokens_per_second", "train_tokens_per_second", "tokens/s"]:
            if k in logs:
                tps = logs[k]
                break
 
        parts = []
        if "loss" in logs:
            parts.append(f"train_loss={float(logs['loss']):.4f}")
        if "eval_loss" in logs:
            parts.append(f"eval_loss={float(logs['eval_loss']):.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={float(logs['learning_rate']):.6g}")
        if "grad_norm" in logs:
            parts.append(f"grad_norm={float(logs['grad_norm']):.4f}")
        if tps is not None:
            parts.append(f"tokens/sec={float(tps):.1f}")
        lps = self._loss_tokens_per_sec(logs)
        if lps is not None:
            parts.append(f"loss_tok/sec~={lps:.1f}")
 
        tr = self._train_tokens_str()
        if tr:
            parts.append(tr)
 
        ev = self._eval_tokens_str()
        if ev and ("eval_loss" in logs):  # show eval_tok only when eval happened
            parts.append(ev)
 
        parts.append(self._gpu_mem_str())
        parts.append(f"elapsed={elapsed/60:.1f}m")
 
        log.info(f"[step {step}] " + " | ".join(parts))
 
        if self.reset_peak_each_log and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
 
 
 
 
class ABSanityCallback(TrainerCallback):
    """
    Runs small fixed-sample generation sanity check on each evaluation.
    Writes ab_eval JSON to out_dir/ab_sanity/step_{global_step}.json
    """
    def __init__(
        self,
        tokenizer,
        val_jsonl: str,
        out_dir: str,
        indices=(0, 1, 2, 10, 25),
        max_new_tokens: int = 512,
        use_bf16: bool = True,
        every_eval: int = 1,   # run each N-th evaluation call
    ):
        self.tokenizer = tokenizer
        self.val_jsonl = val_jsonl
        self.out_dir = out_dir
        self.indices = list(indices)
        self.max_new_tokens = int(max_new_tokens)
        self.use_bf16 = bool(use_bf16)
        self.every_eval = int(every_eval)
        self._eval_calls = 0
 
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._eval_calls += 1
        if self.every_eval > 1 and (self._eval_calls % self.every_eval) != 0:
            return
 
        log = get_logger()
        model = kwargs.get("model", None)  # HF passes model here
        if model is None:
            log.warning("ABSanityCallback: model not found in kwargs; skipping.")
            return
 
        step = int(state.global_step)
        out_path = os.path.join(self.out_dir, "ab_sanity", f"step_{step}.json")
 
        log.info(f"ABSanityCallback: running A/B sanity on indices={self.indices} -> {out_path}")
        run_ab_dump(
            model=model,
            tokenizer=self.tokenizer,
            val_jsonl=self.val_jsonl,
            indices=self.indices,
            out_path=out_path,
            tag=f"sanity_step_{step}",
            max_new_tokens=self.max_new_tokens,
            use_bf16=self.use_bf16,
        )
 
 
 
 
 
class DPOMetricsCallback(TrainerCallback):
    """
    Logs DPO-specific metrics from TRL DPOTrainer.
    - Prints a compact one-line status to logs every N steps
    - Saves raw logs to CSV for later plotting
 
    TRL typically logs keys like (depends on version):
      - loss
      - eval_loss
      - rewards/chosen, rewards/rejected
      - rewards/accuracies
      - rewards/margins
      - logps/chosen, logps/rejected
      - kl (sometimes)
      - learning_rate
    """
 
    def __init__(
        self,
        out_dir: str,
        every_n_steps: int = 10,
        csv_name: str = "dpo_metrics.csv",
    ):
        self.out_dir = out_dir
        self.every_n_steps = max(1, int(every_n_steps))
        self.csv_path = os.path.join(out_dir, csv_name)
 
        self._log = get_logger()
        self._csv_file = None
        self._csv_writer = None
        self._header_written = False
 
        # keep last values to print stable summary even if some keys missing
        self._last: Dict[str, Any] = {}
 
    def _open_csv(self):
        os.makedirs(self.out_dir, exist_ok=True)
        if self._csv_file is None:
            self._csv_file = open(self.csv_path, "a", encoding="utf-8", newline="")
            self._csv_writer = None  # created after header known
 
    def _close_csv(self):
        if self._csv_file:
            try:
                self._csv_file.close()
            except Exception:
                pass
        self._csv_file = None
        self._csv_writer = None
 
    def on_train_begin(self, args, state, control, **kwargs):
        self._open_csv()
        self._log.info("DPOMetricsCallback enabled | csv=%s | every_n_steps=%d", self.csv_path, self.every_n_steps)
 
    def on_train_end(self, args, state, control, **kwargs):
        self._close_csv()
        self._log.info("DPOMetricsCallback finished | csv=%s", self.csv_path)
 
    def _select_keys(self, logs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pick a stable set of metrics if present.
        Normalizes learning rate key (lr / learning_rate).
        """
        out = {}
 
   
        # --- primary loss keys
        for k in ["loss", "eval_loss"]:
            if k in logs:
                out[k] = logs[k]
   
        # --- DPO / TRL specific keys (version-dependent)
        dpo_keys = [
            "rewards/accuracies",
            "rewards/margins",
            "rewards/chosen",
            "rewards/rejected",
            "logps/chosen",
            "logps/rejected",
            "kl",
            "eval_rewards/accuracies",
            "eval_rewards/margins",
            "eval_rewards/chosen",
            "eval_rewards/rejected",
            "eval_logps/chosen",
            "eval_logps/rejected",
            "eval_kl",
        ]
   
        for k in dpo_keys:
            if k in logs:
                out[k] = logs[k]
   
        return out
 
 
    def _fmt(self, x: Any) -> str:
        try:
            if isinstance(x, (int, float)):
                return f"{x:.4f}"
        except Exception:
            pass
        return str(x)
 
    def _write_csv_row(self, step: int, logs: Dict[str, Any]):
        self._open_csv()
 
        # merge with last values so columns remain stable over time
        merged = dict(self._last)
        merged.update(logs)
        self._last = merged
 
        row = {"step": step, **merged}
 
        # init writer + header once (based on current merged keys)
        if not self._header_written:
            fieldnames = list(row.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()
            self._header_written = True
        else:
            # If new keys appear later, we won't rewrite header (safe in prod).
            # They will be ignored in CSV to keep it consistent.
            pass
 
        # write only known columns
        safe_row = {k: row.get(k, "") for k in self._csv_writer.fieldnames}
        self._csv_writer.writerow(safe_row)
        self._csv_file.flush()
 
    def _print_compact(self, step: int, selected: Dict[str, Any], prefix: str = "DPO"):
        # create compact stable output
        parts = [f"{prefix} step={step}"]
        for k in ["loss", "rewards/accuracies", "rewards/margins", "rewards/chosen", "rewards/rejected", "kl"]:
            if k in selected:
                parts.append(f"{k}={self._fmt(selected[k])}")
        self._log.info(" | ".join(parts))
 
    def on_log(self, args, state, control, logs: Optional[Dict[str, Any]] = None, **kwargs):
        if not logs:
            return
        step = int(getattr(state, "global_step", 0))
 
        # Throttle printing
        if step % self.every_n_steps != 0 and "eval_loss" not in logs and not any(k.startswith("eval_") for k in logs.keys()):
            # still write CSV (optional). For prod, keep it lightweight:
            return
 
        selected = self._select_keys(logs)
 
        # Print compact line
        if "eval_loss" in logs or any(k.startswith("eval_") for k in logs.keys()):
            self._print_compact(step, selected, prefix="DPO[EVAL]")
        else:
            self._print_compact(step, selected, prefix="DPO")
 
        # Save to CSV (includes more keys via merged last-values)
        self._write_csv_row(step, selected)
 
 
 
 
 
 
@dataclass
class EarlyStopCfg:
    # eval-loss early stop
    metric_name: str = "eval_loss"
    minimize: bool = True          # True for loss; False for accuracy-like
    patience: int = 3              # how many evals without improvement
    min_delta: float = 0.0         # required improvement magnitude
    # hard stop guard (optional)
    max_steps: Optional[int] = None
 
 
class EarlyStoppingOnMetricCallback(TrainerCallback):
    """
    Stops training when a chosen eval metric stops improving.
    Works for both SFT and DPO as long as metric is logged (e.g. eval_loss).
    """
    def __init__(self, cfg: EarlyStopCfg):
        self.cfg = cfg
        self._log = get_logger()
        self.best: Optional[float] = None
        self.bad_count: int = 0
 
    def _is_improved(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.cfg.minimize:
            return current < (self.best - self.cfg.min_delta)
        else:
            return current > (self.best + self.cfg.min_delta)
 
    def on_evaluate(self, args, state, control, metrics: Optional[Dict[str, Any]] = None, **kwargs):
        if not metrics:
            return control
 
        m = metrics.get(self.cfg.metric_name, None)
        if m is None:
            # metric not present → do nothing
            return control
 
        try:
            m = float(m)
        except Exception:
            return control
 
        if self._is_improved(m):
            old = self.best
            self.best = m
            self.bad_count = 0
            self._log.info(
                "EarlyStop(metric=%s): improved from %s to %.6f",
                self.cfg.metric_name,
                "None" if old is None else f"{old:.6f}",
                m,
            )
        else:
            self.bad_count += 1
            self._log.info(
                "EarlyStop(metric=%s): no improvement (current=%.6f best=%.6f) | bad=%d/%d",
                self.cfg.metric_name,
                m,
                self.best,
                self.bad_count,
                self.cfg.patience,
            )
            if self.bad_count >= self.cfg.patience:
                self._log.warning(
                    "EarlyStop(metric=%s): STOP training (patience reached).",
                    self.cfg.metric_name,
                )
                control.should_training_stop = True
 
        # Optional hard stop by steps
        if self.cfg.max_steps is not None and state.global_step >= int(self.cfg.max_steps):
            self._log.warning("EarlyStop: reached max_steps=%d → STOP", int(self.cfg.max_steps))
            control.should_training_stop = True
 
        return control
 
 
class DPOOverOptimizationStopCallback(TrainerCallback):
    """
    DPO-specific guard: stop if training becomes trivially perfect and margins explode.
    This prevents over-optimization that often harms JSON format / generalization.
 
    Triggers if for N logs in a row:
      rewards/accuracies >= acc_threshold
      AND rewards/margins >= margin_threshold
    """
    def __init__(
        self,
        acc_threshold: float = 0.98,
        margin_threshold: float = 8.0,
        consecutive: int = 5,
    ):
        self.acc_threshold = float(acc_threshold)
        self.margin_threshold = float(margin_threshold)
        self.consecutive = int(consecutive)
        self._log = get_logger()
        self._streak = 0
 
    def on_log(self, args, state, control, logs: Optional[Dict[str, Any]] = None, **kwargs):
        if not logs:
            return control
 
        acc = logs.get("rewards/accuracies", None) or logs.get("eval_rewards/accuracies", None)
        margin = logs.get("rewards/margins", None) or logs.get("eval_rewards/margins", None)
 
        if acc is None or margin is None:
            return control
 
        try:
            acc = float(acc)
            margin = float(margin)
        except Exception:
            return control
 
        if (acc >= self.acc_threshold) and (margin >= self.margin_threshold):
            self._streak += 1
            self._log.warning(
                "DPOOverOptGuard: streak=%d/%d | acc=%.4f (>=%.2f) | margin=%.4f (>=%.2f)",
                self._streak,
                self.consecutive,
                acc, self.acc_threshold,
                margin, self.margin_threshold,
            )
            if self._streak >= self.consecutive:
                self._log.warning("DPOOverOptGuard: STOP training (over-optimization detected).")
                control.should_training_stop = True
        else:
            if self._streak > 0:
                self._log.info("DPOOverOptGuard: reset streak (acc=%.4f margin=%.4f)", acc, margin)
            self._streak = 0
 
        return control
 
 
 