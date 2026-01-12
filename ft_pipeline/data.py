# ft_pipeline/data.py
import json
from datasets import load_dataset
from .logger import get_logger
 
def load_prompt_completion_dataset(train_jsonl: str, val_jsonl: str, train_limit: int | None = None, val_limit: int | None = None):
    """
    Expects each JSONL row to contain:
      - messages: list[{"role": "...", "content": "..."}]  (or a JSON string)
      - target: {"content": "..."} OR string (optionally JSON string)
    Returns HF DatasetDict with columns: prompt, completion
      prompt: list[dict] (chat messages)
      completion: list[dict] (single assistant message)
    """
    log = get_logger()
    log.info("Loading datasets")
    log.info(f"  train: {train_jsonl}")
    log.info(f"  val:   {val_jsonl}")
    ds = load_dataset("json", data_files={"train": train_jsonl, "validation": val_jsonl})
    log.info("Converting to prompt/completion format")
 
    def _to_prompt_completion(batch):
        prompts, completions = [], []
        for messages, target in zip(batch["messages"], batch["target"]):
            if isinstance(messages, str):
                messages = json.loads(messages)
 
            # extract target_text
            if isinstance(target, dict):
                target_text = target.get("content", "")
            else:
                try:
                    t = json.loads(target)
                    target_text = t.get("content", "") if isinstance(t, dict) else str(target)
                except Exception:
                    target_text = str(target)
 
            target_text = target_text.lstrip()
           
            prompts.append(messages)
            completions.append([{"role": "assistant", "content": target_text}])
 
        return {"prompt": prompts, "completion": completions}
 
    # Remove original columns -> keep only prompt/completion
    ds = ds.map(_to_prompt_completion, batched=True, remove_columns=ds["train"].column_names)
 
    if train_limit is not None:
        ds["train"] = ds["train"].select(range(min(train_limit, len(ds["train"]))))
 
    if val_limit is not None:
        ds["validation"] = ds["validation"].select(range(min(val_limit, len(ds["validation"]))))
 
    log.info(
        f"Dataset ready | train={len(ds['train'])} | val={len(ds['validation'])}"
    )
    return ds
 
 
 
 
 
def load_dpo_dataset(
    train_jsonl: str,
    val_jsonl: str,
    tokenizer,
    train_limit: int | None = None,
    val_limit: int | None = None,
):
    """
    Expects each JSONL row:
      - messages: list[{"role","content"}, ...] (or JSON string)
      - chosen: str (assistant answer)
      - rejected: str (assistant answer)
 
    Returns DatasetDict with columns:
      - prompt: str  (chat template with add_generation_prompt=True)
      - chosen: str  (assistant completion ONLY)
      - rejected: str
    """
    log = get_logger()
    log.info("Loading DPO datasets")
    log.info(f"  train: {train_jsonl}")
    log.info(f"  val:   {val_jsonl}")
 
    ds = load_dataset("json", data_files={"train": train_jsonl, "validation": val_jsonl})
 
    def _to_dpo(batch):
        prompts, chosens, rejecteds = [], [], []
        for messages, chosen, rejected in zip(batch["messages"], batch["chosen"], batch["rejected"]):
            if isinstance(messages, str):
                messages = json.loads(messages)
 
            # Prompt text must end at assistant generation boundary
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if not prompt_text.endswith("\n"):
                prompt_text += "\n"
 
            c = (chosen or "").lstrip()
            r = (rejected or "").lstrip()
 
            prompts.append(prompt_text)
            chosens.append(c)
            rejecteds.append(r)
 
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}
 
    ds = ds.map(_to_dpo, batched=True, remove_columns=ds["train"].column_names)
 
    if train_limit is not None:
        ds["train"] = ds["train"].select(range(min(train_limit, len(ds["train"]))))
 
    if val_limit is not None:
        ds["validation"] = ds["validation"].select(range(min(val_limit, len(ds["validation"]))))
 
    log.info(f"DPO dataset ready | train={len(ds['train'])} | val={len(ds['validation'])}")
    return ds
 
 
 
 
 