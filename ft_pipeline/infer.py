# ft_pipeline/infer.py
import re
import json
import torch
from .logger import get_logger
 
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
 
 
def build_prompt(tokenizer, messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not prompt.endswith("\n"):
        prompt += "\n"
    return prompt
 
 
def extract_json_candidate(text: str) -> str:
    t = text.strip()
    m = JSON_FENCE_RE.search(t)
    if m:
        return m.group(1).strip()
 
    start_obj = t.find("{")
    start_arr = t.find("[")
    if start_obj == -1 and start_arr == -1:
        return t
 
    start = min([x for x in [start_obj, start_arr] if x != -1])
    return t[start:].strip()
 
 
def try_parse_json(text: str):
    candidate = extract_json_candidate(text)
    try:
        parsed = json.loads(candidate)
        return True, parsed, None, candidate
    except Exception as e:
        return False, None, str(e), candidate
 
 
@torch.no_grad()
def generate(
    model,
    tokenizer,
    messages,
    max_new_tokens: int,
    use_bf16: bool = True,
    do_sample: bool = False,
    log_first_call: bool = False,
):
    """
    Gemma3 + SDPA can throw dtype mismatch (query fp32, key/value bf16) if you run without autocast.
    The robust fix is to ALWAYS run generate under CUDA autocast (bf16 or fp16).
    """
    log = get_logger()
    model.eval()
 
    # Disable gradient checkpointing for inference
    was_gc = getattr(model, "is_gradient_checkpointing", False)
    if was_gc:
        model.gradient_checkpointing_disable()
 
    # Enable cache for faster generation
    model.config.use_cache = True
 
    prompt = build_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
 
    # Move to model device (works for both plain + peft)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
 
    # attention_mask should be integer type
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
 
    # Choose autocast dtype
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
 
    if log_first_call:
        p = next(model.parameters())
        log.info(
            "Inference start | device=%s | model_param_dtype=%s | input_ids_dtype=%s | autocast=%s",
            str(device),
            str(p.dtype),
            str(inputs["input_ids"].dtype),
            str(amp_dtype),
        )
 
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,      
        num_beams=1,            
        temperature=None,      
        top_p=None,
        top_k=None,
        typical_p=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
 
    # ✅ Critical: autocast prevents Gemma3 SDPA dtype mismatch
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        out = model.generate(**inputs, **gen_kwargs)
 
    # Restore training-ish settings
    model.config.use_cache = False
    if was_gc:
        model.gradient_checkpointing_enable()
 
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
 
 
 
 