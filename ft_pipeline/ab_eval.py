# ft_pipeline/ab_eval.py
import os, json
from datasets import load_dataset
from .infer import generate, try_parse_json
from .logger import get_logger
 
def run_ab_dump(
    model,
    tokenizer,
    val_jsonl: str,
    indices,
    out_path: str,
    tag: str,
    max_new_tokens: int,
    use_bf16: bool,
):
    val_raw = load_dataset("json", data_files={"val": val_jsonl})["val"]
 
    rows = []
    ok_cnt = 0
 
    for i in indices:
        raw = val_raw[i]
        messages = raw["messages"]
        if isinstance(messages, str):
            import json as _json
            messages = _json.loads(messages)
 
        pred_text = generate(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
            use_bf16=use_bf16,
            do_sample=False,
            log_first_call=(i == indices[0]),
        )
        ok, parsed, err, extracted = try_parse_json(pred_text)
        ok_cnt += int(ok)
 
        gold = raw.get("target", "")
        if isinstance(gold, dict):
            gold = gold.get("content", "")
        rows.append({
            "idx": int(i),
            "gold": gold,
            "pred_raw": pred_text,
            "pred_json_ok": bool(ok),
            "pred_json_error": err,
            "pred_json_extracted": extracted,
        })
 
    payload = {
        "tag": tag,
        "indices": list(indices),
        "max_new_tokens": int(max_new_tokens),
        "valid_json_rate": ok_cnt / max(1, len(indices)),
        "rows": rows,
    }
 
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
 
    print(f"[{tag}] Saved: {out_path}")
    print(f"[{tag}] valid_json_rate: {payload['valid_json_rate']:.2%}")
 
def make_ab_report(before_path: str, after_path: str, report_path: str):
    before = json.load(open(before_path, "r", encoding="utf-8"))
    after  = json.load(open(after_path, "r", encoding="utf-8"))
 
    b_rows = {r["idx"]: r for r in before["rows"]}
    a_rows = {r["idx"]: r for r in after["rows"]}
 
    md = []
    md.append("# A/B comparison (strict JSON)\n")
    md.append(f"- Indices: {before['indices']}\n")
    md.append(f"- max_new_tokens: {before['max_new_tokens']}\n")
    md.append(f"- valid_json_rate BEFORE: {before['valid_json_rate']:.2%}\n")
    md.append(f"- valid_json_rate AFTER:  {after['valid_json_rate']:.2%}\n\n")
 
    for idx in before["indices"]:
        b = b_rows[idx]
        a = a_rows[idx]
 
        md.append(f"## Sample idx = {idx}\n")
 
        md.append("### Gold\n")
        md.append("```text\n" + (b.get("gold") or "") + "\n```\n")
 
        md.append("### BEFORE\n")
        md.append(f"- json_ok: {b['pred_json_ok']}\n")
        if not b["pred_json_ok"]:
            md.append(f"- error: {b['pred_json_error']}\n")
        md.append("```text\n" + (b.get("pred_raw") or "") + "\n```\n")
        md.append("**Extracted JSON:**\n")
        md.append("```json\n" + (b.get("pred_json_extracted") or "") + "\n```\n")
 
        md.append("### AFTER\n")
        md.append(f"- json_ok: {a['pred_json_ok']}\n")
        if not a["pred_json_ok"]:
            md.append(f"- error: {a['pred_json_error']}\n")
        md.append("```text\n" + (a.get("pred_raw") or "") + "\n```\n")
        md.append("**Extracted JSON:**\n")
        md.append("```json\n" + (a.get("pred_json_extracted") or "") + "\n```\n")
 
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
 
    print("Wrote report:", report_path)