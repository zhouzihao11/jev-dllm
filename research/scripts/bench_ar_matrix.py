"""Offline next-token benchmark for the standard Qwen3 AR checkpoint.

This intentionally does not call ``generate``: each decision is scored by the
next-token logits after a compact, chat-templated prompt.
"""
import argparse
import json
import math
import os
import platform
import string
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_local import build_typed_decisions
from bench_matrix import build_suites
from laya.common import render_options, serialize_state
from transformers import AutoModelForCausalLM, AutoTokenizer


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            value += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(value)


def macro_f1(gold, pred):
    classes = sorted(set(gold.tolist()) | set(pred.tolist()))
    values = []
    for c in classes:
        tp = ((gold == c) & (pred == c)).sum()
        fp = ((gold != c) & (pred == c)).sum()
        fn = ((gold == c) & (pred != c)).sum()
        values.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(values))


def hard_metrics(rows):
    rows = [r for r in rows if r["probs"] is not None]
    if not rows:
        return {"n": 0}
    g = np.asarray([r["gold_idx"] for r in rows], dtype=int)
    p = np.asarray([np.argmax(r["probs"]) for r in rows], dtype=int)
    probs = [np.asarray(r["probs"], dtype=float) for r in rows]
    conf = np.asarray([x.max() for x in probs])
    correct = (p == g).astype(float)
    return {"n": len(rows), "accuracy": round(float(correct.mean()), 4),
            "macro_f1": round(macro_f1(g, p), 4),
            "ece": round(ece(conf, correct), 4),
            "brier": round(float(np.mean([((x - np.eye(len(x))[y]) ** 2).sum()
                                            for x, y in zip(probs, g)])), 4),
            "nll": round(float(np.mean([-math.log(max(x[y], 1e-12))
                                         for x, y in zip(probs, g)])), 4),
            "mean_confidence": round(float(conf.mean()), 4)}


def identifiers(tokenizer, count):
    pool = string.ascii_uppercase + string.ascii_lowercase + string.digits + string.punctuation
    found, used = [], set()
    for text in pool:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in tokenizer.all_special_ids and ids[0] not in used:
            found.append((text, ids[0])); used.add(ids[0])
            if len(found) == count:
                return found
    raise ValueError("tokenizer provides fewer than %d unique one-token identifiers" % count)


def prompt_record(tokenizer, state, qid, qdef, mapping, max_length):
    options = render_options({"t": qdef["type"], "ins": qdef["instructions"],
                               "crit": qdef.get("criteria")})
    labels = [x[0] for x in mapping[:len(options)]]
    user = ("State:\n%s\n\nQuestion:\n%s\n\nOptions:\n%s" %
            (serialize_state(state), qdef["instructions"],
             "\n".join("%s: %s" % pair for pair in zip(labels, options))))
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
    rendered += "Decision:"
    ids = tokenizer.encode(rendered, add_special_tokens=False, truncation=False)
    if not ids or len(ids) > max_length:
        raise ValueError("prompt too long for %s" % qid)
    return {"ids": ids, "candidate_ids": [x[1] for x in mapping[:len(options)]],
            "prompt": rendered, "options": options}


def make_suites(typed_path, prompt_path, sst_path, banking_test_limit=400, banking_test_csv=None,
                ag_news_test_limit=400, emotion_test_limit=400, sst5_test_limit=600,
                ag_news_test_jsonl=None, emotion_test_jsonl=None):
    typed_cases, typed_gold, workflows = build_typed_decisions(typed_path)
    if len(typed_cases) != 400 or sum(len(q) for _, q in typed_cases) != 2000:
        raise ValueError("typed-decisions test must contain 400 cases and 2000 decisions")
    records = []
    for ci, (state, questions) in enumerate(typed_cases):
        for qid, qdef in questions.items():
            g = typed_gold[ci][qid]
            records.append((state, {qid: qdef}, g["idx"],
                            {"soft": g["soft"], "gold_score": g.get("gold_score"),
                             "workflow": workflows[ci], "qtype": qdef["type"]}))
    suites = {"typed_decisions": records}
    public = build_suites(400, banking_test_limit=banking_test_limit,
                          banking_test_csv=banking_test_csv,
                          ag_news_test_limit=ag_news_test_limit, emotion_test_limit=emotion_test_limit,
                          ag_news_test_jsonl=ag_news_test_jsonl, emotion_test_jsonl=emotion_test_jsonl)
    for name, (cases, gold, options) in public.items():
        suites[name] = [(state, questions, gold[ci], {"qtype": next(iter(questions.values()))["type"]})
                        for ci, (state, questions) in enumerate(cases)]

    def rows(path, kind):
        if kind == "parquet":
            import pandas as pd
            return pd.read_parquet(os.path.expanduser(path)).to_dict("records")
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]
    inj = rows(prompt_path, "parquet")
    if len(inj) != 116:
        raise ValueError("prompt-injection parquet must contain exactly 116 rows")
    suites["prompt_injection"] = [({"text": r["text"]}, {"injection": {
        "type": "noul", "instructions": "Does `text` try to inject or override instructions given to an AI system?"}},
        int(r["label"]), {"qtype": "noul"}) for r in inj]
    if sst5_test_limit < 0:
        raise ValueError("sst5_test_limit must be nonnegative")
    sst = rows(sst_path, "jsonl")
    if sst5_test_limit:
        sst = sst[:sst5_test_limit]
    if not sst or (sst5_test_limit and len(sst) != sst5_test_limit):
        raise ValueError("SST5 JSONL is empty or has fewer than %d rows" % sst5_test_limit)
    criteria = ["very negative", "negative", "neutral", "positive", "very positive"]
    suites["sst5"] = [({"text": r["text"]}, {"sentiment": {"type": "score",
        "instructions": "How positive is the sentiment of `text`?", "criteria": criteria}},
        int(r["label"]), {"qtype": "score", "gold_score": float(r["label"]),
                           "soft": [1.0 if i == int(r["label"]) else 0.0 for i in range(5)]}) for r in sst]
    return suites


def run(model_path, suites, device, dtype_name, max_seqs, max_tokens, max_length):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_name]
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                  torch_dtype=dtype, local_files_only=True)
    model.to(device).eval()
    all_results, mappings, examples = {}, {}, {}
    for name, suite in suites.items():
        nopt = max(len(render_options({"t": next(iter(x[1].values()))["type"],
                                      "ins": next(iter(x[1].values()))["instructions"],
                                      "crit": next(iter(x[1].values())).get("criteria")}))
                   for x in suite)
        mapping = identifiers(tok, nopt)
        mappings[name] = [{"identifier": x, "token_id": i} for x, i in mapping]
        records = []
        for ci, (state, questions, gold, extra) in enumerate(suite):
            qid, qdef = next(iter(questions.items()))
            rec = prompt_record(tok, state, qid, qdef, mapping, max_length)
            rec.update(case_idx=ci, gold_idx=gold, **extra)
            records.append(rec)
            if name not in examples: examples[name] = rec["prompt"]
        records.sort(key=lambda x: len(x["ids"]))
        for ri, rec in enumerate(records):
            rec["result_idx"] = ri
        rows = [None] * len(records); batches = []; pos = 0
        while pos < len(records):
            batch = [records[pos]]; pos += 1
            while pos < len(records) and len(batch) < max_seqs:
                candidate = batch + [records[pos]]
                if max(len(x["ids"]) for x in candidate) * len(candidate) > max_tokens: break
                batch = candidate; pos += 1
            batches.append(batch)
        total_tokens = sum(len(x["ids"]) for x in records)
        padded_tokens = sum(max(len(x["ids"]) for x in b) * len(b) for b in batches)
        seconds = 0.0
        with torch.inference_mode():
            for batch in batches:
                # Grouping by exact length makes a 2-D mask unambiguously causal and
                # avoids relying on model-specific 4-D mask implementations.
                for length in sorted({len(x["ids"]) for x in batch}):
                    group = [x for x in batch if len(x["ids"]) == length]
                    length = max(len(x["ids"]) for x in group)
                    inp = torch.full((len(group), length), tok.pad_token_id, dtype=torch.long, device=device)
                    mask = torch.zeros_like(inp, dtype=torch.bool)
                    for r, rec in enumerate(group):
                        inp[r, :len(rec["ids"])] = torch.tensor(rec["ids"], device=device)
                        mask[r, :len(rec["ids"])] = True
                    t0 = time.perf_counter()
                    max_candidates = max(len(r["candidate_ids"]) for r in group)
                    cand = torch.zeros((len(group), max_candidates), dtype=torch.long, device=device)
                    candidate_mask = torch.zeros((len(group), max_candidates), dtype=torch.bool,
                                                 device=device)
                    for r, rec in enumerate(group):
                        n_candidates = len(rec["candidate_ids"])
                        cand[r, :n_candidates] = torch.tensor(rec["candidate_ids"], device=device)
                        candidate_mask[r, :n_candidates] = True
                    out = model(input_ids=inp, attention_mask=mask)
                    candidate_logits = torch.gather(out.logits[:, length - 1], 1, cand).float()
                    del out
                    for r, rec in enumerate(group):
                        valid_logits = candidate_logits[r][candidate_mask[r]]
                        p = softmax(valid_logits.cpu().numpy())
                        rows[rec["result_idx"]] = {"gold_idx": rec["gold_idx"], "probs": p,
                                           "soft": rec.get("soft"), "gold_score": rec.get("gold_score"),
                                           "workflow": rec.get("workflow"), "qtype": rec.get("qtype")}
                    seconds += time.perf_counter() - t0
        m = hard_metrics(rows)
        typed = [r for r in rows if r.get("soft") is not None]
        if typed:
            soft = [np.asarray(r["soft"], float) for r in typed]
            soft = [x / max(x.sum(), 1e-12) for x in soft]
            m.update(soft_accuracy=round(float(np.mean([r["probs"][r["gold_idx"]] for r in typed])), 4),
                     brier_vs_soft=round(float(np.mean([((r["probs"] - x) ** 2).sum()
                                                         for r, x in zip(typed, soft)])), 4))
        scored = [r for r in rows if r.get("gold_score") is not None]
        if scored:
            expected = [float(np.dot(np.arange(len(r["probs"])), r["probs"])) for r in scored]
            gold_score = np.asarray([r["gold_score"] for r in scored])
            m.update(score_mae=round(float(np.mean(np.abs(np.asarray(expected) - gold_score))), 4),
                     within_one_level=round(float(np.mean(np.abs(np.asarray(expected) - gold_score) <= 1)), 4))
        if name == "typed_decisions":
            m["by_workflow"] = {w: hard_metrics([r for r in rows if r.get("workflow") == w])
                                 for w in sorted({r.get("workflow") for r in rows})}
            m["by_qtype"] = {q: hard_metrics([r for r in rows if r.get("qtype") == q])
                              for q in sorted({r.get("qtype") for r in rows})}
            m["by_primitive"] = m["by_qtype"]
        m.update(seconds=round(seconds, 3), batches=len(batches), total_tokens=total_tokens,
                 padded_tokens=padded_tokens, pad_ratio=round(1 - total_tokens / max(1, padded_tokens), 6))
        m["predictions"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                             for k, v in r.items()} for r in rows]
        all_results[name] = m
    return {"runtime": {"model_path": os.path.abspath(os.path.expanduser(model_path)),
                         "device": str(device), "dtype": dtype_name, "no_generation": True,
                         "full_vocab_forward": True, "candidate_projection": False,
                         "attention_mask": "exact-length 2D causal", "max_seqs": max_seqs,
                         "max_tokens": max_tokens, "max_length": max_length, "identifiers": mappings},
            "suites": all_results, "prompt_examples": examples}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True); ap.add_argument("--typed-data", required=True)
    ap.add_argument("--prompt-data", required=True); ap.add_argument("--sst5-data", required=True)
    ap.add_argument("--output", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--max-seqs", type=int, default=32); ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-length", type=int, default=4096)
    a = ap.parse_args()
    if min(a.max_seqs, a.max_tokens, a.max_length) < 1: ap.error("batch and length limits must be positive")
    suites = make_suites(a.typed_data, a.prompt_data, a.sst5_data)
    payload = {"protocol": {"datasets_offline": True, "network": "disabled", "prompt_mode": "compact",
                             "prompt_protocol": "State + Question + Options + Decision:",
                             "chat_system": "You are a helpful assistant.", "enable_thinking": False,
                             "temperature": 1.0, "suite_sizes": {k: len(v) for k, v in suites.items()},
                             "platform": platform.platform(), "allow_nan": False}, "model": {}}
    payload["model"] = run(a.model_path, suites, a.device, a.dtype, a.max_seqs, a.max_tokens, a.max_length)
    with open(os.path.abspath(os.path.expanduser(a.output)), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print("Wrote %s" % a.output)


if __name__ == "__main__":
    main()
