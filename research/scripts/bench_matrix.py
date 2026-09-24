"""Offline four-checkpoint matrix on the three Jev-comparable suites."""
import argparse
import csv
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research", "scripts"))

import laya
import laya.agent as laya_agent
from bench_local import build_sequence, metrics, score_cases, temp_for
from laya.common import render_options, serialize_state
from transformers import AutoModelForMaskedLM, AutoTokenizer

SEED = 13
MASK_ID = 151669
PREFIX = "Decision: "
JEV_REFS = {
    "ag_news": {"accuracy": 0.910, "n": 100, "source": "AbdelStark/jev-benchmarks v0.1.0"},
    "emotion": {"accuracy": 0.480, "brier": 0.846, "nll": 5.588, "n": 100,
                 "source": "AbdelStark/jev-benchmarks v0.1.0"},
    "banking77": {"accuracy": 0.870, "n": 100, "labels": 72,
                  "source": "AbdelStark/jev-benchmarks v0.1.0"},
}


def build_suites(n, banking_test_limit=None, banking_test_csv=None,
                 ag_news_test_limit=None, emotion_test_limit=None,
                 ag_news_test_jsonl=None, emotion_test_jsonl=None):
    def load_dataset(*args, **kwargs):
        from datasets import load_dataset as load
        return load(*args, **kwargs)

    def test_rows(path, dataset, config, limit, classes):
        limit = n if limit is None else limit
        if limit < 0:
            raise ValueError("%s test limit must be nonnegative" % dataset)
        if path is not None:
            with open(os.path.expanduser(path), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if any(not isinstance(r, dict) or not isinstance(r.get("text"), str)
                   or type(r.get("label")) is not int or not 0 <= r["label"] < classes
                   for r in rows):
                raise ValueError("%s JSONL requires text strings and valid integer labels" % dataset)
        else:
            rows = list(load_dataset(dataset, *config, split="test"))
        if limit:
            rows = rows[:limit]
        if not rows or (limit and len(rows) != limit):
            raise ValueError("%s test source is empty or has fewer than %d examples" % (dataset, limit))
        return rows

    banking_limit = n if banking_test_limit is None else banking_test_limit
    if banking_limit < 0:
        raise ValueError("banking_test_limit must be nonnegative")
    suites = {}

    rows = test_rows(ag_news_test_jsonl, "fancyzhx/ag_news", (), ag_news_test_limit, 4)
    crit = {"world": "world news and international politics", "sports": "sports",
            "business": "business and economy", "sci_tech": "science and technology"}
    keys = list(crit)
    cases = [({"article": r["text"]}, {"topic": {"type": "choice",
             "instructions": "What is the topic of `article`?", "criteria": dict(crit)}})
             for r in rows]
    suites["ag_news"] = (cases, [keys.index(keys[int(r["label"])]) for r in rows], keys)

    rows = test_rows(emotion_test_jsonl, "dair-ai/emotion", ("split",), emotion_test_limit, 6)
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    cases = [({"text": r["text"]}, {"emotion": {"type": "choice",
             "instructions": "Which emotion is most strongly expressed in `text`?",
             "criteria": {x: None for x in names}}}) for r in rows]
    suites["emotion"] = (cases, [int(r["label"]) for r in rows], names)

    if banking_test_csv is not None:
        with open(os.path.expanduser(banking_test_csv), encoding="utf-8", newline="") as f:
            rows = [{"text": r["text"], "label_text": r["category"]} for r in csv.DictReader(f)]
        labels = sorted({r["label_text"] for r in rows})
    else:
        d = load_dataset("mteb/banking77", split="test")
        labels = sorted(set(d["label_text"]))
        rows = list(d)
    if banking_limit:
        rows = rows[:banking_limit]
        if len(rows) != banking_limit or len(labels) != 77:
            raise ValueError("Banking77 must provide 77 labels and %d examples" % banking_limit)
    options = [x.replace("_", " ") for x in labels]
    cases, gold = [], []
    for r in rows:
        cases.append(({"message": r["text"]}, {"intent": {"type": "choice",
                     "instructions": "Which banking intent does `message` express?",
                     "criteria": dict(zip(options, [None] * len(options)))}}))
        gold.append(options.index(r["label_text"].replace("_", " ")))
    suites["banking77"] = (cases, gold, options)
    return suites


def one_hot_metrics(rows):
    """Metrics for hard labels; soft fields report probability mass on gold."""
    out = metrics(rows)
    valid = [x for x in rows if x[1] is not None]
    if valid:
        soft_acc = np.mean([float(x[1][x[0]]) for x in valid])
        soft_brier = np.mean([float(((np.asarray(x[1]) - np.eye(len(x[1]))[x[0]]) ** 2).sum())
                              for x in valid])
        out["soft_accuracy"] = round(float(soft_acc), 4)
        out["brier_vs_soft"] = round(float(soft_brier), 4)
    else:
        out["soft_accuracy"] = None
        out["brier_vs_soft"] = None
    return out


def load_laya(path, device):
    # ModernBERT SDPA can produce nonfinite values with padded batches. Keep this
    # evaluator-local and do not alter the library's default loader behavior.
    original = laya_agent.build_model
    def eager(cfg, encoder_dir=None):
        return original(cfg, encoder_dir=encoder_dir, attention_implementation="eager")
    laya_agent.build_model = eager
    try:
        agent = laya.load(path, device=device)
    finally:
        laya_agent.build_model = original
    agent.model.eval()
    return agent


def run_laya(path, suites, device):
    agent = load_laya(path, device)
    results = {}
    for name, (cases, gold, _) in suites.items():
        logits, index, seconds, dropped = score_cases(agent, cases, max_tokens=8192,
                                                       max_seqs=64, tag=name)
        rows = []
        for (ci, _, qt, k), z in zip(index, logits):
            rows.append((gold[ci], None if z is None else _softmax(z, temp_for(agent, qt, k))))
        m = one_hot_metrics(rows)
        m.update(seconds=round(seconds, 3), dropped=dropped,
                 ms_per_case=round(1000 * seconds / len(cases), 3))
        results[name] = m
    runtime = {"model_path": os.path.abspath(os.path.expanduser(path)), "device": str(agent.device),
               "temperature": [float(x) for x in agent.temperature],
               "temperature_by_options": {k: float(v) for k, v in agent.temperature_by_options.items()},
               "temperature_raw": [float(x) for x in agent.temperature_raw],
               "temperature_by_options_raw": {k: float(v) for k, v in agent.temperature_by_options_raw.items()},
               "max_len": agent.cfg.get("max_len"), "head_max_len": agent.cfg.get("head_max_len")}
    del agent
    return {"runtime": runtime, "suites": results}


def _softmax(z, temperature=1.0):
    z = np.asarray(z, dtype=float) / max(1e-3, float(temperature))
    e = np.exp(z - np.max(z))
    return e / e.sum()


def qwen_identifiers(tok, count):
    pool = list(string.ascii_uppercase + string.ascii_lowercase + string.digits +
                string.punctuation)
    found = []
    for text in pool:
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in tok.all_special_ids and ids[0] not in [x[1] for x in found]:
            found.append((text, ids[0]))
        if len(found) == count:
            return found
    raise ValueError("Tokenizer provides fewer than %d unique one-token identifiers" % count)


def qwen_input(tok, state, qid, qdef, max_length, mapping, prompt_mode="full"):
    options = render_options({"t": qdef["type"], "ins": qdef["instructions"],
                              "crit": qdef.get("criteria")})
    identifiers = [x[0] for x in mapping[:len(options)]]
    candidates = [x[1] for x in mapping[:len(options)]]
    if prompt_mode == "state_question":
        user = ("State:\n%s\n\n%s\n\nDecision: " %
                (serialize_state(state), qdef["instructions"]))
    elif prompt_mode == "compact":
        user = ("State:\n%s\n\nQuestion:\n%s\n\nOptions:\n%s" %
                (serialize_state(state), qdef["instructions"],
                 "\n".join("%s: %s" % pair for pair in zip(identifiers, options))))
    else:
        user = ("Evaluate this decision using the full state and original question below. "
                "Choose exactly one option, and fill the decision with only its letter identifier.\n\n"
                "State:\n%s\n\nQuestion ID: %s\nType: %s\nInstructions:\n%s\n"
                "Original criteria (original order):\n%s\n\nOptions (original label order):\n%s"
                % (serialize_state(state), qid, qdef["type"], qdef["instructions"],
                   json.dumps(qdef.get("criteria"), ensure_ascii=False),
                   "\n".join("%s: %s" % pair for pair in zip(identifiers, options))))
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False) + PREFIX
    ids = tok.encode(prompt, add_special_tokens=False, truncation=False) + [MASK_ID]
    if ids.count(MASK_ID) != 1 or len(ids) > max_length:
        raise ValueError("invalid mask or sequence length for %s" % qid)
    return ids, identifiers, candidates, options, prompt


def _qwen_additive_mask(attention, dtype):
    """Build the 4D mask expected by the custom Qwen SDPA implementation."""
    valid = attention.bool()
    invalid = (~valid[:, None, None, :]) | (~valid[:, None, :, None])
    return torch.zeros(invalid.shape, dtype=dtype, device=attention.device).masked_fill(
        invalid, -1e4)


def _qwen_hidden(model, inputs, attention, mask_mode):
    if mask_mode == "4d":
        dtype = next(model.parameters()).dtype
        model_attention = _qwen_additive_mask(attention, dtype)
    else:
        model_attention = torch.ones_like(attention, dtype=torch.bool)
    return model.model(input_ids=inputs, attention_mask=model_attention).last_hidden_state


def _qwen_mask_error(exc):
    text = str(exc).lower()
    return isinstance(exc, (RuntimeError, TypeError, ValueError, IndexError)) and any(
        word in text for word in ("attention", "mask", "shape", "dimension", "size", "4d")
    )


def _qwen_grouped_hidden(model, inputs, attention, batch, seq_len, device):
    hidden_parts = []
    for length in sorted(set(len(x["ids"]) for x in batch)):
        members = [i for i, x in enumerate(batch) if len(x["ids"]) == length]
        grouped_inputs = inputs[members, :length]
        grouped_attention = attention[members, :length]
        hidden_parts.append((members, _qwen_hidden(
            model, grouped_inputs, grouped_attention, "grouped_2d")))
    hidden_all = torch.empty(
        (len(batch), seq_len, hidden_parts[0][1].shape[-1]),
        dtype=hidden_parts[0][1].dtype, device=device)
    for members, part in hidden_parts:
        hidden_all[members, :part.shape[1]] = part
    return hidden_all


def run_qwen(path, suites, device, dtype_name, max_length, prompt_mode="full",
             max_seqs=32, max_tokens=8192):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_name]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tok.encode(tok.convert_ids_to_tokens(MASK_ID), add_special_tokens=False) != [MASK_ID]:
        raise ValueError("Qwen tokenizer does not round-trip mask ID %d" % MASK_ID)
    model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True,
                                                  torch_dtype=dtype, local_files_only=True)
    model.to(device).eval()
    results, mappings = {}, {}
    for name, (cases, gold, options) in suites.items():
        mapping = qwen_identifiers(tok, len(options))
        mappings[name] = [{"identifier": x, "token_id": i} for x, i in mapping]
        records, dropped = [], 0
        for ci, (state, questions) in enumerate(cases):
            for qid, qdef in questions.items():
                ids, labels, candidate_ids, _, _ = qwen_input(
                    tok, state, qid, qdef, max_length, mapping, prompt_mode)
                records.append({"ids": ids, "qid": qid, "candidate_ids": candidate_ids,
                                "candidate_mask": [True] * len(candidate_ids),
                                "gold": gold[ci], "labels": labels, "ci": ci})
        records.sort(key=lambda x: len(x["ids"]))
        rows, batches = [], []
        pos = 0
        while pos < len(records):
            batch = [records[pos]]
            pos += 1
            while pos < len(records):
                proposed = batch + [records[pos]]
                padded = max(len(x["ids"]) for x in proposed)
                if len(proposed) > max_seqs or padded * len(proposed) > max_tokens:
                    break
                batch = proposed
                pos += 1
            batches.append(batch)
        total_tokens = sum(len(x["ids"]) for x in records)
        padded_tokens = sum(max(len(x["ids"]) for x in b) * len(b) for b in batches)
        inference_seconds = 0.0
        mask_mode = "4d"
        with torch.inference_mode():
            for batch in batches:
                seq_len = max(len(x["ids"]) for x in batch)
                inputs = torch.full((len(batch), seq_len), tok.pad_token_id,
                                    dtype=torch.long, device=device)
                attention = torch.zeros((len(batch), seq_len), dtype=torch.bool, device=device)
                mask_positions = []
                for row, record in enumerate(batch):
                    length = len(record["ids"])
                    inputs[row, :length] = torch.tensor(record["ids"], dtype=torch.long,
                                                         device=device)
                    attention[row, :length] = True
                    mask_positions.append(record["ids"].index(MASK_ID))
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize(device)
                batch_start = time.perf_counter()
                if mask_mode == "grouped_2d":
                    hidden_all = _qwen_grouped_hidden(
                        model, inputs, attention, batch, seq_len, device)
                else:
                    try:
                        hidden_all = _qwen_hidden(model, inputs, attention, mask_mode)
                    except Exception as exc:
                        if not _qwen_mask_error(exc):
                            raise
                        # Older/custom remote-code revisions only accept 2D masks.
                        # Keep every fallback call unpadded, so its all-true mask is valid.
                        mask_mode = "grouped_2d"
                        hidden_all = _qwen_grouped_hidden(
                            model, inputs, attention, batch, seq_len, device)
                hidden = hidden_all[torch.arange(len(batch), device=device),
                                    torch.tensor(mask_positions, device=device)].float()
                candidate_count = max(len(x["candidate_ids"]) for x in batch)
                cand = torch.zeros((len(batch), candidate_count), dtype=torch.long, device=device)
                candidate_mask = torch.zeros((len(batch), candidate_count), dtype=torch.bool,
                                             device=device)
                for row, record in enumerate(batch):
                    count = len(record["candidate_ids"])
                    cand[row, :count] = torch.tensor(record["candidate_ids"], device=device)
                    candidate_mask[row, :count] = torch.tensor(record["candidate_mask"],
                                                               dtype=torch.bool, device=device)
                rows_w = model.lm_head.weight[cand].float()
                z_batch = torch.einsum("bd,bkd->bk", hidden, rows_w)
                bias = getattr(model.lm_head, "bias", None)
                if bias is not None:
                    z_batch = z_batch + bias[cand].float()
                for row, record in enumerate(batch):
                    z = z_batch[row][candidate_mask[row]].cpu().numpy()
                    if not np.isfinite(z).all():
                        raise ValueError("non-finite candidate logits at %s/%d" % (name, record["ci"]))
                    rows.append((record["gold"], _softmax(z)))
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter() - batch_start
        seconds = inference_seconds
        m = one_hot_metrics(rows)
        m.update(seconds=round(seconds, 3), dropped=dropped,
                 ms_per_case=round(1000 * seconds / len(cases), 3), batches=len(batches),
                 mean_batch=round(len(records) / len(batches), 3),
                 pad_ratio=round(1 - total_tokens / padded_tokens, 6),
                 max_length=max(len(x["ids"]) for x in records), total_tokens=total_tokens)
        results[name] = m
    runtime = {"model_path": os.path.abspath(os.path.expanduser(path)), "device": str(device),
               "dtype": dtype_name, "mask_id": MASK_ID, "enable_thinking": False,
               "attention_mask": "4d additive (grouped 2d fallback)", "candidate_projection": True,
               "max_length": max_length, "prompt_mode": prompt_mode,
               "max_seqs": max_seqs, "max_tokens": max_tokens, "identifiers": mappings}
    del model
    return {"runtime": runtime, "suites": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("laya-base", "laya-finetuned"):
        ap.add_argument("--" + name)
    for name in ("qwen-base", "qwen-finetuned"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--qwen-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--qwen-prompt-mode", choices=("full", "compact", "state_question"), default="full")
    ap.add_argument("--qwen-max-seqs", type=int, default=32)
    ap.add_argument("--qwen-max-tokens", type=int, default=8192)
    ap.add_argument("--qwen-only", action="store_true")
    a = ap.parse_args()
    if a.n != 400 or a.max_length < 1 or a.qwen_max_seqs < 1 or a.qwen_max_tokens < 1:
        ap.error("--n must be exactly 400; lengths and Qwen batching limits must be positive")
    if not a.qwen_only and (not a.laya_base or not a.laya_finetuned):
        ap.error("--laya-base and --laya-finetuned are required unless --qwen-only is used")
    output = os.path.abspath(os.path.expanduser(a.output))
    suites = build_suites(a.n)
    payload = {"protocol": {"seed": SEED, "n": a.n, "datasets_offline": True,
                            "suites": list(suites), "jev_refs": JEV_REFS,
                            "banking_jev_labels": 72, "allow_nan": False,
                            "qwen_prompt_mode": a.qwen_prompt_mode,
                            "qwen_max_seqs": a.qwen_max_seqs,
                            "qwen_max_tokens": a.qwen_max_tokens,
                            "qwen_only": a.qwen_only,
                            "labels": {k: v[2] for k, v in suites.items()},
                            "platform": platform.platform()}, "models": {}}
    if not a.qwen_only:
        for key, path in (("laya_base", a.laya_base), ("laya_finetuned", a.laya_finetuned)):
            payload["models"][key] = {"kind": "laya", **run_laya(path, suites, a.device)}
    for key, path in (("qwen_base", a.qwen_base), ("qwen_finetuned", a.qwen_finetuned)):
        payload["models"][key] = {"kind": "qwen", **run_qwen(path, suites, a.device,
                                                                  a.qwen_dtype, a.max_length,
                                                                  a.qwen_prompt_mode,
                                                                  a.qwen_max_seqs,
                                                                  a.qwen_max_tokens)}
    with open(output, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print("Wrote %s" % output)


if __name__ == "__main__":
    main()
