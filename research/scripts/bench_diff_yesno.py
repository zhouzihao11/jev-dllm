"""Offline diffusion BASE zero-shot benchmark with shared Yes/No mask scoring.

Use --limit-per-suite 3 for a remote smoke run, or 0 for all decisions.
Each forward contains an original-order batch of decisions and all of their masks.
"""
import argparse
import json
import os
import time
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from bench_ar_matrix import make_suites
from bench_local import metrics, np, softmax_t, to_internal, torch
from laya.common import render_options, serialize_state
from transformers import AutoModelForMaskedLM, AutoTokenizer


def token_mapping(tok, base):
    mapping = {}
    for text in ("Yes", "No"):
        ids = tok.encode(text, add_special_tokens=False)
        if (len(ids) != 1 or ids[0] in tok.all_special_ids
                or tok.decode(ids, clean_up_tokenization_spaces=False) != text):
            raise ValueError("%s must round-trip as a single non-special token" % text)
        mapping[text] = ids[0]
    if mapping["Yes"] == mapping["No"]:
        raise ValueError("Yes and No must have distinct token IDs")
    mask_id = getattr(base.config, "mask_token_id", None)
    if mask_id is None:
        mask_id = tok.mask_token_id if tok.mask_token_id is not None else 151669
    if tok.mask_token_id not in (None, mask_id):
        raise ValueError("Tokenizer and model mask IDs disagree")
    mask_text = tok.convert_ids_to_tokens(mask_id)
    if not mask_text or tok.encode(mask_text, add_special_tokens=False) != [mask_id]:
        raise ValueError("Mask token must round-trip to its exact ID")
    if mask_id in mapping.values():
        raise ValueError("Mask and answer tokens must differ")
    return mapping, mask_id, mask_text


def decision_input(tok, state, qdef, mask_id, mask_text, max_length):
    q = to_internal(qdef)
    if q["t"] not in ("choice", "score", "noul"):
        raise ValueError("Unsupported question type: %s" % q["t"])
    options = render_options(q)
    if not options:
        raise ValueError("Decision has no options")
    state_text = serialize_state(state)
    if any(mask_text in text for text in [state_text, q["ins"]] + options):
        raise ValueError("Source data contains the mask token")
    user = "State:\n%s\n\nQuestion:\n%s\n\n" % (state_text, q["ins"])
    if q["t"] == "noul":
        user += "Answer Yes or No.\nAnswer:" + mask_text
        count = 1
    else:
        user += "For each option, mark Yes if it answers the question, otherwise No.\n"
        user += "\n".join(option + ": " + mask_text for option in options)
        count = len(options)
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False)
    parts = prompt.split(mask_text)
    if len(parts) != count + 1:
        raise ValueError("Chat template changed the number of masks")
    ids, positions = [], []
    # Insert IDs explicitly: never rely on a textual placeholder being a mask.
    for i, part in enumerate(parts):
        ids.extend(tok.encode(part, add_special_tokens=False, truncation=False))
        if i < count:
            positions.append(len(ids))
            ids.append(mask_id)
    if ids.count(mask_id) != count:
        raise ValueError("Expected exactly %d mask IDs" % count)
    if len(ids) > max_length:
        raise ValueError("Prompt has %d tokens, context limit %d; no truncation"
                         % (len(ids), max_length))
    return ids, positions, options, prompt


def validate_probs(p):
    if (not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or not np.isclose(p.sum(), 1.0)):
        raise ValueError("Non-finite or unnormalized probabilities")


def aggregate(records):
    result = metrics([(r["gold_idx"], r["probs"]) for r in records])
    soft_accuracy, brier, errors = [], [], []
    for r in records:
        p = np.asarray(r["probs"], float)
        if r["soft"] is not None:
            target = np.asarray(r["soft"], float)
            if target.sum() > 0:
                target = target / target.sum()
                soft_accuracy.append(float(np.dot(p, target)))
                brier.append(float(((p - target) ** 2).sum()))
        if r["gold_score"] is not None:
            errors.append(abs(float(np.dot(np.arange(len(p)), p)) - r["gold_score"]))
    result.update(
        soft_accuracy=round(float(np.mean(soft_accuracy)), 4) if soft_accuracy else None,
        brier_vs_soft=round(float(np.mean(brier)), 4) if brier else None,
        n_soft=len(soft_accuracy),
        score_mae=round(float(np.mean(errors)), 4) if errors else None,
        within_1_level=round(float(np.mean(np.asarray(errors) <= 1)), 4) if errors else None,
        n_scored=len(errors))
    return result


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def forward_masks(base, inputs, attention_mask, rows, positions, weight, bias, timed=True):
    synchronize(inputs.device)
    start = time.perf_counter() if timed else 0.0
    output = base.model(input_ids=inputs, attention_mask=attention_mask, return_dict=True)
    hidden = output.last_hidden_state[rows, positions]
    if hidden.dtype != weight.dtype:
        raise ValueError("Hidden and lm_head dtypes differ; refusing implicit conversion")
    logits = torch.nn.functional.linear(hidden, weight, bias).float()
    synchronize(inputs.device)
    elapsed = time.perf_counter() - start if timed else 0.0
    return logits, elapsed


def banking_coverage(suite):
    counts = Counter(int(row[2]) for row in suite)
    options = list(next(iter(suite[0][1].values()))["criteria"]) if suite else []
    return {"actual_cases": len(suite), "gold_class_coverage": len(counts),
            "gold_counts_by_class": {label: counts[i] for i, label in enumerate(options)},
            "candidate_count": len(options)}


def evaluation_data(suites, args, banking_test_limit):
    sources = {
        "typed_decisions": (args.typed_data, None, "parquet", 0),
        "ag_news": (args.ag_news_test_jsonl, "fancyzhx/ag_news", "jsonl", args.ag_news_test_limit),
        "emotion": (args.emotion_test_jsonl, "dair-ai/emotion", "jsonl", args.emotion_test_limit),
        "banking77": (args.banking_test_csv, "mteb/banking77", "csv", banking_test_limit),
        "prompt_injection": (args.prompt_data, None, "parquet", 0),
        "sst5": (args.sst5_data, None, "jsonl", args.sst5_test_limit),
    }

    def coverage(name, suite):
        cases, previous, classes, candidates = 0, None, {}, {}
        for state, questions, gold, extra in suite:
            if name != "typed_decisions" or cases == 0 or state is not previous:
                cases += 1
            previous = state
            qdef = next(iter(questions.values()))
            counts = classes.setdefault(qdef["type"], Counter())
            counts[int(gold)] += 1
            candidates.setdefault(qdef["type"], set()).add(len(render_options(to_internal(qdef))))
        return {"actual_cases": cases, "actual_decisions": len(suite),
                "candidate_counts_by_question_type": {k: sorted(v) for k, v in candidates.items()},
                "gold_class_coverage_by_question_type": {k: len(v) for k, v in classes.items()},
                "gold_counts_by_question_type": {k: dict(sorted(v.items())) for k, v in classes.items()}}

    result = {}
    for name, suite in suites.items():
        path, dataset, source_type, limit = sources[name]
        result[name] = {
            "source_path": path, "dataset_name": dataset if path is None else None,
            "source_type": "huggingface_dataset" if path is None else source_type, "split": "test",
            "requested_test_limit": limit,
            "selection_policy": "all_source_test_rows_original_order" if limit == 0
                                else "first_n_test_rows_original_order_not_stratified",
            "evaluation_decision_limit": args.limit_per_suite,
            "constructed": coverage(name, suite),
            "evaluated": coverage(name, suite[:args.limit_per_suite] if args.limit_per_suite else suite)}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("model-path", "typed-data", "prompt-data", "sst5-data", "output", "predictions"):
        ap.add_argument("--" + key, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Contiguous original-order decisions per forward; final partial batch allowed")
    ap.add_argument("--warmup-batches", type=int, default=0,
                    help="Untimed repetitions of the first real batch per suite; not scored")
    ap.add_argument("--limit-per-suite", type=int, default=0,
                    help="Maximum decisions per suite; 0 = full suite, 3 = smoke")
    ap.add_argument("--banking-test-limit", type=int, default=400,
                    help="Banking77 test prefix size (legacy default 400); 0 = all source test rows")
    ap.add_argument("--banking-test-csv", default=None,
                    help="Optional Banking77 test CSV with text/category columns; overrides mteb cache")
    ap.add_argument("--ag-news-test-limit", type=int, default=400, help="Test prefix size; 0 = all source test rows")
    ap.add_argument("--emotion-test-limit", type=int, default=400, help="Test prefix size; 0 = all source test rows")
    ap.add_argument("--sst5-test-limit", type=int, default=600, help="Test prefix size; 0 = all source test rows")
    ap.add_argument("--ag-news-test-jsonl", default=None, help="Optional frozen test JSONL with text/label fields")
    ap.add_argument("--emotion-test-jsonl", default=None, help="Optional frozen test JSONL with text/label fields")
    a = ap.parse_args()
    if a.batch_size < 1 or a.warmup_batches < 0:
        ap.error("batch-size must be positive and warmup-batches nonnegative")
    if a.max_length < 1 or a.limit_per_suite < 0:
        ap.error("max-length must be positive and limit-per-suite nonnegative")
    if a.banking_test_limit < 0:
        ap.error("banking-test-limit must be nonnegative")
    if min(a.ag_news_test_limit, a.emotion_test_limit, a.sst5_test_limit) < 0:
        ap.error("test limits must be nonnegative")
    for key in ("model_path", "typed_data", "prompt_data", "sst5_data", "output", "predictions"):
        setattr(a, key, os.path.abspath(os.path.expanduser(getattr(a, key))))
    if a.banking_test_csv is not None:
        a.banking_test_csv = os.path.abspath(os.path.expanduser(a.banking_test_csv))
    for key in ("ag_news_test_jsonl", "emotion_test_jsonl"):
        if getattr(a, key) is not None:
            setattr(a, key, os.path.abspath(os.path.expanduser(getattr(a, key))))
    if not os.path.isdir(a.model_path):
        ap.error("model-path must be a local checkpoint directory")
    if a.output == a.predictions:
        ap.error("output and predictions must differ")
    for path in (a.output, a.predictions):
        if not os.path.isdir(os.path.dirname(path)) or os.path.exists(path):
            ap.error("Output parent must exist and file must not exist: %s" % path)

    suites = make_suites(a.typed_data, a.prompt_data, a.sst5_data,
                         banking_test_limit=a.banking_test_limit, banking_test_csv=a.banking_test_csv,
                         ag_news_test_limit=a.ag_news_test_limit, emotion_test_limit=a.emotion_test_limit,
                         sst5_test_limit=a.sst5_test_limit, ag_news_test_jsonl=a.ag_news_test_jsonl,
                         emotion_test_jsonl=a.emotion_test_jsonl)
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True, local_files_only=True)
    base = AutoModelForMaskedLM.from_pretrained(
        a.model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=getattr(torch, a.dtype))
    device = torch.device(a.device)
    base.to(device).eval()
    mapping, mask_id, mask_text = token_mapping(tok, base)
    limits = [a.max_length]
    for limit in (getattr(base.config, "max_position_embeddings", None),
                  getattr(tok, "model_max_length", None)):
        if isinstance(limit, int) and 0 < limit < 10**9:
            limits.append(limit)
    context_limit = min(limits)
    if not hasattr(base, "model") or not hasattr(base, "lm_head"):
        raise ValueError("Checkpoint must expose base.model and base.lm_head")
    head = base.lm_head
    pad_id = tok.pad_token_id
    if a.batch_size > 1 and (not isinstance(pad_id, int)
                            or not 0 <= pad_id < base.get_input_embeddings().weight.shape[0]):
        raise ValueError("Batching requires a valid tokenizer pad_token_id")
    candidate_ids = torch.tensor([mapping["Yes"], mapping["No"]], device=head.weight.device)
    with torch.inference_mode():
        weight = head.weight.index_select(0, candidate_ids)
        bias = head.bias.index_select(0, candidate_ids) if getattr(head, "bias", None) is not None else None
    payload = {
        "settings": dict(vars(a), model_class=type(base).__name__, mask_id=mask_id,
                         mask_token=mask_text, answer_token_ids=mapping,
                         logit_columns=["Yes", "No"], context_limit=context_limit,
                         temperature=1.0, sequences_per_forward=a.batch_size, padding=a.batch_size > 1,
                         padding_side="right", pad_token_id=pad_id,
                         padding_policy="to actual batch maximum length; none for singleton batches",
                         batch_order="contiguous original suite order; no sorting or token budget",
                         warmup_policy="repeat first real batch per suite; untimed and not scored",
                         attention_implementation=getattr(base.config, "_attn_implementation", None),
                         backbone_attention_implementation=getattr(base.model.config, "_attn_implementation", None)
                         if hasattr(base.model, "config") else None,
                         attention_module_classes=sorted({type(module).__name__ for module in base.model.modules()
                                                          if "attention" in type(module).__name__.lower()}),
                         projection_dtype=str(weight.dtype),
                         truncation=False, enable_thinking=False, generation=False,
                         full_vocab_forward=False, candidate_projection=True,
                         attention_mask=("4D bool [B,1,1,L] valid keys; True=allowed, right padding False; bidirectional"
                                         if a.batch_size > 1 else "2D bool [1,L] all-valid; bidirectional"),
                         system="You are a helpful assistant.",
                         limit_unit="decisions",
                         timing="synchronized backbone + mask gather + two-row projection + float32 cast; "
                                "excludes tokenization, collation, index allocation, H2D, D2H, CPU softmax and IO",
                         prediction_seconds="batch_seconds / actual batch_size; amortized, not per-request latency",
                         performance_scope="timed batches only; token counts, latency and throughput exclude warmup",
                         peak_cuda_memory_scope="per suite, reset after model load; includes resident weights, warmup and timed batches"),
        "suite_sizes": {}, "suites": {}, "prompt_examples": {}}
    banking = suites["banking77"]
    payload["evaluation_data"] = evaluation_data(suites, a, a.banking_test_limit)
    payload["suite_metadata"] = {"banking77": {
        "dataset_name": None if a.banking_test_csv is not None else "mteb/banking77", "split": "test",
        "source_type": "csv" if a.banking_test_csv is not None else "huggingface_dataset",
        "source_path": a.banking_test_csv,
        "selection_policy": "all_source_test_rows_original_order" if a.banking_test_limit == 0
                            else "first_n_test_rows_original_order_not_stratified",
        "requested_test_limit": a.banking_test_limit,
        "constructed": banking_coverage(banking),
        "evaluation_selection_policy": "first_n_suite_decisions_original_order" if a.limit_per_suite
                                       else "all_constructed_suite_decisions_original_order",
        "evaluated": banking_coverage(banking[:a.limit_per_suite] if a.limit_per_suite else banking)}}
    with open(a.predictions, "x", encoding="utf-8") as predictions, torch.inference_mode():
        for name, suite in suites.items():
            # make_suites retains the same state object for each typed case's questions.
            indexed, case_idx, previous = [], -1, None
            for decision_idx, (state, questions, gold, extra) in enumerate(suite):
                if name != "typed_decisions" or decision_idx == 0 or state is not previous:
                    case_idx += 1
                previous = state
                indexed.append((case_idx, decision_idx, state, questions, gold, extra))
            payload["suite_sizes"][name] = {"cases": case_idx + 1, "decisions": len(indexed)}
            if a.limit_per_suite:
                indexed = indexed[:a.limit_per_suite]
            print("[%s] starting %d decisions / %d cases" %
                  (name, len(indexed), len({x[0] for x in indexed})), flush=True)
            records, seconds, total_tokens = [], 0.0, 0
            batch_sizes, batch_seconds = [], []
            padded_tokens, max_batch_seq_length, warmup_forwards = 0, 0, 0
            if device.type == "cuda":
                synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            for batch_id, offset in enumerate(range(0, len(indexed), a.batch_size)):
                batch = []
                for ci, di, state, questions, gold, extra in indexed[offset:offset + a.batch_size]:
                    if len(questions) != 1:
                        raise ValueError("Expected one question per suite record")
                    qid, qdef = next(iter(questions.items()))
                    ids, positions, options, prompt = decision_input(
                        tok, state, qdef, mask_id, mask_text, context_limit)
                    batch.append((ci, di, qid, qdef, gold, extra, ids, positions, options, prompt))
                batch_size = len(batch)
                length = max(len(row[6]) for row in batch)
                inputs = torch.tensor([row[6] + [pad_id] * (length - len(row[6])) for row in batch],
                                      dtype=torch.long, device=device)
                attention_mask = torch.tensor([[True] * len(row[6]) + [False] * (length - len(row[6]))
                                               for row in batch], dtype=torch.bool, device=device)
                if a.batch_size > 1:
                    # The inherited BlockMask=Tensor fallback bypasses 2D mask preparation.
                    # Explicit key-mask axes broadcast across heads and queries in SDPA.
                    attention_mask = attention_mask[:, None, None, :]
                mask_rows = torch.tensor([i for i, row in enumerate(batch) for _ in row[7]],
                                         dtype=torch.long, device=device)
                mask_positions = torch.tensor([p for row in batch for p in row[7]],
                                              dtype=torch.long, device=device)
                if batch_id == 0:
                    for _ in range(a.warmup_batches):
                        warmup_logits, _ = forward_masks(
                            base, inputs, attention_mask, mask_rows, mask_positions, weight, bias, timed=False)
                        del warmup_logits
                        warmup_forwards += 1
                z_tensor, elapsed = forward_masks(
                    base, inputs, attention_mask, mask_rows, mask_positions, weight, bias)
                batch_logits = z_tensor.cpu().numpy().astype(float)
                del z_tensor, inputs, attention_mask, mask_rows, mask_positions
                batch_sizes.append(batch_size)
                batch_seconds.append(elapsed)
                padded_tokens += batch_size * length
                max_batch_seq_length = max(max_batch_seq_length, length)
                seconds += elapsed
                mask_offset = 0
                for ci, di, qid, qdef, gold, extra, ids, positions, options, prompt in batch:
                    z = batch_logits[mask_offset:mask_offset + len(positions)]
                    mask_offset += len(positions)
                    if z.shape != (len(positions), 2) or not np.isfinite(z).all():
                        raise ValueError("Invalid Yes/No logits for %s/%d/%s" % (name, ci, qid))
                    logodds = z[:, 0] - z[:, 1]
                    pairs = np.asarray([softmax_t(pair, 1.0) for pair in z])
                    for pair in pairs:
                        validate_probs(pair)
                    p = pairs[0, ::-1] if qdef["type"] == "noul" else softmax_t(logodds, 1.0)
                    validate_probs(p)
                    if not np.isfinite(logodds).all() or not 0 <= gold < len(p):
                        raise ValueError("Invalid logodds or gold index")
                    soft = extra.get("soft")
                    if soft is not None:
                        soft = np.asarray(soft, float)
                        if soft.shape != p.shape or not np.isfinite(soft).all() or np.any(soft < 0):
                            raise ValueError("Invalid soft target")
                        soft = soft.tolist()
                    score = extra.get("gold_score")
                    if score is not None and not np.isfinite(score):
                        raise ValueError("Invalid gold score")
                    record = {"id": "%s/%d/%s" % (name, ci, qid), "suite": name,
                              "case_idx": ci, "decision_idx": di, "qid": qid,
                              "workflow": extra.get("workflow"), "type": qdef["type"],
                              "gold_idx": int(gold), "soft": soft, "gold_score": score,
                              "options": options, "probs": p.tolist(),
                              "prediction_index": int(np.argmax(p)),
                              "yes_no_logits": z.tolist(), "logodds": logodds.tolist(),
                              "yes_probs": pairs[:, 0].tolist(), "answer_token_ids": mapping,
                              "mask_id": mask_id, "mask_positions": positions,
                              "input_ids": ids, "token_length": len(ids), "seconds": elapsed / batch_size,
                              "batch_id": batch_id, "batch_size": batch_size, "batch_seconds": elapsed,
                              "seconds_kind": "amortized_batch_time" if batch_size > 1 else "single_decision_forward"}
                    predictions.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    predictions.flush()
                    records.append(record)
                    total_tokens += len(ids)
                    examples = payload["prompt_examples"].setdefault(name, {})
                    examples.setdefault(qdef["type"], {"id": record["id"], "prompt": prompt,
                                                       "input_ids": ids, "mask_positions": positions})
                if len(records) // 50 != offset // 50 or len(records) == len(indexed):
                    print("[%s] %d/%d decisions; %.3fs inference" %
                          (name, len(records), len(indexed), seconds), flush=True)
            result = aggregate(records)
            for field, key in (("workflow", "by_workflow"), ("type", "by_question_type")):
                result[key] = {value: aggregate([r for r in records if r[field] == value])
                               for value in sorted({r[field] for r in records if r[field] is not None})}
            n_cases = len({r["case_idx"] for r in records})
            result.update(n_cases=n_cases, n_decisions=len(records), forwards=len(batch_sizes),
                          requested_batch_size=a.batch_size, batch_sizes=batch_sizes,
                          requested_warmup_batches=a.warmup_batches, warmup_forwards=warmup_forwards,
                          warmup_decisions_scored=0,
                          total_nonpadding_tokens=total_tokens, padded_tokens=padded_tokens,
                          padding_tokens=padded_tokens - total_tokens,
                          max_batch_seq_length=max_batch_seq_length,
                          batch_latency_seconds_p50=float(np.percentile(batch_seconds, 50)),
                          batch_latency_seconds_p95=float(np.percentile(batch_seconds, 95)),
                          decisions_per_second=len(records) / seconds,
                          peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(device)
                          if device.type == "cuda" else None,
                          seconds=seconds, ms_per_decision=1000 * seconds / len(records),
                          ms_per_case=1000 * seconds / n_cases, total_tokens=total_tokens,
                          dropped=0, case_timing_note="Time for evaluated decisions in each represented case; smoke may include a partial case")
            payload["suites"][name] = result
            print(json.dumps({name: result}, allow_nan=False), flush=True)
    with open(a.output, "x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False, allow_nan=False)
    print("Wrote %s and %s" % (a.output, a.predictions), flush=True)


if __name__ == "__main__":
    main()
