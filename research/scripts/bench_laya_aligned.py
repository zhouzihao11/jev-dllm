"""Native Laya, fixed input/temperature views of the shared Yes/No test protocol."""
import argparse
import json
import math
import os
import time
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from bench_ar_matrix import make_suites
from bench_diff_yesno import aggregate, banking_coverage, evaluation_data, synchronize, validate_probs
from bench_local import np, softmax_t, temp_for, to_internal, torch
from bench_matrix import load_laya
from laya.common import QTYPES, build_sequence, collate_items, render_options, serialize_state, temp_bucket
from shared_yesno_supervised import prediction_record, summarize_predictions


CONDITIONS = {
    "aligned_unit": {"input_mode": "aligned", "temperature_view": "unit", "primary": True},
    "aligned_sdk_temperature": {"input_mode": "aligned", "temperature_view": "sdk", "primary": False},
    "sdk_unit": {"input_mode": "sdk_default", "temperature_view": "unit", "primary": False},
    "sdk_native": {"input_mode": "sdk_default", "temperature_view": "sdk", "primary": False},
}


def _target(row, qid, q):
    # This baseline's supervised loader validates targets inline, not via _target.
    primitive, criteria = q["type"], q.get("criteria")
    if primitive == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice requires ordered criteria dictionary")
        keys = list(criteria)
    elif primitive == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score requires ordered criteria list")
        keys = [str(i) for i in range(len(criteria))]
    elif primitive == "noul":
        keys = ["false", "true"]
    else:
        raise ValueError("unsupported primitive")
    if set(row["gold"]) != {qid} or set(row["option_ids"]) != {qid}:
        raise ValueError("gold/option_ids must match question")
    option_ids = row["option_ids"][qid]
    if (not isinstance(option_ids, list) or len(option_ids) != len(keys)
            or any(not isinstance(x, str) or not x for x in option_ids)
            or len(set(option_ids)) != len(keys)):
        raise ValueError("invalid option identities")
    gold = row["gold"][qid]
    probabilities = gold["probabilities"]
    if not isinstance(probabilities, dict) or set(probabilities) != set(keys):
        raise ValueError("gold must be exhaustive in natural option order")
    target = [probabilities[k] for k in keys]
    if (any(isinstance(x, bool) or not isinstance(x, (int, float))
            or not math.isfinite(x) or x < 0 for x in target)
            or abs(math.fsum(target) - 1.0) > 1e-6):
        raise ValueError("invalid target probabilities")
    if gold["kind"] not in ("hard", "known_distribution"):
        raise ValueError("unknown target kind")
    if gold["kind"] == "hard" and (target.count(1) != 1 or any(x not in (0, 1) for x in target)):
        raise ValueError("hard target must be one-hot")
    return target


def internal_rows(path):
    rows, identities = [], set()
    with open(path, encoding="utf-8") as stream:
        for ci, line in enumerate(stream):
            row = json.loads(line)
            if row["schema_version"] != "shared_yesno_data_v1" or row["split"] != "test":
                raise ValueError("internal data must be canonical test")
            for key in ("case_id", "group_id", "view_id"):
                if not isinstance(row[key], str) or not row[key]:
                    raise ValueError("invalid " + key)
            identity = (row["case_id"], row["view_id"])
            if identity in identities:
                raise ValueError("duplicate internal case/view")
            identities.add(identity)
            if not isinstance(row["source"]["name"], str) or not row["source"]["name"]:
                raise ValueError("invalid source.name")
            if not isinstance(row["questions"], dict) or len(row["questions"]) != 1:
                raise ValueError("expected one internal question")
            qid, original = next(iter(row["questions"].items()))
            qdef = {k: original[k] for k in ("type", "instructions", "criteria") if k in original}
            target = _target(row, qid, qdef)
            options = render_options(to_internal(qdef))
            if len(options) != len(target):
                raise ValueError("target/options mismatch")
            item = {k: row[k] for k in ("case_id", "group_id", "view_id", "split")}
            item.update(source=row["source"]["name"], primitive=qdef["type"],
                        target_kind=row["gold"][qid]["kind"], target=target)
            rows.append(dict(id="internal/%s/%s/%s" % (*identity, qid), suite="internal",
                             case_idx=ci, decision_idx=ci, qid=qid, state=row["state"],
                             qdef=qdef, options=options, internal_item=item,
                             gold=row["gold"][qid], option_ids=row["option_ids"][qid]))
    if not rows:
        raise ValueError("canonical internal test is empty")
    return rows


def external_rows(name, suite):
    indexed, case_idx, previous = [], -1, None
    for decision_idx, (state, questions, gold, extra) in enumerate(suite):
        if name != "typed_decisions" or decision_idx == 0 or state is not previous:
            case_idx += 1
        previous = state
        if len(questions) != 1:
            raise ValueError("expected one question per suite record")
        qid, qdef = next(iter(questions.items()))
        options = render_options(to_internal(qdef))
        if not 0 <= gold < len(options):
            raise ValueError("invalid gold index")
        soft, score = extra.get("soft"), extra.get("gold_score")
        if soft is not None:
            soft = np.asarray(soft, float)
            if soft.shape != (len(options),) or not np.isfinite(soft).all() or (soft < 0).any():
                raise ValueError("invalid soft target")
            soft = soft.tolist()
        if score is not None and not math.isfinite(score):
            raise ValueError("invalid gold score")
        indexed.append(dict(id="%s/%d/%s" % (name, case_idx, qid), suite=name,
                            case_idx=case_idx, decision_idx=decision_idx, qid=qid,
                            state=state, qdef=qdef, options=options, type=qdef["type"],
                            workflow=extra.get("workflow"), gold_idx=int(gold),
                            soft=soft, gold_score=score))
    return indexed


def sequence(agent, row, mode, context_limit):
    tok = agent.tok
    q: dict[str, Any] = to_internal(row["qdef"])
    if mode == "aligned" and q["t"] == "noul":
        q = dict(q, crit=None)
    options = render_options(q)
    texts = [str(q["ins"])] + options + [serialize_state(row["state"])]
    replacements = sum(text.count(tok.mask_token) for text in texts)
    clean = [text.replace(tok.mask_token, " ") for text in texts]
    encode = lambda text: tok(text, add_special_tokens=False, truncation=False)["input_ids"]
    head = encode("%s question: %s" % (q["t"], clean[0]))
    opts = [encode(" " + text) for text in clean[1:-1]]
    state = encode(clean[-1])
    full_head, full_opts, full_state = len(head), [len(x) for x in opts], len(state)
    if mode == "aligned":
        ids = [tok.cls_token_id] + head + [tok.sep_token_id]
        markers = []
        for option in opts:
            markers.append(len(ids))
            ids += [tok.mask_token_id] + option
        ids += [tok.sep_token_id] + state + [tok.sep_token_id]
        kept_head, kept_opts, kept_state = full_head, full_opts, full_state
    else:
        max_len, head_max_len = agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192)
        ids, markers = build_sequence(tok, row["state"], q, max_len, head_max_len)
        clipped = [[tok.mask_token_id] + option[:48] for option in opts]
        budget = head_max_len - sum(map(len, clipped))
        if budget < 16:
            per = max(4, (head_max_len - 16) // max(1, len(clipped)))
            clipped = [option[:per] for option in clipped]
            budget = head_max_len - sum(map(len, clipped))
        kept_head = min(full_head, max(8, budget), max(0, max_len - 1))
        kept_opts = [max(0, min(len(option) - 1, max_len - marker - 1))
                     for option, marker in zip(clipped, markers)]
        prefix_len = 3 + min(full_head, max(8, budget)) + sum(map(len, clipped))
        kept_state = min(full_state, max(0, max_len - prefix_len - 1))
    if (len(markers) != len(options) or ids.count(tok.mask_token_id) != len(options)
            or any(ids[m] != tok.mask_token_id for m in markers)):
        raise ValueError("marker/options mismatch: " + row["id"])
    if len(ids) > context_limit:
        raise ValueError("%s/%s: %d tokens exceeds %d; refusing truncation" %
                         (mode, row["id"], len(ids), context_limit))
    actual_opts = [option[:n] for option, n in zip(opts, kept_opts)]
    flags = dict(instruction_truncated=kept_head < full_head,
                 option_truncated=kept_opts != full_opts, state_truncated=kept_state < full_state)
    flags["head_truncated"] = flags["instruction_truncated"] or flags["option_truncated"]
    metadata = dict(token_length=len(ids), raw_instruction_tokens=full_head,
                    raw_option_tokens=full_opts, raw_state_tokens=full_state,
                    kept_instruction_tokens=kept_head, kept_option_tokens=kept_opts,
                    kept_state_tokens=kept_state, **flags,
                    literal_mask_replacements=replacements,
                    candidate_prefix_collisions=len(actual_opts) - len({tuple(x) for x in actual_opts}),
                    input_options=options,
                    input_option_texts=[tok.decode(x) for x in actual_opts],
                    noul_extra_rubric=mode == "sdk_default" and q["t"] == "noul"
                    and options != render_options(dict(q, crit=None)))
    return dict(ids=ids, markers=markers, qtype=QTYPES[q["t"]]), metadata


def forwards(agent, items, max_tokens, max_seqs):
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    output: list[Any] = [None] * len(items)
    i, batches, seconds = 0, 0, 0.0
    with torch.inference_mode():
        while i < len(order):
            j, length = i, 0
            while j < len(order) and j - i < max_seqs:
                length = max(length, len(items[order[j]]["ids"]))
                if length * (j - i + 1) > max_tokens:
                    break
                j += 1
            if j == i:
                raise ValueError("single sequence exceeds max-tokens; increase runtime budget")
            selected = [items[k] for k in order[i:j]]
            batch = collate_items([selected], agent.tok.pad_token_id)
            if batch is None:
                raise ValueError("empty native batch")
            tensors = [batch[k].to(agent.device) for k in
                       ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
            synchronize(agent.device)
            start = time.perf_counter()
            logits, act_logits = agent.model(*tensors)
            synchronize(agent.device)
            seconds += time.perf_counter() - start
            if not torch.isfinite(logits).all() or not torch.isfinite(act_logits).all():
                raise ValueError("nonfinite native forward; retain attempt and rerun with max-seqs=1")
            for r, k in enumerate(order[i:j]):
                output[k] = logits[r, :len(items[k]["markers"])].float().cpu().numpy().copy()
            i, batches = j, batches + 1
    if any(z is None for z in output):
        raise ValueError("missing forward rows")
    return output, dict(forwards=batches, seconds=seconds, total_tokens=sum(len(x["ids"]) for x in items),
                        timing_note="Synchronized native model forward; excludes tokenization, transfer and IO; shared by both T views")


def input_summary(rows, metadata):
    fields = ("head_truncated", "instruction_truncated", "option_truncated", "state_truncated",
              "noul_extra_rubric")
    return dict(actual_max_length=max(m["token_length"] for m in metadata),
                decisions=len(metadata),
                decisions_with_flags={k: sum(bool(m[k]) for m in metadata) for k in fields},
                cases_with_flags={k: len({r["case_idx"] for r, m in zip(rows, metadata) if m[k]})
                                 for k in fields},
                literal_mask_replacements=sum(m["literal_mask_replacements"] for m in metadata),
                decisions_with_candidate_prefix_collisions=sum(m["candidate_prefix_collisions"] > 0 for m in metadata))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    paths = ("model-path", "internal-data", "typed-data", "prompt-data", "sst5-data",
             "banking-test-csv", "output", "predictions")
    for key in paths:
        ap.add_argument("--" + key, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-seqs", type=int, default=16)
    ap.add_argument("--limit-per-suite", type=int, default=0)
    ap.add_argument("--ag-news-test-limit", type=int, default=400, help="Test prefix size; 0 = all source test rows")
    ap.add_argument("--emotion-test-limit", type=int, default=400, help="Test prefix size; 0 = all source test rows")
    ap.add_argument("--sst5-test-limit", type=int, default=600, help="Test prefix size; 0 = all source test rows")
    ap.add_argument("--ag-news-test-jsonl", default=None, help="Optional frozen test JSONL with text/label fields")
    ap.add_argument("--emotion-test-jsonl", default=None, help="Optional frozen test JSONL with text/label fields")
    ap.add_argument("--head-math-sdpa", action="store_true",
                    help="Use math-only Torch SDPA for the native head if remote numerics require it")
    a = ap.parse_args()
    if min(a.max_length, a.max_tokens, a.max_seqs) < 1 or a.limit_per_suite < 0:
        ap.error("positive budgets and nonnegative limit required")
    if min(a.ag_news_test_limit, a.emotion_test_limit, a.sst5_test_limit) < 0:
        ap.error("test limits must be nonnegative")
    for key in paths:
        key = key.replace("-", "_")
        setattr(a, key, os.path.abspath(os.path.expanduser(getattr(a, key))))
    for key in ("ag_news_test_jsonl", "emotion_test_jsonl"):
        if getattr(a, key) is not None:
            setattr(a, key, os.path.abspath(os.path.expanduser(getattr(a, key))))
    if a.output == a.predictions:
        ap.error("output and predictions must differ")
    for path in (a.output, a.predictions):
        if os.path.exists(path) or not os.path.isdir(os.path.dirname(path)):
            ap.error("output parent must exist and file must not exist: " + path)
    if not os.path.isdir(a.model_path):
        ap.error("model-path must be an existing local checkpoint on the execution host")
    payload: dict[str, Any] = dict(status="running", settings=vars(a), conditions={}, suite_sizes={}, input_metadata={},
                   input_examples={}, condition_definitions=CONDITIONS)
    # Reserve both artifacts before loading/forward; failures leave a readable attempt.
    with open(a.output, "x", encoding="utf-8") as out, open(a.predictions, "x", encoding="utf-8") as pred:
        try:
            suites = make_suites(a.typed_data, a.prompt_data, a.sst5_data,
                                 banking_test_limit=0, banking_test_csv=a.banking_test_csv,
                                 ag_news_test_limit=a.ag_news_test_limit, emotion_test_limit=a.emotion_test_limit,
                                 sst5_test_limit=a.sst5_test_limit, ag_news_test_jsonl=a.ag_news_test_jsonl,
                                 emotion_test_jsonl=a.emotion_test_jsonl)
            payload["evaluation_data"] = evaluation_data(suites, a, 0)
            coverage = banking_coverage(suites["banking77"])
            payload["banking77_coverage"] = coverage
            datasets = {name: external_rows(name, suite) for name, suite in suites.items()}
            datasets["internal"] = internal_rows(a.internal_data)
            payload["evaluation_data"]["internal"] = dict(
                source_path=a.internal_data, source_type="jsonl", split="test", requested_test_limit=0,
                selection_policy="all_source_test_rows_original_order",
                evaluation_decision_limit=a.limit_per_suite)
            for selection, selected in (
                    ("constructed", datasets["internal"]),
                    ("evaluated", datasets["internal"][:a.limit_per_suite] if a.limit_per_suite
                     else datasets["internal"])):
                payload["evaluation_data"]["internal"][selection] = dict(
                    actual_cases=len({r["case_idx"] for r in selected}), actual_decisions=len(selected),
                    candidate_counts_by_question_type={
                        qt: sorted({len(r["options"]) for r in selected if r["qdef"]["type"] == qt})
                        for qt in sorted({r["qdef"]["type"] for r in selected})})
            agent = load_laya(a.model_path, a.device)
            if agent.device.type != torch.device(a.device).type:
                raise ValueError("loader device fallback is not permitted")
            agent.model.float().eval()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if a.head_math_sdpa:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
                if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                    torch.backends.cuda.enable_cudnn_sdp(False)
            encoder_cfg = agent.model.encoder.config
            encoder_limit = getattr(encoder_cfg, "max_position_embeddings", None)
            if not isinstance(encoder_limit, int) or encoder_limit < 1:
                raise ValueError("encoder must declare max_position_embeddings")
            tok_limit = getattr(agent.tok, "model_max_length", None)
            context_limit = min(a.max_length, encoder_limit)
            payload["runtime"] = dict(model_path=a.model_path, device=str(agent.device),
                dtype=str(next(agent.model.parameters()).dtype), amp=False, tf32=False,
                encoder_attention="eager", reference_compile=getattr(encoder_cfg, "reference_compile", None),
                encoder_context_limit=encoder_limit, context_limit=context_limit,
                tokenizer_length_hint=tok_limit,
                training_context_note="Checkpoint configuration below is verbatim; absent training-context fields are unknown.",
                checkpoint_config=agent.cfg, max_len=agent.cfg.get("max_len", 512),
                head_max_len=agent.cfg.get("head_max_len", 192),
                temperature_raw=list(map(float, agent.temperature_raw)),
                temperature=list(map(float, agent.temperature)),
                temperature_by_options_raw={k: float(v) for k, v in agent.temperature_by_options_raw.items()},
                temperature_by_options={k: float(v) for k, v in agent.temperature_by_options.items()},
                training_changed=False, calibration_fitted=False)
            payload["protocol_notes"] = [
                "aligned_unit is primary; all four conditions are predeclared, not selected on test.",
                "Aligned uses full native format, generic noul descriptions, two false/true markers; DLM uses one Yes/No mask.",
                "SDK input uses original build_sequence and criteria, including noul rubrics and checkpoint truncation budgets.",
                "T=1 is an unscaled view, not a claim of comparable calibration; SDK temperatures retain loader clamps.",
                "FP32 batched Laya timing is not architecture-matched throughput against BF16 batch-one DLM.",
                "Checkpoint max_len/head_max_len are recorded, not endorsed SDK settings or proof of training context.",
                "DLM S0 10k and Laya pretraining differ; historical typedFT is a separate target-train reference; AG News is in Laya training mix.",
                "Smoke prefixes are execution checks, not statistical validation; no dropped decisions."]
            for condition in CONDITIONS:
                payload["conditions"][condition] = {"suites": {}}
            for name, all_rows in datasets.items():
                rows = all_rows[:a.limit_per_suite] if a.limit_per_suite else all_rows
                payload["suite_sizes"][name] = dict(cases=len({r["case_idx"] for r in all_rows}),
                    decisions=len(all_rows), evaluated_decisions=len(rows),
                    evaluated_cases=len({r["case_idx"] for r in rows}))
                payload["input_metadata"][name] = {}
                for mode in ("aligned", "sdk_default"):
                    built = [sequence(agent, row, mode, context_limit) for row in rows]
                    items, metadata = zip(*built)
                    zs, timing = forwards(agent, items, a.max_tokens, a.max_seqs)
                    payload["input_metadata"][name][mode] = dict(input_summary(rows, metadata), **timing)
                    for row, item in zip(rows, items):
                        key = mode + "/" + row["qdef"]["type"]
                        payload["input_examples"].setdefault(key, dict(id=row["id"], input_ids=item["ids"],
                            mask_positions=item["markers"], decoded=agent.tok.decode(item["ids"])))
                    for condition, definition in CONDITIONS.items():
                        if definition["input_mode"] != mode:
                            continue
                        records = []
                        for row, z, meta in zip(rows, zs, metadata):
                            primitive, k = row["qdef"]["type"], len(row["options"])
                            qt = QTYPES[primitive]
                            bucket = temp_bucket(qt, k)
                            preset = temp_for(agent, qt, k)
                            temperature = 1.0 if definition["temperature_view"] == "unit" else preset
                            if not math.isfinite(temperature) or temperature <= 0 or z.shape != (k,):
                                raise ValueError("invalid temperature/logit shape")
                            record = {key: value for key, value in row.items()
                                      if key not in ("state", "qdef", "internal_item")}
                            if name == "internal":
                                record.update(prediction_record(row["internal_item"],
                                    torch.tensor(z).float().unsqueeze(0) / temperature))
                                p = np.asarray(record["probabilities"])
                            else:
                                p = softmax_t(z, temperature)
                            validate_probs(p)
                            record.update(condition=condition, input_mode=mode, native_logits=z.tolist(),
                                probs=p.tolist(), prediction_index=int(np.argmax(p)), temperature=temperature,
                                temperature_bucket=bucket, sdk_temperature=preset,
                                sdk_temperature_raw=float(agent.temperature_by_options_raw.get(
                                    bucket, agent.temperature_raw[qt])), **meta)
                            pred.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                            records.append(record)
                        pred.flush()
                        if len(records) != len(rows):
                            raise ValueError("missing prediction rows")
                        if name == "internal":
                            result = summarize_predictions(records)
                        else:
                            result: Any = aggregate(records)
                            for field, key in (("workflow", "by_workflow"), ("type", "by_question_type")):
                                result[key] = {v: aggregate([r for r in records if r[field] == v])
                                    for v in sorted({r[field] for r in records if r[field] is not None})}
                        result.update(n_decisions=len(records), n_cases=len({r["case_idx"] for r in records}), dropped=0)
                        payload["conditions"][condition]["suites"][name] = result
                    print("[%s/%s] %d decisions complete" % (name, mode, len(rows)), flush=True)
            payload["status"] = "complete"
        except Exception as exc:
            payload.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
            raise
        finally:
            json.dump(payload, out, indent=2, ensure_ascii=False, allow_nan=False)
            out.write("\n")


if __name__ == "__main__":
    main()
