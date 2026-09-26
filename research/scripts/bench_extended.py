"""Evaluate frozen full_eval_v2 JSONL decisions with inherited aligned T=1 scoring."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from _benchmark_bootstrap import bootstrap

bootstrap()

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from bench_diff_yesno import decision_input, forward_masks, synchronize, token_mapping, validate_probs
from bench_laya_aligned import forwards, sequence
from bench_local import metrics, np, softmax_t, to_internal, torch
from bench_matrix import load_laya
from laya.common import render_options, serialize_state
from transformers import AutoModelForMaskedLM, AutoTokenizer


METRIC_DEFINITIONS = {
    "hard": "Inherited metrics on successful hard-target rows only; n is the probability denominator. "
            "ECE: 15 equal-width confidence bins (lo, hi]. Brier: sum over classes, not mean. "
            "NLL: -log(max(p_gold, 1e-12)). Argmax ties select first option. "
            "Index-based macro F1 omitted because label identities can vary by decision.",
    "all_attempted_accuracy": "Correct hard-target decisions / all attempted hard-target decisions; "
                              "unsupported counts incorrect; null-gold decisions are not hard truth.",
    "soft": "CE=-sum(target*log(max(p,1e-12))); Brier=sum((p-target)^2); no target renormalization.",
    "binary_ptrue_mse": "Noul only: (p[1]-target_true)^2; soft target preferred, else hard truth. "
                        "Separate from multiclass sum Brier.",
    "score": "Levels are zero-based option indexes. MAE compares E_p[level] with gold_score, "
             "else soft expected level, else gold_idx. RPS is sum of squared CDF differences "
             "over K-1 boundaries, unnormalized; soft preferred, else hard one-hot. "
             "A scalar-only score does not define an RPS target.",
    "group_complete_accuracy": "All decisions correct in a fully selected group with hard truth "
                               "on every row. Unsupported rows fail the group. Groups partially "
                               "selected by smoke or report partition are excluded, not completed.",
    "timing": "Inherited synchronized forward seconds, excluding tokenization, transfers and IO; batch=1.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, "duplicate JSON key: " + key)
        out[key] = value
    return out


def reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def finite_float(value):
    result = float(value)
    require(math.isfinite(result), "nonfinite JSON number")
    return result


def read_json(value):
    return json.loads(value, object_pairs_hook=unique_object, parse_constant=reject_constant,
                      parse_float=finite_float)


def bundled(root, value):
    require(text(value), "bundle path must be nonempty")
    relative = Path(value)
    require(not relative.is_absolute() and ".." not in relative.parts, "bundle path must be relative")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "bundle path escapes profile directory")
    require(path.is_file(), "missing bundle file: " + str(path))
    return path


def validate_row(row, identities):
    keys = {"id", "group_id", "state", "qdef", "gold_idx", "soft", "gold_score", "option_ids", "metadata"}
    require(isinstance(row, dict) and set(row) == keys, "row keys must match v2 schema")
    require(text(row["id"]) and text(row["group_id"]), "invalid id/group_id")
    require(row["id"] not in identities, "duplicate decision id: " + row["id"])
    identities.add(row["id"])
    require(isinstance(row["state"], (str, dict, list)), "state must be string/object/list")
    q = row["qdef"]
    require(isinstance(q, dict) and set(q) == {"type", "instructions", "criteria"}, "invalid qdef keys")
    require(q["type"] in ("choice", "score", "noul"), "unsupported question type")
    require(q["instructions"] is not None, "instructions must not be null")
    crit = q["criteria"]
    if q["type"] == "choice":
        require(isinstance(crit, (dict, list)) and len(crit) >= 2, "choice needs >=2 options")
        require(all(text(k) for k in crit) and len(set(crit)) == len(crit), "invalid choice keys")
    elif q["type"] == "score":
        require(isinstance(crit, list) and len(crit) >= 2, "score needs ordered >=2 levels")
    else:
        require(crit is None or isinstance(crit, dict) and set(crit) <= {"false", "true"},
                "noul criteria must be null or false/true object")
    options = render_options(to_internal(q))
    k = len(options)
    ids = row["option_ids"]
    require(isinstance(ids, list) and len(ids) == k and all(text(x) for x in ids)
            and len(set(ids)) == k, "option_ids must uniquely identify rendered options in order")
    gold, soft, score = row["gold_idx"], row["soft"], row["gold_score"]
    require(gold is None or type(gold) is int and 0 <= gold < k, "gold_idx out of range")
    if soft is not None:
        require(isinstance(soft, list) and len(soft) == k
                and all(number(x) and 0 <= x <= 1 for x in soft)
                and abs(math.fsum(soft) - 1) <= 1e-6, "invalid soft target")
    if score is not None:
        require(q["type"] == "score" and number(score) and 0 <= score <= k - 1,
                "gold_score must be a zero-based expected level in range")
    require(gold is not None or soft is not None or score is not None, "decision needs a target")
    meta = row["metadata"]
    require(isinstance(meta, dict) and "source" in meta and "adaptation" in meta,
            "metadata requires source and adaptation semantics")
    require(all(text(meta[k]) or isinstance(meta[k], dict) and bool(meta[k])
                for k in ("source", "adaptation")), "source/adaptation must be nonempty text or objects")
    require("task_id" not in meta or text(meta["task_id"]), "invalid metadata.task_id")
    if soft is not None:
        require(meta.get("soft_target_kind") in ("human-votes", "exact-mechanism"),
                "soft targets require metadata.soft_target_kind")
    return dict(row, options=options, primitive=q["type"])


def load_profile(path, selected):
    profile = read_json(path.read_text(encoding="utf-8"))
    require(isinstance(profile, dict) and profile.get("schema_version") == "full_eval_v2",
            "expected full_eval_v2 profile")
    require(text(profile.get("profile_id")), "invalid profile_id")
    require(isinstance(profile.get("suites"), list) and profile["suites"], "empty suites")
    datasets, names, identities = [], set(), set()
    for spec in profile["suites"]:
        require(isinstance(spec, dict) and {"name", "path", "category", "expected_count", "source", "protocol"}
                <= set(spec), "invalid suite specification")
        name = spec["name"]
        require(text(name) and all(c.isascii() and (c.isalnum() or c in "_-") for c in name)
                and name not in names, "suite names must be unique safe filename stems")
        names.add(name)
        require(text(spec["category"]) and isinstance(spec["source"], dict)
                and isinstance(spec["protocol"], dict), "invalid suite metadata")
        require(type(spec["expected_count"]) is int and spec["expected_count"] > 0, "invalid expected_count")
        source = bundled(path.parent, spec["path"])
        rows = []
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                rows.append(validate_row(read_json(line), identities))
        require(len(rows) == spec["expected_count"], "source count mismatch: " + name)
        if not selected or name in selected:
            datasets.append((spec, rows))
    require(not selected or len(set(selected)) == len(selected) and set(selected) <= names,
            "unknown or repeated --suites")
    return profile, datasets


def summarize(records, full_groups):
    completed = [r for r in records if r["status"] == "ok"]
    hard = [r for r in completed if r["gold_idx"] is not None]
    attempted_hard = [r for r in records if r["gold_idx"] is not None]
    correct = lambda r: r["status"] == "ok" and r["pred_idx"] == r["gold_idx"]
    hard_metrics = metrics([(r["gold_idx"], r["probs"]) for r in hard])
    hard_metrics.pop("macro_f1", None)
    values = defaultdict(list)
    for r in completed:
        p = np.asarray(r["probs"], float)
        target = np.asarray(r["soft"], float) if r["soft"] is not None else None
        if target is not None:
            values["soft_ce"].append(float(-np.dot(target, np.log(np.maximum(p, 1e-12)))))
            values["soft_brier"].append(float(np.square(p - target).sum()))
        elif r["gold_idx"] is not None:
            target = np.eye(len(p))[r["gold_idx"]]
        if r["primitive"] == "noul" and target is not None:
            values["binary_ptrue_mse"].append(float((p[1] - target[1]) ** 2))
        if r["primitive"] == "score":
            level = r["gold_score"]
            if level is None and target is not None:
                level = float(np.dot(np.arange(len(p)), target))
            if level is not None:
                values["score_mae"].append(abs(float(np.dot(np.arange(len(p)), p)) - level))
            if target is not None:
                values["score_rps"].append(float(np.square(np.cumsum(p - target)[:-1]).sum()))
    groups = defaultdict(list)
    for r in records:
        groups[r["group_id"]].append(r)
    eligible = [rs for group, rs in groups.items()
                if {r["id"] for r in rs} == full_groups[group]
                and all(r["gold_idx"] is not None for r in rs)]
    out: dict[str, Any] = dict(attempted=len(records), completed=len(completed),
               coverage=len(completed) / len(records) if records else None,
               status_counts=dict(Counter(r["status"] for r in records)), hard=hard_metrics,
               attempted_hard=len(attempted_hard),
               all_attempted_accuracy=sum(map(correct, attempted_hard)) / len(attempted_hard)
               if attempted_hard else None,
               group_complete_accuracy=sum(all(map(correct, rs)) for rs in eligible) / len(eligible)
               if eligible else None, n_groups_eligible=len(eligible),
               n_groups_excluded=len(groups) - len(eligible),
               forward_seconds=sum(r["forward_seconds"] for r in records))
    for key in ("soft_ce", "soft_brier", "binary_ptrue_mse", "score_mae", "score_rps"):
        out[key] = {"value": float(np.mean(values[key])) if values[key] else None, "n": len(values[key])}
    return out


def forward_masks_batch(base, sequences, positions, pad_id, device, weight, bias, batched):
    # Same padding and explicit SDPA key-mask axes as bench_diff_yesno.main.
    length = max(map(len, sequences))
    inputs = torch.tensor([ids + [pad_id] * (length - len(ids)) for ids in sequences],
                          dtype=torch.long, device=device)
    attention = torch.tensor([[True] * len(ids) + [False] * (length - len(ids)) for ids in sequences],
                             dtype=torch.bool, device=device)
    if batched:
        attention = attention[:, None, None, :]
    mask_rows = torch.tensor([i for i, ps in enumerate(positions) for _ in ps], dtype=torch.long, device=device)
    mask_positions = torch.tensor([p for ps in positions for p in ps], dtype=torch.long, device=device)
    logits, seconds = forward_masks(base, inputs, attention, mask_rows, mask_positions, weight, bias)
    logits = logits.cpu().numpy().astype(float)
    offsets = np.cumsum([0] + [len(ps) for ps in positions])
    return [logits[start:end] for start, end in zip(offsets[:-1], offsets[1:])], seconds


class Evaluator:
    def __init__(self, args):
        self.backend = args.backend
        self.batch_size = args.batch_size
        self.runtime: dict[str, Any]
        self.device = torch.device(args.device)
        if args.backend == "dllm":
            self.tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
            self.base = AutoModelForMaskedLM.from_pretrained(
                args.model_path, trust_remote_code=True, local_files_only=True, torch_dtype=getattr(torch, args.dtype))
            self.base.to(self.device).eval()
            mapping, self.mask_id, self.mask_text = token_mapping(self.tok, self.base)
            head = self.base.lm_head
            candidates = torch.tensor([mapping["Yes"], mapping["No"]], device=head.weight.device)
            with torch.inference_mode():
                self.weight = head.weight.index_select(0, candidates)
                self.bias = head.bias.index_select(0, candidates) if getattr(head, "bias", None) is not None else None
            limits = [args.max_length]
            for limit in (getattr(self.base.config, "max_position_embeddings", None), self.tok.model_max_length):
                if type(limit) is int and 0 < limit < 10**9:
                    limits.append(limit)
            self.limit = min(limits)
            self.runtime = dict(dtype=str(self.weight.dtype), answer_token_ids=mapping,
                                logit_columns=["Yes", "No"], mask_id=self.mask_id)
        else:
            self.agent = load_laya(args.model_path, args.device)
            require(self.agent.device.type == self.device.type and
                    (self.device.index is None or self.agent.device.index == self.device.index),
                    "loader device fallback is not permitted")
            self.agent.model.float().eval()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                torch.backends.cuda.enable_cudnn_sdp(False)
            self.tok = self.agent.tok
            self.mask_text = self.tok.mask_token
            limit = getattr(self.agent.model.encoder.config, "max_position_embeddings", None)
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
                raise ValueError("encoder must declare context limit")
            self.limit = min(args.max_length, limit)
            self.runtime = dict(dtype="float32", encoder_attention="eager", head_sdpa="math", tf32=False)
        self.runtime.update(context_limit=self.limit, device=str(self.device), temperature=1.0,
                            batch_size=args.batch_size, truncation=False, input_mode="aligned",
                            warmup_forwards=0, batching="suite-local contiguous windows; supported rows only",
                            native_head_batching="fully batched" if self.backend == "laya" else None)

    def prepare(self, row):
        record = {k: v for k, v in row.items() if k not in ("state", "qdef")}
        record.update(status="ok", length=None, forward_seconds=0.0, raw_logits=None,
                      probs=None, pred_idx=None)
        q = to_internal(row["qdef"])
        texts = [serialize_state(row["state"]), q["ins"]] + row["options"]
        if self.backend == "dllm" and any(self.mask_text in value for value in texts):
            record.update(status="unsupported_source_mask", error="Source contains literal mask token")
            return record, None
        # Build without clipping, then gate the forward at the effective model limit.
        ids: Any
        item: Any = None
        positions: Any = None
        if self.backend == "dllm":
            ids, positions, options, _ = decision_input(
                self.tok, row["state"], row["qdef"], self.mask_id, self.mask_text, sys.maxsize)
            record["input_options"] = options
        else:
            item, meta = sequence(self.agent, {"id": row["id"], "state": row["state"], "qdef": row["qdef"]},
                                  mode="aligned", context_limit=sys.maxsize)
            ids = item["ids"]
            record["input_metadata"] = meta
        record["length"] = len(ids)
        if len(ids) > self.limit:
            record.update(status="unsupported_length", error="Input exceeds context limit %d; no truncation" % self.limit)
            return record, None
        return record, dict(ids=ids, positions=positions, item=item)

    def predict_batch(self, rows, batch_id):
        prepared = [self.prepare(row) for row in rows]
        selected = [i for i, (_, item) in enumerate(prepared) if item is not None]
        if not selected:
            return [record for record, _ in prepared], None
        items: list[Any] = [prepared[i][1] for i in selected]
        size = len(items)
        with torch.inference_mode():
            if self.backend == "dllm":
                pad_id = self.tok.pad_token_id
                if self.batch_size > 1:
                    require(pad_id is not None, "batched DLLM requires a padding token")
                zs, elapsed = forward_masks_batch(
                    self.base, [x["ids"] for x in items], [x["positions"] for x in items],
                    pad_id, self.device, self.weight, self.bias, self.batch_size > 1)
            else:
                zs, timing = forwards(self.agent, [x["item"] for x in items],
                                     max_tokens=self.limit * size, max_seqs=size)
                require(timing["forwards"] == 1, "native batch unexpectedly split")
                elapsed = float(timing["seconds"])
        for i, item, z in zip(selected, items, zs):
            row, record = rows[i], prepared[i][0]
            if self.backend == "dllm":
                require(z.shape == (len(item["positions"]), 2) and np.isfinite(z).all(), "invalid Yes/No logits")
                logodds = z[:, 0] - z[:, 1]
                require(np.isfinite(logodds).all(), "nonfinite logodds")
                pairs = np.asarray([softmax_t(pair, 1.0) for pair in z])
                for pair in pairs:
                    validate_probs(pair)
                p = pairs[0, ::-1] if row["primitive"] == "noul" else softmax_t(logodds, 1.0)
                record.update(yes_no_probs=pairs.tolist(), logodds=logodds.tolist())
            else:
                require(z.shape == (len(row["options"]),) and np.isfinite(z).all(), "invalid native logits")
                p = softmax_t(z, 1.0)
            validate_probs(p)
            require(len(p) == len(row["options"]), "probability/options mismatch")
            record.update(raw_logits=z.tolist(), probs=p.tolist(), pred_idx=int(np.argmax(p)),
                          forward_seconds=elapsed / size)
            if self.batch_size > 1:
                record.update(batch_id=batch_id, actual_batch_size=size)
        timing = dict(batch_id=batch_id, batch_size=size, forward_seconds=elapsed,
                      nonpad_tokens=sum(len(x["ids"]) for x in items),
                      padded_tokens=size * max(len(x["ids"]) for x in items))
        return [record for record, _ in prepared], timing


def save(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("dllm", "laya"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--smoke-limit", type=int, default=3)
    parser.add_argument("--smoke", action="store_true", help="First smoke-limit decisions per suite; NOT a full evaluation")
    parser.add_argument("--suites", nargs="+")
    parser.add_argument("--include-legacy", action="store_true")
    args = parser.parse_args()
    require(args.max_length > 0, "max-length must be positive")
    require(args.batch_size > 0 and args.smoke_limit > 0, "batch-size and smoke-limit must be positive")
    require(not args.include_legacy or args.batch_size == 1,
            "legacy runner has no runtime batch-size support; omit --include-legacy for batching")
    args.profile = args.profile.expanduser().resolve()
    args.model_path = str(Path(args.model_path).expanduser().resolve())
    require(Path(args.model_path).is_dir(), "model-path must be a local checkpoint directory")
    profile, datasets = load_profile(args.profile, args.suites)
    legacy = bundled(args.profile.parent, profile.get("legacy_profile")) if args.include_legacy else None
    output = args.output_dir.expanduser().absolute()
    output.mkdir(exist_ok=False)
    run: dict[str, Any] = dict(schema_version="full_eval_v2", profile_id=profile["profile_id"], profile_path=str(args.profile),
               backend=args.backend, model_path=args.model_path, smoke=args.smoke, smoke_limit=args.smoke_limit, status="running",
               full_status="not_full_smoke" if args.smoke else "running", max_length=args.max_length,
               metric_definitions=METRIC_DEFINITIONS, suites={},
               requested_sources={spec["name"]: dict(expected_count=spec["expected_count"], source_count=len(rows),
                    planned_count=min(args.smoke_limit, len(rows)) if args.smoke else len(rows), attempted=0)
                    for spec, rows in datasets})
    if args.batch_size > 1:
        run["metric_definitions"] = dict(METRIC_DEFINITIONS, timing=
            "Synchronized forward seconds, excluding tokenization, transfers and IO; "
            "row seconds are batch seconds / actual decisions, not request latency. No warmup.")
    save(output / "run.json", run)
    # A fatal failure deliberately leaves status=running and streamed evidence, never a quality completion.
    evaluator = Evaluator(args)
    run["runtime"] = evaluator.runtime
    save(output / "run.json", run)
    for spec, all_rows in datasets:
        name = spec["name"]
        rows = all_rows[:args.smoke_limit] if args.smoke else all_rows
        full_groups = defaultdict(set)
        for row in all_rows:
            full_groups[row["group_id"]].add(row["id"])
        records = []
        batches = []
        if evaluator.device.type == "cuda":
            synchronize(evaluator.device)
            torch.cuda.reset_peak_memory_stats(evaluator.device)
        print("Starting %s: %d decisions" % (name, len(rows)), flush=True)
        with (output / (name + "_predictions.jsonl")).open("x", encoding="utf-8") as stream, \
                (output / (name + "_batches.jsonl")).open("x", encoding="utf-8") as batch_stream:
            for batch_id, offset in enumerate(range(0, len(rows), args.batch_size)):
                batch_records, timing = evaluator.predict_batch(rows[offset:offset + args.batch_size], batch_id)
                for record in batch_records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                records.extend(batch_records)
                if timing is not None:
                    batches.append(timing)
                    batch_stream.write(json.dumps(timing, allow_nan=False) + "\n")
                    batch_stream.flush()
                if len(records) % 100 == 0 or len(records) == len(rows):
                    print("%s: %d/%d attempted" % (name, len(records), len(rows)), flush=True)
        report: dict[str, Any] = dict(suite=spec, source_count=len(all_rows), planned_count=len(rows),
                      metrics=summarize(records, full_groups), by_task={}, by_primitive={})
        seconds = math.fsum(b["forward_seconds"] for b in batches)
        decisions = sum(b["batch_size"] for b in batches)
        report["performance"] = dict(
            requested_batch_size=args.batch_size, forwards=len(batches),
            encoder_forwards=len(batches), native_head_forwards=len(batches) if args.backend == "laya" else 0,
            batch_sizes=[b["batch_size"] for b in batches], warmup_forwards=0,
            nonpad_tokens=sum(b["nonpad_tokens"] for b in batches),
            padded_tokens=sum(b["padded_tokens"] for b in batches),
            forward_seconds=seconds, ms_per_decision=1000 * seconds / decisions if decisions else None,
            batch_p50_ms=float(np.percentile([b["forward_seconds"] for b in batches], 50) * 1000) if batches else None,
            batch_p95_ms=float(np.percentile([b["forward_seconds"] for b in batches], 95) * 1000) if batches else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(evaluator.device) if evaluator.device.type == "cuda" else None,
            memory_scope="suite peak allocated CUDA bytes, includes resident weights; reset at suite start")
        for field, key in (("by_task", lambda r: r["metadata"].get("task_id", "__unspecified__")),
                           ("by_primitive", lambda r: r["primitive"])):
            partitions = defaultdict(list)
            for record in records:
                partitions[key(record)].append(record)
            report[field] = {k: summarize(rs, full_groups) for k, rs in partitions.items()}
        report["status"] = "complete" if all(r["status"] == "ok" for r in records) else "completed_with_unsupported"
        save(output / (name + "_report.json"), report)
        run["suites"][name] = report
        run["requested_sources"][name]["attempted"] = len(records)
        save(output / "run.json", run)
    del evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if legacy is not None:
        command = [sys.executable, str(Path(__file__).with_name("run_full_benchmark.py")),
                   "--profile", str(legacy), "--backend", args.backend, "--model-path", args.model_path,
                   "--output-dir", str(output / "legacy"), "--device", args.device, "--dtype", args.dtype]
        if args.smoke:
            command.append("--smoke")
        run["legacy"] = dict(profile=str(legacy), report="legacy/run.json", status="running")
        save(output / "run.json", run)
        subprocess.run(command, check=True)
        run["legacy"]["status"] = "complete"
    run["status"] = ("complete" if all(r["status"] == "complete" for r in run["suites"].values())
                     else "completed_with_unsupported")
    run["full_status"] = "not_full_smoke" if args.smoke else run["status"]
    save(output / "run.json", run)


if __name__ == "__main__":
    main()
