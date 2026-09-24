"""Canonical shared Yes/No supervision and offline, T=1 evaluation.

Smoke limits select whole groups, never a source-ordered prefix, and may return
fewer than the requested number of decisions. Full runs use limit=0.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
import random

import torch

from laya.common import QTYPES


def _spread_indices(items, limit):
    if not limit or limit >= len(items):
        return list(range(len(items)))
    groups = defaultdict(list)
    for i, item in enumerate(items):
        groups[item["group_id"]].append(i)
    rng = random.Random(42)
    buckets = defaultdict(list)
    for group, indices in groups.items():
        keys = set()
        for i in indices:
            item = items[i]
            target = item["target"]
            # Spread ordinal examples across low/middle/high expected levels.
            level = (min(2, int(3 * sum(j * p for j, p in enumerate(target))
                                    / (len(target) - 1)))
                     if item["primitive"] == "score" else -1)
            keys.add((item["source"], item["primitive"], item["target_kind"],
                      len(target), level))
        for key in sorted(keys):
            buckets[key].append(group)
    keys = sorted(buckets)
    rng.shuffle(keys)
    for queue in buckets.values():
        rng.shuffle(queue)
    selected, selected_groups = set(), set()

    def add(group):
        if group in selected_groups or len(selected) + len(groups[group]) > limit:
            return False
        selected_groups.add(group)
        selected.update(groups[group])
        return True

    # Guarantee these broad coverage axes when present; fail rather than silently
    # calling an impossibly small or oversized-group selection representative.
    requirements = [("primitive", p) for p in sorted({x["primitive"] for x in items})]
    requirements += [("target_kind", k) for k in sorted({x["target_kind"] for x in items})]
    if any(len(x["target"]) == 120 for x in items):
        requirements.insert(0, ("K", 120))
    for field, value in requirements:
        matches = lambda x: (len(x["target"]) if field == "K" else x[field]) == value
        if any(matches(items[i]) for i in selected):
            continue
        candidates = [g for g, indices in groups.items() if any(matches(items[i]) for i in indices)]
        rng.shuffle(candidates)
        candidates.sort(key=lambda g: len(groups[g]))
        if not any(add(g) for g in candidates):
            raise ValueError("limit cannot retain whole groups and coverage of %s=%s" % (field, value))
    while True:
        progress = False
        for key in keys:
            queue = buckets[key]
            while queue:
                if add(queue.pop()):
                    progress = True
                    break
        if not progress:
            break
    return sorted(selected)


def load_structured_items(path, tok, max_length, yesno_tokens, limit=0):
    """Read canonical one-question JSONL; preserve display order and lineage.

    Split tags must be consistent within the file, but dev/test are allowed.
    No target normalization, missing-gold defaults, truncation, or invalid-row
    dropping is performed, including on rows excluded by the smoke limit.
    """
    from bench_diff_yesno import decision_input

    if max_length < 1 or limit < 0:
        raise ValueError("max_length must be positive and limit nonnegative")
    mapping, mask_id, mask_text = yesno_tokens
    items, identities, splits = [], set(), set()
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if row.get("schema_version") != "shared_yesno_data_v1":
                    raise ValueError("expected canonical shared_yesno_data_v1")
                split = row["split"]
                if split not in ("train", "dev", "validation", "test"):
                    raise ValueError("unknown split")
                splits.add(split)
                if len(splits) != 1:
                    raise ValueError("mixed split tags in one file")
                for key in ("case_id", "group_id", "view_id"):
                    if not isinstance(row[key], str) or not row[key]:
                        raise ValueError("%s must be a nonempty string" % key)
                identity = (row["case_id"], row["view_id"])
                if identity in identities:
                    raise ValueError("duplicate case/view identity")
                identities.add(identity)
                source = row["source"]["name"]
                if not isinstance(source, str) or not source:
                    raise ValueError("source.name must be a nonempty string")
                questions = row["questions"]
                if not isinstance(questions, dict) or len(questions) != 1:
                    raise ValueError("expected exactly one question")
                qid, original = next(iter(questions.items()))
                primitive = original["type"]
                if primitive not in QTYPES:
                    raise ValueError("unsupported primitive")
                # Only model-visible fields reach the inherited renderer.
                q = {key: original[key] for key in ("type", "instructions", "criteria")
                     if key in original}
                criteria = q.get("criteria")
                if primitive == "choice":
                    if not isinstance(criteria, dict) or len(criteria) < 2:
                        raise ValueError("choice requires ordered criteria dictionary")
                    keys = list(criteria)
                elif primitive == "score":
                    if not isinstance(criteria, list) or len(criteria) < 2:
                        raise ValueError("score requires ordered criteria list")
                    keys = [str(i) for i in range(len(criteria))]
                else:
                    keys = ["false", "true"]
                if set(row["gold"]) != {qid} or set(row["option_ids"]) != {qid}:
                    raise ValueError("gold/option_ids must match the question")
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
                kind = gold["kind"]
                if kind not in ("hard", "known_distribution"):
                    raise ValueError("unknown target kind")
                if kind == "hard" and (target.count(1) != 1 or any(x not in (0, 1) for x in target)):
                    raise ValueError("hard target must be one-hot")
                ids, positions, options, prompt = decision_input(
                    tok, row["state"], q, mask_id, mask_text, max_length)
                if len(options) != len(target) or len(positions) != (1 if primitive == "noul" else len(target)):
                    raise ValueError("target/options/masks mismatch")
                items.append(dict(ids=ids, candidates=[mapping["Yes"], mapping["No"]],
                                  target=target, qtype=QTYPES[primitive], labels=options,
                                  options=options, prompt=prompt, positions=positions,
                                  noul=primitive == "noul", primitive=primitive, source=source,
                                  case_id=row["case_id"], group_id=row["group_id"],
                                  view_id=row["view_id"], target_kind=kind, split=split))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("%s:%d: %s" % (path, line_number, exc)) from exc
    selected = [items[i] for i in _spread_indices(items, limit)]
    print(json.dumps({"data": str(path), "rows": len(items), "selected_rows": len(selected),
                      "limit": limit, "split": next(iter(splits), None),
                      "source_counts": dict(Counter(x["source"] for x in selected)),
                      "primitive_counts": dict(Counter(x["primitive"] for x in selected)),
                      "target_kind_counts": dict(Counter(x["target_kind"] for x in selected)),
                      "K_counts": dict(Counter(len(x["target"]) for x in selected))}), flush=True)
    return selected


def supervised_terms(logits, target, primitive, rps_weight=0.0):
    """Scalar FP32 decision means; retains gradients through candidate logits."""
    if logits.ndim != 2 or target.ndim != 2 or logits.shape != target.shape or min(logits.shape) < 1:
        raise ValueError("logits and target must be matching nonempty [B,K] tensors")
    if primitive not in QTYPES or not math.isfinite(rps_weight) or rps_weight < 0:
        raise ValueError("invalid primitive or RPS weight")
    z, t = logits.float(), target.to(device=logits.device, dtype=torch.float32)
    if (not torch.isfinite(z).all() or not torch.isfinite(t).all() or (t < 0).any()
            or ((t.sum(-1) - 1).abs() > 1e-6).any()):
        raise ValueError("nonfinite logits or invalid targets")
    if primitive == "noul" and z.shape[-1] != 2:
        raise ValueError("noul logits must be [No, Yes]")
    if primitive == "score" and z.shape[-1] < 2:
        raise ValueError("score requires at least two levels")
    logp = torch.nn.functional.log_softmax(z, dim=-1)
    ce = -(t * logp).sum(-1).mean()
    entropy = -torch.xlogy(t, t).sum(-1).mean()
    rps = z.new_zeros(())
    if primitive == "score":
        rps = (logp.exp().cumsum(-1)[:, :-1] - t.cumsum(-1)[:, :-1]).square().mean()
    terms = dict(loss=ce + rps_weight * rps, ce=ce, kl=ce - entropy, rps=rps)
    if any(not torch.isfinite(value) for value in terms.values()):
        raise ValueError("nonfinite supervised terms")
    return terms


def prediction_record(item, logits):
    """One decision at T=1. Ordinal error uses expectations, not argmax levels."""
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("prediction_record requires [1,K] logits")
    z = logits.detach().float()
    t = torch.tensor([item["target"]], device=z.device, dtype=torch.float32)
    terms = supervised_terms(z, t, item["primitive"])
    p = z.softmax(-1)[0]
    probabilities = p.cpu().tolist()
    if not all(math.isfinite(x) for x in probabilities):
        raise ValueError("nonfinite prediction")
    record = {key: item[key] for key in
              ("case_id", "view_id", "group_id", "source", "primitive", "target_kind")}
    record.update(target=list(item["target"]), probabilities=probabilities, K=len(probabilities),
                  ce=terms["ce"].item(), kl=terms["kl"].item(),
                  brier=(p - t[0]).square().sum().item(), prediction_index=int(p.argmax()),
                  rps=terms["rps"].item() if item["primitive"] == "score" else None)
    hard = item["target_kind"] == "hard"
    if hard:
        record.update(hard_correct=int(p.argmax()) == int(t[0].argmax()),
                      confidence=p.max().item(), hard_target_index=int(t[0].argmax()))
    if item["primitive"] == "score":
        expected_p = math.fsum(i * x for i, x in enumerate(probabilities))
        expected_t = math.fsum(i * x for i, x in enumerate(item["target"]))
        error = abs(expected_p - expected_t)
        record.update(expected_score=expected_p, target_expected_score=expected_t,
                      mean_abs_error_of_means=error)
        if hard:
            record.update(mae_hard_score=error, within_one_hard_score=error <= 1.0)
        else:
            record["soft_score_abs_error_of_means"] = error
    return record


def _metrics(records):
    def mean(key, rows=records):
        values = [r[key] for r in rows if r.get(key) is not None]
        return math.fsum(values) / len(values) if values else None

    hard = [r for r in records if r["target_kind"] == "hard"]
    noul = [r for r in hard if r["primitive"] == "noul"]
    tp = sum(r["prediction_index"] == 1 and r["hard_target_index"] == 1 for r in noul)
    fp = sum(r["prediction_index"] == 1 and r["hard_target_index"] == 0 for r in noul)
    fn = sum(r["prediction_index"] == 0 and r["hard_target_index"] == 1 for r in noul)
    bins = defaultdict(list)
    for r in hard:
        bins[min(14, int(r["confidence"] * 15))].append(r)
    ece = (math.fsum(len(rows) * abs(mean("confidence", rows) - mean("hard_correct", rows))
                     for rows in bins.values()) / len(hard)) if hard else None
    result = {key: mean(key) for key in
              ("ce", "kl", "brier", "rps", "mean_abs_error_of_means", "mae_hard_score", "within_one_hard_score")}
    result.update(count=len(records), hard_count=len(hard),
                  score_count=sum(r["primitive"] == "score" for r in records),
                  hard_score_count=sum(r["primitive"] == "score" for r in hard),
                  hard_accuracy=mean("hard_correct", hard), hard_ece=ece,
                  hard_noul={"count": len(noul), "tp": tp, "fp": fp, "fn": fn,
                             "precision": tp / (tp + fp) if tp + fp else None,
                             "recall": tp / (tp + fn) if tp + fn else None,
                             "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None})
    return result


def summarize_predictions(records):
    """Aggregate actual decisions (gather records first in distributed callers).

    ECE uses 15 equal-width confidence bins, hard labels only. Noul's positive
    class is true/Yes; undefined precision/recall/F1 are null. Score MAE and
    within-one compare E[p] with the hard level; soft error compares E[p], E[t].
    """
    records = list(records)
    # Refuse nonfinite values rather than dropping invalid decisions.
    json.dumps(records, allow_nan=False)
    result = _metrics(records)
    for field in ("source", "primitive", "K", "target_kind"):
        groups = defaultdict(list)
        for record in records:
            value = len(record["target"]) if field == "K" else record[field]
            groups[str(value)].append(record)
        result["by_" + field] = {key: _metrics(rows) for key, rows in sorted(groups.items())}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "data", "output", "predictions"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.max_length < 1 or a.limit < 0:
        ap.error("max-length must be positive and limit nonnegative")
    for name in ("model_path", "data", "output", "predictions"):
        setattr(a, name, os.path.abspath(os.path.expanduser(getattr(a, name))))
    if not os.path.isdir(a.model_path):
        ap.error("model-path must be a local directory")
    if a.output == a.predictions:
        ap.error("output and predictions must differ")
    for path in (a.output, a.predictions):
        if os.path.lexists(path) or not os.path.isdir(os.path.dirname(path)):
            ap.error("output parent must exist and output must not exist: " + path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from bench_diff_yesno import token_mapping
    from train_qwen_masked_typed import CandidateModel, amp_context

    device = torch.device(a.device)
    dtype = getattr(torch, a.amp_dtype)
    if device.type != "cuda" and dtype != torch.float32:
        ap.error("inherited AMP context requires CUDA; use float32 for CPU")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True, trust_remote_code=True)
    base = AutoModelForMaskedLM.from_pretrained(
        a.model_path, torch_dtype=dtype, local_files_only=True, trust_remote_code=True)
    tokens = token_mapping(tok, base)
    limits = [a.max_length]
    for value in (getattr(base.config, "max_position_embeddings", None),
                  getattr(tok, "model_max_length", None)):
        if isinstance(value, int) and 0 < value < 10**9:
            limits.append(value)
    items = load_structured_items(a.data, tok, min(limits), tokens, a.limit)
    model = CandidateModel(base.to(device)).eval()
    records = []
    with open(a.output, "x", encoding="utf-8") as output, \
            open(a.predictions, "x", encoding="utf-8") as predictions, torch.inference_mode():
        for item in items:
            ids = torch.tensor([item["ids"]], dtype=torch.long, device=device)
            candidates = torch.tensor([item["candidates"]], dtype=torch.long, device=device)
            with amp_context(dtype):
                logits = model(ids, torch.ones_like(ids, dtype=torch.bool), candidates,
                               item["positions"], item["noul"])
            record = prediction_record(item, logits)
            predictions.write(json.dumps(record, allow_nan=False) + "\n")
            records.append(record)
        result = summarize_predictions(records)
        result["settings"] = dict(vars(a), temperature=1.0, context_limit=min(limits),
                                  protocol="shared_yesno", answer_token_ids=tokens[0],
                                  mask_id=tokens[1], sequences_per_forward=1,
                                  full_vocab_forward=False, candidate_projection=True,
                                  attention_mask="2D bool all-valid", truncation=False,
                                  generation=False, enable_thinking=False,
                                  hard_ece_bins=15, noul_positive_class="true/Yes",
                                  score_error="abs(E[p]-E[t]); hard within-one iff error <= 1",
                                  smoke_selection="seed42 whole-group stratified spread; original row order")
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
