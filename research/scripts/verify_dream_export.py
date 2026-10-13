"""Standalone BF16 native Dream export evaluation; reports differences, not quality gates."""
import argparse
import json
import math
import os
from pathlib import Path
import time

from _benchmark_bootstrap import bootstrap

bootstrap()

import torch
from transformers import AutoModel, AutoTokenizer

from train_dream_masked_typed import (
    CandidateWrapper, data_inputs, forward, memory, prediction_record,
    load_structured_items, summarize_predictions, write_json,
)


def compare(records, reference):
    identity = lambda row: (row["case_id"], row["view_id"])
    current, previous = ({identity(row): row for row in rows} for rows in (records, reference))
    common = sorted(current.keys() & previous.keys())
    compatible = [key for key in common if len(current[key]["probabilities"]) == len(previous[key]["probabilities"])]
    differences = [abs(x - y) for key in compatible for x, y in
                   zip(current[key]["probabilities"], previous[key]["probabilities"])]
    return dict(matched=len(common), shape_mismatches=len(common) - len(compatible),
                missing_identities=sorted(previous.keys() - current.keys()),
                extra_identities=sorted(current.keys() - previous.keys()),
                duplicate_current=len(records) - len(current), duplicate_reference=len(reference) - len(previous),
                target_mismatches=sum(current[key]["target"] != previous[key]["target"] for key in common),
                max_abs_probability_difference=max(differences, default=None),
                mean_abs_probability_difference=math.fsum(differences) / len(differences) if differences else None,
                argmax_disagreements=sum(current[key]["prediction_index"] != previous[key]["prediction_index"] for key in compatible),
                dev_kl_difference=summarize_predictions(records)["kl"] - summarize_predictions(reference)["kl"],
                max_abs_decision_kl_difference=max((abs(current[key]["kl"] - previous[key]["kl"])
                                                    for key in compatible), default=None))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "dev-data", "output-dir"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--reference-predictions")
    ap.add_argument("--repeat-eval", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=4096)
    a = ap.parse_args()
    if min(a.repeat_eval, a.max_length) < 1:
        ap.error("repeat-eval and max-length must be positive")
    output = Path(a.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        ap.error("output-dir must be empty")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    started = time.perf_counter()

    def emit(event, **values):
        print(json.dumps(dict(event=event, elapsed_seconds=time.perf_counter() - started,
                              memory=memory(), **values), allow_nan=False), flush=True)
    emit("init", settings=vars(a))
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True, local_files_only=True)
    base = AutoModel.from_pretrained(a.model_path, trust_remote_code=True, local_files_only=True,
                                    torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    base.config.use_cache = base.model.config.use_cache = False
    tokens, limit, renderer = data_inputs(tok, base, a.max_length)
    dev = load_structured_items(a.dev_data, tok, limit, tokens, limit=0, renderer=renderer)
    if not dev or any(row["split"] not in ("dev", "validation") for row in dev):
        raise ValueError("Expected nonempty canonical dev split")
    emit("tokenized", dev_count=len(dev), context_limit=limit)
    model = CandidateWrapper(base.to(device)).eval()
    reference = None
    if a.reference_predictions:
        with open(a.reference_predictions, encoding="utf-8") as stream:
            reference = [json.loads(line) for line in stream]
        if not reference:
            raise ValueError("Empty reference predictions")
    first, results = None, []
    for repeat in range(a.repeat_eval):
        emit("dev_begin", repeat=repeat)
        records = []
        with torch.no_grad(), open(output / ("eval-%d-predictions.jsonl" % repeat), "x", encoding="utf-8") as stream:
            for item in dev:
                row = prediction_record(item, forward(model, item, device))
                records.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        result = dict(repeat=repeat, metrics=summarize_predictions(records))
        if reference is not None:
            result["versus_reference"] = compare(records, reference)
        if first is not None:
            result["versus_first_eval"] = compare(records, first)
        else:
            first = records
        write_json(output / ("eval-%d.json" % repeat), result)
        results.append(result)
        emit("dev", **result)
    write_json(output / "summary.json", dict(settings=vars(a), results=results, temperature=1.0,
                                             renderer="oldgeneric_noul_v1", precision="native_bf16_projection_fp32_terms"))


if __name__ == "__main__":
    main()
