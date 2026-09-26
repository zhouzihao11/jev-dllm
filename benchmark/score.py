"""Score a complete ID set, or explicitly rescore validated historical artifacts."""

import argparse
from collections import defaultdict
import math
from pathlib import Path

from common import (bundled, exact, indexed, inherited, input_bundle, load, require,
                    rows, save, selected_inputs, source_profile, validate_target)


def validate_prediction(prediction, row):
    require(prediction["id"] == row["id"] and prediction["suite"] == row["suite"], "Prediction identity mismatch")
    status = prediction["status"]
    require(status in ("ok", "unsupported_length", "unsupported_source_mask"), "Unknown prediction status")
    length = prediction["length"]
    require(length is None and status == "unsupported_source_mask"
            or type(length) is int and length > 0, "Invalid token length")
    p, pred = prediction["probs"], prediction["pred_idx"]
    if status == "ok":
        require(isinstance(p, list) and len(p) == len(row["option_ids"])
                and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in p)
                and abs(math.fsum(p) - 1) <= 1.001e-5, "Invalid probabilities; no renormalization")
        require(type(pred) is int and pred == max(range(len(p)), key=p.__getitem__), "Wrong first-argmax index")
    else:
        require(p is None and pred is None, "Unsupported row must not carry predictions")
    seconds = prediction.get("forward_seconds", 0.0)
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0, "Invalid timing")


def historical(args, manifest, inputs, targets, provenance, backend):
    require(args.source_profile is not None, "Historical reuse requires --source-profile")
    profile, datasets = source_profile(args.source_profile)
    require(profile["profile_id"] == manifest["source_profile_id"], "Original profile mismatch")
    original = {r["id"]: (spec, r) for spec, data in datasets for r in data}
    root = args.historical_root.expanduser().resolve()
    run = load(root / "run.json")
    require(run["profile_id"] == profile["profile_id"] and not run["smoke"]
            and run["status"] in ("complete", "completed_with_unsupported")
            and run["full_status"] in ("complete", "completed_with_unsupported"),
            "Historical source must be a completed original full run")
    predictions = {}
    for spec, data in datasets:
        name = spec["name"]
        report = run["suites"][name]
        require(exact(report["suite"], spec), "Historical suite descriptor differs")
        old = indexed(rows(bundled(root, name + "_predictions.jsonl")))
        require(set(old) == {r["id"] for r in data}, "Historical suite missing/foreign IDs")
        require(report["source_count"] == report["planned_count"] == len(data), "Historical count mismatch")
        predictions.update(old)
    result = []
    for item in inputs:
        identity = item["id"]
        require(identity in original, "Unknown retained original ID")
        spec, source = original[identity]
        require(item["suite"] == spec["name"] and exact(item["state"], source["state"])
                and exact(item["qdef"], source["qdef"]), "Historical input/order differs: " + identity)
        target, prov = targets[identity], provenance[identity]
        require(all(exact(target[k], source[k]) for k in ("gold_idx", "soft", "gold_score", "option_ids")),
                "Historical target/order differs: " + identity)
        require(prov["original_id"] == identity and prov["group_id"] == source["group_id"]
                and exact(prov["metadata"], source["metadata"]) and exact(prov["source"], spec),
                "Historical provenance differs: " + identity)
        pred = predictions[identity]
        require(all(exact(pred[k], source[k]) for k in
                    ("gold_idx", "soft", "gold_score", "option_ids", "group_id", "metadata")),
                "Historical artifact target/provenance differs: " + identity)
        require(pred["primitive"] == source["qdef"]["type"]
                and exact(pred["options"], backend.render_options(backend.to_internal(source["qdef"]))),
                "Historical artifact options differ: " + identity)
        if "state" in pred or "qdef" in pred:
            require(exact(pred.get("state"), source["state"])
                    and exact(pred.get("qdef"), source["qdef"]), "Historical artifact input differs")
        result.append(dict(pred, suite=spec["name"]))
    return result, {"label": "historical_rescore", "new_inference": False,
                    "original_backend": run["backend"], "runtime": run.get("runtime"),
                    "input_validation": "explicit source profile; old artifacts omit state/qdef"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path)
    mode.add_argument("--historical-root", type=Path)
    parser.add_argument("--source-profile", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-manifest", type=Path)
    args = parser.parse_args()
    manifest, inputs = input_bundle(args.data_root)
    targets = indexed(rows(bundled(args.data_root, manifest["files"]["targets"])))
    provenance = indexed(rows(bundled(args.data_root, manifest["files"]["provenance"])))
    all_ids = {r["id"] for r in inputs}
    require(set(targets) == set(provenance) == all_ids, "Bundle target/provenance IDs differ")
    selected = inputs
    if args.run_manifest is not None:
        run = load(args.run_manifest)
        require(run["profile_id"] == manifest["profile_id"]
                and run["source_profile_id"] == manifest["source_profile_id"]
                and run["smoke"] == args.smoke and run["phases"]["inference"] == "complete",
                "Run manifest mismatch or unfinished inference")
        selected = selected_inputs(inputs, args.smoke, run["smoke_limit"])
        require(run["expected_ids"] == [r["id"] for r in selected], "Invalid expected ID selection")
    else:
        require(not args.smoke, "Smoke scoring requires explicit --run-manifest expected_ids")
    require(args.historical_root is not None or args.source_profile is None,
            "--source-profile is only for historical reuse")
    backend = inherited()
    validated, identities = {}, set()
    groups = defaultdict(lambda: defaultdict(set))
    decisions = indexed(rows(bundled(args.data_root, manifest["audit"]["decisions"])))
    require(len(decisions) == manifest["source_count"]
            and {identity for identity, d in decisions.items() if d["decision"] == "retain"} == all_ids,
            "Audit/package selection mismatch")
    for identity, decision in decisions.items():
        groups[decision["suite"]][decision["group_id"]].add(identity)
    for item in inputs:
        identity = item["id"]
        target, prov = targets[identity], provenance[identity]
        require(set(target) == {"id", "gold_idx", "soft", "gold_score", "option_ids"}, "Invalid target fields")
        require(set(prov) == {"id", "suite", "group_id", "metadata", "source", "original_id"}
                and prov["suite"] == item["suite"] and prov["original_id"] == identity,
                "Invalid provenance")
        require(decisions[identity]["group_id"] == prov["group_id"]
                and decisions[identity]["suite"] == item["suite"], "Audit group mismatch")
        validate_target(target, item["qdef"])
        row = dict(id=identity, state=item["state"], qdef=item["qdef"],
                   group_id=prov["group_id"], metadata=prov["metadata"],
                   **{k: target[k] for k in ("gold_idx", "soft", "gold_score", "option_ids")})
        validated[identity] = dict(backend.validate_row(row, identities), suite=item["suite"])
    if args.historical_root is not None:
        require(not args.smoke and args.run_manifest is None, "Historical mode rescoring is full retained core only")
        predictions, label = historical(args, manifest, inputs, targets, provenance, backend)
    else:
        predictions = rows(args.predictions)
        label = dict(label="input_only_inference" if args.run_manifest else "external_predictions",
                     new_inference=None if args.run_manifest is None else True)
    predictions = indexed(predictions)
    require(set(predictions) == {r["id"] for r in selected}, "Missing/foreign predictions; full runs cannot drop rows")
    records = []
    allowed = {"id", "suite", "status", "length", "probs", "pred_idx", "raw_logits", "forward_seconds",
               "yes_no_probs", "logodds", "batch_id", "actual_batch_size", "input_options", "input_metadata", "error"}
    for item in selected:
        row, pred = validated[item["id"]], predictions[item["id"]]
        if args.historical_root is None:
            require(set(pred) <= allowed, "Unexpected prediction fields (possible target leakage)")
        validate_prediction(pred, row)
        records.append(dict(row, **{k: pred[k] for k in ("status", "length", "probs", "pred_idx")},
                            forward_seconds=pred.get("forward_seconds", 0.0)))
    reports = {}
    for spec in manifest["suites"]:
        suite_rows = [r for r in records if r["suite"] == spec["name"]]
        report = dict(source_count=spec["count"], retained_count=spec["retained_count"],
                      planned_count=len(suite_rows), metrics=backend.summarize(suite_rows, groups[spec["name"]]))
        for field, key in (("by_primitive", lambda r: r["primitive"]),
                           ("by_task", lambda r: r["metadata"].get("task_id", "__unspecified__"))):
            partitions = defaultdict(list)
            for row in suite_rows:
                partitions[key(row)].append(row)
            report[field] = {name: backend.summarize(part, groups[spec["name"]]) for name, part in partitions.items()}
        reports[spec["name"]] = report
    status = "complete" if all(r["status"] == "ok" for r in records) else "completed_with_unsupported"
    definitions = dict(backend.METRIC_DEFINITIONS)
    definitions["timing"] = ("Inherited synchronized forward seconds, excluding tokenization, transfers and IO; "
                             "batched row seconds are elapsed / actual supported batch size, not request latency.")
    output = args.output_dir.expanduser().absolute()
    output.mkdir(exist_ok=False)
    save(output / "report.json", dict(schema_version="benchmark_core_score_v1", profile_id=manifest["profile_id"],
                                     source_profile_id=manifest["source_profile_id"], status=status,
                                     full_status="not_full_smoke" if args.smoke else status,
                                     smoke=args.smoke, mode=label, attempted=len(records),
                                     metric_definitions=definitions, suites=reports,
                                     overall_winner=None, selection="full retained core" if not args.smoke else "smoke prefix only"))


if __name__ == "__main__":
    main()
