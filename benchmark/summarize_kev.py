"""Summarize scored Kev artifacts without models, datasets, or metric recomputation."""

import argparse
import json
from pathlib import Path


EXCLUDED = {"nimble_multinli": "training source", "nimble_aegis2": "training source",
            "nimble_paws": "development exposure"}
ORDINAL = ("nimble_helpsteer2", "nimble_summeval-relevance", "nimble_summeval-consistency")
METRICS = ("accuracy", "ece", "nll", "brier")
TITLE = "Kev native success denominator; not five-model common set"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_run(path, counts, profile):
    run, report = load(path / "run.json"), load(path / "scores/report.json")
    for artifact in (run, report):
        require(artifact["smoke"] is False and artifact["profile_id"] == profile,
                "Expected a full, nonsmoke " + profile + " run")
        require(all(artifact[k] in ("complete", "completed_with_unsupported")
                    for k in ("status", "full_status")), "Run/scoring is not complete")
    require(run["backend"] == "kev" and run["requested_dtype"] == "float32"
            and run["batch_size"] == 1, "Expected native Kev FP32 batch one")
    require(run["phases"] == {"inference": "complete", "scoring": "complete"},
            "Both evaluation phases must be complete")
    require(run["source_profile_id"] == report["source_profile_id"], "Source profiles differ")
    ids = run["expected_ids"]
    require(len(ids) == len(set(ids)) == report["attempted"] == sum(counts.values()),
            "Expected-ID or attempted count mismatch")
    require(set(report["suites"]) == set(counts), "Suite set differs from requested profile")
    for name, count in counts.items():
        suite = report["suites"][name]
        require(suite["retained_count"] == suite["planned_count"]
                == suite["metrics"]["attempted"] == count, "Count mismatch: " + name)
        m = suite["metrics"]
        require(0 < m["hard"]["n"] <= m["completed"] <= count, "Invalid denominator: " + name)
    metadata = {k: v for k, v in run.items() if k != "expected_ids"}
    metadata["artifact_directory"] = str(path.resolve())
    metadata["condition_label"] = "temperature unknown; not identity-certified"
    runtime = run.get("runtime", {})
    require(runtime.get("dtype", "float32") == "float32", "Recorded runtime is not FP32")
    if runtime.get("temperature") is not None:
        metadata["condition_label"] = ("native checkpoint temperature" if
            runtime.get("calibration") == "checkpoint" else "temperature diagnostic/other")
    return metadata, report


def aggregate(suites, names):
    n = sum(suites[name]["metrics"]["hard"]["n"] for name in names)
    return {"suites": names, "hard_n": n,
            "attempted": sum(suites[name]["planned_count"] for name in names),
            "completed": sum(suites[name]["metrics"]["completed"] for name in names),
            "equal_suite_mean_stored_metrics": {
                key: sum(suites[name]["metrics"]["hard"][key] for name in names) / len(names)
                for key in METRICS},
            "weighted_accuracy_stored_precision": sum(
                suites[name]["metrics"]["hard"]["accuracy"] * suites[name]["metrics"]["hard"]["n"]
                for name in names) / n}


def fmt(value):
    return "-" if value is None else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-run", required=True, type=Path)
    parser.add_argument("--typed-run", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load(Path(__file__).parent / "manifests/core_v1.json")
    counts = {s["name"]: s["retained_count"] for s in spec["suites"]}
    metadata, report = read_run(args.core_run, counts, spec["profile_id"])
    require(report["source_profile_id"] == spec["source_profile_id"], "Wrong Core source profile")
    suites = report["suites"]
    names = [name for name in counts if name not in EXCLUDED]
    nimble = [name for name in names if name.startswith("nimble_")]
    require(len(counts) == 17 and sum(counts.values()) == 6744
            and len(names) == 14 and len(nimble) == 9, "Unexpected packaged Core definition")
    summary = {"title": TITLE, "core_metadata": metadata, "excluded": EXCLUDED,
               "metric_definitions": report["metric_definitions"],
               "core14": {name: suites[name] for name in names},
               "core14_aggregate": aggregate(suites, names), "nimble9": aggregate(suites, nimble)}
    rows = [(name, suites[name]) for name in names]
    if args.typed_run is not None:
        typed_meta, typed = read_run(args.typed_run, {"typed_decisions": 2000},
                                     "legacy_typed_decisions_v1")
        summary["typed_separate"] = {"metadata": typed_meta, "suite": typed["suites"]["typed_decisions"],
                                     "metric_definitions": typed["metric_definitions"]}
        rows.append(("typed_decisions (legacy; separate)", typed["suites"]["typed_decisions"]))
    lines = ["# " + TITLE, "", "Artifact-only summary; no new inference or scoring.", "",
             "Full Core17 attempted denominator: 6744. Core14 excludes: " +
             ", ".join(f"{k} ({v})" for k, v in EXCLUDED.items()) + ".",
             "Retained sources: overlap not confirmed, not certified unseen or blind.",
             "Stored hard metrics are rounded by score.py (four decimals); means and weighted accuracy",
             "use that precision, not reconstructed exact correct counts or pooled ECE.",
             "Typed results are separate and never enter Core/Nimble aggregates.", "",
             "## Recorded conditions (not checkpoint identity certification)", "", "```json",
             json.dumps(dict(core=metadata, typed=summary.get("typed_separate", {}).get("metadata")),
                        indent=2), "```", "",
             "## Native-success hard metrics (fractions, not percentages)", "",
             "| Suite | Successful / attempted | Hard n | Coverage | Accuracy | ECE15 | NLL | Brier(sum) | All-attempted accuracy |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, suite in rows:
        m = suite["metrics"]
        cells = [name, f'{m["completed"]}/{suite["planned_count"]}', str(m["hard"]["n"]),
                 fmt(m["coverage"])] + [fmt(m["hard"][k]) for k in METRICS]
        lines.append("| " + " | ".join(cells + [fmt(m["all_attempted_accuracy"])]) + " |")
    lines += ["", "ContractNLI uses this run's native-success denominator, not the historical",
              "five-model common 1819/2074 set (historical Kev alone: 1853/2074).", "",
              "## Equal-suite diagnostic means", "",
              "| Scope | Successful / attempted | Hard n | Accuracy | ECE15 | NLL | Brier(sum) | Weighted accuracy (stored precision) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in ("core14_aggregate", "nimble9"):
        a = summary[key]
        lines.append("| " + " | ".join([key, f'{a["completed"]}/{a["attempted"]}', str(a["hard_n"])] +
                     [fmt(a["equal_suite_mean_stored_metrics"][k]) for k in METRICS] +
                     [fmt(a["weighted_accuracy_stored_precision"])]) + " |")
    lines += ["", "## Ordinal target diagnostics", "",
              "MAE compares predicted expected level to the scorer's target; RPS is unnormalized.",
              "Values and target denominators are copied from score.py, not recomputed target means.", "",
              "| Suite | MAE n | MAE | RPS n | RPS |", "|---|---:|---:|---:|---:|"]
    for name in ORDINAL:
        m = suites[name]["metrics"]
        lines.append(f'| {name} | {m["score_mae"]["n"]} | {fmt(m["score_mae"]["value"])} | '
                     f'{m["score_rps"]["n"]} | {fmt(m["score_rps"]["value"])} |')
    args.output_dir.expanduser().mkdir(exist_ok=False)
    output = args.output_dir.expanduser()
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
