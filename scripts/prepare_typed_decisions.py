#!/usr/bin/env python3
"""Package the bundled legacy typed decisions, not the audited Core."""

import argparse
from collections import Counter
import gzip
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

from common import (bundled, exact, indexed, input_bundle, load, require, rows,
                    save, save_rows, validate_target)


SCOPE = "legacy_typed_decisions_not_audited_core"
SOURCE_PROFILE = "full_test_v1_20260923_r2"
SUITE = "typed_decisions"


def typed_source(explicit, scratch):
    manifest = load(REPO_ROOT / "datasets" / "manifest.json")
    require(manifest.get("quality_profile_origin") == SOURCE_PROFILE,
            "Unexpected bundled quality profile origin")
    entries = [entry for entry in manifest["files"]
               if entry["output_path"] in ("bench/typed_decisions_test.parquet",
                                           "bench/typed_decisions_test.jsonl")]
    require(len(entries) == 1, "Expected one bundled typed fixture")
    entry = entries[0]
    require(entry["rows"] == 400 and entry["decisions"] == 2000,
            "Expected the 400-case, 2000-decision fixture")
    source = (explicit.expanduser().resolve() if explicit is not None
              else bundled(REPO_ROOT / "datasets", entry["path"]))
    require(source.is_file(), "Missing typed data")
    provenance = dict(manifest="datasets/manifest.json", release_id=manifest["release_id"],
                      profile_id=SOURCE_PROFILE, entry=entry,
                      upstream=manifest["upstreams"]["bench_typed_decisions"],
                      input_mode="explicit_prepared_fixture" if explicit is not None else "bundled_fixture")
    if explicit is None and entry["compression"] == "gzip":
        destination = scratch / Path(entry["output_path"]).name
        with gzip.open(source, "rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        source = destination
    if source.suffix == ".jsonl":
        import pandas as pd

        raw = rows(source)
        require(len(raw) == 400, "Expected 400 raw typed JSONL cases, not decision rows")
        source = scratch / "typed.parquet"
        pd.DataFrame(raw).to_parquet(source, index=False)
    require(source.suffix == ".parquet", "--typed-data must be prepared raw typed JSONL or Parquet")
    return source, provenance


def convert(source, source_info):
    sys.path[:0] = [str(REPO_ROOT / "research" / "scripts"), str(REPO_ROOT)]
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    from bench_local import build_typed_decisions
    from bench_laya_aligned import external_rows

    cases, gold, workflows = build_typed_decisions(str(source))
    require(len(cases) == len(gold) == len(workflows) == 400
            and all(len(questions) == 5 for _, questions in cases),
            "Expected 400 intact five-question cases")
    records = []
    for ci, (state, questions) in enumerate(cases):
        for qid, qdef in questions.items():
            target = gold[ci][qid]
            records.append((state, {qid: qdef}, target["idx"],
                            dict(soft=target["soft"], gold_score=target.get("gold_score"),
                                 workflow=workflows[ci], qtype=qdef["type"])))
    converted = external_rows(SUITE, records)
    require(len(converted) == 2000, "Expected 2000 decisions")
    inputs, targets, provenance, decisions = [], [], [], []
    groups = Counter()
    for position, (row, original) in enumerate(zip(converted, records)):
        state, questions, gold_idx, extra = original
        qid, original_qdef = next(iter(questions.items()))
        identity = row["id"]
        require(row["case_idx"] == position // 5 and row["decision_idx"] == position
                and identity == "%s/%d/%s" % (SUITE, position // 5, qid)
                and row["qid"] == qid, "Legacy identity/order mismatch")
        require(exact(row["state"], state) and exact(row["qdef"], original_qdef)
                and row["gold_idx"] == gold_idx and exact(row["soft"], extra["soft"])
                and exact(row["gold_score"], extra["gold_score"]), "Legacy conversion differs")
        qdef = dict(row["qdef"])
        qdef.setdefault("criteria", None)
        if qdef["type"] == "choice":
            require(isinstance(qdef["criteria"], dict), "Legacy choice criteria must be an ordered object")
            option_ids = list(qdef["criteria"])
        elif qdef["type"] == "noul":
            option_ids = ["false", "true"]
        else:
            option_ids = [str(i) for i in range(len(qdef["criteria"]))]
        require(len(option_ids) == len(row["options"]), "Legacy option count differs")
        workflow = row["workflow"]
        require(isinstance(workflow, str) and bool(workflow.strip()), "Missing workflow")
        metadata = dict(extra)
        metadata.update(source=source_info,
                        adaptation="Legacy conversion; only absent criteria becomes null",
                        task_id=workflow, qid=qid, workflow=workflow,
                        case_idx=row["case_idx"], decision_idx=position,
                        options=row["options"], legacy_extra=extra, dataset_scope=SCOPE,
                        soft_target_kind="reference-distribution", soft_target_rounding_decimals=6,
                        soft_target_provenance="Upstream legacy reference probabilities; not asserted "
                                               "to be human votes or an exact mechanism")
        require(isinstance(row["soft"], list)
                and all(type(v) in (int, float) and math.isfinite(v)
                        and abs(v - round(v, 6)) <= 1e-12 for v in row["soft"]),
                "Expected six-decimal legacy reference probabilities")
        item = dict(id=identity, suite=SUITE, state=row["state"], qdef=qdef)
        target = {key: row[key] for key in ("id", "gold_idx", "soft", "gold_score")}
        target["option_ids"] = option_ids
        validate_target(target, qdef, metadata=metadata)
        group = "%s/%d" % (SUITE, row["case_idx"])
        groups[group] += 1
        inputs.append(item)
        targets.append(target)
        provenance.append(dict(id=identity, suite=SUITE, group_id=group, original_id=identity,
                               metadata=metadata, source=source_info))
        decisions.append(dict(id=identity, suite=SUITE, group_id=group, decision="retain",
                              reasons=["legacy_full_membership"]))
    indexed(inputs)
    require(len(groups) == 400 and set(groups.values()) == {5}, "Expected 400 intact groups")
    return inputs, targets, provenance, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed-data", type=Path,
                        help="Prepared raw 400-case typed JSONL or Parquet; default: bundled manifest entry")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().absolute()
    require(not os.path.lexists(output), "output-dir must be NEW")
    require(output.parent.is_dir(), "output-dir parent must exist")
    resolved = output.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        require(resolved.is_relative_to(REPO_ROOT / "local_data"),
                "Inside the repository, output must be under ignored local_data/")
    with tempfile.TemporaryDirectory(prefix="typed-decisions-") as temporary:
        source, source_info = typed_source(args.typed_data, Path(temporary))
        inputs, targets, provenance, decisions = convert(source, source_info)
    manifest = dict(
        schema_version="benchmark_core_v1", profile_id="legacy_typed_decisions_v1",
        source_profile_id=SOURCE_PROFILE, dataset_scope=SCOPE,
        source_count=2000, retained_count=2000,
        suites=[dict(name=SUITE, original_name=SUITE, count=2000, retained_count=2000,
                     path="inputs.jsonl")],
        files=dict(inputs="inputs.jsonl", targets="targets.jsonl", provenance="provenance.jsonl"),
        conversion=dict(helpers=["research/scripts/bench_local.py:build_typed_decisions",
                                 "research/scripts/bench_laya_aligned.py:external_rows"],
                        source=source_info),
        audit=dict(decisions="decisions.jsonl", selection_version="legacy_full_membership",
                   scope="Legacy membership and group coverage only; not a source-isolation audit",
                   counts_by_suite={SUITE: 2000}, decision_counts=dict(retain=2000)),
        target_scope="Original hard targets, reference distributions and scalar scores; no renormalization",
        release_status=SCOPE)
    output.mkdir(exist_ok=False)
    for filename, data in (("inputs.jsonl", inputs), ("targets.jsonl", targets),
                           ("provenance.jsonl", provenance), ("decisions.jsonl", decisions)):
        save_rows(output / filename, data)
    save(output / "manifest.json", manifest)
    _, loaded = input_bundle(output)
    require(exact(loaded, inputs), "Written inputs/order differ")
    for filename, expected in (("targets.jsonl", targets), ("provenance.jsonl", provenance),
                               ("decisions.jsonl", decisions)):
        actual = rows(output / filename)
        require(exact(actual, expected) and set(indexed(actual)) == set(indexed(loaded)),
                "Written data/order/IDs differ: " + filename)


if __name__ == "__main__":
    main()
