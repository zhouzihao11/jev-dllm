"""Package an audited original v2 profile without changing inputs or targets."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from common import indexed, load, require, rows, save, save_rows, source_profile


def excluded_by_policy(spec, row):
    source = json.dumps([spec, row["metadata"].get("source"), row["id"], row["group_id"]],
                        ensure_ascii=False).casefold()
    normalized = "".join(c for c in source if c.isalnum())
    return any(name in normalized for name in ("clinc", "boolq", "forecastbench"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    profile, datasets = source_profile(args.profile)
    audit_dir = args.audit_dir.expanduser().resolve()
    retained = load(audit_dir / "retained_ids.json")
    audit = load(audit_dir / "audit.json")
    require(audit.get("status") == "complete", "Audit must be complete before packaging")
    require((audit_dir / "AUDIT_REPORT.md").is_file(), "Missing actual audit report")
    require(retained["profile_id"] == profile["profile_id"], "Audit/source profile mismatch")
    if "profile_id" in audit:
        require(audit["profile_id"] == profile["profile_id"], "Audit report profile mismatch")
    ids = retained["retained_ids"]
    require(isinstance(ids, list) and all(isinstance(x, str) for x in ids)
            and len(set(ids)) == len(ids), "Invalid/duplicate retained IDs")
    keep = set(ids)
    decisions = indexed(rows(audit_dir / "decisions.jsonl"))
    originals = {row["id"]: (spec, row) for spec, data in datasets for row in data}
    require(set(decisions) == set(originals), "Audit must decide every original ID exactly once")
    for identity, decision in decisions.items():
        spec, row = originals[identity]
        require(decision["suite"] == spec["name"] and decision["group_id"] == row["group_id"],
                "Audit identity/group mismatch: " + identity)
        require(decision["decision"] in ("retain", "exclude", "quarantine")
                and isinstance(decision["reasons"], list), "Invalid audit decision")
        require((identity in keep) == (decision["decision"] == "retain"),
                "retained_ids/decisions disagree: " + identity)
    require(keep <= set(originals) and keep, "Unknown or empty retained IDs")
    counts = Counter(originals[identity][0]["name"] for identity in keep)
    audit_counts = retained["counts_by_suite"]
    require(set(audit_counts) <= {spec["name"] for spec, _ in datasets}
            and all(type(value) is int and value >= 0 for value in audit_counts.values())
            and {k: v for k, v in audit_counts.items() if v} == dict(counts),
            "Audited suite counts disagree")
    inputs, targets, provenance, suites = [], [], [], []
    for spec, data in datasets:
        name = spec["name"]
        if counts[name]:
            suites.append(dict(name=name, original_name=name, count=len(data),
                               retained_count=counts[name], path="inputs.jsonl"))
        for row in data:
            if row["id"] not in keep:
                continue
            require(not excluded_by_policy(spec, row), "Policy-excluded source cannot enter core: " + row["id"])
            inputs.append(dict(id=row["id"], suite=name, state=row["state"], qdef=row["qdef"]))
            targets.append({key: row[key] for key in ("id", "gold_idx", "soft", "gold_score", "option_ids")})
            provenance.append(dict(id=row["id"], group_id=row["group_id"], suite=name,
                                   metadata=row["metadata"], source=spec, original_id=row["id"]))
    output = args.output_dir.expanduser().absolute()
    output.mkdir(exist_ok=False)
    save_rows(output / "inputs.jsonl", inputs)
    save_rows(output / "targets.jsonl", targets)
    save_rows(output / "provenance.jsonl", provenance)
    for name in ("retained_ids.json", "decisions.jsonl", "audit.json", "AUDIT_REPORT.md"):
        shutil.copyfile(audit_dir / name, output / name)
    manifest = dict(schema_version="benchmark_core_v1", profile_id="core_v1_candidate",
                    source_profile_id=profile["profile_id"], source_count=len(originals),
                    retained_count=len(inputs), suites=suites,
                    files=dict(inputs="inputs.jsonl", targets="targets.jsonl", provenance="provenance.jsonl"),
                    audit=dict(report="AUDIT_REPORT.md", details="audit.json", decisions="decisions.jsonl",
                               retained_ids="retained_ids.json", selection_version="core_v1_candidate",
                               source_selection_version=audit.get("selection_version"),
                               reported_scope=audit.get("scope", audit.get("scope_limitations")),
                               scope="relative known S0/S1 training, selection and diagnostic exposure; not pretraining clearance",
                               counts_by_suite=dict(counts),
                               decision_counts=dict(Counter(d["decision"] for d in decisions.values())),
                                policy_excluded_source_families={"CLINC": "seen-source",
                                    "BoolQ": "seen-source", "ForecastBench": "retrospective forecasting"}),
                    release_status="candidate_not_release_ready")
    save(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
