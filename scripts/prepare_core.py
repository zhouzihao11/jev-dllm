"""Acquire pinned public sources and prepare the frozen 17-suite core candidate."""

import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from core_sources import acquire, require


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "benchmark/manifests/core_v1.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def run(script, *arguments, scratch):
    environment = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(scratch))
    subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)],
                   cwd=ROOT, env=environment, check=True)


def frozen_decisions(manifest):
    excluded = [dict(item) for item in manifest["excluded_ids"]]
    for group in manifest["excluded_groups"]:
        for suffix in group["member_suffixes"]:
            excluded.append(dict(id=group["group_id"] + "/" + suffix,
                                 suite=group["suite"], group_id=group["group_id"],
                                 decision=group["decision"], reason=group["reason"],
                                 canonical_id=group["canonical_group_id"] + "/" + suffix))
    indexed = {item["id"]: item for item in excluded}
    require(len(indexed) == len(excluded), "Duplicate frozen exclusion ID")
    return indexed


def replay_membership(profile_path, audit_dir, manifest):
    # Packaging uses the same original-v2 validation as benchmark/prepare.py.
    sys.path.insert(0, str(ROOT / "benchmark"))
    common = importlib.import_module("common")

    profile, datasets = common.source_profile(profile_path)
    expected = {suite["name"]: suite for suite in manifest["suites"]}
    require([spec["name"] for spec, _ in datasets] == list(expected), "Candidate suite order differs")
    exclusions = frozen_decisions(manifest)
    decisions, retained, counts, seen_exclusions = [], [], {}, set()
    original_ids = {row["id"] for _, rows in datasets for row in rows}
    for spec, rows in datasets:
        name = spec["name"]
        require(len(rows) == expected[name]["candidate_count"], "Candidate count differs: " + name)
        suite_counts = Counter({"retain": 0, "exclude": 0, "quarantine": 0})
        for row in rows:
            identity = row["id"]
            frozen = exclusions.get(identity)
            status = "retain"
            decision = dict(id=identity, suite=name, group_id=row["group_id"], decision="retain", reasons=[])
            if frozen is not None:
                require(frozen["suite"] == name and frozen["group_id"] == row["group_id"],
                        "Frozen exclusion identity/group mismatch: " + identity)
                reason = {"kind": frozen["reason"]}
                if "canonical_id" in frozen:
                    require(frozen["canonical_id"] in original_ids, "Missing canonical duplicate ID")
                    reason["canonical_id"] = frozen["canonical_id"]
                decision.update(decision=frozen["decision"], reasons=[reason])
                status = frozen["decision"]
                seen_exclusions.add(identity)
            else:
                retained.append(identity)
            suite_counts[status] += 1
            decisions.append(decision)
        require(suite_counts["retain"] == expected[name]["retained_count"], "Retained count differs: " + name)
        counts[name] = dict(suite_counts)
    require(seen_exclusions == set(exclusions), "Frozen exclusion IDs missing from sources")
    require(len(decisions) == manifest["candidate_count"] and len(retained) == manifest["retained_count"],
            "Core total differs from the frozen membership")
    frozen_retained = read_json(CONTROL.parent / manifest["retained_ids_file"])
    require(retained == frozen_retained, "Retained IDs or their order differ from the frozen research membership")
    require(dict(Counter(item["decision"] for item in decisions)) == manifest["decision_counts"],
            "Frozen decision counts differ")
    audit_dir.mkdir()
    write_json(audit_dir / "retained_ids.json", {
        "profile_id": profile["profile_id"], "retained_ids": retained,
        "counts_by_suite": {name: value["retain"] for name, value in counts.items()},
    })
    common.save_rows(audit_dir / "decisions.jsonl", decisions)
    write_json(audit_dir / "audit.json", {
        "status": "complete", "profile_id": profile["profile_id"],
        "selection_version": manifest["selection_version"], "mode": "frozen_public_membership_replay",
        "raw_rows": len(decisions), "candidate_rows": len(decisions), "candidate_suites": len(counts),
        "retained_rows": len(retained), "counts_by_suite": counts,
        "decision_counts": manifest["decision_counts"], "scope": manifest["scope"],
        "historical_parent": manifest["historical_parent"],
        "historical_audit_method": manifest["historical_audit_method"],
        "policy_excluded_source_families": manifest["policy_excluded_source_families"],
        "membership_method": manifest["membership_method"],
        "exclusion_counts": manifest["exclusion_counts"],
    })
    report = ["# Core v1 public membership replay", "",
              "This is the public summary of frozen selection, not a rerun of the private exposure audit.",
              "Status complete means all candidate IDs were assigned their frozen decision and counts validated.",
              "No private training files, exposure paths, text snippets, predictions or score logs are required.",
              "", manifest["membership_method"], "", *manifest["scope"], "",
              "6768 candidates -> 6744 retained: 4 exposure/group exclusions, 18 duplicate exclusions,",
              "and 2 target-ambiguity quarantines. Policy exclusions are applied before acquisition.",
              "", "| Suite | Candidate | Retained |", "| --- | ---: | ---: |"]
    report += [f"| {s['name']} | {s['candidate_count']} | {s['retained_count']} |" for s in manifest["suites"]]
    (audit_dir / "AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def prepare(raw, output, manifest):
    require(not output.exists(), "--output-dir must be NEW: " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".prepare-core-") as temporary:
        scratch = Path(temporary)
        converted = scratch / "v2"
        revisions = {s["repo"]: s["revision"] for s in manifest["sources"] if s["kind"] == "github"}
        run("research/scripts/prepare_eval_v2_core.py", "--raw-root", raw["core"],
            "--output-root", converted, "--include-source", "jevbench", "--include-source", "jabr",
            "--jevbench-revision", revisions["fstandhartinger/jevbench"],
            "--jabr-revision", revisions["jabr/classifier-benchmark"], scratch=scratch)
        run("research/scripts/prepare_eval_v2_nimble.py", "--raw-root", raw["nimble"],
            "--output-dir", converted, "--exclude-subset", "boolq", scratch=scratch)
        run("research/scripts/prepare_eval_v2_documents.py", "--raw-root", raw["documents"],
            "--output-root", converted, "--include-source", "contractnli", scratch=scratch)
        suites = read_json(converted / "core_suites.json")["suites"]
        suites += read_json(converted / "full_eval_v2_nimble.json")["suites"]
        suites.append(read_json(converted / "contractnli.suite.json"))
        profile = converted / "full_eval_v2.json"
        write_json(profile, {"schema_version": "full_eval_v2", "profile_id": manifest["source_profile_id"],
                             "scope": "Original core candidate subset only", "suites": suites})
        audit_dir = scratch / "audit"
        replay_membership(profile, audit_dir, manifest)
        packaged = scratch / "package"
        run("benchmark/prepare.py", "--profile", profile, "--audit-dir", audit_dir,
            "--output-dir", packaged, scratch=scratch)
        produced = read_json(packaged / "manifest.json")
        require(produced["source_count"] == manifest["candidate_count"]
                and produced["retained_count"] == manifest["retained_count"]
                and [(s["name"], s["retained_count"]) for s in produced["suites"]]
                == [(s["name"], s["retained_count"]) for s in manifest["suites"]],
                "Packaged membership/counts differ")
        shutil.copyfile(converted / "nimble_upstream_checksums.json", packaged / "nimble_upstream_checksums.json")
        require(not output.exists(), "Refusing to replace an existing output")
        packaged.rename(output)
    print(f"Prepared {manifest['retained_count']} decisions across {len(manifest['suites'])} suites: {output}")
    print("Frozen membership replay complete; not a new exposure audit or publication approval.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "shared-yesno-dllm",
                        help="External source cache (default: XDG_CACHE_HOME/shared-yesno-dllm)")
    parser.add_argument("--raw-root", type=Path, help="Reuse a complete pinned raw tree; never download into it")
    parser.add_argument("--offline", action="store_true", help="Forbid source acquisition; require complete cached files")
    parser.add_argument("--output-dir", type=Path, help="New benchmark bundle directory (required unless --download-only)")
    parser.add_argument("--download-only", action="store_true", help="Acquire/validate source files only; no converter imports or execution")
    args = parser.parse_args()
    if not args.download_only and args.output_dir is None:
        parser.error("--output-dir is required unless --download-only is used")
    if args.output_dir is not None and (args.output_dir.expanduser().exists()
                                        or args.output_dir.expanduser().is_symlink()):
        parser.error("--output-dir must be NEW")
    cache = args.cache_dir.expanduser().resolve()
    root = args.raw_root.expanduser().resolve() if args.raw_root is not None else cache / "core-v1/raw"
    manifest = read_json(CONTROL)
    raw = acquire(root, cache, manifest, offline=args.offline, explicit_raw=args.raw_root is not None)
    if args.download_only:
        print(f"All required pinned source files present: {root}")
        print("Acquisition only; conversion and output equivalence have not been verified.")
        return
    prepare(raw, args.output_dir.expanduser().resolve(), manifest)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Core preparation failed: {error}") from None
