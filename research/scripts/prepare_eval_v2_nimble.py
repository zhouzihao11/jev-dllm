"""Build the official Nimble Public13 from local raw files; run remotely, offline."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile


UPSTREAM_REVISION = "62076b4f2d365b5879dafcf7f6dd072a1fe76df7"
# name: (local file, raw split, immutable repository revision or archive SHA-256)
SOURCES = {
    "vitaminc": ("vitaminc.zip", "dev", "sha256:49d82dc1690cbee420d18e2c26f687a7937710bb211845d2571430dfd4dc0337"),
    "massive": ("amazon-massive-dataset-1.1.tar.gz", "test", "sha256:4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"),
    "boolq": ("boolq/data/validation-00000-of-00001.parquet", "validation", "35b264d03638db9f4ce671b711558bf7ff0f80d5"),
    "squad2": ("squad2/squad_v2/validation-00000-of-00001.parquet", "validation", "3ffb306f725f7d2ce8394bc1873b24868140c412"),
    "paws": ("paws/labeled_final/test-00000-of-00001.parquet", "labeled_final/test", "161ece9501cf0a11f3e48bd356eaa82de46d6a09"),
    "multinli": ("multinli_1.0.zip", "dev_matched", "sha256:049f507b9e36b1fcb756cfd5aeb3b7a0cfcb84bf023793652987f7e7e0957822"),
    "civil_comments": ("civil_comments/data/test-00000-of-00001.parquet", "test", "f2970eb3a55777454c94069077cc8d9b5866312d"),
    "aegis2": ("aegis2/test.json", "test", "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"),
    "helpsteer2": ("helpsteer2/validation.jsonl.gz", "validation", "990b2711a36180dd19d9c94b8627844866f8982a"),
    "summeval": ("summeval/data/test-00000-of-00001-35901af5f6649399.parquet", "test", "bfc121155064afa2d81b5505682ffc0d96f4334c"),
    "pubmedqa": ("pubmedqa/pqa_labeled/train-00000-of-00001.parquet", "pqa_labeled/train", "9001f2853fb87cab8d220904e0de81ac6973b318"),
}
SUBSETS = {
    "vitaminc-dev": 599, "massive-en-US": 350, "massive-de-DE": 350,
    "boolq": 300, "squad2": 299, "paws": 250, "multinli": 299,
    "civil_comments": 300, "aegis2": 250, "helpsteer2": 249,
    "summeval-relevance": 240, "summeval-consistency": 144, "pubmedqa": 250,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def materialize(raw_root, name, scratch):
    """Only unpack the needed archive member/split, retaining source row order."""
    source = raw_root / SOURCES[name][0]
    require(source.is_file(), f"Missing raw source: {source}")
    if name == "massive":
        return source
    target = scratch / (name + ".jsonl")
    if target.exists():
        return target
    if source.suffix == ".zip":
        suffix = "dev.jsonl" if name == "vitaminc" else "multinli_1.0_dev_matched.jsonl"
        with zipfile.ZipFile(source) as archive:
            members = [n for n in archive.namelist() if n == suffix or n.endswith("/" + suffix)]
            require(len(members) == 1, f"Expected exactly one {suffix} in {source}")
            with archive.open(members[0]) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    elif source.suffix == ".gz":
        with gzip.open(source, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        if source.suffix == ".parquet":
            import pyarrow.parquet as pq

            records = (row for batch in pq.ParquetFile(source).iter_batches()
                       for row in batch.to_pylist())
        else:
            records = json.loads(source.read_text(encoding="utf-8"))
            require(isinstance(records, list), f"Expected JSON array: {source}")
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
    return target


def adapt(record, suite, source, note):
    qdef = record["input"]["questions"]["decision"]
    reference = record["reference"]
    kind, criteria, target = qdef["type"], qdef["criteria"], reference["target"]
    if kind == "noul":
        options, gold = ["false", "true"], int(target)
    elif kind == "score":
        options, gold = [str(i) for i in range(len(criteria))], target
    else:
        options = list(criteria)
        gold = options.index(target)
    soft = None
    metadata = {
        "task_id": suite, "source": dict(source, row_id=record["id"]),
        "adaptation": note, "upstream_reference": reference,
        "upstream_family": record["family"], "domain": record["domain"],
        "pretraining_overlap": "unknown",
    }
    if suite == "boolq":
        metadata["source_overlap"] = {"S1": True, "S0": False}
    if "distribution" in reference:
        require(suite in ("multinli", "civil_comments"), "Unexpected distribution source")
        if suite == "multinli":
            require(reference["annotations"] == 5, "MultiNLI requires five human votes")
        soft = [reference["distribution"][key] for key in options]
        metadata["soft_target_kind"] = "human-votes"
        metadata["soft_target_semantics"] = (
            "Five original annotator labels; count per option divided by five."
            if suite == "multinli" else
            "Published toxicity rater fraction; rater count unpublished; false = 1 - toxicity."
        )
    score = None
    if kind == "score":
        score = reference["expert_mean"] - 1 if "expert_mean" in reference else float(target)
    if suite.startswith("summeval-"):
        metadata["target_is_derived_rounded_mean"] = True
        metadata["adaptation"] += (
            " Gold index preserves the official upstream rounded-half-up expert-mean target, "
            "not an individual vote. Gold score is expert_mean - 1; soft remains null. "
            "Hard accuracy measures agreement with the official rounded target; "
            "MAE measures error against the unrounded mean on the zero-based scale."
        )
    return {
        "id": f"nimble/{suite}/{record['id']}",
        "group_id": f"nimble/{suite}/{record['family']}",
        "state": record["input"]["state"], "qdef": qdef,
        "gold_idx": gold, "soft": soft, "gold_score": score,
        "option_ids": options, "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-subset", action="append", choices=tuple(SUBSETS), default=[])
    args = parser.parse_args()
    subsets = {name: count for name, count in SUBSETS.items() if name not in args.exclude_subset}
    require(bool(subsets), "No selected Nimble subsets")
    raw_root = args.raw_root.resolve()
    upstream = raw_root / "upstream"
    sys.path.insert(0, str(upstream))
    from nimble.datasets.public_benchmarks import jsonl_line, load_sources

    modules = load_sources()
    manifests = upstream / "docs/assets/public-benchmarks/subsets"
    specs, checksums = [], []
    prepared = {}
    with tempfile.TemporaryDirectory(prefix="nimble-public13-") as scratch_dir:
        for suite, count in subsets.items():
            manifest = json.loads((manifests / (suite + "-manifest.json")).read_text())
            name, subset, ids = manifest["dataset"], manifest["subset"], manifest["ids"]
            require(len(ids) == len(set(ids)) == manifest["count"] == count,
                    f"Invalid official manifest count: {suite}")
            if suite == "massive-de-DE":
                en = json.loads((manifests / "massive-en-US-manifest.json").read_text())
                require(ids == en["ids"], "MASSIVE locales must use identical IDs")
            module = modules[name]
            source_path = materialize(raw_root, name, Path(scratch_dir))
            wanted, selected, seen = set(ids), {}, set()
            for raw in module.rows(source_path, subset):
                record = module.record(raw, subset)
                if record is None:
                    continue
                identifier = record["id"]
                require(identifier not in seen, f"Duplicate source ID: {identifier}")
                seen.add(identifier)
                if identifier in wanted:
                    selected[identifier] = record
            require(set(selected) == wanted, f"{suite}: missing official IDs {sorted(wanted - set(selected))[:10]}")
            records = [selected[identifier] for identifier in ids]
            require(len({r["family"] for r in records}) == manifest["families"],
                    f"Official family count differs: {suite}")
            digest = hashlib.sha256("".join(jsonl_line(r) for r in records).encode()).hexdigest()
            checksums.append({"subset": suite, "official_dataset_sha256": manifest["dataset_sha256"],
                              "rebuilt_dataset_sha256": digest, "exact_match": digest == manifest["dataset_sha256"]})
            source = {"dataset": module.SOURCE_URL, "revision": SOURCES[name][2],
                      "split": SOURCES[name][1], "subset": subset, "license": module.LICENSE,
                      "nimble_revision": UPSTREAM_REVISION, "raw_file": SOURCES[name][0]}
            note = (module.NOTE + " Official pinned converter preserves state, instructions and criteria; "
                    "exact committed IDs in manifest order, no resampling or tokenizer refiltering.")
            rows = [adapt(r, suite, source, note) for r in records]
            prepared[suite] = rows
            specs.append({"name": "nimble_" + suite, "path": f"data/nimble_{suite}.jsonl",
                          "category": rows[0]["qdef"]["type"], "expected_count": count,
                          "source": source, "protocol": {"selection": "Exact official manifest IDs",
                          "adaptation": rows[0]["metadata"]["adaptation"],
                          "families": manifest["families"], "pretraining_overlap": "unknown"}})
    total = sum(subsets.values())
    require(sum(map(len, prepared.values())) == total, "Selected Nimble subset count differs")
    output = args.output_dir
    paths = [output / spec["path"] for spec in specs]
    profile_path = output / "full_eval_v2_nimble.json"
    report_path = output / "nimble_upstream_checksums.json"
    require(not any(p.exists() for p in paths + [profile_path, report_path]), "Refusing to overwrite Nimble outputs")
    (output / "data").mkdir(parents=True, exist_ok=True)
    for suite, path in zip(subsets, paths):
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in prepared[suite]:
                stream.write(jsonl_line(row))
    profile_path.write_text(json.dumps({"schema_version": "full_eval_v2",
                            "profile_id": "nimble_public13_official", "suites": specs}, indent=2) + "\n")
    report_path.write_text(json.dumps(checksums, indent=2) + "\n")
    print(json.dumps({"profile": str(profile_path), "count": total,
                      "upstream_checksum_matches": sum(c["exact_match"] for c in checksums),
                      "checksums": str(report_path)}))


if __name__ == "__main__":
    main()
