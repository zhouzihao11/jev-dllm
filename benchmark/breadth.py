"""Bridge input-only inference to the frozen official Breadth-v1 report."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from common import exact, indexed, input_bundle, load, option_count, require, rows, save, save_rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen(args):
    source = args.kev_source.expanduser().resolve()
    suite = args.suite.expanduser().resolve()
    expected = load(source / "evals" / "breadth-v1" / "manifest.json")
    manifest = load(suite / "manifest.json")
    require(exact(manifest, expected), "Suite differs from supplied official expected manifest")
    require(manifest["version"] == "breadth-v1" and len(manifest["datasets"]) == 14
            and len(manifest["areas"]) == 5, "Expected full original Breadth-v1")
    require(args.split != "test" or args.allow_test, "Test requires --allow-test")
    path = suite / (args.split + ".jsonl")
    spec = manifest["files"][path.name]
    require(path.is_file(), "Frozen partition must already exist; no acquisition is performed")
    require(digest(path) == spec["sha256"], "Official partition SHA mismatch")
    records = rows(path)
    require(len(records) == spec["records"]
            and sum(len(r["questions"]) for r in records) == spec["questions"],
            "Official partition counts mismatch")
    counts = Counter(r["_meta"]["source"] for r in records)
    require(dict(counts) == spec["by_source"] and set(counts) == set(manifest["datasets"]),
            "Official dataset counts mismatch")
    identities = set()
    for record in records:
        meta = record["_meta"]
        require(isinstance(meta["id"], str) and meta["id"] and meta["id"] not in identities,
                "Invalid/duplicate frozen record ID")
        identities.add(meta["id"])
        require(meta["variant"] == "clean" and isinstance(meta["group_id"], str),
                "Expected original clean record/group")
        for qid, question in record["questions"].items():
            require(isinstance(qid, str) and qid, "Invalid question ID")
            definition(question)
            labels(question)
    return source, suite, manifest, records


def identity(rid, qid):
    # JSON pair encoding is reversible even when either ID contains delimiters.
    return json.dumps([rid, qid], ensure_ascii=True, separators=(",", ":"))


def definition(question):
    require(question["type"] in ("choice", "noul"), "Breadth supports choice/noul only")
    qdef = dict(type=question["type"], instructions=question["instructions"], criteria=question.get("criteria"))
    option_count(qdef)
    return qdef


def labels(question):
    # kev.benchmark.labels / kev.api.question_keys for these two data classes.
    keys = list(question["criteria"]) if question["type"] == "choice" else ["false", "true"]
    label = keys.index(question["label"]) if question["type"] == "choice" else int(question["label"])
    require(0 <= label < len(keys), "Invalid frozen label")
    return keys, label


def decisions(manifest, records):
    for dataset in manifest["datasets"]:
        for record in records:
            if record["_meta"]["source"] == dataset:
                for qid, question in record["questions"].items():
                    yield identity(record["_meta"]["id"], qid), record, qid, question


def prepare(args):
    _, suite, official, records = frozen(args)
    inputs, metadata, targets = [], [], []
    for uid, record, qid, question in decisions(official, records):
        meta = record["_meta"]
        keys, label = labels(question)
        inputs.append(dict(id=uid, suite=meta["source"], state=record["state"], qdef=definition(question)))
        metadata.append(dict(id=uid, record_id=meta["id"], question_id=qid,
                             group_id=meta["group_id"], record_metadata=meta, task=question["src"]))
        targets.append(dict(id=uid, option_ids=keys, gold_idx=label))
    indexed(inputs)
    counts = Counter(r["suite"] for r in inputs)
    manifest = dict(schema_version="benchmark_core_v1", profile_id="breadth-v1__" + args.split,
                    source_profile_id="breadth-v1", dataset_scope="breadth_external_full_original",
                    split=args.split, retained_count=len(inputs),
                    official_manifest_sha256=digest(suite / "manifest.json"),
                    partition=official["files"][args.split + ".jsonl"],
                    suites=[dict(name=name, retained_count=counts[name], **spec)
                            for name, spec in official["datasets"].items()],
                    files=dict(inputs="inputs.jsonl", metadata="metadata.jsonl", targets="targets.jsonl"))
    args.out.mkdir(parents=True, exist_ok=False)
    save_rows(args.out / "inputs.jsonl", inputs)
    save_rows(args.out / "metadata.jsonl", metadata)
    save_rows(args.out / "targets.jsonl", targets)
    save(args.out / "manifest.json", manifest)


def vector(values, count):
    require(isinstance(values, list) and len(values) == count
            and all(type(v) in (int, float) and math.isfinite(v) for v in values),
            "Invalid/nonfinite logit or probability vector")
    return values


def softmax(logits):
    maximum = max(logits)
    values = [math.exp(v - maximum) for v in logits]
    total = math.fsum(values)
    return [v / total for v in values]


def distribution(values, count):
    p = vector(values, count)
    total = math.fsum(p)
    # Exactly kev.benchmark.validate_distribution: no flooring or log(p) fallback.
    require(all(0 <= v <= 1 for v in p) and total > 0
            and abs(total - 1) <= max(1e-5, count * 0.005 + 1e-8), "Invalid probability sum/range")
    return [v / total for v in p], total


def native(record, qid, question, p, logits, temperature, total=1.0):
    meta = record["_meta"]
    keys, label = labels(question)
    return dict(id=meta["id"], question=qid, group=meta["group_id"], source=meta["source"],
                task=question["src"], type=question["type"], variant=meta["variant"],
                keys=keys, label=label, p=p, logits=logits, inference_temperature=temperature,
                control_id=meta.get("control_id"), pair_id=meta.get("pair_id"), sibling=meta.get("sibling"),
                parent=meta.get("parent_id") or meta["id"], raw_probability_sum=total,
                zero_count=sum(v == 0 for v in p))


def export(args):
    _, suite, official, records = frozen(args)
    manifest, inputs = input_bundle(args.data_root)
    expected = list(decisions(official, records))
    require(manifest["profile_id"] == "breadth-v1__" + args.split
            and manifest.get("official_manifest_sha256") == digest(suite / "manifest.json")
            and len(inputs) == len(expected), "Bundle differs from frozen partition")
    for row, (uid, record, _, question) in zip(inputs, expected):
        require(exact(row, dict(id=uid, suite=record["_meta"]["source"], state=record["state"],
                                qdef=definition(question))), "Bundle input/ID/order mismatch")
    predictions = indexed(rows(args.predictions))
    require(set(predictions) == {uid for uid, _, _, _ in expected},
            "Prediction IDs must match full partition, including unsupported rows")
    output, unsupported, statuses = [], [], Counter()
    for uid, record, qid, question in expected:
        prediction = predictions[uid]
        require(prediction["suite"] == record["_meta"]["source"], "Prediction suite mismatch")
        status = prediction["status"]
        statuses[status] += 1
        if status != "ok":
            require(isinstance(status, str) and status.startswith("unsupported_"), "Unexpected prediction status")
            unsupported.append(dict(id=uid, status=status, source=prediction["suite"]))
            continue
        keys, _ = labels(question)
        p, total = distribution(prediction["probs"], len(keys))
        temperature = 1.0
        if args.backend == "kev":
            meta = prediction["input_metadata"]
            require(meta["option_keys"] == keys, "Kev option order mismatch")
            logits = vector(meta["logits"], len(keys))
            temperature = meta["inference_temperature"]
        elif args.backend in ("dream", "dllm"):
            raw = prediction["raw_logits"]
            require(isinstance(raw, list) and len(raw) == (1 if question["type"] == "noul" else len(keys)),
                    "Yes/No logit shape mismatch")
            pairs = [vector(pair, 2) for pair in raw]
            logits = pairs[0][::-1] if question["type"] == "noul" else [yes - no for yes, no in pairs]
        else:
            logits = vector(prediction["raw_logits"], len(keys))
        vector(logits, len(keys))
        require(type(temperature) in (int, float) and math.isfinite(temperature) and temperature > 0,
                "Invalid inference temperature")
        require(all(abs(a - b) <= 1e-5 for a, b in zip(p, softmax(logits))),
                "Saved probabilities disagree with backend logits")
        if args.temperature is not None:
            # Kev stores served z/T; restore raw z before applying the absolute fixed T.
            logits = [v * temperature / args.temperature for v in logits]
            vector(logits, len(keys))
            temperature, p, total = args.temperature, softmax(logits), 1.0
        output.append(native(record, qid, question, p, logits, temperature, total))
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "rows.json", output)
    save(args.out / "export.json", dict(split=args.split, backend=args.backend, population=len(expected),
                                       answered=len(output), status_counts=dict(statuses), unsupported=unsupported,
                                       temperature_override=args.temperature, temperature_fit=False))


def named(specs):
    result = {}
    for spec in specs:
        name, separator, path = spec.partition("=")
        require(separator and name and path and name not in result, "Expected unique NAME=PATH")
        result[name] = Path(path).expanduser().resolve()
    return result


def validate_native(data, expected):
    seen = set()
    for row in data:
        key = row["id"], row["question"]
        require(key in expected and key not in seen, "Unknown/duplicate native row ID")
        seen.add(key)
        record, question = expected[key]
        keys, label = labels(question)
        for field, value in dict(keys=keys, label=label, source=record["_meta"]["source"],
                                 group=record["_meta"]["group_id"], task=question["src"],
                                 type=question["type"], variant="clean").items():
            require(row[field] == value, "Native row metadata mismatch: " + field)
        p, total = distribution(row["p"], len(keys))
        require(abs(total - 1) < 1e-6, "Native rows must already be normalized")
        logits = vector(row["logits"], len(keys))
        require(all(abs(a - b) <= 1e-5 for a, b in zip(p, softmax(logits))), "Native p/logits mismatch")
        temperature = row["inference_temperature"]
        require(type(temperature) in (int, float) and math.isfinite(temperature) and temperature > 0,
                "Invalid native inference temperature")


def comparison_markdown(comparison):
    def cell(value):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    def table(headers, data):
        return "\n".join(["| " + " | ".join(headers) + " |",
                          "| " + " | ".join(["---"] * len(headers)) + " |"]
                         + ["| " + " | ".join(cell(v) for v in row) + " |" for row in data])

    systems = comparison["systems"]
    provenance = comparison["provenance"]
    first = next(iter(systems.values()))
    sections = ["# Breadth-v1 full original test comparison",
                "All values are displayed from the merged official report JSON; no metrics are recomputed. "
                "All 14 datasets and five areas retain the original scoring population, with no training-overlap "
                "exclusions. Missing answers count as wrong in coverage-adjusted scores. Calibration "
                "(ECE10, NLL, Brier) uses answered questions only.",
                "Dataset `accuracy` is question accuracy over all original questions. `case_exact` is record "
                "accuracy: every question in the original record must be answered correctly. Coverage is the "
                "official answered share of these scoring units, not necessarily question coverage. "
                "Area scores and skills are official dataset means; index and raw_index use the official 0-100 scale.",
                "## Overall"]
    data = []
    for name, system in systems.items():
        overall = system["overall"]
        calibration = overall["calibration"]
        data.append([name, provenance[name], overall["index"], overall["raw_index"],
                     calibration["ece"], overall["mean_dataset_ece"], calibration["nll"],
                     calibration["brier"], overall["answered_questions"], overall["questions"]])
    sections.append(table(["System", "Provenance", "Index", "Raw index", "Pooled ECE10",
                           "Mean dataset ECE10", "Pooled NLL", "Pooled Brier", "Answered questions",
                           "Original questions"], data))
    sections.append("## Five areas")
    for area, spec in first["areas"].items():
        sections.append("### " + cell(spec["label"]) + " (" + cell(area) + ")")
        sections.append(table(["System", "Provenance", "Score", "Skill", "Pooled ECE10"],
                              [[name, provenance[name], system["areas"][area]["score"],
                                system["areas"][area]["skill"], system["areas"][area]["calibration"]["ece"]]
                               for name, system in systems.items()]))
    sections.append("## All 14 datasets")
    for dataset, spec in first["datasets"].items():
        sections.append("### " + cell(dataset) + " (" + cell(spec["area"]) + "; "
                        + cell(spec["metric"]) + "; original scoring units: " + cell(spec["units"]) + ")")
        data = []
        for name, system in systems.items():
            scored = system["datasets"][dataset]
            calibration = scored["calibration"]
            data.append([name, provenance[name], scored["score"], scored["skill"], scored["answered"],
                         calibration["ece"], calibration["nll"], calibration["brier"]])
        sections.append(table(["System", "Provenance", "Score", "Skill", "Coverage", "ECE10", "NLL", "Brier"], data))
    sections.extend(["## Uncertainty", comparison["uncertainty_scope"],
                     "Published official references have no paired uncertainty without local rerun rows. "
                     "The following is the unchanged official local-only bootstrap output:",
                     "```json\n" + json.dumps(comparison["uncertainty"], ensure_ascii=True,
                                              allow_nan=False, indent=2) + "\n```"])
    return "\n\n".join(sections) + "\n"


def report(args):
    source, suite, official, records = frozen(args)
    require(args.split == "test" and args.allow_test, "Comparison reports require full test --allow-test")
    local, references = named(args.result), named(args.reference)
    selected = set(args.reference_system) if args.reference_system is not None else None
    reference_reports = {tag: load(path) for tag, path in references.items()}
    if selected is not None:
        available = {name for reference in reference_reports.values() for name in reference["systems"]}
        require(selected <= available, "Requested reference systems absent: " + ", ".join(sorted(selected - available)))
    expected = {(r["_meta"]["id"], qid): (r, q) for r in records for qid, q in r["questions"].items()}
    for directory in local.values():
        validate_native(load(directory / "rows.json"), expected)
    require(not args.out.exists(), "Report output must be new")
    command = [sys.executable, str(source / "scripts" / "breadth_report.py"),
               "--suite", str(suite), "--split", "test", "--allow-test", "--bootstrap", "2000",
               "--out", str(args.out)]
    for name, directory in local.items():
        command.extend(["--result", name + "=" + str(directory)])
    subprocess.run(command, check=True)
    comparison = load(args.out / "report.json")
    comparison["provenance"] = {name: "local rerun" for name in local}
    for tag, path in references.items():
        reference = reference_reports[tag]
        require(reference["split"] == "test" and reference["suite_manifest_sha256"] == digest(suite / "manifest.json")
                and exact(reference["formula"], official["scoring"]), "Reference suite/split/formula mismatch")
        for name, system in reference["systems"].items():
            if selected is not None and name not in selected:
                continue
            label = tag + ":" + name
            require(label not in comparison["systems"] and set(system["datasets"]) == set(official["datasets"])
                    and set(system["areas"]) == set(official["areas"])
                    and system["overall"]["questions"] == official["files"]["test.jsonl"]["questions"],
                    "Reference collision/incomplete panel")
            comparison["systems"][label] = system
            comparison["sources"][label] = dict(report=str(path), provenance="official reference", paired_uncertainty=False)
            comparison["provenance"][label] = "official reference"
    comparison["uncertainty_scope"] = "Official paired bootstrap2000 for local rerun rows only; no paired reference uncertainty"
    save(args.out / "comparison.json", comparison)
    (args.out / "comparison.md").write_text(comparison_markdown(comparison), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("prepare", prepare), ("export", export), ("report", report)):
        sub = commands.add_parser(name)
        sub.set_defaults(function=function)
        sub.add_argument("--kev-source", type=Path, required=True)
        sub.add_argument("--suite", type=Path, required=True)
        sub.add_argument("--split", choices=("development", "test"), default="test")
        sub.add_argument("--allow-test", action="store_true")
        sub.add_argument("--out", type=Path, required=True)
        if name == "export":
            sub.add_argument("--data-root", type=Path, required=True)
            sub.add_argument("--predictions", type=Path, required=True)
            sub.add_argument("--backend", choices=("dllm", "dream", "laya", "kev"), required=True)
            sub.add_argument("--temperature", type=float, help="Absolute fixed raw-logit temperature; never fitted")
        if name == "report":
            sub.add_argument("--result", action="append", required=True, help="NAME=local native rows directory")
            sub.add_argument("--reference", action="append", default=[], help="TAG=published official report JSON")
            sub.add_argument("--reference-system", nargs="+", help="Existing system names to include across reference reports only")
    args = parser.parse_args()
    if getattr(args, "temperature", None) is not None:
        require(math.isfinite(args.temperature) and args.temperature > 0, "Temperature must be positive/finite")
    args.function(args)


if __name__ == "__main__":
    main()
