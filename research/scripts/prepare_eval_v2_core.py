"""Convert staged public core sources offline; run on the remote preparation host."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(text):
    return json.loads(text, object_pairs_hook=unique_object)


def option_ids(question):
    kind, criteria = question["type"], question.get("criteria")
    require(question["instructions"] is not None, "Missing instructions")
    if kind == "choice":
        require(isinstance(criteria, (dict, list)), "Invalid choice criteria")
        ids = list(criteria)
    elif kind == "score":
        require(isinstance(criteria, list), "Invalid score criteria")
        ids = [str(i) for i in range(len(criteria))]
    elif kind == "noul":
        require(criteria is None or isinstance(criteria, dict)
                and set(criteria) <= {"false", "true"}, "Invalid noul criteria")
        ids = ["false", "true"]
    else:
        raise ValueError(f"Unsupported primitive: {kind}")
    require(len(ids) >= 2 and all(isinstance(x, str) and x.strip() for x in ids)
            and len(ids) == len(set(ids)), "Invalid option identities")
    return ids


def question(raw, kind=None):
    require(set(raw) <= {"type", "instructions", "criteria"}, "Unsupported question fields")
    return {"type": kind or raw["type"], "instructions": raw["instructions"],
            "criteria": raw.get("criteria")}


def row(identity, group, state, qdef, gold, metadata, soft=None):
    ids = option_ids(qdef)
    require(isinstance(state, (str, dict, list)), f"{identity}: invalid state")
    require(gold is None or type(gold) is int and 0 <= gold < len(ids),
            f"{identity}: invalid hard label")
    if soft is not None:
        require(len(soft) == len(ids) and all(type(p) in (int, float) and math.isfinite(p)
                and 0 <= p <= 1 for p in soft) and abs(math.fsum(soft) - 1) <= 1e-6,
                f"{identity}: invalid upstream distribution (not renormalized)")
    require(gold is not None or soft is not None, f"{identity}: no supported target")
    return dict(id=identity, group_id=group, state=state, qdef=qdef, gold_idx=gold,
                soft=soft, gold_score=None, option_ids=ids, metadata=metadata)


def source(dataset, revision, split, path, license_name):
    require(len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
            f"{dataset}: supply the full frozen Git commit")
    return dict(dataset=dataset, revision=revision, split=split, file=path, license=license_name)


def jevbench(root, revision):
    suites = []
    for tier, count in (("original", 72), ("easy", 48), ("hard", 111)):
        name = "jevbench_public_" + tier
        path = f"datasets/public/{tier}.jsonl"
        provenance = source("fstandhartinger/jevbench", revision, "public", path, "MIT")
        rows = []
        for line_number, line in enumerate((root / path).read_text(encoding="utf-8").splitlines(), 1):
            require(bool(line.strip()), f"{path}:{line_number}: blank source row")
            raw = read_json(line)
            identity = name + "/" + raw["id"]
            require(raw["split"] == "public", f"{identity}: not a public item")
            qdef = question(raw["question"])
            ids = option_ids(qdef)
            labels = ["no", "yes"] if qdef["type"] == "noul" else ids
            require(len(raw["labels"]) == len(labels) and set(raw["labels"]) == set(labels),
                    f"{identity}: labels do not match rendered criteria")
            expected = raw.get("expected")
            if qdef["type"] == "score" and expected is not None:
                require(type(expected) is int, f"{identity}: score must be an index")
                gold = expected
            else:
                gold = labels.index(expected) if expected is not None else None
            upstream = raw.get("provenance", {})
            adaptation = (
                "State, instructions and criteria retain source order and values. Choice targets "
                "are mapped by label to criterion order, which can differ from source labels order. "
                "Noul no/yes map to false/true; score expected is a zero-based level index. "
                "Hard truth is only upstream expected, never derived from a soft target."
            )
            meta = dict(task_id=raw["family"], source=dict(provenance, row_id=raw["id"],
                        line=line_number), adaptation=adaptation, tier=tier,
                        upstream_labels=raw["labels"], upstream_expected=expected,
                        upstream_group=raw.get("group"), upstream_provenance=upstream)
            soft = None
            if "gold_probs" in upstream:
                require(raw["family"] == "probability", f"{identity}: unclassified soft semantics")
                probabilities = upstream["gold_probs"]
                require(isinstance(probabilities, dict) and set(probabilities) == set(labels),
                        f"{identity}: mismatched probability labels")
                soft = [probabilities[label] for label in labels]
                meta["soft_target_kind"] = "exact-mechanism"
                meta["adaptation"] += (
                    " Soft values are the published gold_probs, including upstream rounding, "
                    "mapped by label without renormalization; HARD-TIER.md defines their countable "
                    "mechanism. The original arithmetic rationale is in upstream_provenance. "
                    "The separate expected label is the upstream most-likely outcome, not a "
                    "sampled outcome or a human vote."
                )
            group = name + "/" + (raw.get("group") or raw["id"])
            rows.append(row(identity, group, raw["state"], qdef, gold, meta, soft))
        require(len(rows) == count, f"{name}: expected {count} public decisions, got {len(rows)}")
        suites.append((name, rows, "typed_decision", provenance, {
            "selection": "All rows of this one public tier, in file order; no historical copies",
            "adaptation": "Native typed questions, label-key mapping to unchanged criterion order",
            "scope": "Public diagnostic only; not official JevBench 534 or v1.4.2 headline scoring",
            "exclusions": "No published tier rows excluded; sealed and nonpublic sources not requested",
        }))
    return suites


def jabr(root, revision):
    path = "cases/v2.toml"
    with (root / path).open("rb") as stream:
        tasks = tomllib.load(stream)["task"]
    require(len(tasks) == 49 and len({task["id"] for task in tasks}) == 49,
            "jabr v2 must contain 49 unique tasks")
    provenance = source("jabr/classifier-benchmark", revision, "v2", path, "CC0-1.0")
    name = "jabr_v2"
    rows = []
    for task in tasks:
        qdef = question(task["question"], task["type"])
        ids = option_ids(qdef)
        for index, case in enumerate(task["cases"]):
            identity = f"{name}/{task['id']}/{index}"
            expected = case["expected"]
            if task["type"] == "noul":
                require(type(expected) is bool, f"{identity}: noul expected must be boolean")
                gold = int(expected)
            elif task["type"] == "score":
                require(type(expected) is int, f"{identity}: score expected must be level index")
                gold = expected
            else:
                gold = ids.index(expected)
            meta = dict(task_id=task["id"], source=dict(provenance, row_id=f"{task['id']}/{index}"),
                        adaptation="Unchanged state, question instructions and ordered rubric; "
                        "choice key to criterion index; boolean to false/true; score expected "
                        "already denotes zero-based level index, not the numeral in the level text.",
                        upstream_expected=expected, extends=task.get("extends"),
                        label_origin="Synthetic LLM committee generated and cross-checked; not human labels")
            rows.append(row(identity, identity, case["state"], qdef, gold, meta))
    require(len(rows) == 866, f"jabr v2: expected 866 cases, got {len(rows)}")
    return [(name, rows, "typed_decision", provenance, {
        "selection": "All 49 tasks and 866 cases in TOML order, including inline grammar_issue cases",
        "adaptation": "Native typed rubric and expected label semantics; no v1 duplication or sampling",
        "label_origin": "Synthetic LLM committee, not human annotation",
        "exclusions": "None from v2; v1 and external generated source suites not included",
    })]


def clinc(root, revision):
    path = "data/data_full.json"
    raw = read_json((root / path).read_text(encoding="utf-8"))
    counts = Counter(label for _, label in raw["test"])
    require(len(counts) == 150 and set(counts.values()) == {30},
            "CLINC test must contain 150 intents with 30 examples each")
    require(len(raw["oos_test"]) == 1000 and all(label == "oos" for _, label in raw["oos_test"]),
            "CLINC oos_test must contain 1000 oos examples")
    require("oos" not in counts, "CLINC test unexpectedly contains OOS")
    labels = list(counts) + ["oos"]
    criteria = dict.fromkeys(labels)
    criteria["oos"] = "Out of scope: the utterance matches none of the 150 supported intents."
    qdef = dict(type="choice", instructions="Classify the intent of this utterance. Select oos "
                "if it matches none of the supported intents.", criteria=criteria)
    name = "clinc150_oos_seen_source"
    provenance = source("clinc/oos-eval", revision, "test+oos_test", path, "CC-BY-3.0")
    rows = []
    for split in ("test", "oos_test"):
        for index, (state, label) in enumerate(raw[split]):
            identity = f"{name}/{split}/{index}"
            meta = dict(task_id="intent_classification", source=dict(provenance, split=split,
                        row_id=str(index)), adaptation="Exact utterance and intent label; no upstream "
                        "question/rubric exists. Added fixed neutral classification instruction. "
                        "All 150 original label strings, in first-occurrence test order, followed "
                        "by explicit 151st oos candidate; only oos gets a definition.",
                        intent=label, diagnostic="seen-source for both S0 and S1; not unseen-source evidence")
            rows.append(row(identity, identity, state, qdef, labels.index(label), meta))
    return [(name, rows, "classification", provenance, {
        "selection": "All 4500 test rows followed by all 1000 oos_test rows; original within-split order",
        "adaptation": "151-way choice; exact intent strings; fixed neutral instruction; explicit OOS",
        "option_ids": labels,
        "label_order": "First occurrence in frozen test split, then oos; never sorted or sampled",
        "diagnostic": "Seen-source for S0 and S1, not an unseen-source generalization benchmark",
        "exclusions": "No test rows excluded; train, val, oos_train and oos_val not evaluated",
    })]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path,
                        help="Contains jevbench/, classifier-benchmark/, and optionally oos-eval/")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="Parent bundle directory; creates data/ and core_suites.json")
    parser.add_argument("--jevbench-revision")
    parser.add_argument("--jabr-revision")
    parser.add_argument("--clinc-revision")
    parser.add_argument("--include-source", action="append", choices=("jevbench", "jabr", "clinc"))
    args = parser.parse_args()
    selected = args.include_source or ["jevbench", "jabr", "clinc"]
    suites = []
    for name, builder, directory, revision in (
        ("jevbench", jevbench, "jevbench", args.jevbench_revision),
        ("jabr", jabr, "classifier-benchmark", args.jabr_revision),
        ("clinc", clinc, "oos-eval", args.clinc_revision),
    ):
        if name in selected:
            require(revision is not None, f"Missing --{name}-revision")
            suites += builder(args.raw_root / directory, revision)
    identities = [item["id"] for _, rows, _, _, _ in suites for item in rows]
    require(len(identities) == len(set(identities)), "Duplicate decision IDs")
    destinations = [args.output_root / "data" / (name + ".jsonl") for name, *_ in suites]
    descriptor = args.output_root / "core_suites.json"
    require(all(not path.exists() for path in [*destinations, descriptor]),
            "Refusing to overwrite prepared core artifacts")
    (args.output_root / "data").mkdir(parents=True, exist_ok=True)
    specs = []
    for (name, rows, category, provenance, protocol), destination in zip(suites, destinations):
        with destination.open("x", encoding="utf-8") as stream:
            for item in rows:
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        specs.append(dict(name=name, path="data/" + destination.name, category=category,
                          expected_count=len(rows), source=provenance,
                          protocol=dict(protocol, raw_example_count=len(rows),
                                        primitive_counts=dict(Counter(r["qdef"]["type"] for r in rows)),
                                        soft_target_count=sum(r["soft"] is not None for r in rows))))
    with descriptor.open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": "full_eval_v2", "suites": specs}, stream,
                  indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"descriptor": str(descriptor), "counts": {
        spec["name"]: spec["expected_count"] for spec in specs}}))


if __name__ == "__main__":
    main()
