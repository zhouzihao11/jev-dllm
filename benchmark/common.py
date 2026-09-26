"""Standard-library bundle IO; deliberately no model or dataset imports."""

from collections import Counter
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Nonfinite JSON number: " + value)


def finite_float(value):
    result = float(value)
    require(math.isfinite(result), "Nonfinite JSON number")
    return result


def decode(value):
    return json.loads(value, object_pairs_hook=unique_object,
                      parse_constant=reject_constant, parse_float=finite_float)


def load(path):
    return decode(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [decode(line) for line in stream]


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    indent=2) + "\n", encoding="utf-8")


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def save_rows(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in data:
            write_row(stream, row)


def bundled(root, name):
    root = Path(root).resolve()
    require(isinstance(name, str) and bool(name), "Missing relative bundle path")
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe bundle path")
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Missing/escaping bundle file: " + name)
    return path


def indexed(data):
    result = {}
    for row in data:
        identity = row["id"]
        require(isinstance(identity, str) and bool(identity) and identity not in result,
                "Invalid/duplicate ID: " + str(identity))
        result[identity] = row
    return result


def exact(a, b):
    # Mapping insertion order is semantically significant for choice criteria.
    return json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)


def option_count(qdef):
    require(set(qdef) == {"type", "instructions", "criteria"}, "Invalid qdef")
    require(qdef["type"] in ("choice", "score", "noul"), "Invalid primitive")
    require(qdef["instructions"] is not None, "Missing instructions")
    criteria = qdef["criteria"]
    if qdef["type"] == "noul":
        require(criteria is None or isinstance(criteria, dict)
                and set(criteria) <= {"false", "true"}, "Invalid noul criteria")
        return 2
    require(isinstance(criteria, (dict, list)) and len(criteria) >= 2, "Invalid criteria")
    if qdef["type"] == "score":
        require(isinstance(criteria, list), "Score levels must be ordered list")
    else:
        require(all(isinstance(x, str) and x.strip() for x in criteria)
                and len(set(criteria)) == len(criteria), "Invalid choice keys")
    return len(criteria)


def validate_target(row, qdef):
    k = option_count(qdef)
    ids = row["option_ids"]
    require(isinstance(ids, list) and len(ids) == k
            and all(isinstance(x, str) and x.strip() for x in ids)
            and len(set(ids)) == k, "Invalid ordered option IDs")
    gold, soft, score = (row[key] for key in ("gold_idx", "soft", "gold_score"))
    require(gold is None or type(gold) is int and 0 <= gold < k, "Invalid hard target")
    numeric = lambda x: type(x) in (int, float) and math.isfinite(x)
    if soft is not None:
        require(isinstance(soft, list) and len(soft) == k
                and all(numeric(x) and 0 <= x <= 1 for x in soft)
                and abs(math.fsum(soft) - 1) <= 1e-6, "Invalid soft target")
    require(score is None or qdef["type"] == "score" and numeric(score)
            and 0 <= score <= k - 1, "Invalid scalar target")
    require(any(x is not None for x in (gold, soft, score)), "Missing target")


def source_profile(path):
    path = Path(path).expanduser().resolve()
    profile = load(path)
    require(profile.get("schema_version") == "full_eval_v2", "Expected original full_eval_v2 profile")
    require(isinstance(profile.get("profile_id"), str) and profile["profile_id"], "Missing profile ID")
    datasets, names, identities = [], set(), set()
    require(isinstance(profile.get("suites"), list) and profile["suites"], "Missing suites")
    for spec in profile["suites"]:
        name = spec["name"]
        require(isinstance(name, str) and name and name not in names
                and all(c.isascii() and (c.isalnum() or c in "_-") for c in name), "Invalid suite name")
        names.add(name)
        data = rows(bundled(path.parent, spec["path"]))
        require(len(data) == spec["expected_count"], "Original count mismatch: " + name)
        for row in data:
            require(set(row) == {"id", "group_id", "state", "qdef", "gold_idx", "soft",
                                 "gold_score", "option_ids", "metadata"}, "Invalid original row keys")
            require(isinstance(row["id"], str) and row["id"] and row["id"] not in identities,
                    "Invalid/duplicate original ID")
            identities.add(row["id"])
            require(isinstance(row["group_id"], str) and row["group_id"], "Missing original group")
            require(isinstance(row["state"], (str, dict, list)), "Invalid original state")
            validate_target(row, row["qdef"])
            meta = row["metadata"]
            require(isinstance(meta, dict) and {"source", "adaptation"} <= set(meta), "Missing provenance")
            if row["soft"] is not None:
                require(meta.get("soft_target_kind") in ("human-votes", "exact-mechanism"),
                        "Unknown soft-target semantics")
        datasets.append((spec, data))
    return profile, datasets


def input_bundle(root):
    root = Path(root).expanduser().resolve()
    manifest = load(root / "manifest.json")
    require(manifest.get("schema_version") == "benchmark_core_v1"
            and manifest.get("profile_id") == "core_v1_candidate", "Wrong bundle schema/profile")
    data = rows(bundled(root, manifest["files"]["inputs"]))
    indexed(data)
    counts = Counter()
    for row in data:
        require(set(row) == {"id", "suite", "state", "qdef"}, "Input contains non-input fields")
        require(isinstance(row["state"], (str, dict, list)), "Invalid state")
        option_count(row["qdef"])
        counts[row["suite"]] += 1
    suites = manifest["suites"]
    require(len({s["name"] for s in suites}) == len(suites), "Duplicate suite")
    require(set(counts) <= {s["name"] for s in suites}
            and all(counts[s["name"]] == s["retained_count"] for s in suites)
            and len(data) == manifest["retained_count"], "Bundle counts mismatch")
    require([r["id"] for s in suites for r in data if r["suite"] == s["name"]]
            == [r["id"] for r in data], "Inputs must preserve suite-local order")
    return manifest, data


def selected_inputs(data, smoke, limit):
    require(type(limit) is int and limit > 0, "Smoke limit must be positive")
    counts, selected = Counter(), []
    for row in data:
        counts[row["suite"]] += 1
        if not smoke or counts[row["suite"]] <= limit:
            selected.append(row)
    return selected


def inherited():
    import importlib
    import sys

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "research" / "scripts"))
    sys.path.insert(0, str(repo))
    return importlib.import_module("bench_extended")
