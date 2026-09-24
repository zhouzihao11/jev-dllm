"""Construct canonical decisions from local, prepared training sources only.

Public interface: load_source_pools(source_dir, seed=42,
max_records_per_source=24000) -> (pools, report). No files are written. Missing
or unusable inputs produce diagnostics and empty affected pools, not substitutes.
The caller owns splitting, token-length filtering, and manifest revision metadata.
Only state and questions are model inputs; option_ids and metadata are internal.
"""

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA_VERSION = "shared_yesno_data_v1"
_SOURCES = ("clinc", "snli", "arc", "sgd")
_EXCLUDED_DOMAINS = {"banking", "credit_cards"}
_SGD_MAX_VIEWS = 8
_CLINC_ALIASES = {
    "pto_request": "request paid time off",
    "pto_request_status": "paid time off request status",
    "pto_balance": "paid time off balance",
    "pto_used": "paid time off used",
    "w2": "W-2 tax form",
    "w9": "W-9 tax form",
    "rollover_401k": "roll over a 401(k) retirement account",
    "uber": "request an Uber ride",
    "mpg": "vehicle miles per gallon",
    "what_is_your_name": "ask the assistant's name",
}
_OPTION_REFERENCE = re.compile(
    r"\b(?:all|none|both|neither|either)\s+(?:of\s+)?(?:the\s+)?"
    r"(?:above|below|these|those|options|answers|choices|statements)\b"
    r"|\b(?:all|none)\s+of\s+(?:them|these|those)\b"
    r"|\b(?:both|neither|either)\s+[A-E1-5]\s*(?:,|and|or|&)\s*[A-E1-5]\b"
    r"|\b[A-E1-5]\s*(?:and|or|&)\s*[A-E1-5]\b"
    r"|\([A-E1-5]\)\s*(?:,|and|or|&)\s*\([A-E1-5]\)"
    r"|\b(?:option|choice|answer|statement)\s+[A-E1-5]\b"
    r"|\b(?:above|below)\b", re.IGNORECASE)
_BARE_OPTION_REFERENCE = re.compile(r"\s*(?:all|none|both|neither)(?:\s+of\s+them)?[.!]?\s*", re.IGNORECASE)
_VISUAL_REFERENCE = re.compile(
    r"\b(?:figure|diagram|image|illustration|photograph|picture|table|graph|chart)\b"
    r"|\b(?:shown|pictured|depicted|illustrated)\b|<img\b", re.IGNORECASE)


def _digest(*parts):
    text = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rng(seed, *parts):
    return random.Random(int(_digest(seed, *parts), 16))


def _normal(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _identifier(value):
    return (_text(value) or isinstance(value, int)) and not isinstance(value, bool)


def _reject(report, reason, record_id, **extra):
    report["rejection_counts"][reason] += 1
    report["rejections"].append(dict(record_id=record_id, reason=reason, **extra))


def _read_json(path, report):
    if not path.is_file():
        report["missing_inputs"].append(str(path))
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if value is None:
            report["input_errors"].append({"path": str(path), "error": "Expected JSON data, got null"})
        return value
    except (OSError, UnicodeError, ValueError) as exc:
        report["input_errors"].append({"path": str(path), "error": str(exc)})
        return None


def _reservoir(path, name, seed, cap, report, observe=None):
    """Sample distinct source IDs, retaining file order, before view expansion."""
    sample, seen, eligible = [], set(), 0
    rng = _rng(seed, name, "reservoir")
    if not path.is_file():
        report["missing_inputs"].append(str(path))
        return sample
    try:
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                report["lines_read"] += 1
                try:
                    row = json.loads(line.decode("utf-8"))
                except (UnicodeError, ValueError) as exc:
                    _reject(report, "malformed_json", None, line=line_number, error=str(exc))
                    continue
                if not isinstance(row, dict):
                    _reject(report, "record_not_object", None, line=line_number)
                    continue
                rid = row.get("dialogue_id" if name == "sgd" else "record_id")
                if not _identifier(rid):
                    _reject(report, "missing_record_id", None, line=line_number)
                    continue
                rid = str(rid)
                if any(row[key] != "train" for key in ("original_split", "split") if key in row):
                    _reject(report, "non_train_split", rid, line=line_number)
                    continue
                # ARC IDs can legitimately recur in Easy and Challenge exports.
                identity = (str(row.get("config", "")), rid) if name == "arc" else rid
                if identity in seen:
                    _reject(report, "duplicate_source_id", rid, line=line_number)
                    continue
                seen.add(identity)
                if observe is not None and not observe(row, rid):
                    continue
                eligible += 1
                item = (line_number, row, rid)
                if len(sample) < cap:
                    sample.append(item)
                else:
                    index = rng.randrange(eligible)
                    if index < cap:
                        sample[index] = item
    except OSError as exc:
        report["input_errors"].append({"path": str(path), "error": str(exc)})
        return []
    report["reservoir_eligible_records"] = eligible
    report["reservoir_retained_records"] = len(sample)
    report["reservoir_omitted_records"] = eligible - len(sample)
    return [(row, rid) for _, row, rid in sorted(sample)]


def _decision(name, rid, group, view, family, state, instructions, options,
              correct, stratum, metadata=None, qtype="choice"):
    # options are (stable source ID, natural display key, description) triples.
    keys = [item[1] for item in options]
    ids = [item[0] for item in options]
    if not keys or len(set(keys)) != len(keys) or len(set(ids)) != len(ids):
        raise ValueError("Decision options must have unique IDs and display keys")
    if correct not in ids:
        raise ValueError("Decision has no explicitly annotated correct option")
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": "%s:%s" % (name, _digest(rid, view)),
        "group_id": "%s:%s" % (name, _digest(group)),
        "view_id": view,
        "source": {"name": name, "record_id": rid, "original_split": "train"},
        "family": family,
        "language": "en",
        "state": state,
        "questions": {"decision": {"type": qtype, "instructions": instructions,
                                    "criteria": {key: desc for _, key, desc in options}}},
        "gold": {"decision": {"kind": "hard", "probabilities": {
            key: float(oid == correct) for oid, key, _ in options}}},
        "option_ids": {"decision": ids},
        "metadata": dict(metadata or {}, sampling_stratum=stratum),
    }


def _clinc(rows, ontology, seed, report, k_choices=None):
    labels = sorted(ontology)
    display = {label: _CLINC_ALIASES.get(label, label.replace("_", " ")) for label in labels}
    report["ontology"] = [{"intent": x, "domain": ontology[x], "display": display[x]} for x in labels]
    if len(set(display.values())) != len(labels):
        report["input_errors"].append({"error": "CLINC verbalizations are not unique"})
        return []
    if len(labels) < 2:
        report["input_errors"].append({"error": "CLINC needs at least two allowed source intents"})
        return []
    choices = (4, 8, 16, 32, 64, 77, len(labels)) if k_choices is None else k_choices
    buckets = sorted({k for k in choices if 2 <= k <= len(labels)})
    if not buckets:
        raise ValueError("CLINC needs at least one valid candidate count")
    report["k_buckets"] = buckets
    report["candidate_policy"] = (
        "One view per retained record. Balanced shuffled K schedule. Equal mixture of "
        "uniformly shuffled and domain-clustered directories; choose uniformly among "
        "the K circular windows containing the target, then shuffle displayed options. "
        "For a fixed directory each label belongs to exactly K windows: candidate "
        "sets do not designate a special target position or unique target-domain member. "
        "Class priors and source semantics can still affect difficulty.")
    schedule = [buckets[i % len(buckets)] for i in range(len(rows))]
    _rng(seed, "clinc", "k_schedule").shuffle(schedule)
    pools = []
    for (row, rid), k in zip(rows, schedule):
        label = row["intent"]
        rng = _rng(seed, "clinc", rid)
        directory = labels[:]
        clustered = bool(rng.randrange(2))
        if clustered:
            domains = defaultdict(list)
            for item in directory:
                domains[ontology[item]].append(item)
            domain_order = sorted(domains)
            rng.shuffle(domain_order)
            directory = []
            for domain in domain_order:
                rng.shuffle(domains[domain])
                directory.extend(domains[domain])
        else:
            rng.shuffle(directory)
        start = directory.index(label) - rng.randrange(k)
        candidates = [directory[(start + i) % len(directory)] for i in range(k)]
        rng.shuffle(candidates)
        pools.append(_decision(
            "clinc", rid, _normal(row["text"]), "intent_k%d" % k, "intent_classification",
            row["text"], "Which original intent is closest to the user's utterance?",
            [(item, display[item], None) for item in candidates], label, "intent:" + label,
            {"domain": row["domain"], "k": k, "directory_policy": "domain_clustered" if clustered else "uniform"}))
    return pools


def _snli(rows, seed, report):
    labels = ("entailment", "contradiction", "neutral")
    descriptions = {
        "entailment": "The premise entails the hypothesis.",
        "contradiction": "The premise contradicts the hypothesis.",
        "neutral": "The premise neither entails nor contradicts the hypothesis.",
    }
    valid = []
    for row, rid in rows:
        if row.get("label") not in labels or not all(_text(row.get(k)) for k in ("premise", "hypothesis")):
            _reject(report, "invalid_label_or_empty_text", rid)
            continue
        valid.append((row, rid))
    # Alternate within each gold label after a seeded shuffle, not from text cues.
    tasks, by_label = {}, defaultdict(list)
    for row, rid in valid:
        by_label[row["label"]].append(rid)
    for label, ids in sorted(by_label.items()):
        rng = _rng(seed, "snli", label, "binary_tasks")
        rng.shuffle(ids)
        offset = rng.randrange(2)
        for index, rid in enumerate(ids):
            tasks[rid] = ("entailment", "contradiction")[(index + offset) % 2]
    pool = []
    for row, rid in valid:
        explicit = row.get("group_key")
        has_group = _identifier(explicit)
        group = ("source_group", str(explicit)) if has_group else ("premise", _normal(row["premise"]))
        report["grouping_counts"]["source_group" if has_group else "normalized_premise_fallback"] += 1
        state = {"premise": row["premise"], "hypothesis": row["hypothesis"]}
        options = [(label, label, descriptions[label]) for label in labels]
        _rng(seed, "snli", rid, "order").shuffle(options)
        label = row["label"]
        pool.append(_decision("snli", rid, group, "three_way", "natural_language_inference",
                              state, "How does the premise relate to the hypothesis?", options,
                              label, "three_way:" + label))
        task = tasks[rid]
        verb = "entail" if task == "entailment" else "contradict"
        answer = "true" if label == task else "false"
        pool.append(_decision(
            "snli", rid, group, "binary_" + task, "natural_language_inference", state,
            "Does the premise %s the hypothesis? Answer about this relation, not whether "
            "the hypothesis is true in the real world." % verb,
            [("false", "false", "The specified relation does not hold."),
             ("true", "true", "The specified relation holds.")],
            answer, "binary_%s:%s" % (task, answer),
            {"source_label": label, "binary_relation": task}, qtype="noul"))
    return pool


def _arc(rows, seed, report):
    pool = []
    for row, rid in rows:
        question, choices = row.get("question"), row.get("choices")
        if not _text(question) or not isinstance(choices, list) or len(choices) < 2:
            _reject(report, "invalid_question_or_choices", rid)
            continue
        if any(not isinstance(c, dict) or not _identifier(c.get("id")) or not _text(c.get("text")) for c in choices):
            _reject(report, "invalid_choice", rid)
            continue
        ids = [str(c["id"]) for c in choices]
        texts = [c["text"] for c in choices]
        if len(set(ids)) != len(ids):
            _reject(report, "duplicate_option_ids", rid)
            continue
        if len({_normal(t) for t in texts}) != len(texts):
            _reject(report, "duplicate_option_texts", rid)
            continue
        if not _identifier(row.get("answerKey")) or str(row["answerKey"]) not in ids:
            _reject(report, "missing_or_unmapped_answer", rid)
            continue
        if not _text(row.get("config")):
            _reject(report, "missing_config", rid)
            continue
        if (any(_OPTION_REFERENCE.search(t) for t in [question] + texts)
                or any(_BARE_OPTION_REFERENCE.fullmatch(t) for t in texts)):
            _reject(report, "option_or_position_reference", rid, config=row["config"])
            continue
        if any(_VISUAL_REFERENCE.search(t) for t in [question] + texts):
            _reject(report, "possible_missing_visual_context", rid, config=row["config"])
            continue
        options = [(oid, text, None) for oid, text in zip(ids, texts)]
        _rng(seed, "arc", row["config"], rid).shuffle(options)
        pool.append(_decision(
            "arc", rid, _normal(question), "answer:" + row["config"], "science_question_answering",
            question, "Which answer correctly answers the question?", options,
            str(row["answerKey"]), "config:" + row["config"], {"config": row["config"]}))
    return pool


def _sgd_schema(raw, report):
    if not isinstance(raw, list):
        if raw is not None:
            report["input_errors"].append({"error": "SGD schema must be the original list of services"})
        return {}
    if not raw:
        report["input_errors"].append({"error": "SGD schema contains no services"})
    schemas, seen = {}, set()
    for service in raw:
        if not isinstance(service, dict) or not _text(service.get("service_name")):
            _reject(report, "invalid_schema_service", None)
            continue
        name = service["service_name"]
        if name in seen:
            schemas.pop(name, None)
            _reject(report, "duplicate_schema_service", name)
            continue
        seen.add(name)
        intents = service.get("intents")
        if (not _text(service.get("description")) or not isinstance(intents, list) or not intents
                or any(not isinstance(i, dict) or not _text(i.get("name"))
                       or not _text(i.get("description")) for i in intents)):
            _reject(report, "invalid_schema_intents_or_description", name)
            continue
        names = [i["name"] for i in intents]
        displays = [n.replace("_", " ") for n in names]
        if len(set(names)) != len(names) or len(set(displays)) != len(displays):
            _reject(report, "duplicate_schema_intents", name)
            continue
        schemas[name] = service
    return schemas


def _sgd(rows, schemas, none_services, seed, report):
    pool = []
    for dialogue, rid in rows:
        turns, services = dialogue.get("turns"), dialogue.get("services")
        if (not isinstance(turns, list) or not turns or not isinstance(services, list)
                or not all(_text(s) for s in services)):
            _reject(report, "invalid_dialogue", rid)
            continue
        if any(not isinstance(t, dict) or t.get("speaker") not in ("USER", "SYSTEM")
               or not _text(t.get("utterance")) for t in turns):
            _reject(report, "invalid_turn", rid)
            continue
        candidates, prefix = [], []
        for ti, turn in enumerate(turns):
            prefix.append({"speaker": turn["speaker"], "utterance": turn["utterance"]})
            if turn["speaker"] != "USER":
                continue
            frames = turn.get("frames", [])
            if not isinstance(frames, list):
                _reject(report, "invalid_frames", rid, turn=ti)
                continue
            service_counts = Counter(f.get("service") for f in frames
                                     if isinstance(f, dict) and _text(f.get("service")))
            for fi, frame in enumerate(frames):
                if not isinstance(frame, dict) or not _text(frame.get("service")):
                    _reject(report, "invalid_frame", rid, turn=ti, frame=fi)
                    continue
                name = frame["service"]
                if service_counts[name] != 1:
                    _reject(report, "duplicate_service_frame", rid, turn=ti, frame=fi)
                    continue
                if name not in schemas or name not in services:
                    _reject(report, "unknown_service", rid, turn=ti, service=name)
                    continue
                state = frame.get("state")
                if not isinstance(state, dict) or "active_intent" not in state:
                    _reject(report, "unannotated_active_intent", rid, turn=ti, service=name)
                    continue
                active = state["active_intent"]
                definition = schemas[name]
                options = [(i["name"], i["name"].replace("_", " "), i["description"])
                           for i in definition["intents"]]
                if name in none_services and not any(o[0] == "NONE" for o in options):
                    options.append(("NONE", "No active intent", "No service intent is active at this point."))
                if not _text(active) or active not in [o[0] for o in options]:
                    _reject(report, "unknown_active_intent", rid, turn=ti, service=name)
                    continue
                if len(options) < 2 or len({o[1] for o in options}) != len(options):
                    _reject(report, "invalid_intent_options", rid, turn=ti, service=name)
                    continue
                _rng(seed, "sgd", rid, ti, name).shuffle(options)
                candidates.append(_decision(
                    "sgd", rid, rid, "turn%d:%s:active_intent" % (ti, name), "dialogue_active_intent",
                    {"dialogue_prefix": prefix[:], "service_definition": {
                        "description": definition["description"],
                        "intents": [{"intent": o[1], "description": o[2]} for o in options]}},
                    "What is the current active intent for the described service at the last user turn? "
                    "Use the full conversation prefix, not a future turn.",
                    options, active, "service:%s:intent:%s" % (name, active),
                    {"service": name, "turn_index": ti, "prefix_turns": len(prefix)}))
        report["views_before_dialogue_cap"] += len(candidates)
        if len(candidates) > _SGD_MAX_VIEWS:
            indices = sorted(_rng(seed, "sgd", rid, "view_cap").sample(range(len(candidates)), _SGD_MAX_VIEWS))
            report["views_omitted_by_dialogue_cap"] += len(candidates) - len(indices)
            candidates = [candidates[i] for i in indices]
        if not candidates:
            _reject(report, "dialogue_has_no_eligible_views", rid)
        pool.extend(candidates)
    return pool


def load_source_pools(source_dir, seed=42, max_records_per_source=24000):
    """Read prepared train JSONL and return four decision lists plus diagnostics.

    The cap is a positive number of original records, not expanded decisions.
    Auxiliary JSON is accepted in prepared/ or at source_dir/ (prepared wins).
    CLINC ontology and explicitly observed SGD NONE labels are collected over the
    complete train file before expansion, not inferred from the reservoir alone.
    Rejection lists are complete; they may be large for heavily malformed input.
    """
    if isinstance(max_records_per_source, bool) or not isinstance(max_records_per_source, int) or max_records_per_source < 1:
        raise ValueError("max_records_per_source must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    root = Path(source_dir).expanduser()
    prepared = root / "prepared"
    report = {
        "schema_version": SCHEMA_VERSION, "seed": seed,
        "max_records_per_source": max_records_per_source,
        "missing_inputs": [], "input_errors": [], "sources": {},
        "sampling_policy": "Seeded uniform reservoir of unique original IDs before expansion; semantic filtering follows sampling except CLINC ontology validation.",
        "duplicate_policy": "First record per source ID wins (ARC: config plus ID). Different IDs with duplicate content are retained. CLINC/ARC group by normalized utterance/stem; SNLI by source group_key or normalized premise; SGD by dialogue ID.",
        "limitations": [
            "Only prepared train files are used; absent split fields rely on resource-preparation provenance.",
            "No synthetic requests, soft labels, score decisions, or annotation-confidence estimates are created.",
            "The parent must split by group_id, filter token lengths without truncation, and attach source revisions.",
            "Missing or malformed source_manifest is reported but does not block valid source records.",
            "Semantic rejection counts for SNLI/ARC/SGD cover retained reservoir records only; JSON, identity and split checks cover all lines. Duplicate-ID tracking and complete rejection reports use memory proportional to input size.",
        ],
    }
    for name in _SOURCES:
        report["sources"][name] = {
            "path": str(prepared / (name + "_train.jsonl")), "lines_read": 0,
            "missing_inputs": [], "input_errors": [], "rejection_counts": Counter(),
            "rejections": [], "grouping_counts": Counter(), "reservoir_eligible_records": 0,
            "reservoir_retained_records": 0, "reservoir_omitted_records": 0,
            "views_before_dialogue_cap": 0, "views_omitted_by_dialogue_cap": 0,
        }
    def auxiliary(filename):
        candidate = prepared / filename
        return candidate if candidate.is_file() else root / filename

    manifest_path = auxiliary("source_manifest.json")
    manifest = _read_json(manifest_path, report)
    if manifest is not None and not isinstance(manifest, dict):
        report["input_errors"].append({"path": str(manifest_path), "error": "Manifest must be an object"})
        manifest = None
    report["source_manifest_path"] = str(manifest_path)
    report["source_manifest"] = manifest
    cr = report["sources"]["clinc"]
    ontology, conflicting = {}, set()

    def observe_clinc(row, rid):
        if not all(_text(row.get(k)) for k in ("text", "intent", "domain")):
            _reject(cr, "invalid_clinc_record", rid)
            return False
        label, domain = row["intent"], row["domain"]
        if label in ontology and ontology[label] != domain:
            conflicting.add(label)
        ontology[label] = domain
        if domain in _EXCLUDED_DOMAINS or label in ("oos", "out_of_scope"):
            _reject(cr, "excluded_domain_or_out_of_scope", rid)
            return False
        return True

    none_services = set()

    def observe_sgd(row, rid):
        turns = row.get("turns")
        for turn in turns if isinstance(turns, list) else []:
            if not isinstance(turn, dict) or turn.get("speaker") != "USER":
                continue
            frames = turn.get("frames")
            for frame in frames if isinstance(frames, list) else []:
                if (isinstance(frame, dict) and _text(frame.get("service"))
                        and isinstance(frame.get("state"), dict)
                        and frame["state"].get("active_intent") == "NONE"):
                    none_services.add(frame["service"])
        return True

    raw = {}
    for name in _SOURCES:
        raw[name] = _reservoir(prepared / (name + "_train.jsonl"), name, seed,
                               max_records_per_source, report["sources"][name],
                               {"clinc": observe_clinc, "sgd": observe_sgd}.get(name))
    allowed = {label: domain for label, domain in ontology.items()
               if domain not in _EXCLUDED_DOMAINS and label not in conflicting
               and label not in ("oos", "out_of_scope")}
    clinc_rows = []
    for row, rid in raw["clinc"]:
        if row["intent"] not in allowed:
            _reject(cr, "conflicting_intent_domain", rid)
        else:
            clinc_rows.append((row, rid))
    cr["conflicting_intents"] = sorted(conflicting)
    cr["semantics"] = "Closest original intent; allowed ontology collected from the complete prepared training file; banking, credit_cards and OOS excluded. Labels use underscore verbalization plus a small reviewed alias table. No completeness claim beyond supplied file."
    report["sources"]["snli"]["semantics"] = "Exact premise/hypothesis; three-way choice plus one relation-specific binary task per row, balanced within each gold label. Neutral means neither entailment nor contradiction, not a false real-world fact. Missing group_key falls back to normalized premise; different captions of the same image may then cross groups."
    report["sources"]["arc"]["semantics"] = "Original choices only, shuffled with synchronized source IDs. Conservative lexical rejection of option references and potentially missing visuals can over-filter self-contained questions and cannot detect every implicit visual dependency. Stem grouping spans configs."
    sr = report["sources"]["sgd"]
    sr["semantics"] = "Active-intent choice only at explicitly annotated USER frames. Prefix includes all prior USER/SYSTEM utterances and current USER utterance, never frame annotations or future content. NONE is a source sentinel included only when schema-defined or explicitly observed for that service in training. No negative labels inferred from absent annotations; no categorical-slot views. Original service/intent descriptions retained; service IDs remain metadata only."
    sr["max_views_per_dialogue"] = _SGD_MAX_VIEWS
    sr["none_services_observed"] = sorted(none_services)
    schema_path = auxiliary("sgd_schema.json")
    sr["schema_path"] = str(schema_path)
    schemas = _sgd_schema(_read_json(schema_path, sr), sr)
    pools = {"clinc": _clinc(clinc_rows, allowed, seed, cr),
             "snli": _snli(raw["snli"], seed, report["sources"]["snli"]),
             "arc": _arc(raw["arc"], seed, report["sources"]["arc"]),
             "sgd": _sgd(raw["sgd"], schemas, none_services, seed, sr)}
    for name, pool in pools.items():
        source_report = report["sources"][name]
        source_report["decisions"] = len(pool)
        source_report["original_records_with_decisions"] = len({
            (r["source"]["record_id"], r["metadata"].get("config")) for r in pool})
        source_report["groups"] = len({r["group_id"] for r in pool})
        source_report["sampling_strata"] = dict(Counter(r["metadata"]["sampling_stratum"] for r in pool))
        source_report["task_counts"] = dict(Counter(r["questions"]["decision"]["type"] for r in pool))
        source_report["rejection_counts"] = dict(source_report["rejection_counts"])
        source_report["grouping_counts"] = dict(source_report["grouping_counts"])
        report["missing_inputs"].extend(source_report["missing_inputs"])
        report["input_errors"].extend(dict(error, source=name) for error in source_report["input_errors"])
    report["complete_inputs"] = not report["missing_inputs"] and not report["input_errors"]
    return pools, report
