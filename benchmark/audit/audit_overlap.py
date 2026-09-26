"""Offline, prediction-independent lexical exposure audit of a full_eval_v2 profile."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
import unicodedata


FIXED_EXCLUDES = {
    "clinc150_oos_seen_source": "known_source_overlap:clinc",
    "nimble_boolq": "known_source_overlap:boolq",
    "forecastbench": "retrospective_forecast_scope_exclusion",
}
ALIASES = {
    "clinc": ["clinc", "clinc150", "clinc_oos", "clinc/oos-eval", "clinc_oos/plus"],
    "boolq": ["boolq", "google/boolq", "super_glue/boolq"],
    "snli": ["snli", "stanfordnlp/snli"],
    "multinli": ["multinli", "multi_nli", "nyu-mll/multi_nli", "multinli_1.0"],
    "arc": ["arc", "ai2_arc", "allenai/ai2_arc", "arc_challenge", "arc_easy"],
    "sgd": ["sgd", "schema_guided_dstc8", "schema-guided-dialogue"],
    "dbpedia": ["dbpedia", "dbpedia_14", "fancyzhx/dbpedia_14"],
    "goemotions": ["goemotions", "go_emotions", "google-research-datasets/go_emotions"],
    "massive": ["massive", "massive-en-us", "massive-de-de", "amazon_massive",
                "amazon-massive-dataset-1.1", "amazon-science/massive", "amazon/massive"],
    "summeval": ["summeval", "summeval-relevance", "summeval-consistency", "mteb/summeval"],
    "vitaminc": ["vitaminc", "vitaminc-dev", "tals/vitaminc"],
    "squad2": ["squad2", "squad_v2", "rajpurkar/squad_v2"],
    "paws": ["paws", "google-research-datasets/paws"],
    "civil_comments": ["civil_comments", "google/civil_comments"],
    "aegis2": ["aegis2", "nvidia/aegis-ai-content-safety-dataset-2.0"],
    "helpsteer2": ["helpsteer2", "nvidia/helpsteer2"],
    "pubmedqa": ["pubmedqa", "qiaojin/pubmed_qa"],
    "contractnli": ["contractnli", "contract-nli", "stanfordnlp/contract-nli"],
    "forecastbench": ["forecastbench", "forecastingresearch/forecastbench-datasets"],
    "jevbench": ["jevbench", "fstandhartinger/jevbench"],
    "classifier-benchmark": ["classifier-benchmark", "jabr/classifier-benchmark"],
}
ALIAS_MAP = {alias: name for name, aliases in ALIASES.items() for alias in aliases}
IGNORED_FIELDS = {
    "instructions", "instruction", "criteria", "candidates", "options", "option_ids",
    "labels", "label", "rubric", "rules", "policy", "task", "task_description",
    "system", "system_prompt", "choices", "answer_choices",
}
DOCUMENT_FIELDS = {"document", "article", "passage", "context", "contract", "text",
                   "source_text", "source", "premise", "dialogue", "dialogue_prefix"}
LABEL = re.compile(r"^\s*([A-Za-z][A-Za-z _-]{0,45}):\s*(.*)$")


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
    raise ValueError("Nonfinite JSON constant: " + value)


def loads(text):
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)


def read_json(path):
    return loads(path.read_text(encoding="utf-8"))


def records(path):
    with path.open(encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            require(bool(text.strip()), f"{path}:{line}: blank line")
            value = loads(text)
            require(isinstance(value, dict), f"{path}:{line}: expected object")
            yield value


def dump(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def save(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, allow_nan=False, indent=2)
        stream.write("\n")


def progress(event, **fields):
    print(dump(dict(event=event, **fields)), flush=True)


def normal(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def words(text):
    return re.findall(r"\w+", normal(text))


def ordered(value):
    if isinstance(value, str):
        return normal(value)
    if isinstance(value, dict):
        return {key: ordered(item) for key, item in value.items()}
    if isinstance(value, list):
        return [ordered(item) for item in value]
    return value


def resolve(base, value):
    path = Path(value).expanduser()
    return (base / path).resolve()


def namespace(descriptor, suite=None):
    if suite and suite.startswith("jevbench_public_"):
        return "jevbench"
    if suite == "jabr_v2":
        return "classifier-benchmark"
    if suite in FIXED_EXCLUDES:
        return {"clinc150_oos_seen_source": "clinc", "nimble_boolq": "boolq",
                "forecastbench": "forecastbench"}[suite]
    if suite and suite.startswith("nimble_"):
        name = suite[len("nimble_"):].casefold()
        if name in ALIAS_MAP:
            return ALIAS_MAP[name]
    if suite == "contractnli":
        return "contractnli"
    if isinstance(descriptor, dict):
        values = [descriptor[k] for k in ("name", "dataset", "repo_id", "dataset_name", "url")
                  if isinstance(descriptor.get(k), str)]
    else:
        values = [descriptor] if isinstance(descriptor, str) else []
    for value in values:
        key = value.casefold().rstrip("/")
        for prefix in ("https://huggingface.co/datasets/", "http://huggingface.co/datasets/",
                       "https://github.com/"):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key in ALIAS_MAP:
            return ALIAS_MAP[key]
    # Unlisted solver families stay distinct instead of being conflated as "synthetic".
    return "unmapped:" + normal(values[0]) if values else "unknown:" + str(suite)


def raw_identity(source, ns):
    if not isinstance(source, dict):
        return None
    rid = source.get("record_id", source.get("row_id"))
    if rid is None:
        return None
    rid = str(rid) if isinstance(rid, (str, int)) else dump(rid)
    split = source.get("original_split", source.get("split"))
    # Bare integer row offsets are split-local. Never strip dataset prefixes from IDs.
    if rid.isdecimal():
        if not isinstance(split, str) or not split:
            return None
        return ns, normal(split), rid
    return ns, "nonlocal_id", rid


def extract(state, args):
    """Return eligible spans, keeping paths but never key names in content text."""
    leaves = []

    def walk(value, path, field):
        if isinstance(value, str):
            leaves.append((path, field, value))
        elif isinstance(value, dict):
            for key, item in value.items():
                if normal(key).replace(" ", "_") not in IGNORED_FIELDS:
                    walk(item, path + "." + key, key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]", field)

    walk(state, "state", "state")
    spans = []
    for path, field, text in leaves:
        lines, block, active, section = [], [], True, "unlabelled"
        for index, line in enumerate(text.splitlines()):
            match = LABEL.match(line)
            if match:
                if block:
                    spans.append((f"{path}:section:{section}", field, "\n".join(block), False))
                    block = []
                label, line = match.groups()
                section = f"{index + 1}:{label}"
                active = normal(label).replace(" ", "_") not in IGNORED_FIELDS
            if active:
                lines.append(line)
                block.append(line)
                if line.strip():
                    spans.append((f"{path}:line:{index + 1}", field, line, False))
        if block:
            spans.append((f"{path}:section:{section}", field, "\n".join(block), False))
        clean = "\n".join(lines)
        spans.append((path, field, clean, True))
        for index, paragraph in enumerate(re.split(r"\n\s*\n", clean)):
            spans.append((f"{path}:paragraph:{index + 1}", field, paragraph, False))
    clean_leaves = [text for _, _, text, whole_leaf in spans if whole_leaf]
    content = "\n".join(clean_leaves)
    # The original state is evidence; full-state equality separately preserves structure.
    spans.insert(0, ("state:content", "state", content, True))
    spans.sort(key=lambda item: not item[3])
    result, seen = [], set()
    total_words = len(words(content))
    for path, field, text, whole_leaf in spans:
        key = normal(text)
        tokens = words(text)
        if len(tokens) < args.exact_min_words or key in seen:
            continue
        seen.add(key)
        link = (path == "state:content" or
                (whole_leaf and normal(field) in DOCUMENT_FIELDS and
                 len(tokens) >= args.group_min_words and len(tokens) >= total_words * 0.5))
        result.append(dict(field=path, text=text, normalized=key, words=tokens,
                           group_link=link))
    return result


class Groups:
    def __init__(self, count):
        self.parents = list(range(count))

    def root(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def join(self, left, right):
        left, right = self.root(left), self.root(right)
        self.parents[max(left, right)] = min(left, right)


class LexicalIndex:
    """Index unique evaluation spans; scan each unique exposure span exactly once."""
    def __init__(self, args):
        self.args = args
        self.units, self.exact, self.postings = [], {}, defaultdict(list)
        self.comparisons = 0

    def add(self, span, owner):
        key = span["normalized"]
        if key not in self.exact:
            index = len(self.units)
            tokens = span["words"]
            shingles = set(zip(*(tokens[i:] for i in range(5)))) if len(tokens) >= self.args.near_min_words else set()
            self.exact[key] = index
            self.units.append(dict(normalized=key, shingle_count=len(shingles),
                                   words=set(tokens), owners=[]))
            for shingle in shingles:
                self.postings[shingle].append(index)
        self.units[self.exact[key]]["owners"].append(owner)

    def match(self, span):
        key, tokens = span["normalized"], span["words"]
        exact = self.exact.get(key)
        if exact is not None:
            yield exact, "exact_content", dict(jaccard=1.0, short_containment=1.0,
                                               shared_words=len(set(tokens)))
        if len(tokens) < self.args.near_min_words:
            return
        shingles = set(zip(*(tokens[i:] for i in range(5))))
        intersections = Counter()
        for shingle in shingles:
            intersections.update(self.postings.get(shingle, ()))
        token_set = set(tokens)
        for index in sorted(intersections):
            if index == exact:
                continue
            self.comparisons += 1
            other = self.units[index]
            intersection = intersections[index]
            jaccard = intersection / (len(shingles) + other["shingle_count"] - intersection)
            containment = intersection / min(len(shingles), other["shingle_count"])
            shared = len(token_set & other["words"])
            if jaccard >= self.args.jaccard or (containment >= self.args.containment and
                                               shared >= self.args.shared_words):
                yield index, "near_content", dict(jaccard=jaccard,
                                                  short_containment=containment,
                                                  shared_words=shared)


def load_profile(path):
    profile = read_json(path)
    require(profile.get("schema_version") == "full_eval_v2", "Expected full_eval_v2")
    require(isinstance(profile.get("profile_id"), str), "Missing profile_id")
    rows, suites, identities = [], set(), set()
    for spec in profile["suites"]:
        suite = spec["name"]
        require(suite not in suites, "Duplicate suite: " + suite)
        suites.add(suite)
        source_path = resolve(path.parent, spec["path"])
        require(source_path.is_relative_to(path.parent), "Suite path escapes profile directory")
        count = 0
        for raw in records(source_path):
            require(isinstance(raw.get("id"), str) and raw["id"] not in identities,
                    "Missing/duplicate evaluation ID")
            require(isinstance(raw.get("group_id"), str) and raw["group_id"], "Missing group_id")
            require(isinstance(raw.get("state"), (str, dict, list)), "Invalid evaluation state")
            require(isinstance(raw.get("qdef"), dict), "Invalid qdef")
            identities.add(raw["id"])
            source = raw["metadata"].get("source", spec["source"])
            rows.append(dict(id=raw["id"], group_id=raw["group_id"], suite=suite,
                             state=raw["state"], qdef=raw["qdef"], source=source,
                             namespace=namespace(source, suite),
                             upstream_family=raw["metadata"].get("upstream_family"),
                             target_signature={key: raw[key] for key in
                                               ("gold_idx", "soft", "gold_score", "option_ids")},
                             source_overlap=raw["metadata"].get("source_overlap"),
                             reasons=[], base="retain"))
            count += 1
        require(count == spec["expected_count"], f"Suite count mismatch: {suite}: {count}")
    return profile, rows


def load_exposures(path, args):
    config = read_json(path)
    require(isinstance(config, list) and config, "Exposures must be a nonempty JSON list")
    names, reports, unique, manifests = set(), [], {}, {}
    for spec in config:
        require({"name", "role", "path", "model", "expected_count"} <= set(spec),
                "Exposure missing required fields")
        require(spec["name"] not in names, "Duplicate exposure name")
        names.add(spec["name"])
        require(spec["role"] in {"gradient_train", "selection_dev", "diagnostic_dev"}, "Invalid role")
        require(isinstance(spec["model"], str) and spec["model"], "Missing model provenance")
        source_path = resolve(path.parent, spec["path"])
        manifest_path = resolve(path.parent, spec["training_manifest"]) if spec.get("training_manifest") else None
        checked, diagnostic_checked = None, None
        if spec.get("diagnostic_report"):
            require(spec["role"] == "diagnostic_dev", "Diagnostic report requires diagnostic_dev role")
            diagnostic_path = resolve(path.parent, spec["diagnostic_report"])
            diagnostic = read_json(diagnostic_path)
            settings = diagnostic["settings"]
            require(Path(settings["data"]).is_absolute(), "Diagnostic data path must be absolute")
            require(resolve(diagnostic_path.parent, settings["data"]) == source_path,
                    "Diagnostic/exposure path mismatch")
            require(settings["limit"] == 0, "Limited diagnostic cannot establish full-file exposure")
            require(diagnostic["count"] == spec["expected_count"], "Diagnostic/exposure count mismatch")
            require(isinstance(settings.get("model_path"), str) and
                    Path(settings["model_path"]).is_absolute(), "Missing absolute diagnostic model path")
            expected_model = spec.get("model_path")
            if expected_model is None and Path(spec["model"]).is_absolute():
                expected_model = spec["model"]
            if expected_model is not None:
                require(resolve(path.parent, expected_model) ==
                        resolve(diagnostic_path.parent, settings["model_path"]),
                        "Diagnostic/exposure model path mismatch")
            diagnostic_checked = dict(path=str(diagnostic_path), data=settings["data"],
                count=diagnostic["count"], limit=settings["limit"], role=spec["role"],
                model=spec["model"], model_path=settings["model_path"],
                model_check="path_verified" if expected_model is not None else
                            "declared_alias_only: supply model_path to verify alias association")
        if manifest_path:
            manifest = read_json(manifest_path)
            cfg = manifest["config"]
            require(not cfg.get("train_limit", 0) and not cfg.get("dev_limit", 0),
                    "Limited run manifests cannot establish full-file exposure")
            field = "train_data" if spec["role"] == "gradient_train" else "dev_data"
            require(Path(cfg[field]["path"]).is_absolute(), "Manifest data paths must be absolute")
            require(resolve(manifest_path.parent, cfg[field]["path"]) == source_path,
                    f"Manifest/exposure path mismatch: {spec['name']}")
            count_key = "train_count" if field == "train_data" else "dev_count"
            require(cfg[count_key] == spec["expected_count"], "Manifest/exposure count mismatch")
            checked = dict(path=str(manifest_path), run_id=manifest.get("run_id"),
                           code_commit=manifest.get("code_commit"), field=field,
                           data=cfg[field], count=cfg[count_key])
            manifests[(spec["model"], str(manifest_path))] = cfg
        count, source_counts, splits, local_ids = 0, Counter(), Counter(), set()
        for row in records(source_path):
            require(row.get("schema_version") == "shared_yesno_data_v1", "Expected canonical exposure")
            source = row["source"]
            require(isinstance(source, dict) and isinstance(source.get("name"), str),
                    "Canonical source must be an object with name")
            require(source.get("record_id") is not None and source.get("original_split") is not None,
                    "Canonical source missing record_id/original_split")
            identity = (row["case_id"], row["view_id"])
            require(identity not in local_ids, "Repeated case/view in exposure file")
            local_ids.add(identity)
            ns = namespace(source)
            key = (ns, dump(source.get("record_id")), source.get("original_split"),
                   dump(row["state"]))
            if key not in unique:
                unique[key] = dict(source=source, namespace=ns, state=row["state"], references=[])
            unique[key]["references"].append(dict(exposure=spec["name"], role=spec["role"],
                model=spec["model"], case_id=row["case_id"], view_id=row["view_id"],
                group_id=row["group_id"], record_id=source.get("record_id"),
                original_split=source.get("original_split"), split=row["split"]))
            source_counts[ns] += 1
            splits[row["split"]] += 1
            count += 1
            if count % args.progress_every == 0:
                progress("load_exposure", exposure=spec["name"], rows=count)
        require(count == spec["expected_count"], f"Exposure count mismatch: {spec['name']}: {count}")
        report = dict(spec, path=str(source_path), actual_count=count,
                      source_counts=dict(source_counts), split_counts=dict(splits), manifest_check=checked,
                      diagnostic_check=diagnostic_checked,
                      provenance_note=None if checked or diagnostic_checked else
                          "No training_manifest or diagnostic_report supplied")
        reports.append(report)
        progress("exposure_loaded", exposure=spec["name"], rows=count, unique_states=len(unique))
    for (model, manifest_path), cfg in manifests.items():
        for field, roles in (("train_data", {"gradient_train"}),
                             ("dev_data", {"selection_dev", "diagnostic_dev"})):
            target = str(resolve(Path(manifest_path).parent, cfg[field]["path"]))
            require(any(r["model"] == model and r["path"] == target and r["role"] in roles
                        and r["manifest_check"] and r["manifest_check"]["path"] == manifest_path
                        for r in reports), f"Manifest {manifest_path}: missing {field} exposure for {model}")
    return reports, list(unique.values())


def audit(args):
    require(not args.output_dir.exists(), "Output directory must be new")
    profile_path, exposure_path = args.profile.resolve(), args.exposures.resolve()
    profile, rows = load_profile(profile_path)
    exposures, training = load_exposures(exposure_path, args)
    reviews = read_json(args.review_decisions) if args.review_decisions else []
    require(isinstance(reviews, list), "Review decisions must be a JSON list")
    review_map = {}
    for review in reviews:
        require(set(review) == {"candidate_id", "disposition", "reason"}, "Invalid review fields")
        require(review["candidate_id"] not in review_map, "Duplicate review candidate_id")
        require(review["disposition"] in {"dismiss", "confirm_overlap", "quarantine"}, "Invalid disposition")
        require(isinstance(review["reason"], str) and review["reason"].strip(), "Review requires reason")
        review_map[review["candidate_id"]] = review
    output = args.output_dir.resolve()
    output.mkdir(exist_ok=False)
    save(output / "audit.json", dict(status="running", profile_id=profile["profile_id"]))
    groups, index = Groups(len(rows)), LexicalIndex(args)
    exposure_namespaces = {record["namespace"] for record in training}
    group_first, family_first, state_first, raw_ids = {}, {}, {}, defaultdict(list)
    duplicate_members, ambiguous_keys = defaultdict(list), set()
    whole_states = defaultdict(list)
    links, duplicate_count, candidate_count = [], 0, 0
    duplicate_cases = []
    for i, row in enumerate(rows):
        if row["suite"] in FIXED_EXCLUDES or (row["namespace"] in exposure_namespaces and
                not row["namespace"].startswith(("unknown:", "unmapped:"))):
            continue
        key = dump(ordered(row["state"])), dump(ordered(row["qdef"]))
        members = duplicate_members[key]
        if members:
            duplicate_count += 1
            row["base"] = "exclude"
            row["reasons"].append(dict(kind="duplicate_decision", canonical_id=rows[members[0]]["id"]))
        for previous in members:
            same = row["target_signature"] == rows[previous]["target_signature"]
            duplicate_cases.append(dict(id=row["id"], counterpart=rows[previous]["id"],
                canonical_id=rows[members[0]]["id"], target_same=same,
                target_signature=row["target_signature"],
                counterpart_target_signature=rows[previous]["target_signature"]))
            if not same:
                ambiguous_keys.add(key)
        members.append(i)
    ambiguous_ids = set()
    for key in ambiguous_keys:
        members = duplicate_members[key]
        for i in members:
            rows[i]["base"] = "quarantine"
            rows[i]["reasons"].append(dict(kind="duplicate_target_ambiguity",
                canonical_id=rows[members[0]]["id"], member_ids=[rows[j]["id"] for j in members]))
            ambiguous_ids.add(rows[i]["id"])
    with (output / "duplicate_cases.jsonl").open("x", encoding="utf-8") as stream:
        for case in duplicate_cases:
            stream.write(dump(case) + "\n")
    for i, row in enumerate(rows):
        if row["suite"] in FIXED_EXCLUDES:
            row["base"] = "exclude"
            row["reasons"].append(dict(kind=FIXED_EXCLUDES[row["suite"]]))
            continue
        if row["namespace"] in exposure_namespaces and not row["namespace"].startswith(("unknown:", "unmapped:")):
            row["base"] = "exclude"
            row["reasons"].append(dict(kind="observed_exposure_source_overlap", namespace=row["namespace"]))
            continue
        candidate_count += 1
        group_key = row["suite"], row["group_id"]
        if group_key in group_first:
            groups.join(i, group_first[group_key])
        else:
            group_first[group_key] = i
        family = row["upstream_family"]
        if family is not None and family != "":
            require(not row["namespace"].startswith(("unknown:", "unmapped:")),
                    f"Explicit upstream_family requires a mapped source namespace: {row['id']}")
            family_key = row["namespace"], dump(family)
            if family_key in family_first:
                previous = family_first[family_key]
                groups.join(i, previous)
                links.append(dict(id=row["id"], counterpart=rows[previous]["id"],
                    kind="upstream_family_group", namespace=row["namespace"], upstream_family=family))
            else:
                family_first[family_key] = i
        spans = extract(row["state"], args)
        if spans:
            whole_states[dump(ordered(row["state"]))].append(i)
        for span in spans:
            index.add(span, (i, span["field"], span["text"]))
            # Short identical full states link only inside the same source namespace.
            if span["group_link"]:
                key = (row["namespace"], span["normalized"])
                if len(span["words"]) >= args.group_min_words:
                    key = ("cross_source", span["normalized"])
                if key in state_first:
                    previous = state_first[key]
                    groups.join(i, previous)
                    links.append(dict(id=row["id"], counterpart=rows[previous]["id"],
                                      field=span["field"], kind="shared_content_group"))
                else:
                    state_first[key] = i
        rid = raw_identity(row["source"], row["namespace"])
        if rid is not None:
            raw_ids[rid].append(i)
    progress("candidate_index_ready", decisions=candidate_count, unique_spans=len(index.units))
    training_units, identity_matches, whole_matches = {}, [], defaultdict(list)
    for t, record in enumerate(training):
        rid = raw_identity(record["source"], record["namespace"])
        if rid is not None and rid in raw_ids:
            identity_matches.append((t, raw_ids[rid]))
        spans = extract(record["state"], args)
        whole_key = dump(ordered(record["state"]))
        if spans and whole_key in whole_states:
            whole_matches[whole_key].append(t)
        for span in spans:
            key = span["normalized"]
            if key not in training_units:
                training_units[key] = dict(span=span, owners=[])
            training_units[key]["owners"].append((t, span["field"], span["text"]))
        if (t + 1) % args.progress_every == 0:
            progress("prepare_exposure_spans", records=t + 1, unique_spans=len(training_units))
    hit_counts, direct, used_reviews = Counter(), defaultdict(list), set()
    candidate_number = 0
    with (output / "candidates.jsonl").open("x", encoding="utf-8") as stream:
        def emit(kind, eval_owners, train_owners, similarity):
            nonlocal candidate_number
            candidate_number += 1
            cid = f"candidate/{candidate_number:08d}"
            review = review_map.get(cid)
            if review:
                require(kind in {"near_content", "exact_content"},
                        "Only content candidates may be hand reviewed; identity matches cannot be dismissed")
                used_reviews.add(cid)
            disposition = (review["disposition"] if review else
                           "quarantine" if kind in {"near_content", "exact_content"} else "confirm_overlap")
            affected = sorted({owner[0] for owner in eval_owners})
            exposure_evidence = [dict(source=training[t]["source"], namespace=training[t]["namespace"],
                field=field, matched_span=text, original_state=training[t]["state"],
                references=training[t]["references"]) for t, field, text in train_owners]
            event = dict(candidate_id=cid, kind=kind, similarity=similarity,
                classification="dismissed_generic_or_distinct_context" if disposition == "dismiss" else
                               "unresolved_lexical_similarity" if disposition == "quarantine" else
                               "source_record_identity" if kind == "raw_id" else "substantive_shared_content",
                label_seen="not_inferred", disposition=disposition, review=review,
                evaluation=[dict(id=rows[i]["id"], suite=rows[i]["suite"], group_id=rows[i]["group_id"],
                    source=rows[i]["source"], field=field, matched_span=text,
                    original_state=rows[i]["state"]) for i, field, text in eval_owners],
                exposures=exposure_evidence)
            stream.write(dump(event) + "\n")
            hit_counts[kind] += 1
            if disposition != "dismiss":
                for i in affected:
                    direct[i].append(dict(candidate_id=cid, kind=kind,
                        decision="exclude" if disposition == "confirm_overlap" else "quarantine"))
            if candidate_number % args.progress_every == 0:
                stream.flush()
                progress("matching", evidence_records=candidate_number)

        for t, matched in identity_matches:
            emit("raw_id", [(i, "source.record_id", None) for i in matched],
                 [(t, "source.record_id", None)], None)
        for key, owners in whole_matches.items():
            emit("exact_whole_state", [(i, "state", rows[i]["state"]) for i in whole_states[key]],
                 [(t, "state", training[t]["state"]) for t in owners], dict(normalized_equal=True))
        for n, item in enumerate(training_units.values(), 1):
            for matched, kind, similarity in index.match(item["span"]):
                emit(kind, index.units[matched]["owners"], item["owners"], similarity)
            if n % args.progress_every == 0:
                stream.flush()
                progress("scan_exposure_spans", processed=n, total=len(training_units), hits=candidate_number)
    require(used_reviews == set(review_map), "Review IDs absent or not reviewable content; use identical inputs/order")
    propagation = defaultdict(list)
    for i, hits in direct.items():
        propagation[groups.root(i)].append(dict(id=rows[i]["id"], hits=hits))
    decisions, retained, counts = [], [], defaultdict(Counter)
    for spec in profile["suites"]:
        counts[spec["name"]].update(retain=0, exclude=0, quarantine=0)
    for i, row in enumerate(rows):
        decision, reasons = row["base"], list(row["reasons"])
        events = propagation.get(groups.root(i), [])
        if events:
            overlap = any(hit["decision"] == "exclude" for event in events for hit in event["hits"])
            propagated = "exclude" if overlap else "quarantine"
            if decision != "exclude":
                decision = propagated
            reasons.append(dict(kind="group_exposure_overlap" if overlap else "group_unresolved_content",
                                component_id=rows[groups.root(i)]["id"],
                                direct_hits=direct.get(i, []),
                                evidence_member_ids=[event["id"] for event in events]))
        decisions.append(dict(id=row["id"], suite=row["suite"], group_id=row["group_id"],
                              decision=decision, reasons=reasons))
        counts[row["suite"]][decision] += 1
        if decision == "retain":
            retained.append(row["id"])
    with (output / "decisions.jsonl").open("x", encoding="utf-8") as stream:
        for decision in decisions:
            stream.write(dump(decision) + "\n")
    save(output / "retained_ids.json", dict(profile_id=profile["profile_id"], retained_ids=retained,
         counts_by_suite={spec["name"]: counts[spec["name"]]["retain"] for spec in profile["suites"]}))
    limitations = [
        "Lexical audit, not semantic equivalence detection; paraphrases and translations may be missed.",
        "Pretraining exposure is unknown; supplied fine-tuning/dev files are not all model history.",
        "Not a blind benchmark: these v2 suites have already been evaluated; no predictions or scores are read here.",
        "Exact content demonstrates shared content, not automatically shared labels or identical decisions.",
        "Instruction/option/rule fields and short content are filtered; numeric-only solver states are not lexically audited.",
        "Raw IDs are compared only in mapped source namespaces; local integers also require matching split.",
        "Run manifests establish declared paths/counts, not historical file immutability or the model/checkpoint lineage.",
        "Shared-component grouping is heuristic; original suite groups can be broad and cause conservative exclusions.",
        "All retained IDs are conditional on this exposure inventory and these thresholds, never a clean certification.",
    ]
    report = dict(status="complete", profile_id=profile["profile_id"], profile_path=str(profile_path),
        exposures_path=str(exposure_path), raw_rows=len(rows), candidate_rows=candidate_count,
        candidate_suites=len({r["suite"] for r in rows if r["suite"] not in FIXED_EXCLUDES}),
        fixed_exclusions=FIXED_EXCLUDES, counts_by_suite=dict(counts),
        decision_counts=dict(Counter(d["decision"] for d in decisions)),
        known_source_overlap_metadata_rows=sum(
            any(value is True for value in r["source_overlap"].values())
            if isinstance(r["source_overlap"], dict) else r["source_overlap"] is True for r in rows),
        fixed_excluded_rows_by_suite=dict(Counter(r["suite"] for r in rows if r["suite"] in FIXED_EXCLUDES)),
        source_overlap_metadata=[dict(id=r["id"], suite=r["suite"], value=r["source_overlap"])
                                 for r in rows if r["source_overlap"]],
        exposure_raw_rows=sum(r["actual_count"] for r in exposures),
        exposure_unique_source_record_states=len(training), exposure_unique_spans=len(training_units),
        exposure_unique_source_records=len({(r["namespace"], dump(r["source"].get("record_id")),
                                             r["source"].get("original_split")) for r in training}),
        exposure_rows_by_role=dict(Counter({role: sum(e["actual_count"] for e in exposures if e["role"] == role)
            for role in ("gradient_train", "selection_dev", "diagnostic_dev")})),
        exposure_rows_by_source=dict(sum((Counter(e["source_counts"]) for e in exposures), Counter())),
        exposure_unique_states_by_source=dict(Counter(r["namespace"] for r in training)),
        exposures=exposures, exposures_without_manifest=[e["name"] for e in exposures if not e["manifest_check"]],
        exposures_without_provenance=[e["name"] for e in exposures
                                     if not e["manifest_check"] and not e["diagnostic_check"]],
        source_aliases=ALIASES,
        resolved_sources={"evaluation": sorted({(r["suite"], r["namespace"]) for r in rows}),
                          "exposure": sorted({(r["source"]["name"], r["namespace"]) for r in training})},
        observed_source_overlap=sorted({r["namespace"] for r in rows} & {r["namespace"] for r in training}),
        related_not_identical_sources=[["snli", "multinli"]],
        thresholds={key: getattr(args, key) for key in ("exact_min_words", "near_min_words", "jaccard",
                                                       "containment", "shared_words", "group_min_words")},
        shingle_words=5, candidate_caps=None, lexical_pair_comparisons=index.comparisons,
        hit_counts=dict(hit_counts), duplicate_decisions=duplicate_count, group_links=links,
        duplicate_pairs=len(duplicate_cases), duplicate_target_ambiguity_groups=len(ambiguous_keys),
        duplicate_target_ambiguity_count=len(ambiguous_ids),
        duplicate_target_ambiguity_ids=[r["id"] for r in rows if r["id"] in ambiguous_ids],
        group_components=len({groups.root(i) for i, r in enumerate(rows) if r["suite"] not in FIXED_EXCLUDES}),
        review_decisions=reviews, scope_limitations=limitations)
    save(output / "audit.json", report)
    lines = ["# Overlap audit", "", "Status: complete (lexical audit only; not a clean certification).", "",
             f"Profile: `{profile['profile_id']}`. Raw decisions: {len(rows)}; candidate core: {candidate_count}; "
             f"retained: {len(retained)}.", "", "## Decisions", "",
             "| Suite | Retain | Exclude | Quarantine |", "| --- | ---: | ---: | ---: |"]
    for spec in profile["suites"]:
        name = spec["name"]
        lines.append(f"| {name} | {counts[name]['retain']} | {counts[name]['exclude']} | {counts[name]['quarantine']} |")
    lines += ["", "## Evidence and provenance", "",
              f"Exposure rows: {report['exposure_raw_rows']}; unique source-record/state views: {len(training)}.",
              f"Evidence records: {candidate_number}; hit types: `{dump(dict(hit_counts))}`.",
              f"Duplicate decisions: {duplicate_count}; pairs: {len(duplicate_cases)}; "
              f"target-ambiguous IDs: {len(ambiguous_ids)} in {len(ambiguous_keys)} duplicate-input groups.",
              "`duplicate_cases.jsonl` records every duplicate pair and target agreement; ambiguity is not training leakage.",
              "`candidates.jsonl` contains all qualifying matches and full private source spans; roles/models remain attached.",
              "`audit.json` records manifests, source aliases, thresholds, review dispositions and content-group links.",
              "Manifests not supplied for: " + ", ".join(report["exposures_without_manifest"]) + ".",
              "Diagnostic report path/count/limit/role checks: " + ", ".join(
                  e["name"] + " (model: " + e["diagnostic_check"]["model_check"] + ")"
                  for e in exposures if e["diagnostic_check"]) + ".",
              "No manifest or diagnostic report supplied for: " + ", ".join(report["exposures_without_provenance"]) + ".",
              "Fixed exclusions occur before matching, not after inspecting model scores.",
              "", "## Limitations", ""] + ["- " + text for text in limitations]
    (output / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress("complete", output_dir=str(output), decisions=len(decisions), retained=len(retained))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--exposures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-decisions", type=Path)
    parser.add_argument("--exact-min-words", type=int, default=3)
    parser.add_argument("--near-min-words", type=int, default=12)
    parser.add_argument("--jaccard", type=float, default=0.8)
    parser.add_argument("--containment", type=float, default=0.9)
    parser.add_argument("--shared-words", type=int, default=12)
    parser.add_argument("--group-min-words", type=int, default=40)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    require(0 < args.jaccard <= 1 and 0 < args.containment <= 1, "Similarity thresholds must be in (0,1]")
    require(args.exact_min_words >= 3 and args.near_min_words >= 5 and args.shared_words >= 1
            and args.group_min_words >= args.exact_min_words and args.progress_every > 0,
            "Invalid minimum length or progress threshold")
    try:
        audit(args)
    except Exception as exc:
        print(f"Audit failed; partial outputs are not usable: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
