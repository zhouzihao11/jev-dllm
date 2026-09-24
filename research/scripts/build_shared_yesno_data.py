"""Build grouped shared-Yes/No data locally on CPU, without loading model weights."""
import argparse
from collections import Counter, defaultdict, deque
import json
import math
from pathlib import Path
import random
import re
from types import SimpleNamespace
import unicodedata


SCHEMA = "shared_yesno_data_v1"
QUOTAS = dict(clinc=2000, snli=3000, arc=1000, sgd=1000,
              synthetic_constraints=1000, synthetic_ordinal=1500,
              synthetic_probability=500)
REAL = {"clinc", "snli", "arc", "sgd"}
SPLITS = ("train", "dev", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def progress(stage, **details):
    print(dumps(dict(stage=stage, **details)), flush=True)


def write_json(path, value):
    path.write_text(dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(dumps(row) + "\n")


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def normalized(value):
    if isinstance(value, str):
        return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()
    if isinstance(value, dict):
        return {key: normalized(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


def validate_record(row, source):
    require(isinstance(row, dict), "record must be an object")
    require(row.get("schema_version") == SCHEMA, "wrong schema_version")
    for key in ("case_id", "group_id", "view_id", "family"):
        require(isinstance(row.get(key), str) and bool(row[key]), "missing string " + key)
    require(row.get("language") == "en", "only English records are supported")
    require(isinstance(row.get("state"), (str, dict)), "state must be string/object")
    provenance = row.get("source")
    require(isinstance(provenance, dict) and provenance.get("name") == source,
            "source.name must equal pool key")
    require(isinstance(provenance.get("record_id"), str) and bool(provenance["record_id"]),
            "source.record_id must be a nonempty string")
    require(provenance.get("original_split") == ("train" if source in REAL else "generated"),
            "real sources must be train-only; synthetic sources must be generated")
    metadata = row.get("metadata")
    require(isinstance(metadata, dict) and isinstance(metadata.get("sampling_stratum"), str)
            and bool(metadata["sampling_stratum"]), "missing sampling_stratum")
    questions = row.get("questions")
    require(isinstance(questions, dict) and len(questions) == 1, "expected one decision per row")
    qid, question = next(iter(questions.items()))
    require(isinstance(qid, str) and bool(qid) and isinstance(question, dict), "invalid question")
    require(isinstance(question.get("instructions"), str) and bool(question["instructions"]),
            "missing instructions")
    qtype, criteria = question.get("type"), question.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("choice needs display-key criteria")
        require(all(isinstance(k, str) and bool(k) and (v is None or isinstance(v, str))
                    for k, v in criteria.items()), "choice criteria must be descriptions/null")
        order = list(criteria)
        require(len(set(normalized(k) for k in order)) == len(order), "duplicate choice labels")
    elif qtype == "score":
        if (not isinstance(criteria, list) or not criteria
                or not all(isinstance(c, str) for c in criteria)):
            raise ValueError("score needs ordered string levels")
        order = [str(i) for i in range(len(criteria))]
    elif qtype == "noul":
        require(criteria is None or (isinstance(criteria, dict)
                and set(criteria) <= {"false", "true"}
                and all(v is None or isinstance(v, str) for v in criteria.values())),
                "noul criteria must be optional false/true descriptions")
        order = ["false", "true"]
    else:
        raise ValueError("unsupported question type")
    require(len(order) >= 2, "choice/score decisions require at least two options")
    gold, option_ids = row.get("gold"), row.get("option_ids")
    require(isinstance(gold, dict) and set(gold) == {qid}, "gold question IDs mismatch")
    require(isinstance(option_ids, dict) and set(option_ids) == {qid}, "option question IDs mismatch")
    ids = option_ids[qid]
    require(isinstance(ids, list) and len(ids) == len(order)
            and all(isinstance(i, str) and bool(i) for i in ids)
            and len(set(ids)) == len(ids), "option IDs must be unique and match displayed count")
    target = gold[qid]
    require(isinstance(target, dict) and target.get("kind") in ("hard", "known_distribution"),
            "invalid gold kind")
    probabilities = target.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(order):
        raise ValueError("target must exhaustively match displayed keys (score: contiguous 0..K-1)")
    if qtype == "score":
        require(list(probabilities) == order, "score probability keys must be in ascending level order")
    values = [probabilities[key] for key in order]
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v) and 0 <= v <= 1 for v in values),
            "target must be finite and nonnegative")
    require(math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-8), "target must sum to one")
    if target["kind"] == "hard":
        require(values.count(1) == 1 and all(v in (0, 1) for v in values), "hard gold must be one-hot")
    dumps(row)
    return qid, question, order, values


def validate_identity(row, seen_pairs, seen_views, cases):
    pair = (row["case_id"], row["view_id"])
    require(pair not in seen_pairs, "duplicate (case_id, view_id): " + str(pair))
    provenance = (row["group_id"], row["source"])
    require(row["case_id"] not in cases or cases[row["case_id"]] == provenance,
            "case_id has inconsistent group/source: " + row["case_id"])
    view = (row["source"]["name"], row["source"]["record_id"], row["view_id"])
    require(view not in seen_views, "duplicate view_id within source record: " + str(view))
    seen_pairs.add(pair)
    seen_views.add(view)
    cases[row["case_id"]] = provenance


class Components:
    def __init__(self):
        self.parent = {}

    def root(self, key):
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while key != root:
            parent = self.parent[key]
            self.parent[key] = root
            key = parent
        return root

    def join(self, left, right):
        self.parent[self.root(right)] = self.root(left)


def render(row, tok, mask_id, mask_text, max_length, decision_input):
    qid, question, order, target = validate_record(row, row["source"]["name"])
    ids, positions, options, prompt = decision_input(
        tok, row["state"], question, mask_id, mask_text, max_length)
    count = 1 if question["type"] == "noul" else len(order)
    require(len(options) == len(order) and len(positions) == count, "rendered option/mask count mismatch")
    require(positions == [i for i, token in enumerate(ids) if token == mask_id],
            "mask positions disagree with token IDs")
    rendered = {key: row[key] for key in ("case_id", "group_id", "view_id", "source")}
    rendered.update(qid=qid, option_order=order, option_ids=row["option_ids"][qid],
                    options=options, input_ids=ids, mask_positions=positions, target=target,
                    type=question["type"], length=len(ids))
    if "split" in row:
        rendered["split"] = row["split"]
    return rendered, prompt


def semantic_keys(row, rendered, prompt):
    question = next(iter(row["questions"].values()))
    # Noul criteria are not visible in the evaluator prompt; do not use them as a key.
    options = [] if question["type"] == "noul" else rendered["options"]
    structural = [normalized(row["state"]), normalized(question["instructions"]),
                  question["type"], normalized(options)]
    return ("prompt", normalized(prompt)), ("structure", dumps(structural))


def special_literals(tok, mask_text):
    literals = set(tok.all_special_tokens) | {mask_text}
    literals.update(str(token) for token in tok.get_added_vocab()
                    if (str(token).startswith("<") and str(token).endswith(">")))
    template = tok.apply_chat_template(
        [{"role": "system", "content": "probe"}, {"role": "user", "content": "probe"}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False)
    literals.update(re.findall(r"<[^<>\n]+>|\[\/?INST\]", template))
    return sorted(literal for literal in literals if literal)


def prepare(pools, tok, mask_id, mask_text, max_length, decision_input, synthetic_validator,
            quarantine, source_provenance):
    groups, duplicates = Components(), Components()
    candidates, seen_pairs, seen_views, seen_keys, cases = [], set(), set(), {}, {}
    literals = special_literals(tok, mask_text)
    validator_count, render_attempts = 0, 0
    for source in QUOTAS:
        progress("prepare_source", source=source, rows=len(pools[source]))
        for row in pools[source]:
            try:
                validate_record(row, source)
            except (ValueError, TypeError, KeyError) as exc:
                quarantine(row, "invalid_record", str(exc))
                continue
            validate_identity(row, seen_pairs, seen_views, cases)
            if source in REAL and "revision" in row["source"]:
                require(row["source"]["revision"] == source_provenance[source]["revision"],
                        "row revision disagrees with source_manifest: " + source)
            group = ("group", row["group_id"])
            groups.join(group, ("record", source, row["source"]["record_id"]))
            if source not in REAL and synthetic_validator is not None:
                try:
                    result = synthetic_validator(row)
                    require(result is not False, "synthetic validator returned False")
                    validator_count += 1
                except (ValueError, TypeError, KeyError, AssertionError) as exc:
                    quarantine(row, "solver_recomputation_validation", str(exc))
                    continue
            visible = dumps({"state": row["state"], "questions": row["questions"]})
            if any(literal in visible for literal in literals):
                quarantine(row, "special_token_literal")
                continue
            render_attempts += 1
            if render_attempts % 5000 == 0:
                progress("render_candidates", attempted=render_attempts, source=source,
                         accepted_before_dedup=len(candidates))
            try:
                rendered, prompt = render(row, tok, mask_id, mask_text, max_length, decision_input)
            except ValueError as exc:
                if "no truncation" not in str(exc):
                    raise
                quarantine(row, "overlength", str(exc))
                continue
            index = len(candidates)
            duplicates.root(index)
            for key in semantic_keys(row, rendered, prompt):
                if key in seen_keys:
                    other = seen_keys[key]
                    duplicates.join(index, other)
                    groups.join(group, ("group", candidates[other][0]["group_id"]))
                else:
                    seen_keys[key] = index
            candidates.append((row, tuple(rendered["target"])))
        progress("prepared_source", source=source, render_attempts_total=render_attempts,
                 accepted_total_before_dedup=len(candidates))
    sets = defaultdict(list)
    for index in range(len(candidates)):
        sets[duplicates.root(index)].append(index)
    retained = []
    for members in sets.values():
        targets = {candidates[i][1] for i in members}
        if len(targets) > 1:
            for index in members:
                quarantine(candidates[index][0], "ambiguous_semantic_duplicate")
            continue
        members.sort(key=lambda i: (candidates[i][0]["source"]["name"],
                                    candidates[i][0]["case_id"], candidates[i][0]["view_id"]))
        retained.append(candidates[members[0]][0])
        for index in members[1:]:
            quarantine(candidates[index][0], "redundant_semantic_duplicate")
    return retained, groups, validator_count


def split_groups(rows, groups, seed, stratify_sources=()):
    components = defaultdict(list)
    for row in rows:
        components[groups.root(("group", row["group_id"]))].append(row)
    by_sources = defaultdict(list)
    for component, members in components.items():
        signature = tuple(sorted({row["source"]["name"] for row in members}))
        by_sources[signature].append(component)
    assignment = {}
    rng = random.Random(seed)
    for signature in sorted(by_sources):
        keys = sorted(by_sources[signature])
        rng.shuffle(keys)
        buckets = defaultdict(list)
        for key in keys:
            strata = ()
            if len(signature) == 1 and signature[0] in stratify_sources:
                strata = tuple(sorted({row["metadata"]["sampling_stratum"] for row in components[key]}))
            buckets[strata].append(key)
        # Retain the source shuffle order within buckets without perturbing other sources' RNG state.
        for strata in sorted(buckets):
            bucket = buckets[strata]
            heldout = max(1, len(bucket) // 12) if len(bucket) >= 3 else 0
            for index, key in enumerate(bucket):
                assignment[key] = "dev" if index < heldout else "test" if index < 2 * heldout else "train"
    partitions = {split: defaultdict(list) for split in SPLITS}
    for row in rows:
        split = assignment[groups.root(("group", row["group_id"]))]
        partitions[split][row["source"]["name"]].append(row)
    return partitions


def sample_rows(rows, quota, groups, rng):
    strata = defaultdict(lambda: defaultdict(list))
    for row in rows:
        strata[row["metadata"]["sampling_stratum"]][groups.root(("group", row["group_id"]))].append(row)
    queues = {}
    for stratum in sorted(strata):
        group_keys = sorted(strata[stratum])
        rng.shuffle(group_keys)
        queues[stratum] = deque()
        for group in group_keys:
            views = sorted(strata[stratum][group], key=lambda row: (row["case_id"], row["view_id"]))
            rng.shuffle(views)
            queues[stratum].append(deque(views))
    order = sorted(queues)
    rng.shuffle(order)
    active, selected, records = deque(order), [], Counter()
    while active and len(selected) < quota:
        stratum = active.popleft()
        queue = queues[stratum]
        picked = None
        while queue and picked is None:
            views = queue.popleft()
            while views:
                candidate = views.popleft()
                record = (candidate["source"]["name"], candidate["source"]["record_id"])
                if records[record] < 2:
                    picked = candidate
                    records[record] += 1
                    break
            if views:
                queue.append(views)
        if picked is not None:
            selected.append(picked)
        if queue:
            active.append(stratum)
    return selected


def compatibility(row):
    return {"state": dumps(row["state"]), "questions": dumps(row["questions"]),
            "gold": dumps(row["gold"]), "case_id": row["case_id"],
            "group_id": row["group_id"], "view_id": row["view_id"],
            "source": dumps(row["source"]), "split": row["split"]}


def target_statistics(rendered):
    goldpos, classes, class_ids, mass = Counter(), Counter(), Counter(), Counter()
    for item in rendered:
        for index, value in enumerate(item["target"]):
            label = item["option_order"][index]
            mass[label] += value
            if value == max(item["target"]):
                goldpos[str(index)] += 1
                classes[label] += 1
                class_ids[item["option_ids"][index]] += 1
    return dict(rows=len(rendered), goldpos_argmax_including_ties=dict(goldpos),
                class_argmax_including_ties=dict(classes),
                class_id_argmax_including_ties=dict(class_ids), class_target_mass=dict(mass))


def describe(rows, rendered):
    lengths = sorted(item["length"] for item in rendered)
    by_source_k = defaultdict(lambda: defaultdict(list))
    for item in rendered:
        by_source_k[item["source"]["name"]][str(len(item["target"]))].append(item)
    real = sum(row["source"]["name"] in REAL for row in rows)
    return dict(rows=len(rows), sources=dict(Counter(row["source"]["name"] for row in rows)),
                types=dict(Counter(item["type"] for item in rendered)),
                K=dict(Counter(str(len(item["target"])) for item in rendered)),
                lengths=dict(min=lengths[0], max=lengths[-1], mean=sum(lengths) / len(lengths),
                             p50=lengths[len(lengths) // 2], p95=lengths[int((len(lengths) - 1) * .95)]),
                goldpos_argmax_including_ties=target_statistics(rendered)["goldpos_argmax_including_ties"],
                by_source_and_K={source: {k: target_statistics(items) for k, items in buckets.items()}
                                 for source, buckets in by_source_k.items()},
                real_source_rows=real, synthetic_templated_rows=len(rows) - real,
                unique_cases=len({row["case_id"] for row in rows}),
                unique_cases_by_source={source: len({row["case_id"] for row in rows
                                                   if row["source"]["name"] == source}) for source in QUOTAS},
                unique_groups=len({row["group_id"] for row in rows}),
                unique_records=len({(row["source"]["name"], row["source"]["record_id"]) for row in rows}),
                strata={source: dict(Counter(row["metadata"]["sampling_stratum"] for row in rows
                                            if row["source"]["name"] == source)) for source in QUOTAS})


def reopen_validate(output, selected, tok, mask_id, mask_text, max_length, decision_input, pd):
    all_rows, group_sets, record_sets, case_sets, semantic_splits = [], {}, {}, {}, {}
    seen_pairs, seen_views, cases = set(), set(), {}
    counts, statistics = {}, {}
    for split in SPLITS:
        progress("reopen_validate_split", split=split, rows=len(selected[split]))
        rows = read_jsonl(output / "canonical" / (split + ".jsonl"))
        rendered = read_jsonl(output / "rendered" / (split + ".jsonl"))
        table = pd.read_parquet(output / "compatibility" / (split + ".parquet"))
        require(len(rows) == len(rendered) == len(table) == len(selected[split]), "reloaded counts differ")
        require(rows == selected[split], "reloaded canonical rows differ")
        require(table.to_dict("records") == [compatibility(row) for row in rows], "Parquet content mismatch")
        for row, saved in zip(rows, rendered):
            require(row["split"] == split, "split tag mismatch")
            fresh, prompt = render(row, tok, mask_id, mask_text, max_length, decision_input)
            require(fresh == saved, "reloaded render/target/mask correspondence mismatch")
            validate_identity(row, seen_pairs, seen_views, cases)
            for key in semantic_keys(row, fresh, prompt):
                require(key not in semantic_splits, "reloaded semantic duplicate")
                semantic_splits[key] = split
        group_sets[split] = {row["group_id"] for row in rows}
        record_sets[split] = {(row["source"]["name"], row["source"]["record_id"]) for row in rows}
        case_sets[split] = {row["case_id"] for row in rows}
        all_rows.extend(rows)
        counts[split] = len(rows)
        statistics[split] = describe(rows, rendered)
    require(max(Counter((row["source"]["name"], row["source"]["record_id"])
                        for row in all_rows).values()) <= 2, "more than two views per record")
    intersections = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        intersections[left + "/" + right] = {
            "cases": len(case_sets[left] & case_sets[right]),
            "groups": len(group_sets[left] & group_sets[right]),
            "records": len(record_sets[left] & record_sets[right])}
        require(not any(intersections[left + "/" + right].values()), "cross-split group/record leakage")
    return counts, statistics, intersections


def load_manifest(source_dir):
    path = source_dir / "source_manifest.json"
    require(path.is_file(), "missing source provenance: " + str(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    entries = value.get("sources", value) if isinstance(value, dict) else value
    require(isinstance(entries, (dict, list)) and bool(entries),
            "source_manifest must be a source list/mapping or an object with sources list/mapping")
    items = entries.items() if isinstance(entries, dict) else enumerate(entries)
    provenance = {}
    for location, entry in items:
        require(isinstance(entry, dict) and bool(entry),
                "source_manifest source entries must be nonempty provenance objects")
        name = location if isinstance(entries, dict) else entry.get("name")
        require(isinstance(name, str) and bool(name), "source_manifest list entries require name")
        if name not in REAL:
            continue
        require(name not in provenance, "duplicate source_manifest source: " + name)
        revision = entry.get("revision")
        license_declaration = entry.get("license_declaration", entry.get("license"))
        require(isinstance(revision, str) and bool(revision.strip()),
                "source_manifest requires nonempty revision for " + name)
        require(isinstance(license_declaration, (str, list, dict)) and bool(license_declaration)
                and (not isinstance(license_declaration, str) or bool(license_declaration.strip())),
                "source_manifest requires license_declaration or license for " + name)
        entry_path = (["sources"] if isinstance(value, dict) and "sources" in value else []) + [location]
        provenance[name] = dict(manifest_file="source_manifest.json", entry_path=entry_path,
                                revision=revision, license_declaration=license_declaration)
    require(set(provenance) == REAL,
            "source_manifest missing required sources: " + ", ".join(sorted(REAL - set(provenance))))
    return path, value, provenance


def build(args, output):
    import pandas as pd
    from transformers import AutoTokenizer
    from bench_diff_yesno import decision_input, token_mapping
    from shared_yesno_sources import load_source_pools
    import shared_yesno_synthetic

    source_dir = Path(args.source_dir).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    require(source_dir.is_dir() and tokenizer_path.is_dir(), "source/tokenizer paths must be local directories")
    manifest_path, source_manifest, source_provenance = load_manifest(source_dir)
    (output / "source_manifest.json").write_bytes(manifest_path.read_bytes())
    quotas = {split: {source: (2 if args.smoke else max(1, int(count * args.quota_scale /
                                                               (1 if split == "train" else 10))))
                      for source, count in QUOTAS.items()} for split in SPLITS}
    config_path = tokenizer_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=True)
    mapping, mask_id, mask_text = token_mapping(
        tok, SimpleNamespace(config=SimpleNamespace(mask_token_id=config.get("mask_token_id"))))
    manifest = dict(schema_version=SCHEMA, status="building", arguments=vars(args), quotas=quotas,
                    tokenizer_path=str(tokenizer_path), token_mapping=mapping, mask_id=mask_id,
                    mask_text=mask_text, no_training=True, source_provenance=source_provenance,
                    split_scope="Grouped IID heldout records, not template-heldout or semantic-heldout",
                    split_recipe="sorted connected group/record/duplicate components shuffled with the seed "
                                 "within each sorted source signature; single-source components for "
                                 + dumps(sorted(set(args.stratify_source))) +
                                 " additionally bucketed by the sorted set of all member sampling_stratum values, "
                                 "preserving shuffled order; other source signatures remain unstratified; "
                                 "per bucket floor(n/12) dev and test (minimum one when n>=3), remainder train "
                                 "(n<3 entirely train); no components split or views moved after partitioning; "
                                 "round-robin strata and groups, at most two views per source record")
    write_json(output / "build_manifest.json", manifest)
    progress("load_real_pools", max_records_per_source=args.max_records_per_source)
    pools, source_report = load_source_pools(str(source_dir), seed=args.seed,
                                           max_records_per_source=args.max_records_per_source)
    progress("load_synthetic_pools")
    synthetic, synthetic_report = shared_yesno_synthetic.make_synthetic_pools(seed=args.seed)
    require(set(pools) == REAL, "real loader must return exactly clinc/snli/arc/sgd")
    require(set(synthetic) == set(QUOTAS) - REAL, "synthetic loader returned unexpected pools")
    pools = {key: list(value) for key, value in {**pools, **synthetic}.items()}
    progress("pools_loaded", counts={key: len(value) for key, value in pools.items()})
    report: dict = dict(status="building", input_counts={key: len(value) for key, value in pools.items()},
                  source_report=source_report, synthetic_report=synthetic_report,
                  synthetic_scope="Limited templated constraint, ordinal, and known-probability families; "
                                  "not broad semantic coverage or a contamination guarantee.")
    write_json(output / "report.json", report)
    rejections, rejection_sources = Counter(), defaultdict(Counter)
    with (output / "quarantine.jsonl").open("w", encoding="utf-8") as handle:
        def quarantine(row, reason, detail=None):
            row = row if isinstance(row, dict) else {}
            entry = {key: row.get(key) for key in ("case_id", "group_id", "view_id")}
            source = row.get("source", {})
            source = source if isinstance(source, dict) else {}
            entry.update(reason=reason, source=source.get("name"), record_id=source.get("record_id"))
            if detail:
                entry["detail"] = detail[:500]
            handle.write(dumps(entry) + "\n")
            rejections[reason] += 1
            rejection_sources[entry["source"] or "unknown"][reason] += 1

        validator = getattr(shared_yesno_synthetic, "validate_synthetic_record", None)
        rows, groups, checked = prepare(pools, tok, mask_id, mask_text, args.max_length,
                                        decision_input, validator, quarantine, source_provenance)
    progress("split_groups", retained_rows=len(rows), seed=args.seed)
    partitions = split_groups(rows, groups, args.seed, args.stratify_source)
    report.update(rejections=dict(rejections), rejection_sources=dict(rejection_sources),
                  retained_counts=dict(Counter(row["source"]["name"] for row in rows)),
                  solver_recomputation_validation=dict(
                      available=validator is not None, accepted=checked,
                      limitation="Validator shares generator _solve/_presentation implementation; "
                                 "not an independent oracle or correctness proof."),
                  available_by_split={split: {source: len(partitions[split][source]) for source in QUOTAS}
                                      for split in SPLITS})
    write_json(output / "report.json", report)
    progress("groups_split", available_by_split=report["available_by_split"])
    selected, shortages = {}, []
    for split in SPLITS:
        selected[split] = []
        for source in QUOTAS:
            chosen = sample_rows(partitions[split][source], quotas[split][source], groups,
                                 random.Random("%s:%s:%s" % (args.seed, split, source)))
            if len(chosen) < quotas[split][source]:
                shortages.append(dict(source=source, split=split, requested=quotas[split][source],
                                      selectable=len(chosen), candidates=len(partitions[split][source]),
                                      input=report["input_counts"][source]))
            selected[split].extend(dict(row, split=split) for row in chosen)
    report["quota_shortages"] = shortages
    write_json(output / "report.json", report)
    require(not shortages, "source quota shortages (no substitution): " + dumps(shortages))
    progress("quotas_selected", counts={split: len(rows) for split, rows in selected.items()})
    review, review_keys = [], set()
    for directory in ("canonical", "rendered", "compatibility"):
        (output / directory).mkdir()
    for split in SPLITS:
        rows = selected[split]
        progress("render_write_split", split=split, rows=len(rows))
        rendered = [render(row, tok, mask_id, mask_text, args.max_length, decision_input)[0] for row in rows]
        write_jsonl(output / "canonical" / (split + ".jsonl"), rows)
        write_jsonl(output / "rendered" / (split + ".jsonl"), rendered)
        pd.DataFrame([compatibility(row) for row in rows]).to_parquet(
            output / "compatibility" / (split + ".parquet"), engine="pyarrow", index=False)
        progress("split_written", split=split, rows=len(rows))
        for row, item in zip(rows, rendered):
            key = (split, row["source"]["name"], item["type"])
            if key not in review_keys:
                review_keys.add(key)
                review.append(dict(review_status="not_human_reviewed", canonical=row, rendered=item))
    write_jsonl(output / "human_review_samples.jsonl", review)
    counts, statistics, intersections = reopen_validate(
        output, selected, tok, mask_id, mask_text, args.max_length, decision_input, pd)
    total = sum(counts.values())
    real = sum(row["source"]["name"] in REAL for rows in selected.values() for row in rows)
    report.update(status="validated", splits=statistics, intersections=intersections,
                  real_source_rows=real, synthetic_templated_rows=total - real,
                  unique_cases=len({row["case_id"] for rows in selected.values() for row in rows}),
                  real_text_decision_ratio=real / total, reload_validated=True,
                  goldpos_definition="Zero-based argmax positions; all tied maxima counted, not sampled labels.",
                  class_definition="Per source/K: argmax display labels and stable option IDs including ties; "
                                   "class_target_mass sums probabilities by display label, without sampling.")
    write_json(output / "report.json", report)
    card = """# Shared Yes/No dataset build

This is dataset construction, not training or final external benchmark evaluation.
No generative API, model weights, model forward pass, or GPU is used. Human-review
samples are provided for inspection and have NOT been human-reviewed.

Only CLINC, SNLI, ARC, and SGD upstream training records are allowed as real pools.
Synthetic pools are limited templated constraints, ordinal, and known-probability
tasks, not a general semantic reasoning corpus. Source reports are in report.json.
Excluded families are AG News, DAIR Emotion, Banking77, deepset/prompt-injections,
SST/SST5 derivatives, LocalLLaMA/typed-decisions, and Jev evaluation cases, plus
all validation/test splits of the used sources. This is an input-source policy,
not proof of deduplication against those families or absence from upstream texts.

Split recipe: {recipe}. Splitting precedes view selection. Seed: {seed}.
Exact quotas: {quotas}. No quota redistribution or fabricated shortage fillers.
This is a grouped IID reserved-heldout split, NOT a template-heldout split.
Exact selected row counts: {real} real-source; {synthetic} synthetic templated.
Real-text decision ratio (real-source rows / all rows, not token fraction): {ratio}.
This ratio counts source provenance; prompts/questions can be adapter-authored.

Deduplication covers exact rendered prompts and conservatively normalized
model-visible inputs (NFC, whitespace, sorted structured-state keys; punctuation,
case, numbers, and negation preserved). IDs and non-visible metadata are ignored.
Option order is preserved in duplicate keys: option-reordered questions are NOT
matched by deduplication. Adapter group/record links can keep such source variants
together, but do not establish cross-record or cross-source equivalence.
Conflicting-target duplicate components are entirely quarantined; consistent
duplicates keep one deterministic representative. Group/record links, including
duplicate links, stay in one split. Near-semantic contamination is NOT ruled out;
no external heldout benchmark corpus was compared. ARC cross-config stem groups
and all variant groups must be supplied correctly by the source adapter.

Special mask/chat token literals are quarantined, not escaped. The evaluator's
exact decision_input and token_mapping are reused, with no truncation. Noul uses
one mask and targets [false, true]; other types use K masks in displayed order.
All outputs are reopened to check counts, targets, masks, and split disjointness.
The solver_recomputation_validation check shares the synthetic generator's
_solve/_presentation implementation. It is not independent validation and cannot
detect errors common to generation and recomputation.

Public-source license terms are inherited from each upstream source, as recorded
in source_manifest.json, including revision, license documentation, local paths,
and prepared counts. This build does NOT relicense all texts. Check source terms
before redistributing. The source manifest is copied unchanged. Row source.name
maps to build_manifest.json source_provenance, which supplies the declared
revision/license and the entry path in source_manifest.json. Required declarations
are checked for presence, NOT independently verified for legal or factual accuracy.

Compatibility Parquet stores JSON state/questions/gold with one decision per row.
The current train_qwen_masked_typed.py fixed row/item count assumptions remain
UNSUPPORTED. No trainer was modified or invoked. See build_manifest.json for
token mapping and runtime arguments; quarantine.jsonl contains IDs/reasons only.
""".format(recipe=manifest["split_recipe"], seed=args.seed, quotas=dumps(quotas), ratio=real / total,
           real=real, synthetic=total - real)
    (output / "data_card.md").write_text(card, encoding="utf-8")
    manifest["status"] = "validated"
    write_json(output / "build_manifest.json", manifest)
    require(json.loads((output / "source_manifest.json").read_text(encoding="utf-8")) == source_manifest,
            "copied source manifest differs")
    require(json.loads((output / "build_manifest.json").read_text(encoding="utf-8")) == manifest,
            "reloaded build manifest differs")
    # Compare JSON-normalized reports: peer reports may contain tuples or integer keys.
    require(json.loads((output / "report.json").read_text(encoding="utf-8")) == json.loads(dumps(report)),
            "reloaded report differs")
    require(read_jsonl(output / "human_review_samples.jsonl") == review, "review samples differ")
    rejected = read_jsonl(output / "quarantine.jsonl")
    require(dict(Counter(row["reason"] for row in rejected)) == dict(rejections), "quarantine counts differ")
    require((output / "data_card.md").read_text(encoding="utf-8") == card, "data card differs")
    progress("build_complete", status="validated", counts=counts,
             rejected=sum(rejections.values()), output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true", help="Two rows per source per split; full build/reload path")
    parser.add_argument("--quota-scale", type=float, default=1.0,
                        help="Scale quotas, floor each and clamp to one; ignored in smoke mode")
    parser.add_argument("--max-records-per-source", type=int, default=24000)
    parser.add_argument("--stratify-source", action="append", choices=sorted(REAL), default=[],
                        help="Repeatable: partition single-source groups by their full sampling-stratum set")
    args = parser.parse_args()
    if (args.max_length < 1 or args.max_records_per_source < 1
            or not math.isfinite(args.quota_scale) or args.quota_scale <= 0):
        parser.error("length, record limit, and finite quota scale must be positive")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("refusing nonempty existing output directory: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    try:
        build(args, output)
    except Exception as exc:
        for name in ("build_manifest.json", "report.json"):
            path = output / name
            value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            value.update(status="failed", error=str(exc))
            write_json(path, value)
        print(dumps(dict(status="failed", error=str(exc), preserved_output=str(output))))
        raise


if __name__ == "__main__":
    main()
