"""Dataset-only S1 expansion; frozen external fixtures are exclusion-only inputs."""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import re
import shutil
from types import SimpleNamespace
import unicodedata

import build_shared_yesno_data as base
import shared_yesno_sources as sources
import shared_yesno_synthetic as synthetic
import shared_yesno_s1_synthetic as s1_synthetic


PUBLIC = {"dbpedia", "snli", "clinc", "sgd", "goemotions", "boolq"}
S0_SOURCES = base.REAL | set(synthetic.POOL_NAMES)
EXTERNAL = {"typed_decisions", "ag_news", "emotion", "banking77", "prompt_injection", "sst5"}
BUCKETS = (PUBLIC - {"snli"}) | {"snli_choice", "snli_noul"} | set(synthetic.POOL_NAMES)
RENDERER = "bench_diff_yesno.decision_input/shared_yesno_data_v1"


def require(condition, message):
    base.require(condition, message)


def read_json(path):
    require(path.is_file(), "missing required input: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def normal(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def leaves(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from leaves(value[key])
    elif isinstance(value, list):
        for item in value:
            yield from leaves(item)
    elif isinstance(value, (int, float, bool)):
        yield json.dumps(value, allow_nan=False)


def text_units(value):
    parts = list(leaves(value))
    require(bool(parts), "empty original content")
    # Whole original content always participates; long individual fields also block reuse.
    return list(dict.fromkeys([" ".join(parts)] + [p for p in parts if len(normal(p).split()) >= 8]))


def typed_state(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def content(row):
    state = row["state"]
    if row["source"]["name"] in synthetic.POOL_NAMES:
        return [s1_synthetic.canonical_facts(row)]
    if isinstance(state, dict) and "dialogue_prefix" in state:
        return [" ".join(t["utterance"] for t in state["dialogue_prefix"])]
    return text_units(state)


def external_content(row):
    if row["source"]["name"] in synthetic.POOL_NAMES:
        # Compare complete observations plus policy, not isolated shared rule text.
        return [" ".join(leaves(row["state"]))]
    return text_units(row["state"]) + content(row)


class TextIndex:
    """Exact normalized text plus bounded lexical three-word-shingle Jaccard."""
    def __init__(self, config):
        self.threshold = config["near_duplicate_threshold"]
        self.max_postings = config["near_duplicate_max_postings"]
        self.max_candidates = config["near_duplicate_max_candidates"]
        self.exact, self.sets, self.postings = set(), [], defaultdict(list)
        self.stats = Counter()

    @staticmethod
    def shingles(text):
        words = text.split()
        return frozenset(zip(words, words[1:], words[2:])) if len(words) >= 8 else frozenset()

    def add(self, texts):
        for text in texts:
            text = normal(text)
            if not text or text in self.exact:
                continue
            self.exact.add(text)
            tokens = self.shingles(text)
            if not tokens:
                continue
            index = len(self.sets)
            self.sets.append(tokens)
            for token in tokens:
                self.postings[token].append(index)

    def match(self, texts, near=True):
        for text in texts:
            text = normal(text)
            if text in self.exact:
                return "exact"
            if not near:
                continue
            tokens = self.shingles(text)
            if not tokens:
                continue
            postings = []
            for token in sorted(tokens):
                ids = self.postings.get(token, ())
                if len(ids) > self.max_postings:
                    self.stats["skipped_common_postings"] += 1
                elif ids:
                    postings.append(ids)
            candidates = set()
            for ids in sorted(postings, key=len):
                for index in ids:
                    if index not in candidates and len(candidates) >= self.max_candidates:
                        break
                    candidates.add(index)
                if len(candidates) >= self.max_candidates:
                    break
            if len(candidates) >= self.max_candidates:
                self.stats["candidate_cap_queries"] += 1
            for index in sorted(candidates):
                other = self.sets[index]
                if min(len(tokens), len(other)) < self.threshold * max(len(tokens), len(other)):
                    continue
                intersection = len(tokens & other)
                if intersection / (len(tokens) + len(other) - intersection) >= self.threshold:
                    return "near"
        return None


def exclusion_index(profile_path, config):
    """No labels, benchmark examples, or match identities escape this function."""
    profile = read_json(profile_path)
    require(profile.get("schema_version") == "full_eval_v1", "expected frozen full_eval_v1 profile")
    entries = profile.get("sources", {})
    require(EXTERNAL <= set(entries), "profile must contain all six external source families")
    index, counts = TextIndex(config), {}
    for name in sorted(EXTERNAL):
        entry = entries[name]
        path = (profile_path.parent / entry["path"]).resolve()
        require(path.is_file(), "unreadable exclusion fixture for " + name + ": " + str(path))
        field = "state" if name == "typed_decisions" else "text"
        count = 0
        try:
            if entry["format"] == "parquet":
                import pyarrow.parquet as pq
                batches = pq.ParquetFile(path).iter_batches(columns=[field], batch_size=512)
                values = (value for batch in batches for value in batch.column(0).to_pylist())
                for value in values:
                    if field == "state":
                        value = typed_state(value)
                    index.add(text_units(value))
                    count += 1
            elif entry["format"] in ("jsonl", "csv"):
                with path.open(encoding="utf-8", newline="") as stream:
                    rows = csv.DictReader(stream) if entry["format"] == "csv" else (json.loads(x) for x in stream)
                    for row in rows:
                        value = row[field]
                        if field == "state":
                            value = typed_state(value)
                        require(isinstance(value, (str, dict)), "invalid exclusion content")
                        index.add(text_units(value))
                        count += 1
            else:
                raise ValueError("unsupported fixture format")
            require(count > 0 and count == entry["rows"], "fixture row count mismatch")
        except Exception as exc:
            # Never echo fixture values or library errors containing external text.
            raise ValueError("cannot read complete exclusion fixture %s (%s)" %
                             (name, type(exc).__name__)) from None
        counts[name] = count
    return index, counts


def manifest(root, required):
    value = read_json(root / "source_manifest.json")
    entries = value.get("sources", value)
    if isinstance(entries, list):
        entries = {item["name"]: item for item in entries}
    require(isinstance(entries, dict), "source_manifest sources must be a mapping/list")
    result = {}
    for name in sorted(required):
        entry = entries.get(name)
        if not isinstance(entry, dict):
            raise ValueError("missing source_manifest entry: " + name)
        require(isinstance(entry.get("revision"), str) and entry["revision"].strip(),
                "missing revision for " + name)
        require(bool(entry.get("license", entry.get("license_declaration"))), "missing license for " + name)
        lineage = json.dumps({k: entry[k] for k in ("upstream", "dataset", "repo_id", "dataset_name") if k in entry}).casefold()
        require(not re.search(r"typed[-_]decisions|ag[_ -]news|dair-ai/emotion|banking77|"
                              r"prompt[-_]injections|stanford[_ /-]sst|sst[-_]?5|sst[-_]?2", lineage),
                "forbidden source family in declared lineage: " + name)
        result[name] = entry
    return result


def source_report():
    return dict(lines_read=0, missing_inputs=[], input_errors=[], rejection_counts=Counter(),
                rejections=[], grouping_counts=Counter(), views_before_dialogue_cap=0,
                views_omitted_by_dialogue_cap=0)


def raw_pools(args, cap):
    reports, pools, ontology, none_services = {}, {}, {}, set()
    for name in sorted(PUBLIC):
        report = reports[name] = source_report()
        root = args.source_dir if name in {"snli", "clinc", "sgd"} else args.new_source_dir
        path = root / "prepared" / (name + "_train.jsonl")
        require(path.is_file(), "missing required prepared source: " + str(path))

        def observe(row, rid):
            if name in {"dbpedia", "goemotions", "boolq"}:
                require(isinstance(row.get("record_id"), str), "new source record_id must be a string: " + name)
            if name == "clinc":
                require(all(sources._text(row.get(k)) for k in ("text", "intent", "domain")),
                        "malformed CLINC record " + rid)
                label, domain = row["intent"], row["domain"]
                require(label not in ontology or ontology[label] == domain, "conflicting CLINC ontology")
                ontology[label] = domain
                if domain in sources._EXCLUDED_DOMAINS or label in ("oos", "out_of_scope"):
                    sources._reject(report, "excluded_domain_or_out_of_scope", rid)
                    return False
            elif name == "sgd":
                for turn in row.get("turns", []):
                    if turn.get("speaker") == "USER":
                        for frame in turn.get("frames", []):
                            if frame.get("state", {}).get("active_intent") == "NONE":
                                none_services.add(frame["service"])
            elif name == "goemotions":
                require(isinstance(row.get("labels"), list), "missing GoEmotions labels: " + rid)
                if len(row["labels"]) != 1:
                    sources._reject(report, "not_single_label", rid)
                    return False
            return True

        pools[name] = sources._reservoir(path, name, args.seed, cap, report, observe)
        require(not report["missing_inputs"] and not report["input_errors"], "unreadable source " + name)
        require(bool(pools[name]), "no usable original train records for " + name)
    ontology = {k: v for k, v in ontology.items()
                if v not in sources._EXCLUDED_DOMAINS and k not in ("oos", "out_of_scope")}
    require(len(ontology) == 120, "expected complete nonfinancial CLINC ontology (120 intents)")
    schema_path = args.source_dir / "prepared" / "sgd_schema.json"
    if not schema_path.is_file():
        schema_path = args.source_dir / "sgd_schema.json"
    schemas = sources._sgd_schema(read_json(schema_path), reports["sgd"])
    require(bool(schemas) and not reports["sgd"]["input_errors"], "unusable SGD schema")
    labels = {}
    for name, count in (("dbpedia", 14), ("goemotions", 28)):
        labels[name] = read_json(args.new_source_dir / "prepared" / (name + "_labels.json"))
        require(isinstance(labels[name], list) and len(labels[name]) == count
                and all(sources._text(x) for x in labels[name])
                and len(set(labels[name])) == count, "invalid complete ontology: " + name)
    require("neutral" in labels["goemotions"], "GoEmotions ontology must include neutral")
    return pools, reports, ontology, schemas, none_services, labels


def original_group(name, raw, rid):
    if name == "snli":
        require(sources._identifier(raw.get("group_key")), "SNLI original image group_key required: " + rid)
        group = ("source_group", str(raw["group_key"]))
    elif name == "clinc":
        group = sources._normal(raw["text"])
    elif name == "boolq":
        group = sources._normal(raw["passage"])
    else:
        group = rid
    return "%s:%s" % (name, sources._digest(group))


def public_decision(name, raw, rid, bucket, seed, ontology, schemas, none_services, labels, report):
    require(all(raw[key] == "train" for key in ("original_split", "split") if key in raw),
            "non-train original split reached converter: " + name + ":" + rid)
    require(str(raw["dialogue_id" if name == "sgd" else "record_id"]) == rid,
            "original raw identity mismatch: " + name)
    if name == "clinc":
        rows = sources._clinc([(raw, rid)], ontology, seed, report,
                              k_choices=[sources._rng(seed, name, rid, "K").choice([2, 4, 8, 16, 32, 64, 120])])
        return rows[0] if rows else None
    if name == "sgd":
        rows = sources._sgd([(raw, rid)], schemas, none_services, seed, report)
        return sources._rng(seed, name, rid, "one_decision").choice(rows) if rows else None
    if name == "snli":
        if raw.get("label") not in ("entailment", "contradiction", "neutral"):
            sources._reject(report, "invalid_snli_label", rid)
            return None
        require(all(sources._text(raw.get(k)) for k in ("premise", "hypothesis")), "empty SNLI text: " + rid)
        group = ("source_group", str(raw["group_key"]))
        state = {k: raw[k] for k in ("premise", "hypothesis")}
        if bucket == "snli_choice":
            # Reuse the inherited ontology and exact relation wording.
            return sources._snli([(raw, rid)], seed, report)[0]
        return sources._decision(
            name, rid, group, "binary_entailment", "natural_language_inference", state,
            "Does the premise entail the hypothesis? Answer about this relation, not whether "
            "the hypothesis is true in the real world.",
            [("false", "false", "The specified relation does not hold."),
             ("true", "true", "The specified relation holds.")],
            str(raw["label"] == "entailment").lower(), "binary_entailment:" + raw["label"], qtype="noul")
    if name == "boolq":
        require(type(raw.get("answer")) is bool, "BoolQ answer must be boolean: " + rid)
        require(all(sources._text(raw.get(k)) for k in ("passage", "question")), "empty BoolQ text: " + rid)
        return sources._decision(name, rid, sources._normal(raw["passage"]), "answer", "reading_comprehension",
                                 {"passage": raw["passage"], "question": raw["question"]}, raw["question"],
                                 [("false", "false", None), ("true", "true", None)],
                                 str(raw["answer"]).lower(), "answer:" + str(raw["answer"]), qtype="noul")
    label = raw["label"] if name == "dbpedia" else raw["labels"][0]
    require(label in labels[name], "label outside original ontology: " + name + ":" + rid)
    if name == "dbpedia":
        require(all(isinstance(raw.get(k), str) for k in ("title", "content"))
                and bool(raw["content"].strip()), "invalid DBpedia text: " + rid)
        state = {k: raw[k] for k in ("title", "content")}
        instruction = "Which original DBpedia ontology class describes this entity?"
    else:
        require(sources._text(raw.get("text")), "empty GoEmotions text: " + rid)
        state, instruction = raw["text"], "Which original emotion label best describes this comment?"
    options = [(x, x, None) for x in labels[name]]
    sources._rng(seed, name, rid, "order").shuffle(options)
    return sources._decision(name, rid, rid, "original_label", name + "_classification", state,
                             instruction, options, label, "label:" + label)


def check_public(row, raw):
    """Independently map original annotations to stable IDs after option permutation."""
    name = row["source"]["name"]
    if name == "snli":
        expected = raw["label"] if row["view_id"] == "three_way" else str(raw["label"] == "entailment").lower()
    elif name == "sgd":
        ti, service = row["metadata"]["turn_index"], row["metadata"]["service"]
        frames = [f for f in raw["turns"][ti]["frames"] if f["service"] == service]
        require(len(frames) == 1, "ambiguous current SGD frame")
        expected = frames[0]["state"]["active_intent"]
        prefix = [{k: t[k] for k in ("speaker", "utterance")} for t in raw["turns"][:ti + 1]]
        require(row["state"]["dialogue_prefix"] == prefix and raw["turns"][ti]["speaker"] == "USER",
                "SGD input must stop at annotated current USER turn")
    elif name == "boolq":
        expected = str(raw["answer"]).lower()
    else:
        expected = raw[{"dbpedia": "label", "clinc": "intent", "goemotions": "labels"}[name]]
        if name == "goemotions":
            require(len(expected) == 1, "multi-label GoEmotions cannot become exclusive choice")
            expected = expected[0]
    probabilities = list(row["gold"]["decision"]["probabilities"].values())
    require(probabilities == [float(oid == expected) for oid in row["option_ids"]["decision"]],
            "original annotation/option permutation mismatch")


def check_synthetic(row):
    """Second implementation; no generator solver or stored fractions used as oracle."""
    synthetic.validate_synthetic_record(row)
    f, q, family = row["metadata"]["solver_facts"], row["metadata"]["query"], row["family"]
    answer, distribution = None, None
    if family == "eligibility":
        candidates = sorted((c["cost"], c["name"]) for c in f["candidates"]
                            if c["certified"] and c["capacity"] >= f["minimum_capacity"] and c["delay"] <= f["maximum_delay"])
        require(candidates and (len(candidates) == 1 or candidates[0][0] != candidates[1][0]), "nonunique minimum")
        answer = candidates[0][1]
    elif family == "preservation":
        violations = [not f["authorized"], f["frozen"], f["balance"] + f["change"] < f["reserve"],
                      f["balance"] + f["change"] > f["ceiling"]]
        answer = str(not any(violations)).lower()
    elif family == "ownership":
        owners = [f["initial_owner"]]
        for event in f["events"]:
            owners.append(event["recipient"] if event["accepted"] and event["sender"] == owners[-1] else owners[-1])
        answer = str(owners[-1] == q["person"]).lower() if q["type"] == "noul" else owners[-1]
    elif family in ("shipment", "laboratory", "deployment"):
        def passed(check, version):
            return (check["version"], check["status"]) == (version, "passed")
        if family == "shipment":
            gates = [f["packed"] >= f["required"] and f["seal_intact"] and f["inspection"] == "passed",
                     f["handed_over"] and f["receipt_validation"] == "passed"]
        elif family == "laboratory":
            gates = [f["collected"] >= f["required"] + f["damaged"],
                     passed(f["assay"], f["batch_version"]) and f["control"] in range(f["control_low"], f["control_high"] + 1),
                     passed(f["review"], f["assay_version"])]
        else:
            gates = [passed(f[k], f["current_version"]) for k in ("build", "tests", "approval", "rollout")]
            gates[1] = gates[1] and f["tests_passed"] == f["tests_required"]
            gates[3] = gates[3] and f["healthy_zones"] == f["required_zones"]
        answer = str(next((i for i, value in enumerate(gates) if not value), len(gates)))
    elif family == "finite_draw":
        outcomes = [color for color, count in f["counts"].items() for _ in range(count)]
        distribution = {color: Fraction(outcomes.count(color), len(outcomes)) for color in f["counts"]}
        if q["type"] == "noul":
            p = distribution[q["color"]]
            distribution = {"false": 1 - p, "true": p}
    elif family == "latent_mixture":
        joint, evidence = Fraction(0), Fraction(0)
        for m in f["mechanisms"]:
            alert = Fraction(m["prior_weight"] * m["signal_count"], m["signal_total"])
            evidence += alert
            joint += alert * Fraction(m["success_count"], m["outcome_total"])
        p = joint / evidence
        distribution = {"false": 1 - p, "true": p} if q["type"] == "noul" else {"Success": p, "Failure": 1 - p}
    elif family == "without_replacement":
        # Dynamic programming over ordered draws, independent of the combinatorial solver.
        mass = {0: Fraction(1)}
        for draw in range(f["draws"]):
            following = defaultdict(Fraction)
            for red, p in mass.items():
                denominator = f["red"] + f["blue"] - draw
                if f["red"] > red:
                    following[red + 1] += p * Fraction(f["red"] - red, denominator)
                if f["blue"] > draw - red:
                    following[red] += p * Fraction(f["blue"] - draw + red, denominator)
            mass = following
        distribution = {str(k): v for k, v in mass.items()}
    else:
        raise ValueError("unsupported independent synthetic check: " + family)
    distribution = distribution if distribution is not None else {answer: Fraction(1)}
    for key, value in row["gold"]["decision"]["probabilities"].items():
        require(math.isclose(value, float(distribution.get(key, 0)), rel_tol=0, abs_tol=1e-12),
                "independent synthetic target mismatch")


def render_check(row, tok, mask_id, mask_text, max_length, decision_input):
    # The inherited validator's real-source allowlist is S0-specific. Validate the
    # same schema through an alias, without modifying the row or its provenance.
    alias = row
    if row["source"]["name"] in PUBLIC - base.REAL:
        alias = dict(row, source=dict(row["source"], name="snli"))
    qid, question, order, target = base.validate_record(alias, alias["source"]["name"])
    ids, positions, options, _ = decision_input(tok, row["state"], question, mask_id, mask_text, max_length)
    require(len(options) == len(target) and len(positions) == (1 if question["type"] == "noul" else len(target)),
            "rendered masks/options/target mismatch")
    return dict(length=len(ids), type=question["type"], target=target,
                option_ids=row["option_ids"][qid], option_order=order)


def describe(rows, rendered):
    return dict(rows=len(rows), sources=dict(Counter(r["source"]["name"] for r in rows)),
                public_rows=sum(r["source"]["name"] in base.REAL | PUBLIC for r in rows),
                synthetic_rows=sum(r["source"]["name"] in synthetic.POOL_NAMES for r in rows),
                original_records=len({(r["source"]["name"], r["source"]["record_id"]) for r in rows}),
                groups=len({r["group_id"] for r in rows}),
                families=dict(Counter(r["family"] for r in rows)),
                types=dict(Counter(r["type"] for r in rendered)),
                K=dict(Counter(str(len(r["target"])) for r in rendered)),
                length_histogram_256=dict(Counter(str((r["length"] // 256) * 256) for r in rendered)),
                max_length=max((r["length"] for r in rendered), default=0),
                gold_by_source={name: base.target_statistics([r for row, r in zip(rows, rendered)
                                                             if row["source"]["name"] == name])
                                for name in sorted({row["source"]["name"] for row in rows})},
                synthetic_family_targets={family: base.target_statistics([r for row, r in zip(rows, rendered)
                                                                          if row["family"] == family])
                                          for family in sorted({row["family"] for row in rows
                                                                if row["source"]["name"] in synthetic.POOL_NAMES})})


def build(args, output, config):
    from transformers import AutoTokenizer
    from bench_diff_yesno import decision_input, token_mapping

    s0_manifest = manifest(args.s0_dir, base.REAL)
    provenance = manifest(args.source_dir, {"snli", "clinc", "sgd"})
    for name in provenance:
        require(provenance[name]["revision"] == s0_manifest[name]["revision"],
                "S0/prepared revision mismatch: " + name)
    new_manifest = manifest(args.new_source_dir, {"dbpedia", "goemotions", "boolq"})
    for name, entry in new_manifest.items():
        require(bool(entry.get("upstream")), "new source requires upstream: " + name)
    provenance.update(new_manifest)
    s0 = {split: base.read_jsonl(args.s0_dir / "canonical" / (split + ".jsonl"))
          for split in ("train", "dev", "test")}
    require(all(s0.values()), "S0 canonical splits must be nonempty")
    all_s0 = list(itertools.chain.from_iterable(s0.values()))
    s0_group_splits, s0_record_splits = {}, {}
    for split, rows in s0.items():
        for row in rows:
            require(row["split"] == split, "S0 split tag mismatch")
            require(row["source"]["name"] in S0_SOURCES, "forbidden/unknown S0 source")
            base.validate_record(row, row["source"]["name"])
            for mapping, key in ((s0_group_splits, row["group_id"]),
                                 (s0_record_splits, (row["source"]["name"], row["source"]["record_id"]))):
                require(mapping.setdefault(key, split) == split, "inherited S0 cross-split group/record overlap")
            if row["source"].get("revision") and row["source"]["name"] in base.REAL:
                require(row["source"]["revision"] == s0_manifest[row["source"]["name"]]["revision"],
                        "S0 row revision disagrees with its manifest")
    s0_build = read_json(args.s0_dir / "build_manifest.json")
    s0_seed = s0_build["arguments"]["seed"]
    require(config["synthetic_seed"] != s0_seed, "synthetic seed must differ from S0 generation seed")
    firewall, fixture_counts = exclusion_index(args.heldout_profile, config)
    blocked = TextIndex(config)
    s0_facts, new_facts = set(), set()
    for row in all_s0:
        if row["source"]["name"] in synthetic.POOL_NAMES:
            s0_facts.add(s1_synthetic.canonical_facts(row))
        else:
            blocked.add(content(row))
    groups_s0 = {r["group_id"] for r in all_s0}
    records_s0 = {(r["source"]["name"], r["source"]["record_id"]) for r in all_s0}
    tokenizer_config = read_json(args.tokenizer_path / "config.json")
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer_path), local_files_only=True, trust_remote_code=True)
    mapping, mask_id, mask_text = token_mapping(
        tok, SimpleNamespace(config=SimpleNamespace(mask_token_id=tokenizer_config.get("mask_token_id"))))
    literals = base.special_literals(tok, mask_text)
    pools, loader_reports, ontology, schemas, none_services, labels = raw_pools(args, config["max_records_per_source"])
    quotas = config["quotas"]
    if args.smoke:
        quotas = {name: {split: min(2, n) for split, n in values.items()} for name, values in quotas.items()}
    rejected, audit = defaultdict(Counter), defaultdict(Counter)
    selected, renders = {"train": [], "dev": []}, {"train": [], "dev": []}
    selected_counts, available, checked = Counter(), Counter(), Counter()
    seen_groups, seen_records = {}, {}
    additions = TextIndex(config)
    report: dict = dict(status="building", arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  config=config, effective_quotas=quotas, exclusion_fixture_rows=fixture_counts,
                  selection_dev="original S0 dev; mean-KL selection unchanged; new_dev diagnostic only")
    base.write_json(output / "report.json", report)
    with (output / "rejections.jsonl").open("w", encoding="utf-8") as rejection_file:
        def reject(row, reason):
            name = row["source"]["name"]
            rejected[name][reason] += 1
            rejection_file.write(base.dumps(dict(source=name, record_id=row["source"]["record_id"],
                                                  case_id=row.get("case_id"), reason=reason)) + "\n")

        retained_s0 = []
        for split, rows in s0.items():
            for row in rows:
                hit = firewall.match(external_content(row),
                                     near=row["source"]["name"] not in synthetic.POOL_NAMES)
                if hit:
                    audit[split]["external_" + hit] += 1
                    if split == "train":
                        reject(row, "s0_train_external_" + hit)
                        continue
                if split == "train":
                    retained_s0.append(row)

        def consider(row, bucket, split, raw=None):
            name = row["source"]["name"]
            available[bucket, split] += 1
            if selected_counts[bucket, split] >= quotas[bucket][split]:
                reject(row, "quota_surplus")
                return
            group, record = row["group_id"], (name, row["source"]["record_id"])
            if name in PUBLIC and (group in groups_s0 or record in records_s0):
                reject(row, "s0_group_or_original_id")
                return
            if group in seen_groups or record in seen_records:
                reject(row, "repeated_new_group_or_record")
                return
            texts = content(row)
            is_synthetic = name in synthetic.POOL_NAMES
            indices = [(firewall, "external")]
            if not is_synthetic:
                indices += [(blocked, "s0_content"), (additions, "new_content")]
            for index, reason in indices:
                hit = index.match(external_content(row) if index is firewall else texts,
                                  near=not is_synthetic)
                if hit:
                    reject(row, reason + "_" + hit)
                    return
            if any(literal in base.dumps({"state": row["state"], "questions": row["questions"]}) for literal in literals):
                reject(row, "special_token_literal")
                return
            if raw is None:
                check_synthetic(row)
            else:
                check_public(row, raw)
            checked[name] += 1
            try:
                rendered = render_check(row, tok, mask_id, mask_text, args.max_length, decision_input)
            except ValueError as exc:
                if "no truncation" not in str(exc):
                    raise
                reject(row, "overlength")
                return
            row.update(split=split, renderer_version=RENDERER)
            row["source"]["revision"] = (provenance[name]["revision"] if name in PUBLIC else
                                          "shared_yesno_synthetic_v1+s1_parameter_supplement_v1"
                                          if row["metadata"].get("generation") == "s1_parameter_supplement_v1"
                                          else "shared_yesno_synthetic_v1")
            row["metadata"].update(s1_bucket=bucket, original_record_id=row["source"]["record_id"],
                                   lineage="original_train_annotation" if name in PUBLIC else "offline_solver_facts",
                                   label_check="original_annotation_mapping" if name in PUBLIC else "independent_solver")
            selected[split].append(row)
            renders[split].append(rendered)
            selected_counts[bucket, split] += 1
            seen_groups[group], seen_records[record] = split, split
            if not is_synthetic:
                additions.add(texts)
            return True

        # Assignment is fixed from original groups before any view conversion or filtering.
        for name in sorted(pools):
            raw_rows = pools[name]
            components = base.Components()
            grouped = defaultdict(list)
            for raw, rid in raw_rows:
                group = original_group(name, raw, rid)
                components.root(group)
                if name == "boolq" and sources._text(raw.get("title")):
                    components.join(group, ("title", sources._normal(raw["title"])))
            for raw, rid in raw_rows:
                grouped[components.root(original_group(name, raw, rid))].append((raw, rid))
            keys = sorted(grouped, key=str)
            loader_reports[name]["candidate_original_groups"] = len(keys)
            loader_reports[name]["records_omitted_by_one_per_group"] = len(raw_rows) - len(keys)
            sources._rng(args.seed, name, "groups").shuffle(keys)
            buckets = ["snli_choice", "snli_noul"] if name == "snli" else [name]
            train = sum(quotas[b]["train"] for b in buckets)
            dev = sum(quotas[b]["dev"] for b in buckets)
            ndev = round(len(keys) * dev / (train + dev)) if train + dev else 0
            for i, group in enumerate(keys):
                split = "dev" if i < ndev else "train"
                raw, rid = sources._rng(args.seed, name, str(group), "record").choice(grouped[group])
                # Exactly one decision per original group, including SNLI image and SGD dialogue.
                if name == "snli":
                    total = sum(quotas[b][split] for b in buckets)
                    draw = sources._rng(args.seed, name, rid, "view").random()
                    bucket = "snli_choice" if total and draw < quotas["snli_choice"][split] / total else "snli_noul"
                else:
                    bucket = name
                if selected_counts[bucket, split] >= quotas[bucket][split]:
                    reject({"source": {"name": name, "record_id": rid}}, "quota_surplus_before_conversion")
                    continue
                row = public_decision(name, raw, rid, bucket, args.seed, ontology, schemas, none_services, labels,
                                      loader_reports[name])
                if row is not None:
                    require(row["group_id"] == original_group(name, raw, rid), "inherited group key mismatch")
                    consider(row, bucket, split, raw)
            base.progress("s1_public_source", source=name, selected=dict(Counter(r["source"]["name"] for r in selected["train"])))

        loader_reports["clinc"]["k_buckets"] = [2, 4, 8, 16, 32, 64, 120]
        loader_reports["clinc"]["candidate_policy"] = (
            "One seeded uniform K draw per original record from k_buckets; inherited uniform/domain-clustered "
            "circular-window candidates and permutation. No exactly balanced K schedule in S1.")

        synthetic_pools, synthetic_report = s1_synthetic.make_s1_pools(
            seed=config["synthetic_seed"], candidates_per_pool=config["synthetic_candidates_per_pool"])
        for name, rows in synthetic_pools.items():
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["group_id"]].append(row)
            keys = sorted(grouped)
            sources._rng(args.seed, name, "groups").shuffle(keys)
            family_views, strata = Counter(), defaultdict(list)
            for key in keys:
                members = sorted(grouped[key], key=lambda r: r["view_id"])
                family = members[0]["family"]
                # Seeded alternating member schedule balances original base/counterfactual
                # views within each family instead of always selecting view zero.
                offset = sources._rng(args.seed, name, family, "member_offset").randrange(len(members))
                member = members[(family_views[family] + offset) % len(members)]
                if len(members) > 1:
                    family_views[family] += 1
                strata[member["metadata"]["sampling_stratum"]].append((key, member))
            partitions = {split: {} for split in ("train", "dev")}
            total = sum(quotas[name].values())
            for stratum, candidates in sorted(strata.items()):
                sources._rng(args.seed, name, stratum, "split").shuffle(candidates)
                ndev = round(len(candidates) * quotas[name]["dev"] / total) if total else 0
                partitions["dev"][stratum] = candidates[:ndev]
                partitions["train"][stratum] = candidates[ndev:]
            schedule = []
            for split, queues in partitions.items():
                # Round-robin target strata; no oversampling or repeated groups.
                order = sorted(queues)
                sources._rng(args.seed, name, split, "strata").shuffle(order)
                while any(queues.values()):
                    for stratum in order:
                        if queues[stratum]:
                            key, row = queues[stratum].pop()
                            schedule.append((split, key, row))
            for split, key, row in schedule:
                members = grouped[key]
                signatures = {s1_synthetic.canonical_facts(member) for member in members}
                reason = ("synthetic_group_s0_facts_exact" if signatures & s0_facts else
                          "synthetic_group_new_facts_exact" if signatures & new_facts else None)
                if reason:
                    for row in members:
                        reject(row, reason)
                    continue
                for field in ("case_id", "group_id", "view_id"):
                    row[field] = "s1:seed%d:" % config["synthetic_seed"] + row[field]
                row["source"]["record_id"] = row["case_id"]
                row["metadata"]["generator_group_id"] = key
                row["metadata"]["generator_seed"] = config["synthetic_seed"]
                if consider(row, name, split):
                    new_facts.update(signatures)

        if (len(retained_s0) + len(selected["train"])) % 2:
            require(bool(selected["train"]), "odd retained S0 count without an added row available for parity exclusion")
            row = selected["train"].pop()
            renders["train"].pop()
            selected_counts[row["metadata"]["s1_bucket"], "train"] -= 1
            reject(row, "two_rank_even_train_count")

    final_train = retained_s0 + selected["train"]
    factual_signatures = [s1_synthetic.canonical_facts(row) for rows in selected.values() for row in rows
                         if row["source"]["name"] in synthetic.POOL_NAMES]
    require(len(factual_signatures) == len(set(factual_signatures)) and not (set(factual_signatures) & s0_facts),
            "duplicate synthetic family/facts survived selection")
    require(bool(final_train) and len(final_train) % 2 == 0, "training rows must be nonempty and even")
    seen_pairs, seen_views, cases = set(), set(), {}
    for row in final_train + s0["dev"] + s0["test"] + selected["dev"]:
        base.validate_identity(row, seen_pairs, seen_views, cases)
    for split, rows in selected.items():
        other = selected["dev" if split == "train" else "train"] + all_s0
        require(not ({r["group_id"] for r in rows} & {r["group_id"] for r in other}), "cross-split group overlap")
        require(not ({(r["source"]["name"], r["source"]["record_id"]) for r in rows}
                     & {(r["source"]["name"], r["source"]["record_id"]) for r in other}), "cross-split record overlap")
    (output / "canonical").mkdir()
    for filename, rows in (("train", final_train), ("new_train", selected["train"]), ("new_dev", selected["dev"])):
        path = output / "canonical" / (filename + ".jsonl")
        base.write_jsonl(path, rows)
        require(base.read_jsonl(path) == rows, "canonical readback mismatch: " + filename)
    shutil.copyfile(args.s0_dir / "canonical/dev.jsonl", output / "canonical/dev.jsonl")
    require((output / "canonical/dev.jsonl").read_bytes() == (args.s0_dir / "canonical/dev.jsonl").read_bytes(),
            "original dev bytes changed")
    s0_rendered = {split: [render_check(row, tok, mask_id, mask_text, args.max_length, decision_input) for row in rows]
                   for split, rows in {"train": retained_s0, "dev": s0["dev"], "test": s0["test"]}.items()}
    shortages = [dict(bucket=b, split=s, requested=quotas[b][s], actual=selected_counts[b, s])
                 for b in sorted(quotas) for s in ("train", "dev") if selected_counts[b, s] < quotas[b][s]]
    report.update(status="complete_with_shortfalls" if shortages else "complete", quota_shortfalls=shortages,
                  s0_source_counts=dict(Counter(r["source"]["name"] for r in all_s0)),
                  s0_input_counts={s: len(rows) for s, rows in s0.items()}, s0_retained_train=len(retained_s0),
                  s0_external_audit={s: dict(audit[s]) for s in s0},
                  s0_dev_unchanged=True, s0_dev_external_collision_warning=bool(audit["dev"]),
                  requires_parent_dev_collision_review=bool(audit["dev"]),
                  heldout_safety="requires_parent_review_of_unchanged_s0_dev" if audit["dev"] else "no_detected_selection_dev_collision_within_audit_scope",
                  s0_test_not_exported=True, rejections={k: dict(v) for k, v in rejected.items()},
                  source_loader_reports=loader_reports, synthetic_report=synthetic_report,
                  independent_label_checks=dict(checked),
                  candidate_decisions={b: {s: available[b, s] for s in ("train", "dev")} for b in sorted(quotas)},
                  splits={"train": describe(final_train, s0_rendered["train"] + renders["train"]),
                          "dev": describe(s0["dev"], s0_rendered["dev"]),
                          "new_train": describe(selected["train"], renders["train"]),
                          "new_dev": describe(selected["dev"], renders["dev"])},
                  tokenizer=dict(path=str(args.tokenizer_path), mapping=mapping, mask_id=mask_id, renderer=RENDERER),
                  dedup=dict(normalization="NFKC casefold, word tokens; punctuation and whitespace folded; numbers retained",
                             threshold=config["near_duplicate_threshold"],
                             method="exact text plus bounded three-word-shingle Jaccard, minimum 8 words",
                             limits={k: config[k] for k in ("near_duplicate_max_postings", "near_duplicate_max_candidates")},
                             index_stats={"external": dict(firewall.stats), "s0": dict(blocked.stats), "new": dict(additions.stats)}),
                  split_scope="grouped IID/parameter-group, NOT family-heldout; one new decision per group",
                  synthetic_dedup="Exact canonical family+facts against all S0 views and all selected linked groups; no lexical near matching. External synthetic screening uses normalized exact complete visible state, not isolated policy fields.",
                  limitations=["External audit covers full supplied fixtures, not all splits of forbidden dataset families.",
                               "Bounded lexical matching can miss near duplicates; no semantic embedding audit.",
                               "Source family/metadata authenticity and licenses rely on prepared resource provenance.",
                               "Synthetic supplements vary registered parameters/statuses/histories, not families; distinct facts do not imply semantic novelty.",
                               "S0 original labels are structurally checked, not reannotated; new labels checked against original annotations."])
    base.write_json(output / "source_manifest.json", dict(sources={**s0_manifest, **provenance},
                    synthetic=dict(revision="shared_yesno_synthetic_v1+s1_parameter_supplement_v1", seed=config["synthetic_seed"],
                                   ordinal="No independent public score source was locked; deliberately selected authorized programmatic operational-rubric fallback, not sentiment or a claim of resource research."),
                    intended_new_train=dict(public=sum(v["train"] for k, v in quotas.items() if k not in synthetic.POOL_NAMES),
                                            synthetic=sum(quotas[k]["train"] for k in synthetic.POOL_NAMES)),
                    renderer_version=RENDERER))
    base.write_json(output / "report.json", report)
    base.progress("s1_complete", status=report["status"], train=len(final_train), new_train=len(selected["train"]),
                  new_dev=len(selected["dev"]), shortages=len(shortages))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("s0-dir", "source-dir", "new-source-dir", "tokenizer-path", "heldout-profile", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "s1_data_config.json")
    parser.add_argument("--seed", type=int, help="Override config seed; synthetic_seed stays explicit in config")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true", help="At most two additions per bucket/split; full S0 and external audit")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    config = read_json(args.config)
    expected = {"seed", "synthetic_seed", "synthetic_candidates_per_pool", "max_records_per_source", "near_duplicate_threshold",
                "near_duplicate_max_postings", "near_duplicate_max_candidates", "quotas"}
    require(set(config) == expected, "config keys must be exactly " + ", ".join(sorted(expected)))
    require(set(config["quotas"]) == BUCKETS, "config must define all ten quota buckets")
    for name, quota in config["quotas"].items():
        require(set(quota) == {"train", "dev"} and all(type(v) is int and v >= 0 for v in quota.values()),
                "invalid train/dev quota for " + name)
    for key in expected - {"quotas", "near_duplicate_threshold"}:
        require(type(config[key]) is int, "config requires integer " + key)
    require(all(config[k] > 0 for k in ("synthetic_candidates_per_pool", "max_records_per_source", "near_duplicate_max_postings", "near_duplicate_max_candidates")),
            "candidate and dedup limits must be positive")
    require(0 < config["near_duplicate_threshold"] <= 1 and args.max_length > 0, "invalid threshold/context length")
    args.seed = config["seed"] if args.seed is None else args.seed
    config["seed"] = args.seed
    output = args.output_dir
    require(not output.exists(), "output-dir must be NEW; refusing overwrite: " + str(output))
    for root in (args.s0_dir, args.source_dir, args.new_source_dir, args.tokenizer_path, args.heldout_profile.parent):
        require(root.is_dir(), "missing input directory: " + str(root))
        require(not output.is_relative_to(root) and not root.is_relative_to(output), "output must be separate from inputs")
    output.mkdir(parents=True)
    try:
        build(args, output, config)
    except Exception as exc:
        path = output / "report.json"
        report = read_json(path) if path.is_file() else {}
        report.update(status="failed", error=str(exc))
        base.write_json(path, report)
        raise


if __name__ == "__main__":
    main()
