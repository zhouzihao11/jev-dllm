"""Independently check final canonical labels against prepared train sources.

No generator, adapter, builder, tokenizer, or model imports. Run after the build
finishes. --output is a new JSON file; failures preserve the report and exit 1.
This verifies source annotation fidelity and reviewed synthetic rules, not the
truth of human-language annotations or the correctness of source preparation.
"""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import re
import sys
import unicodedata


SPLITS = ("train", "dev", "test")
REAL = ("clinc", "snli", "arc", "sgd")
REAL_FAMILIES = dict(clinc="intent_classification", snli="natural_language_inference",
                     arc="science_question_answering", sgd="dialogue_active_intent")
FAMILIES = {
    "synthetic_constraints": ("eligibility", "preservation", "ownership"),
    "synthetic_ordinal": ("shipment", "laboratory", "deployment"),
    "synthetic_probability": ("finite_draw", "latent_mixture", "without_replacement"),
}
ALIASES = {
    "pto_request": "request paid time off",
    "pto_request_status": "paid time off request status",
    "pto_balance": "paid time off balance",
    "pto_used": "paid time off used",
    "w2": "W-2 tax form", "w9": "W-9 tax form",
    "rollover_401k": "roll over a 401(k) retirement account",
    "uber": "request an Uber ride", "mpg": "vehicle miles per gallon",
    "what_is_your_name": "ask the assistant's name",
}

# Reviewed literal contracts, deliberately independent of the generator module.
POLICIES = {
    "eligibility": "A candidate is eligible exactly when certified is true, capacity is at least "
    "minimum_capacity, and delay is at most maximum_delay. Select the eligible "
    "candidate with the lowest cost. This task guarantees a unique minimum.",
    "preservation": "In this synthetic resource policy, balance, change, reserve, and ceiling are measured "
    "in integer resource units, not money. "
    "The proposed change is allowed exactly when authorized is true, frozen is false, "
    "and balance + change is between reserve and ceiling, including both boundaries.",
    "ownership": "Process events in listed chronological order. A transfer changes ownership to "
    "its recipient only if its sender is the current owner and accepted is true. "
    "Otherwise ignore it. These are all ownership events.",
    "shipment": "Ready requires packed >= required, seal_intact true, and inspection passed. "
    "Dispatched additionally requires handed_over true and receipt_validation passed. "
    "Pending and failed validations do not pass. Later evidence cannot bypass readiness.",
    "laboratory": "Use the highest contiguous milestone from collection to qualification to release. "
    "Collection requires collected - damaged >= required. Qualification additionally "
    "requires assay status passed for batch_version and control in [control_low, "
    "control_high], inclusive. Release additionally requires review status passed for "
    "assay_version. Pending, failed, and older-version checks do not pass.",
    "deployment": "Use the highest contiguous milestone: build, tests, approval, rollout. Each check "
    "must have status passed and version equal to current_version. Tests additionally "
    "require tests_passed = tests_required. Rollout additionally requires healthy_zones "
    "= required_zones. Pending, failed, and older-version checks do not pass. Later "
    "checks cannot bypass an earlier incomplete milestone.",
    "finite_draw": "Draw one token uniformly from this finite bag. Counts include every token.",
    "latent_mixture": "Choose one mechanism with probability proportional to prior_weight; its identity is "
    "not observed. Under that mechanism, a uniform draw from signal_total tickets produces "
    "an alert on signal_count tickets. Independently conditional on the same mechanism, "
    "a uniform draw from outcome_total tickets succeeds on success_count tickets. "
    "An alert was observed. Predict the outcome conditional on that alert. The table "
    "lists every possible mechanism, not a secretly selected mechanism.",
}
RUBRICS = {
    "shipment": [
        "Not ready: packing quantity, seal, or inspection requirement is unmet.",
        "Ready: packing quantity, seal, and inspection requirements met, but carrier handoff "
        "incomplete or receipt validation not passed.",
        "Dispatched: ready and carrier handoff complete with passed receipt validation.",
    ],
    "laboratory": [
        "Collection incomplete: fewer than the required usable samples.",
        "Collected: enough usable samples; current-batch assay not yet qualified.",
        "Qualified: collection and assay qualified; current-assay review not passed.",
        "Released: collection, assay qualification, and current-assay review passed.",
    ],
    "deployment": [
        "Unbuilt: current-version build has not passed.",
        "Built: current build passed, but current-version testing is incomplete.",
        "Tested: build and tests complete, but current-version approval not passed.",
        "Approved: build, tests, approval complete, but current rollout incomplete.",
        "Deployed: build, tests, approval, and rollout all complete for current version.",
    ],
}
INSTRUCTIONS = {
    "eligibility": (
        "Which candidate does the policy select?",
        "Select the cheapest candidate satisfying every eligibility rule."),
    "preservation": (
        "Is the proposed change allowed under every policy condition?",
        "Does the proposed operation satisfy the stated authorization and balance policy?"),
    "ownership": (
        "Who is the current owner after all events?",
        "Select the owner obtained by applying the event policy."),
    "ordinal": (
        "Select the highest operational level justified by the observations and rubric.",
        "Which rubric level has been reached without skipping any prerequisite?"),
    "finite_draw": ("What color will the random token have? Predict the distribution over outcomes.",),
    "latent_mixture": (
        "Given the alert, predict the distribution of the outcome draw: success or failure.",),
    "without_replacement": (
        "How many red tokens will be drawn? Predict the distribution over counts.",
        "Predict the uncertain count of red tokens in the sample, using the count rubric."),
}
NLI_DESCRIPTIONS = {
    "entailment": "The premise entails the hypothesis.",
    "contradiction": "The premise contradicts the hypothesis.",
    "neutral": "The premise neither entails nor contradicts the hypothesis.",
}


def normal(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=None):
    require(type(value) is int, "expected integer observation")
    require(minimum is None or value >= minimum, "integer observation out of range")
    return value


def boolean(value):
    require(type(value) is bool, "expected boolean observation")
    return value


def check_pass(check, version):
    require(set(check) == {"status", "version"}, "unexpected versioned check fields")
    require(check["status"] in ("passed", "pending", "failed"), "unknown check status")
    integer(check["version"])
    return check == {"status": "passed", "version": version}


def compact(value, limit=1600):
    text = json.dumps(value, ensure_ascii=True, default=str)
    return text if len(text) <= limit else text[:limit] + " [truncated]"


class Report:
    def __init__(self):
        self.context = {}
        self.checks = defaultdict(Counter)
        self.mismatches = []
        self.errors = Counter()
        self.rows = {split: Counter() for split in SPLITS}
        self.coverage = defaultdict(lambda: defaultdict(Counter))
        self.kinds = Counter()
        self.references = Counter()
        self.exact_inputs = Counter()
        self.samples = defaultdict(list)
        self.source_provenance = {}
        self.source_split_basis = defaultdict(Counter)

    def test(self, name, passed, expected=None, actual=None):
        self.checks[name]["checked"] += 1
        self.checks[name]["passed" if passed else "failed"] += 1
        if not passed:
            self.errors[name] += 1
            detail = dict(self.context, check=name)
            if expected is not None or actual is not None:
                detail.update(expected=compact(expected), actual=compact(actual))
            self.mismatches.append(detail)
        return passed

    def equal(self, name, actual, expected):
        return self.test(name, actual == expected, expected, actual)

    def jsonl(self, path):
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        row = json.loads(line)
                        require(isinstance(row, dict), "JSONL row must be an object")
                        yield line_number, row
                    except (ValueError, TypeError) as exc:
                        previous = self.context
                        self.context = {"path": str(path), "line": line_number}
                        self.test("read_jsonl", False, "JSON object", str(exc))
                        self.context = previous
        except (OSError, UnicodeError) as exc:
            self.test("read_jsonl", False, str(path), str(exc))


def source_key(name, row):
    rid = str(row["dialogue_id" if name == "sgd" else "record_id"])
    return (str(row["config"]), rid) if name == "arc" else rid


def canonical_key(row):
    source = row["source"]
    rid = source["record_id"]
    return (row["metadata"]["config"], rid) if source["name"] == "arc" else rid


def raw_group_key(name, raw):
    if name == "clinc":
        return normal(raw["text"])
    if name == "arc":
        return normal(raw["question"])
    if name == "sgd":
        return str(raw["dialogue_id"])
    group = raw.get("group_key")
    require(isinstance(group, (str, int)) and not isinstance(group, bool) and str(group),
            "SNLI original image group_key missing; premise fallback cannot verify image leakage")
    return str(group)


def load_source_provenance(root, report):
    root = root.resolve()
    manifest_path = root / "source_manifest.json"
    report.context = {"path": str(manifest_path)}
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest["sources"]
    require(isinstance(entries, list), "source_manifest.sources must be a list")
    require(all(isinstance(entry, dict) for entry in entries), "invalid source manifest entry")
    names = [entry["name"] for entry in entries]
    require(len(names) == len(REAL) and set(names) == set(REAL),
            "manifest must contain exactly one entry for each of the four real sources")
    paths = {}
    for entry in entries:
        name = entry["name"]
        report.context = {"path": str(manifest_path), "source": name}
        expected = "prepared/%s_train.jsonl" % name
        require(entry.get("prepared_path") == expected, "unexpected prepared_path for " + name)
        path = (root / expected).resolve()
        require(root in path.parents, "prepared source path resolves outside source root: " + name)
        require(entry.get("original_split") == "train", "manifest source is not original train: " + name)
        require(isinstance(entry.get("revision"), str) and bool(entry["revision"].strip()),
                "missing source revision: " + name)
        require(isinstance(entry.get("license"), (str, list, dict)) and bool(entry["license"]),
                "missing source license declaration: " + name)
        integer(entry["counts"]["exported"], 0)
        report.source_provenance[name] = {
            "manifest_path": str(manifest_path), "prepared_path": expected,
            "resolved_path": str(path), "original_split": entry["original_split"],
            "revision": entry["revision"], "license": entry["license"],
            "declared_exported_records": entry["counts"]["exported"],
        }
        report.test("source_file_train_provenance", True)
        paths[name] = path
    return paths


def load_sources(root, needed, report):
    paths = load_source_provenance(root, report)
    selected = {name: {} for name in REAL}
    domains, none_services = defaultdict(set), set()
    scanned = Counter()
    for name in REAL:
        path = paths[name]
        for line, row in report.jsonl(path):
            scanned[name] += 1
            report.context = {"path": str(path), "line": line, "source": name}
            try:
                key = source_key(name, row)
                split_fields = {field: row[field] for field in ("original_split", "split") if field in row}
                train = all(value == "train" for value in split_fields.values())
                report.test("source_present_split_fields_train", train, "train", split_fields)
                basis = "explicit_row_fields" if split_fields else "manifest_file_provenance_only"
                report.source_split_basis[name][basis] += 1
                if not train:
                    report.source_split_basis[name]["non_train_row_fields"] += 1
                # These two small ontology summaries require a complete stream, not
                # retaining unselected records or loading entire source JSONL files.
                if train and name == "clinc":
                    domains[row["intent"]].add(row["domain"])
                if train and name == "sgd":
                    for turn in row["turns"]:
                        if turn["speaker"] == "USER":
                            for frame in turn.get("frames", []):
                                if frame.get("state", {}).get("active_intent") == "NONE":
                                    none_services.add(frame["service"])
                if key not in needed[name]:
                    continue
                report.test("source_unique_selected_id", key not in selected[name], actual=key)
                report.test("source_original_split_train", train, "manifest train plus any row split fields", split_fields)
                if key not in selected[name]:
                    selected[name][key] = row
            except (KeyError, TypeError, ValueError) as exc:
                report.test("source_record_schema", False, actual=str(exc))
        report.context = {"source": name}
        report.equal("source_exported_record_count", scanned[name],
                     report.source_provenance[name]["declared_exported_records"])
        for key in sorted(needed[name] - selected[name].keys(), key=str):
            report.test("source_selected_id_found", False, actual=key)
    allowed = {label: next(iter(values)) for label, values in domains.items()
               if len(values) == 1 and not values & {"banking", "credit_cards"}
               and label not in ("oos", "out_of_scope")}
    schema_path = root / "prepared" / "sgd_schema.json"
    if not schema_path.is_file():
        schema_path = root / "sgd_schema.json"
    with schema_path.open(encoding="utf-8") as handle:
        services = json.load(handle)
    schemas = {item["service_name"]: item for item in services}
    report.test("sgd_schema_unique_services", len(schemas) == len(services))
    return selected, allowed, schemas, none_services, scanned


def question_parts(row):
    for field in ("questions", "gold", "option_ids"):
        require(set(row[field]) == {"decision"}, "expected one decision in " + field)
    question = row["questions"]["decision"]
    kind, criteria = question["type"], question["criteria"]
    require(kind in ("choice", "score", "noul"), "unknown question type")
    if kind == "score":
        require(isinstance(criteria, list) and all(isinstance(x, str) for x in criteria),
                "score requires string rubric")
        keys = [str(i) for i in range(len(criteria))]
    elif kind == "noul":
        require(criteria is None or isinstance(criteria, dict) and set(criteria) <= {"false", "true"},
                "invalid binary criteria")
        keys = ["false", "true"]
    else:
        require(isinstance(criteria, dict), "choice requires criteria mapping")
        keys = list(criteria)
    ids = row["option_ids"]["decision"]
    require(len(keys) >= 2 and len(ids) == len(keys) and len(set(ids)) == len(ids),
            "option IDs and displayed keys must be unique and aligned")
    require(all(isinstance(x, str) and x for x in ids + keys), "invalid option ID/key")
    return question, keys, ids


def compare_probabilities(row, keys, expected, kind, report):
    gold = row["gold"]["decision"]
    actual = gold["probabilities"]
    report.equal("gold_kind", gold["kind"], kind)
    report.equal("probability_keys", set(actual), set(keys))
    report.test("reference_exhaustive_keys", not set(expected) - set(keys), actual=list(expected))
    values = list(actual.values())
    valid = all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in values)
    report.test("probabilities_finite_bounded", valid)
    report.test("probabilities_normalized", valid and abs(sum(values) - 1) <= 1e-12)
    if kind == "hard":
        report.test("hard_unique_one_hot", valid and values.count(1) == 1
                    and all(p in (0, 1) for p in values))
    reference = {key: expected.get(key, Fraction(0)) for key in keys}
    report.test("reference_normalized", sum(reference.values()) == 1)
    matches = set(actual) == set(keys) and valid and all(
        abs(actual[key] - float(reference[key])) <= 1e-12 for key in keys)
    report.test("entire_reference_distribution", matches,
                {k: str(v) for k, v in reference.items()}, actual)
    return reference, matches


def real_reference(row, raw, allowed, schemas, none_services, question, keys, ids, report):
    name, state, metadata = row["source"]["name"], row["state"], row["metadata"]
    instruction = question["instructions"]
    options = dict(zip(ids, keys))
    criteria = question["criteria"]
    report.equal("real_source_family", row["family"], REAL_FAMILIES[name])
    exact = False
    if name == "clinc":
        exact = report.equal("clinc_exact_utterance", state, raw["text"])
        report.test("clinc_allowed_original_domain", raw["intent"] in allowed
                    and allowed.get(raw["intent"]) == raw["domain"], actual=raw["domain"])
        report.equal("clinc_metadata_domain", metadata["domain"], raw["domain"])
        report.test("clinc_options_source_ontology", set(ids) <= allowed.keys(), actual=ids)
        for oid, display in options.items():
            report.equal("clinc_sanitized_label", display, ALIASES.get(oid, oid.replace("_", " ")))
            report.equal("clinc_no_inferred_description", criteria[display], None)
        report.equal("clinc_instruction", instruction, "Which original intent is closest to the user's utterance?")
        report.equal("clinc_question_type", question["type"], "choice")
        report.equal("clinc_view", row["view_id"], "intent_k%d" % len(ids))
        report.equal("clinc_k", metadata["k"], len(ids))
        target = raw["intent"]
    elif name == "snli":
        exact = report.equal("snli_exact_premise_hypothesis", state,
                             {"premise": raw["premise"], "hypothesis": raw["hypothesis"]})
        target = raw["label"]
        require(target in NLI_DESCRIPTIONS, "invalid original SNLI label")
        group = raw_group_key(name, raw)
        # Prepared group_key is the source image grouping field; never infer an
        # image from a hashed canonical group ID or a normalized caption.
        for field in ("imageID", "image_id"):
            if field in raw:
                report.equal("snli_original_image_group", group, str(raw[field]))
        if "captionID" in raw:
            require(isinstance(raw["captionID"], str) and bool(raw["captionID"]),
                    "invalid original SNLI captionID")
            report.equal("snli_original_caption_group", group, raw["captionID"].split("#", 1)[0])
        view = row["view_id"]
        if view == "three_way":
            report.equal("snli_three_way_type", question["type"], "choice")
            report.equal("snli_three_way_options", options, {k: k for k in NLI_DESCRIPTIONS})
            report.equal("snli_three_way_descriptions", criteria, NLI_DESCRIPTIONS)
            report.equal("snli_three_way_instruction", instruction,
                         "How does the premise relate to the hypothesis?")
        else:
            require(view in ("binary_entailment", "binary_contradiction"), "unknown SNLI view")
            relation = view.removeprefix("binary_")
            report.equal("snli_binary_relation_metadata", metadata["binary_relation"], relation)
            report.equal("snli_original_gold_metadata", metadata["source_label"], target)
            verb = {"entailment": "entail", "contradiction": "contradict"}[relation]
            report.equal("snli_binary_relation_instruction", instruction,
                         "Does the premise %s the hypothesis? Answer about this relation, not whether "
                         "the hypothesis is true in the real world." % verb)
            report.equal("snli_binary_type", question["type"], "noul")
            report.equal("snli_binary_options", options, {"false": "false", "true": "true"})
            report.equal("snli_binary_descriptions", criteria, {
                "false": "The specified relation does not hold.",
                "true": "The specified relation holds."})
            target = "true" if target == relation else "false"
    elif name == "arc":
        exact = report.equal("arc_exact_stem", state, raw["question"])
        expected_options = {str(c["id"]): c["text"] for c in raw["choices"]}
        report.test("arc_unique_original_choices", len(expected_options) == len(raw["choices"]))
        exact = report.equal("arc_exact_choices_by_original_id", options, expected_options) and exact
        report.test("arc_no_generated_descriptions", all(v is None for v in criteria.values()))
        report.equal("arc_config", metadata["config"], raw["config"])
        report.equal("arc_view", row["view_id"], "answer:" + raw["config"])
        report.equal("arc_type", question["type"], "choice")
        report.equal("arc_instruction", instruction, "Which answer correctly answers the question?")
        target = str(raw["answerKey"])
    else:
        ti, service = metadata["turn_index"], metadata["service"]
        integer(ti, 0)
        require(ti < len(raw["turns"]), "SGD turn index outside original dialogue")
        turn = raw["turns"][ti]
        report.equal("sgd_current_turn_user", turn["speaker"], "USER")
        report.test("sgd_original_service", service in raw["services"] and service in schemas)
        frames = [f for f in turn["frames"] if f["service"] == service]
        require(len(frames) == 1, "SGD needs exactly one annotated service frame")
        target = frames[0]["state"]["active_intent"]
        definition = schemas[service]
        source_options = {i["name"]: (i["name"].replace("_", " "), i["description"])
                          for i in definition["intents"]}
        require(len(source_options) == len(definition["intents"]), "duplicate source intent IDs")
        if service in none_services and "NONE" not in source_options:
            source_options["NONE"] = ("No active intent", "No service intent is active at this point.")
        report.equal("sgd_full_source_intent_options", set(ids), set(source_options))
        report.equal("sgd_original_intent_labels_descriptions",
                     {oid: (key, criteria[key]) for oid, key in options.items()}, source_options)
        expected_state = {
            "dialogue_prefix": [{"speaker": t["speaker"], "utterance": t["utterance"]}
                                for t in raw["turns"][:ti + 1]],
            "service_definition": {
                "description": definition["description"],
                "intents": [{"intent": source_options[oid][0], "description": source_options[oid][1]}
                            for oid in ids]},
        }
        exact = report.equal("sgd_exact_prefix_and_source_definitions_no_future", state, expected_state)
        report.equal("sgd_prefix_length", metadata["prefix_turns"], ti + 1)
        report.equal("sgd_view", row["view_id"], "turn%d:%s:active_intent" % (ti, service))
        report.equal("sgd_type", question["type"], "choice")
        report.equal("sgd_instruction", instruction,
                     "What is the current active intent for the described service at the last user turn? "
                     "Use the full conversation prefix, not a future turn.")
    report.exact_inputs[name + ("/matched" if exact else "/mismatched")] += 1
    require(target in options, "original annotated target absent from displayed stable IDs: " + str(target))
    return {options[target]: Fraction(1)}, target


def synthetic_reference(row, question, keys, ids, report):
    family, state = row["family"], row["state"]
    facts = state["observations"]
    qtype, instruction = question["type"], question["instructions"]
    criteria, metadata = question["criteria"], row["metadata"]
    require(family in FAMILIES[row["source"]["name"]], "synthetic family/source mismatch")
    process = family in FAMILIES["synthetic_probability"]
    report.equal("synthetic_visible_state_fields", set(state), {"observations", "process" if process else "policy"})
    if family == "without_replacement":
        policy = ("Draw %d tokens uniformly without replacement from a bag containing exactly %d "
                  "red tokens and %d blue tokens. Every subset of that size is equally likely."
                  % (facts["draws"], facts["red"], facts["blue"]))
    else:
        policy = POLICIES[family]
    report.equal("reviewed_visible_policy", state["process" if process else "policy"], policy)
    report.equal("visible_facts_match_solver_metadata", facts, metadata["solver_facts"])
    report.equal("synthetic_stable_option_ids", ids, [family + ":" + qtype + ":" + k for k in keys])
    query = {"type": qtype}
    if qtype == "choice":
        query["order"] = keys
    if family == "ownership" and qtype == "noul":
        match = re.fullmatch(r"After all events, is (.+) the current owner\?|"
                             r"Does (.+) own the item after applying the transfer policy\?", instruction)
        if match is None:
            raise ValueError("unreviewed visible ownership question")
        query["person"] = next(x for x in match.groups() if x is not None)
    elif family == "finite_draw" and qtype == "noul":
        match = re.fullmatch(r"Will the randomly drawn token be (.+)\? Predict this uncertain event\.", instruction)
        if match is None:
            raise ValueError("unreviewed visible finite-draw question")
        query["color"] = match.group(1)
    elif family == "latent_mixture" and qtype == "noul":
        report.equal("reviewed_visible_instruction", instruction,
                     "Given the observed alert, will the outcome draw succeed? Predict this uncertain event.")
    else:
        accepted = INSTRUCTIONS["ordinal" if family in RUBRICS else family]
        report.test("reviewed_visible_instruction", instruction in accepted, accepted, instruction)
    report.equal("visible_query_matches_metadata", query, metadata["query"])
    if family in RUBRICS:
        report.equal("reviewed_visible_rubric", criteria, RUBRICS[family])
        require(qtype == "score", "ordinal requires score")
    elif family == "without_replacement":
        require(qtype == "score", "count distribution requires score")
        report.equal("reviewed_visible_count_rubric", criteria,
                     ["Exactly %d red tokens in the sample." % n for n in range(facts["draws"] + 1)])
    elif family == "preservation":
        require(qtype == "noul", "preservation requires binary")
        report.equal("reviewed_criteria", criteria,
                     {"false": "The operation is forbidden.", "true": "The operation is allowed."})
    else:
        require(qtype in ("choice", "noul"), "unsupported primitive/question type")
        report.equal("reviewed_criteria", criteria, {key: None for key in keys})

    if family == "eligibility":
        require(qtype == "choice", "eligibility requires choice")
        minimum, maximum = integer(facts["minimum_capacity"]), integer(facts["maximum_delay"])
        candidates = facts["candidates"]
        names = [c["name"] for c in candidates]
        require(len(names) == len(set(names)) and set(keys) == set(names), "candidate option mismatch")
        ranked = []
        for c in candidates:
            certified = boolean(c["certified"])
            capacity, delay, cost = integer(c["capacity"]), integer(c["delay"]), integer(c["cost"])
            if certified and not (capacity < minimum or delay > maximum):
                ranked.append((cost, c["name"]))
        ranked.sort()
        require(ranked and (len(ranked) == 1 or ranked[0][0] < ranked[1][0]), "no unique eligible minimum")
        target = ranked[0][1]
    elif family == "preservation":
        end = integer(facts["balance"]) + integer(facts["change"])
        lower, upper = integer(facts["reserve"]), integer(facts["ceiling"])
        require(lower <= upper, "invalid resource interval")
        authorized, frozen = boolean(facts["authorized"]), boolean(facts["frozen"])
        violations = (not authorized, frozen, end < lower, end > upper)
        target = "false" if any(violations) else "true"
    elif family == "ownership":
        owners = [facts["initial_owner"]]
        people = set(owners)
        for event in facts["events"]:
            people.update((event["sender"], event["recipient"]))
            accepted = boolean(event["accepted"])
            if accepted and event["sender"] == owners[-1]:
                owners.append(event["recipient"])
        if qtype == "noul":
            require(query["person"] in people, "queried owner absent from visible history")
            target = "true" if owners[-1] == query["person"] else "false"
        else:
            require(people <= set(keys), "ownership options omit visible participants")
            target = owners[-1]
    elif family in RUBRICS:
        if family == "shipment":
            require(facts["inspection"] in ("passed", "pending", "failed")
                    and facts["receipt_validation"] in ("passed", "pending", "failed"), "unknown validation status")
            gates = [integer(facts["packed"], 0) >= integer(facts["required"], 0),
                     boolean(facts["seal_intact"]), facts["inspection"] == "passed",
                     boolean(facts["handed_over"]), facts["receipt_validation"] == "passed"]
            stages = [all(gates[:3]), all(gates)]
        elif family == "laboratory":
            collected, damaged = integer(facts["collected"], 0), integer(facts["damaged"], 0)
            require(damaged <= collected, "damaged count exceeds collected")
            low, high, control = (integer(facts[k]) for k in ("control_low", "control_high", "control"))
            require(low <= high, "invalid control interval")
            stages = [collected >= integer(facts["required"], 0) + damaged,
                      check_pass(facts["assay"], integer(facts["batch_version"])) and low <= control <= high,
                      check_pass(facts["review"], integer(facts["assay_version"]))]
        else:
            version = integer(facts["current_version"])
            stages = [check_pass(facts[name], version) for name in ("build", "tests", "approval", "rollout")]
            stages[1] &= integer(facts["tests_passed"], 0) == integer(facts["tests_required"], 0)
            stages[3] &= integer(facts["healthy_zones"], 0) == integer(facts["required_zones"], 0)
        target = str(next((i for i, passed in enumerate(stages) if not passed), len(stages)))
    elif family == "finite_draw":
        counts = {color: integer(n, 0) for color, n in facts["counts"].items()}
        total = sum(counts.values())
        require(total > 0, "empty finite bag")
        if qtype == "noul":
            p = Fraction(counts[query["color"]], total)
            return {"true": p, "false": 1 - p}
        require(set(keys) == set(counts), "finite bag outcome keys mismatch")
        return {color: Fraction(n, total) for color, n in counts.items()}
    elif family == "latent_mixture":
        # Enumerate signal/outcome tickets per mechanism and accumulate joint
        # alert/outcome mass. No stored posterior or generator solver is used.
        joint = defaultdict(Fraction)
        for m in facts["mechanisms"]:
            prior = integer(m["prior_weight"], 0)
            signals, outcomes = integer(m["signal_total"], 1), integer(m["outcome_total"], 1)
            alerts, successes = integer(m["signal_count"], 0), integer(m["success_count"], 0)
            require(alerts <= signals and successes <= outcomes, "ticket count exceeds total")
            mass = Fraction(prior, signals * outcomes)
            for signal_ticket, outcome_ticket in itertools.product(range(signals), range(outcomes)):
                if signal_ticket < alerts:
                    joint[outcome_ticket < successes] += mass
        require(sum(joint.values()) > 0, "observed alert impossible")
        p = joint[True] / sum(joint.values())
        require(qtype == "noul" or set(keys) == {"Success", "Failure"}, "mixture outcome keys mismatch")
        return {"true": p, "false": 1 - p} if qtype == "noul" else {"Success": p, "Failure": 1 - p}
    elif family == "without_replacement":
        red, blue, draws = (integer(facts[k], 0) for k in ("red", "blue", "draws"))
        require(draws <= red + blue, "more draws than tokens")
        # Tokens have distinct physical identities. Enumerating equally likely
        # subsets is independent of the generator's binomial-coefficient formula.
        frequencies = Counter(sum(token < red for token in subset)
                              for subset in itertools.combinations(range(red + blue), draws))
        total = sum(frequencies.values())
        return {str(n): Fraction(frequencies[n], total) for n in range(draws + 1)}
    else:
        raise ValueError("unknown primitive")
    return {target: Fraction(1)}


def verify(dataset, source, report):
    needed = {name: set() for name in REAL}
    for split in SPLITS:
        for line, row in report.jsonl(dataset / "canonical" / (split + ".jsonl")):
            report.context = {"split": split, "line": line}
            try:
                name = row["source"]["name"]
                if name in needed:
                    needed[name].add(canonical_key(row))
            except (KeyError, TypeError, ValueError) as exc:
                report.test("canonical_source_identity", False, actual=str(exc))
    selected, allowed, schemas, none_services, scanned = load_sources(source, needed, report)
    raw_groups = {name: {split: set() for split in SPLITS} for name in REAL}
    identity_splits = {kind: defaultdict(set) for kind in ("group", "case", "record")}
    raw_to_canonical, canonical_to_raw = defaultdict(set), defaultdict(set)
    pairs, source_views, cases, case_states = set(), set(), {}, {}
    clinc_gold = {split: Counter() for split in SPLITS}
    clinc_canonical_gold = {split: Counter() for split in SPLITS}
    for split in SPLITS:
        for line, row in report.jsonl(dataset / "canonical" / (split + ".jsonl")):
            report.context = {"split": split, "line": line,
                              **{k: row.get(k) for k in ("case_id", "view_id", "group_id")}}
            try:
                src = row["source"]
                name, rid = src["name"], src["record_id"]
                report.context.update(source=name, record_id=rid)
                report.rows[split][name] += 1
                require(name in REAL or name in FAMILIES, "unknown source")
                report.equal("canonical_schema", row["schema_version"], "shared_yesno_data_v1")
                report.equal("canonical_language", row["language"], "en")
                report.equal("canonical_split", row["split"], split)
                report.equal("canonical_original_split", src["original_split"], "train" if name in REAL else "generated")
                for field in ("case_id", "view_id", "group_id"):
                    require(isinstance(row[field], str) and row[field], "missing identity " + field)
                require(isinstance(rid, str) and rid, "missing source record ID")
                question, keys, ids = question_parts(row)
                pair = (row["case_id"], row["view_id"])
                view = (name, canonical_key(row), row["view_id"])
                provenance = (row["group_id"], src)
                report.test("unique_case_view", pair not in pairs)
                report.test("unique_source_view", view not in source_views)
                report.test("consistent_case_provenance", row["case_id"] not in cases or cases[row["case_id"]] == provenance)
                if name not in REAL:
                    observed_case = (row["family"], row["state"]["observations"])
                    report.test("consistent_synthetic_case_observations", row["case_id"] not in case_states
                                or case_states[row["case_id"]] == observed_case)
                    case_states[row["case_id"]] = observed_case
                pairs.add(pair)
                source_views.add(view)
                cases[row["case_id"]] = provenance
                for kind, value in (("group", row["group_id"]), ("case", row["case_id"]),
                                    ("record", (name, canonical_key(row)))):
                    identity_splits[kind][value].add(split)
                family = row["family"]
                bucket = name + "/" + family + "/" + question["type"]
                report.references[bucket + "/attempted"] += 1
                gold = row["gold"]["decision"]
                report.kinds[gold["kind"]] += 1
                source_annotation = None
                if name in REAL:
                    report.exact_inputs[name + "/attempted"] += 1
                    raw = selected[name][canonical_key(row)]
                    group = raw_group_key(name, raw)
                    raw_groups[name][split].add(group)
                    raw_to_canonical[(name, group)].add(row["group_id"])
                    canonical_to_raw[(name, row["group_id"])].add(group)
                    expected, target = real_reference(
                        row, raw, allowed, schemas, none_services, question, keys, ids, report)
                    source_annotation = raw["label"] if name == "snli" else target
                    if name == "clinc":
                        clinc_gold[split][target] += 1
                    kind = "hard"
                else:
                    report.equal("synthetic_record_case_identity", rid, row["case_id"])
                    expected = synthetic_reference(row, question, keys, ids, report)
                    kind = "known_distribution" if name == "synthetic_probability" else "hard"
                    target = None
                reference, matched = compare_probabilities(row, keys, expected, kind, report)
                report.references[bucket + "/computed"] += 1
                report.references[bucket + ("/matched" if matched else "/mismatched")] += 1
                if name not in REAL:
                    stored = row["metadata"]["exact_fractions"]
                    report.equal("metadata_exact_fraction_keys", set(stored), set(keys))
                    report.test("independent_metadata_exact_fractions", set(stored) == set(keys) and all(
                        Fraction(stored[key]) == reference[key] for key in keys),
                        {k: str(v) for k, v in reference.items()}, stored)
                probabilities = gold["probabilities"]
                usable = set(probabilities) == set(keys) and all(
                    type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1
                    for p in probabilities.values())
                for key, oid in zip(keys, ids):
                    report.coverage[bucket]["displayed_option_ids"][oid] += 1
                    if usable:
                        p = probabilities[key]
                        report.coverage[bucket]["target_mass_by_id"][oid] += p
                        if p == max(probabilities.values()):
                            report.coverage[bucket]["argmax_ids_including_ties"][oid] += 1
                            if name == "clinc":
                                clinc_canonical_gold[split][oid] += 1
                sample_key = bucket
                if len(report.samples[sample_key]) < 3:
                    report.samples[sample_key].append(dict(
                        report.context, family=family, type=question["type"],
                        state_snippet=compact(row["state"], 2200),
                        instructions=question["instructions"],
                        original_target_id=target,
                        original_source_annotation=source_annotation,
                        target_options=[{"key": key, "original_id": oid, "probability": float(reference[key])}
                                        for key, oid in zip(keys, ids) if reference[key]],
                        review_status="not_human_reviewed"))
            except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError, OverflowError) as exc:
                report.test("row_verification_error", False, actual=str(exc))
    report.context = {}
    intersections = {}
    for name, groups in raw_groups.items():
        intersections[name] = {}
        for left, right in itertools.combinations(SPLITS, 2):
            common = sorted(groups[left] & groups[right])
            intersections[name][left + "/" + right] = {
                "count": len(common), "raw_group_examples": common[:10]}
            report.test("raw_source_groups_disjoint:" + name + ":" + left + "/" + right,
                        not common, actual=common[:10])
    for kind, mapping in identity_splits.items():
        for identity, splits in mapping.items():
            if len(splits) > 1:
                report.test("canonical_cross_split_" + kind, False, actual=[identity, sorted(splits)])
    for (name, raw_group), groups in raw_to_canonical.items():
        report.context = {"source": name, "raw_group": raw_group}
        report.test("raw_group_consistent_canonical_group", len(groups) == 1, actual=sorted(groups))
    for (name, group), originals in canonical_to_raw.items():
        report.context = {"source": name, "group_id": group}
        report.test("canonical_group_consistent_raw_group", len(originals) == 1, actual=sorted(originals))
    report.context = {}
    covered = set().union(*(set(c) for c in clinc_gold.values()))
    canonical_covered = set().union(*(set(c) for c in clinc_canonical_gold.values()))
    missing = sorted(set(allowed) - covered)
    report.test("clinc_full_allowed_gold_coverage", not missing, actual=missing)
    canonical_missing = sorted(set(allowed) - canonical_covered)
    report.test("clinc_full_allowed_canonical_gold_coverage", not canonical_missing, actual=canonical_missing)
    for split in SPLITS:
        report.test("nonempty_canonical_split:" + split, bool(report.rows[split]))
        for name in (*REAL, *FAMILIES):
            report.test("source_present:" + split + ":" + name, report.rows[split][name] > 0)
    samples = []
    for index in range(3):
        for bucket in sorted(report.samples):
            if index < len(report.samples[bucket]) and len(samples) < 30:
                samples.append(report.samples[bucket][index])
    return {
        "source_lines_streamed": scanned,
        "selected_source_records_retained": {name: len(rows) for name, rows in selected.items()},
        "raw_source_group_intersections": intersections,
        "raw_source_group_counts": {name: {s: len(g) for s, g in groups.items()}
                                    for name, groups in raw_groups.items()},
        "identity_counts": {kind: len(mapping) for kind, mapping in identity_splits.items()},
        "clinc_full_allowed_gold_coverage": {
            "allowed_count": len(allowed), "covered_count": len(covered),
            "allowed_ids": sorted(allowed), "missing_ids": missing,
            "source_annotated_gold_counts_by_split": clinc_gold,
            "canonical_argmax_counts_by_split": clinc_canonical_gold,
            "canonical_missing_ids": canonical_missing},
        "review_samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New JSON report file; parent must exist")
    args = parser.parse_args()
    output = args.output.expanduser()
    try:
        handle = output.open("x", encoding="utf-8")
    except OSError as exc:
        parser.error("refusing unavailable/existing output: %s (%s)" % (output, exc))
    report, details = Report(), {}
    with handle:
        try:
            details = verify(args.dataset_dir.expanduser(), args.source_dir.expanduser(), report)
        except Exception as exc:
            report.test("verification_incomplete", False, actual=type(exc).__name__ + ": " + str(exc))
        result = dict(
            status="failed" if report.errors else "passed",
            scope="Prepared-source annotation fidelity and independent visible-state synthetic references only",
            limitations=[
                "Does not prove human-language annotation truth; no blanket semantic-label correctness claim.",
                "Prepared source records are the reference; upstream preparation/revision provenance is not independently proved.",
                "Absent source-row split fields rely on source_manifest.json file-level train provenance, verified against raw sources by resource preparation, not reverified by this checker. Any present row split fields must equal train.",
                "SNLI image grouping uses original prepared group_key (and imageID/image_id if present), not canonical group hashes.",
                "Synthetic policies, instructions and rubrics are checked against manually inspected literal templates; unknown wording fails.",
                "No human review performed; review snippets may be truncated. No token IDs, model, tokenizer, training or performance checks.",
                "CLINC full allowed gold coverage is required across all three splits, not independently in each split.",
            ],
            dataset_dir=str(args.dataset_dir), source_dir=str(args.source_dir),
            source_file_provenance=report.source_provenance,
            source_row_split_provenance_counts=report.source_split_basis,
            total_rows_by_split={s: sum(c.values()) for s, c in report.rows.items()},
            rows_by_split_source=report.rows, hard_soft_counts=report.kinds,
            class_coverage_by_source_primitive_type=report.coverage,
            exact_source_input_match_counts=report.exact_inputs,
            reference_counts=report.references,
            checks=report.checks, mismatch_count=sum(report.errors.values()),
            mismatch_counts=report.errors, mismatch_details=report.mismatches,
            **details)
        json.dump(result, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "mismatch_count": result["mismatch_count"],
                      "output": str(output)}))
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
