"""Bounded, offline synthetic decision pools; no model or dataset dependencies.

Only state and questions are model inputs. Metadata is solver/audit material, not
prompt material. The caller owns split assignment and must split by group_id.
Ordinal labels describe synthetic operational milestones, never sentiment.
"""
import copy
import itertools
import json
import math
import random
from collections import Counter
from fractions import Fraction


SCHEMA_VERSION = "shared_yesno_data_v1"
POOL_NAMES = ("synthetic_constraints", "synthetic_ordinal", "synthetic_probability")
SPLIT_QUOTAS = {
    "synthetic_constraints": {"train": 1000, "dev": 100, "test": 100},
    "synthetic_ordinal": {"train": 1500, "dev": 150, "test": 150},
    "synthetic_probability": {"train": 500, "dev": 50, "test": 50},
}
_PEOPLE = ("Ada", "Bram", "Cleo", "Dara")
_COLORS = ("Red", "Blue", "Green")
_RUBRICS = {
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


def _keys(question):
    if question["type"] == "noul":
        return ["false", "true"]
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return list(question["criteria"])


def _current_pass(check, version):
    return check["status"] == "passed" and check["version"] == version


def _solve(family, facts, query):
    """Derive exact targets from process parameters/observations, not stored gold."""
    if family == "eligibility":
        eligible = [c for c in facts["candidates"]
                    if c["certified"] and c["capacity"] >= facts["minimum_capacity"]
                    and c["delay"] <= facts["maximum_delay"]]
        if not eligible:
            raise ValueError("Eligibility task has no eligible candidate")
        best_cost = min(c["cost"] for c in eligible)
        best = [c["name"] for c in eligible if c["cost"] == best_cost]
        if len(best) != 1:
            raise ValueError("Eligibility task has a tied minimum")
        return {best[0]: Fraction(1)}
    if family == "preservation":
        end = facts["balance"] + facts["change"]
        allowed = (facts["authorized"] and not facts["frozen"]
                   and facts["reserve"] <= end <= facts["ceiling"])
        return {str(allowed).lower(): Fraction(1)}
    if family == "ownership":
        owner = facts["initial_owner"]
        for event in facts["events"]:
            if event["sender"] == owner and event["accepted"]:
                owner = event["recipient"]
        answer = str(owner == query["person"]).lower() if query["type"] == "noul" else owner
        return {answer: Fraction(1)}
    if family == "shipment":
        ready = (facts["packed"] >= facts["required"] and facts["seal_intact"]
                 and facts["inspection"] == "passed")
        dispatched = facts["handed_over"] and facts["receipt_validation"] == "passed"
        level = 0 if not ready else 1 + int(dispatched)
        return {str(level): Fraction(1)}
    if family == "laboratory":
        collected = facts["collected"] - facts["damaged"] >= facts["required"]
        qualified = (_current_pass(facts["assay"], facts["batch_version"])
                     and facts["control_low"] <= facts["control"] <= facts["control_high"])
        reviewed = _current_pass(facts["review"], facts["assay_version"])
        gates = [collected, qualified, reviewed]
    elif family == "deployment":
        version = facts["current_version"]
        gates = [
            _current_pass(facts["build"], version),
            _current_pass(facts["tests"], version)
            and facts["tests_passed"] == facts["tests_required"],
            _current_pass(facts["approval"], version),
            _current_pass(facts["rollout"], version)
            and facts["healthy_zones"] == facts["required_zones"],
        ]
    else:
        gates = None
    if gates is not None:
        level = 0
        for passed in gates:
            if not passed:
                break
            level += 1
        return {str(level): Fraction(1)}
    if family == "finite_draw":
        total = sum(facts["counts"].values())
        distribution = {color: Fraction(count, total) for color, count in facts["counts"].items()}
        if query["type"] == "noul":
            p = distribution[query["color"]]
            return {"false": 1 - p, "true": p}
        return distribution
    if family == "latent_mixture":
        weights, success = [], []
        for mechanism in facts["mechanisms"]:
            likelihood = Fraction(mechanism["signal_count"], mechanism["signal_total"])
            weights.append(Fraction(mechanism["prior_weight"]) * likelihood)
            success.append(Fraction(mechanism["success_count"], mechanism["outcome_total"]))
        normalizer = sum(weights)
        if not normalizer:
            raise ValueError("Conditioning event has zero probability")
        p = sum(w * s for w, s in zip(weights, success)) / normalizer
        if query["type"] == "noul":
            return {"false": 1 - p, "true": p}
        return {"Success": p, "Failure": 1 - p}
    if family == "without_replacement":
        red, blue, draws = facts["red"], facts["blue"], facts["draws"]
        denominator = math.comb(red + blue, draws)
        distribution = {}
        for count in range(draws + 1):
            ways = (math.comb(red, count) * math.comb(blue, draws - count)
                    if count <= red and 0 <= draws - count <= blue else 0)
            distribution[str(count)] = Fraction(ways, denominator)
        return distribution
    raise ValueError("Unknown synthetic family: %s" % family)


def _presentation(family, facts, query, wording):
    """Render only observable facts and explicitly registered rules."""
    state = {"observations": copy.deepcopy(facts)}
    qtype = query["type"]
    if family == "eligibility":
        state["policy"] = (
            "A candidate is eligible exactly when certified is true, capacity is at least "
            "minimum_capacity, and delay is at most maximum_delay. Select the eligible "
            "candidate with the lowest cost. This task guarantees a unique minimum.")
        instructions = ("Which candidate does the policy select?",
                        "Select the cheapest candidate satisfying every eligibility rule.")[wording]
        criteria = {name: None for name in query["order"]}
    elif family == "preservation":
        state["policy"] = (
            "In this synthetic resource policy, balance, change, reserve, and ceiling are measured "
            "in integer resource units, not money. "
            "The proposed change is allowed exactly when authorized is true, frozen is false, "
            "and balance + change is between reserve and ceiling, including both boundaries.")
        instructions = ("Is the proposed change allowed under every policy condition?",
                        "Does the proposed operation satisfy the stated authorization and balance policy?")[wording]
        criteria = {"false": "The operation is forbidden.", "true": "The operation is allowed."}
    elif family == "ownership":
        state["policy"] = (
            "Process events in listed chronological order. A transfer changes ownership to "
            "its recipient only if its sender is the current owner and accepted is true. "
            "Otherwise ignore it. These are all ownership events.")
        if qtype == "noul":
            instructions = ("After all events, is %s the current owner?",
                            "Does %s own the item after applying the transfer policy?")[wording] % query["person"]
            criteria = {"false": None, "true": None}
        else:
            instructions = ("Who is the current owner after all events?",
                            "Select the owner obtained by applying the event policy.")[wording]
            criteria = {name: None for name in query["order"]}
    elif family in _RUBRICS:
        if family == "shipment":
            state["policy"] = (
                "Ready requires packed >= required, seal_intact true, and inspection passed. "
                "Dispatched additionally requires handed_over true and receipt_validation passed. "
                "Pending and failed validations do not pass. Later evidence cannot bypass readiness.")
        elif family == "laboratory":
            state["policy"] = (
                "Use the highest contiguous milestone from collection to qualification to release. "
                "Collection requires collected - damaged >= required. Qualification additionally "
                "requires assay status passed for batch_version and control in [control_low, "
                "control_high], inclusive. Release additionally requires review status passed for "
                "assay_version. Pending, failed, and older-version checks do not pass.")
        else:
            state["policy"] = (
                "Use the highest contiguous milestone: build, tests, approval, rollout. Each check "
                "must have status passed and version equal to current_version. Tests additionally "
                "require tests_passed = tests_required. Rollout additionally requires healthy_zones "
                "= required_zones. Pending, failed, and older-version checks do not pass. Later "
                "checks cannot bypass an earlier incomplete milestone.")
        instructions = ("Select the highest operational level justified by the observations and rubric.",
                        "Which rubric level has been reached without skipping any prerequisite?")[wording]
        criteria = list(_RUBRICS[family])
    elif family == "finite_draw":
        state["process"] = "Draw one token uniformly from this finite bag. Counts include every token."
        if qtype == "noul":
            instructions = "Will the randomly drawn token be %s? Predict this uncertain event." % query["color"]
            criteria = {"false": None, "true": None}
        else:
            instructions = "What color will the random token have? Predict the distribution over outcomes."
            criteria = {color: None for color in query["order"]}
    elif family == "latent_mixture":
        state["process"] = (
            "Choose one mechanism with probability proportional to prior_weight; its identity is "
            "not observed. Under that mechanism, a uniform draw from signal_total tickets produces "
            "an alert on signal_count tickets. Independently conditional on the same mechanism, "
            "a uniform draw from outcome_total tickets succeeds on success_count tickets. "
            "An alert was observed. Predict the outcome conditional on that alert. The table "
            "lists every possible mechanism, not a secretly selected mechanism.")
        if qtype == "noul":
            instructions = "Given the observed alert, will the outcome draw succeed? Predict this uncertain event."
            criteria = {"false": None, "true": None}
        else:
            instructions = "Given the alert, predict the distribution of the outcome draw: success or failure."
            criteria = {outcome: None for outcome in query["order"]}
    elif family == "without_replacement":
        state["process"] = (
            "Draw %d tokens uniformly without replacement from a bag containing exactly %d "
            "red tokens and %d blue tokens. Every subset of that size is equally likely."
            % (facts["draws"], facts["red"], facts["blue"]))
        instructions = ("How many red tokens will be drawn? Predict the distribution over counts.",
                        "Predict the uncertain count of red tokens in the sample, using the count rubric.")[wording]
        criteria = ["Exactly %d red tokens in the sample." % i for i in range(facts["draws"] + 1)]
    else:
        raise ValueError("Unknown synthetic family: %s" % family)
    return state, {"type": qtype, "instructions": instructions, "criteria": criteria}


def _option_ids(family, question):
    return ["%s:%s:%s" % (family, question["type"], key) for key in _keys(question)]


def _record(pool, family, root, variant, facts, query, wording, changed=False):
    group_id = "%s:%s:%04d" % (pool, family, root)
    case_id = group_id + (":counterfactual" if changed else ":base")
    state, question = _presentation(family, facts, query, wording)
    target = _solve(family, facts, query)
    keys = _keys(question)
    if set(target) - set(keys):
        raise ValueError("Solver returned an unrendered option")
    exact = {key: target.get(key, Fraction(0)) for key in keys}
    soft = pool == "synthetic_probability"
    stratum = "%s/%s" % (family, question["type"])
    if not soft:
        stratum += "/" + next(key for key, value in exact.items() if value == 1)
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "group_id": group_id,
        "view_id": group_id + ":view%d" % variant,
        "source": {"name": pool, "record_id": case_id, "original_split": "generated"},
        "family": family,
        "language": "en",
        "state": state,
        "questions": {"decision": question},
        "gold": {"decision": {"probabilities": {k: float(v) for k, v in exact.items()},
                              "kind": "known_distribution" if soft else "hard"}},
        "option_ids": {"decision": _option_ids(family, question)},
        "metadata": {
            "sampling_stratum": stratum,
            "template_id": "%s.%s.v%d" % (family, question["type"], wording + 1),
            "relation": "counterfactual" if changed else ("root" if variant == 0 else "linked_view"),
            "solver_facts": copy.deepcopy(facts),
            "query": copy.deepcopy(query),
            "wording": wording,
            "exact_fractions": {k: "%d/%d" % (v.numerator, v.denominator) for k, v in exact.items()},
        },
    }


def validate_synthetic_record(record):
    """Return None or raise ValueError on schema, presentation, or solver mismatch.

    Reconstructs targets from observations and known process parameters. Stored
    exact_fractions are checked too, but are never used to derive a target.
    Reuses the generation solver and presentation functions; this is a consistency
    check, not an independently implemented verification of their correctness.
    This checks a single row; make_synthetic_pools also checks cross-row identity.
    """
    try:
        if record["schema_version"] != SCHEMA_VERSION or record["language"] != "en":
            raise ValueError("Unsupported schema/language")
        source = record["source"]
        if source["name"] not in POOL_NAMES or source["original_split"] != "generated":
            raise ValueError("Unexpected synthetic source")
        if source["record_id"] != record["case_id"]:
            raise ValueError("Source identity mismatch")
        for field in ("case_id", "group_id", "view_id"):
            if not isinstance(record[field], str) or not record[field]:
                raise ValueError("Missing identity: %s" % field)
        for field in ("questions", "gold", "option_ids"):
            if list(record[field]) != ["decision"]:
                raise ValueError("Expected exactly one decision")
        family, metadata = record["family"], record["metadata"]
        facts, query = metadata["solver_facts"], metadata["query"]
        state, question = _presentation(family, facts, query, metadata["wording"])
        if state != record["state"] or question != record["questions"]["decision"]:
            raise ValueError("Visible input differs from registered factual presentation")
        expected_pool = ("synthetic_ordinal" if family in _RUBRICS else
                         "synthetic_constraints" if family in ("eligibility", "preservation", "ownership")
                         else "synthetic_probability")
        if source["name"] != expected_pool:
            raise ValueError("Family/pool mismatch")
        expected_template = "%s.%s.v%d" % (family, question["type"], metadata["wording"] + 1)
        if metadata["template_id"] != expected_template or not metadata["sampling_stratum"]:
            raise ValueError("Missing or inconsistent template/stratum")
        keys = _keys(question)
        ids = record["option_ids"]["decision"]
        if len(keys) < 2 or ids != _option_ids(family, question) or len(set(ids)) != len(ids):
            raise ValueError("Invalid ordered options")
        target = _solve(family, facts, query)
        if set(target) - set(keys):
            raise ValueError("Target outside exhaustive option space")
        exact = {key: target.get(key, Fraction(0)) for key in keys}
        if sum(exact.values()) != 1 or any(p < 0 or p > 1 for p in exact.values()):
            raise ValueError("Invalid exact distribution")
        gold = record["gold"]["decision"]
        expected_kind = "known_distribution" if expected_pool == "synthetic_probability" else "hard"
        if gold["kind"] != expected_kind or list(gold["probabilities"]) != keys:
            raise ValueError("Gold kind/order mismatch")
        if expected_kind == "hard" and sum(p == 1 for p in exact.values()) != 1:
            raise ValueError("Hard task lacks unique answer")
        if list(metadata["exact_fractions"]) != keys:
            raise ValueError("Exact fractions do not cover the options")
        for key in keys:
            value = gold["probabilities"][key]
            if (not isinstance(value, float) or not math.isfinite(value) or not 0 <= value <= 1
                    or not math.isclose(value, float(exact[key]), rel_tol=0, abs_tol=1e-12)):
                raise ValueError("Recomputed target mismatch: %s" % key)
            if Fraction(metadata["exact_fractions"][key]) != exact[key]:
                raise ValueError("Exact fraction mismatch: %s" % key)
        if not math.isclose(sum(gold["probabilities"].values()), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Exported distribution is not normalized")
    except (KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
        raise ValueError("Malformed synthetic record: %s" % exc) from exc
    return None


def _constraint_rows(rng):
    pool = "synthetic_constraints"
    for i in range(320):
        minimum, maximum = 5 + i % 20, 1 + i // 20
        cheap, middle = 10 + rng.randrange(30), 60 + rng.randrange(30)
        names = ["Cedar", "Maple", "Willow", "Birch"]
        rng.shuffle(names)
        candidates = [
            {"name": names[0], "capacity": minimum, "delay": maximum, "certified": True, "cost": cheap},
            {"name": names[1], "capacity": minimum + rng.randrange(4), "delay": maximum - 1,
             "certified": True, "cost": middle},
            {"name": names[2], "capacity": minimum + 2, "delay": maximum + 1,
             "certified": True, "cost": cheap - 1},
            {"name": names[3], "capacity": minimum + 1, "delay": maximum,
             "certified": False, "cost": cheap - 2},
        ]
        rng.shuffle(candidates)
        facts = {"minimum_capacity": minimum, "maximum_delay": maximum, "candidates": candidates}
        for v in range(2):
            f = copy.deepcopy(facts)
            if v:
                candidate = next(c for c in f["candidates"] if c["name"] == names[0])
                field, value = (("capacity", minimum - 1), ("delay", maximum + 1),
                                ("certified", False))[i % 3]
                candidate[field] = value
            order = [c["name"] for c in candidates]
            rng.shuffle(order)
            yield _record(pool, "eligibility", i, v, f, {"type": "choice", "order": order},
                          (i // 2 + v) % 2, bool(v))

    for i in range(320):
        reserve, width = 2 + i % 20, 5 + i // 20
        balance = reserve + rng.randrange(width + 1)
        boundary = reserve if i % 2 else reserve + width
        facts = {"reserve": reserve, "ceiling": reserve + width, "balance": balance,
                 "change": boundary - balance, "authorized": True, "frozen": False}
        for v in range(2):
            f = dict(facts)
            if v:
                if i % 4 == 0:
                    f["authorized"] = False
                elif i % 4 == 1:
                    f["frozen"] = True
                else:
                    f["change"] += -1 if i % 2 else 1
            yield _record(pool, "preservation", i, v, f, {"type": "noul"},
                          (i // 2 + v) % 2, bool(v))

    # Base-4 histories mix valid transfers, unauthorized senders, and rejected
    # proposals. The tracked owner is used to generate events, never as input.
    for i in range(320):
        owner, events, code = 0, [], i
        for _ in range(5):
            digit, code = code % 4, code // 4
            sender = (owner + 1) % 4 if digit == 2 else owner
            recipient_index = (owner + 1 + int(digit in (1, 2))) % 4
            events.append({"sender": _PEOPLE[sender], "recipient": _PEOPLE[recipient_index],
                           "accepted": digit != 3})
            if digit < 2:
                owner = recipient_index
        recipient = _PEOPLE[(owner + 1 + i % 3) % 4]
        events.append({"sender": _PEOPLE[owner], "recipient": recipient, "accepted": True})
        for v in range(2):
            f = {"initial_owner": _PEOPLE[0], "events": copy.deepcopy(events)}
            if v:
                f["events"][-1]["accepted"] = False
            order = list(_PEOPLE)
            rng.shuffle(order)
            query = ({"type": "noul", "person": recipient} if i % 2
                     else {"type": "choice", "order": order})
            yield _record(pool, "ownership", i, v, f, query, (i // 2 + v) % 2, bool(v))


def _check(status, version):
    return {"status": status, "version": version}


def _ordinal_rows(rng):
    for family, levels in (("shipment", 3), ("laboratory", 4), ("deployment", 5)):
        for i in range(480):
            level, block = i % levels, i // levels
            version = 2 + block % 7
            if family == "shipment":
                required = 8 + block
                facts = {"required": required, "packed": required + rng.randrange(4),
                         "seal_intact": True, "inspection": "passed", "handed_over": True,
                         "receipt_validation": "passed"}
                if level == 0:
                    field, value = (("packed", required - 1), ("seal_intact", False),
                                    ("inspection", "pending"), ("inspection", "failed"))[block % 4]
                    facts[field] = value
                elif level == 1:
                    field, value = (("handed_over", False), ("receipt_validation", "pending"),
                                    ("receipt_validation", "failed"))[block % 3]
                    facts[field] = value
            elif family == "laboratory":
                required, damaged = 5 + block, rng.randrange(4)
                low = 10 + block % 11
                facts = {"required": required, "collected": required + damaged + rng.randrange(3),
                         "damaged": damaged, "batch_version": version, "assay_version": version + 2,
                         "assay": _check("passed", version), "review": _check("passed", version + 2),
                         "control_low": low, "control_high": low + 4, "control": low + block % 5}
                if level == 0:
                    facts["collected"] = required + damaged - 1
                elif level == 1:
                    mode = block % 5
                    if mode < 3:
                        facts["assay"] = _check(("pending", "failed", "passed")[mode],
                                                version - int(mode == 2))
                    else:
                        facts["control"] = low - 1 if mode == 3 else low + 5
                elif level == 2:
                    mode = block % 3
                    facts["review"] = _check(("pending", "failed", "passed")[mode],
                                             version + 2 - int(mode == 2))
            else:
                tests, zones = 12 + block, 2 + block % 9
                facts = {"current_version": version, "build": _check("passed", version),
                         "tests": _check("passed", version), "approval": _check("passed", version),
                         "rollout": _check("passed", version), "tests_required": tests,
                         "tests_passed": tests, "required_zones": zones, "healthy_zones": zones}
                if level < 4:
                    field = ("build", "tests", "approval", "rollout")[level]
                    mode = block % 4
                    if mode == 3 and field in ("tests", "rollout"):
                        facts["tests_passed" if field == "tests" else "healthy_zones"] -= 1
                    else:
                        facts[field] = _check(("pending", "failed", "passed", "pending")[mode],
                                              version - int(mode == 2))
            for v in range(2):
                yield _record("synthetic_ordinal", family, i, v, facts, {"type": "score"}, v)


def _probability_rows(rng):
    pool = "synthetic_probability"
    # Primitive count vectors prevent repeated distributions obtained by scaling.
    vectors = [counts for counts in itertools.product(range(13), repeat=3)
               if sum(counts) and math.gcd(math.gcd(counts[0], counts[1]), counts[2]) == 1]
    zero = [c for c in vectors if 0 in c]
    positive = [c for c in vectors if 0 not in c]
    rng.shuffle(zero)
    rng.shuffle(positive)
    for i, counts in enumerate(zero[:60] + positive[:140]):
        facts = {"counts": dict(zip(_COLORS, counts))}
        order = list(_COLORS)
        rng.shuffle(order)
        for v in range(2):
            query = ({"type": "choice", "order": order} if not v else
                     {"type": "noul", "color": _COLORS[i % 3]})
            yield _record(pool, "finite_draw", i, v, facts, query, 0)

    parameters = list(itertools.product(range(1, 5), range(1, 5), range(6), range(1, 6)))
    rng.shuffle(parameters)
    for i, (prior_a, prior_b, signal_a, signal_b) in enumerate(parameters[:200]):
        # Two hypotheses remain in the input; no selected hidden world is sampled.
        facts = {"mechanisms": [
            {"name": "Copper", "prior_weight": prior_a, "signal_count": signal_a,
             "signal_total": 5, "success_count": i % 7, "outcome_total": 6},
            {"name": "Silver", "prior_weight": prior_b, "signal_count": signal_b,
             "signal_total": 6, "success_count": (i * 3 + 1) % 8, "outcome_total": 7},
        ]}
        rng.shuffle(facts["mechanisms"])
        order = ["Success", "Failure"]
        rng.shuffle(order)
        for v in range(2):
            query = {"type": "noul"} if not v else {"type": "choice", "order": order}
            yield _record(pool, "latent_mixture", i, v, facts, query, 0)

    parameters = [(red, blue, draws) for red in range(13) for blue in range(13)
                  for draws in range(2, 5) if red + blue >= draws]
    boundary = [p for p in parameters if min(p[:2]) < p[2]]
    interior = [p for p in parameters if min(p[:2]) >= p[2]]
    rng.shuffle(boundary)
    rng.shuffle(interior)
    for i, (red, blue, draws) in enumerate(boundary[:80] + interior[:120]):
        facts = {"red": red, "blue": blue, "draws": draws}
        for v in range(2):
            yield _record(pool, "without_replacement", i, v, facts, {"type": "score"}, v)


def _signature(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def make_synthetic_pools(seed=42):
    """Return (pool-name -> canonical rows, JSON-serializable coverage report).

    Produces 1,920 constraint, 2,880 ordinal, and 1,200 probability views with
    exactly two views per independent root. Generation is bounded and uses a
    private random.Random instance. Every row is solver-validated before return.
    No split field is set. All requested split counts are even, allowing exact
    selection of whole two-view groups without group leakage.
    """
    rng = random.Random(seed)
    pools = dict(zip(POOL_NAMES, (list(_constraint_rows(rng)), list(_ordinal_rows(rng)),
                                  list(_probability_rows(rng)))))
    report = {"schema_version": SCHEMA_VERSION, "seed": seed, "pools": {},
              "split_policy": "Parent assigns splits; keep every group_id in exactly one split.",
              "limitations": [
                  "Synthetic controlled English policies and operational rubrics, not real sentiment.",
                  "Two correlated views per root; count groups, not views, as independent samples.",
                  "Counterfactual constraint pairs; ordinal and count-probability pairs are rephrasings.",
                  "Probability targets describe stated ideal processes, not empirical calibration.",
                  "Solver metadata and exact fractions must never be included in model inputs.",
                  "Validation reuses the generation solver; it is not independent solver verification.",
                  "Identifiers are scoped to one generation call; do not merge different seeds.",
                  "No tokenizer/context-length validation; remote smoke owns executable integration.",
              ]}
    all_view_ids = set()
    for name, rows in pools.items():
        groups, cases, signatures = {}, {}, {}
        types, strata, templates, families, hard_labels = (Counter() for _ in range(5))
        zero_mass, nonuniform = 0, 0
        for row in rows:
            validate_synthetic_record(row)
            if row["view_id"] in all_view_ids:
                raise ValueError("Repeated view ID")
            all_view_ids.add(row["view_id"])
            group, case, family = row["group_id"], row["case_id"], row["family"]
            groups.setdefault(group, []).append(row)
            signature = _signature([family, row["metadata"]["solver_facts"]])
            if case in cases and cases[case] != signature:
                raise ValueError("A case ID refers to changed facts")
            cases[case] = signature
            if signature in signatures and signatures[signature] != group:
                raise ValueError("Duplicate factual scenario crosses groups")
            signatures[signature] = group
            question, gold = row["questions"]["decision"], row["gold"]["decision"]
            types[question["type"]] += 1
            strata[row["metadata"]["sampling_stratum"]] += 1
            templates[row["metadata"]["template_id"]] += 1
            families[family] += 1
            values = list(gold["probabilities"].values())
            zero_mass += int(0.0 in values)
            nonuniform += int(len(set(values)) > 1)
            if gold["kind"] == "hard":
                answer = next(k for k, p in gold["probabilities"].items() if p == 1)
                hard_labels["%s/%s/%s" % (family, question["type"], answer)] += 1
        for pair in groups.values():
            if len(pair) != 2:
                raise ValueError("Expected two views per root")
            if name == "synthetic_constraints":
                if pair[0]["case_id"] == pair[1]["case_id"]:
                    raise ValueError("Counterfactual must change case identity")
                first = _solve(pair[0]["family"], pair[0]["metadata"]["solver_facts"], pair[0]["metadata"]["query"])
                second = _solve(pair[1]["family"], pair[1]["metadata"]["solver_facts"], pair[1]["metadata"]["query"])
                if first == second:
                    raise ValueError("Counterfactual did not change the decision")
        rng.shuffle(rows)
        report["pools"][name] = {
            "accepted_views": len(rows), "independent_roots": len(groups), "unique_cases": len(cases),
            "unique_factual_scenarios": len(signatures), "views_per_group": 2,
            "families": dict(families), "question_types": dict(types), "templates": dict(templates),
            "sampling_strata": dict(strata), "hard_labels": dict(hard_labels),
            "views_with_zero_mass": zero_mass, "views_with_nonuniform_target": nonuniform,
            "validated_views": len(rows), "rejected_views": 0,
            "requested_split_views": dict(SPLIT_QUOTAS[name]),
            "requested_split_groups": {split: count // 2 for split, count in SPLIT_QUOTAS[name].items()},
            "surplus_views": len(rows) - sum(SPLIT_QUOTAS[name].values()),
        }
    return pools, report
