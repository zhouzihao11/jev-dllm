"""Bounded S1 fact supplements using only the registered S0 tasks and vocabulary."""
import math
import random

import shared_yesno_synthetic as original


def canonical_facts(row):
    facts = dict(row["metadata"]["solver_facts"])
    for key in ("candidates", "mechanisms"):
        if key in facts:
            facts[key] = sorted(facts[key], key=lambda item: item["name"])
    return original._signature([row["family"], facts])


def constraints(rng, family, index):
    if family == "eligibility":
        minimum, maximum = rng.randint(5, 80), rng.randint(2, 30)
        names = ["Cedar", "Maple", "Willow", "Birch"]
        costs = rng.sample(range(1, 300), len(names))
        candidates = [dict(name=name, capacity=minimum + rng.randint(-5, 10),
                           delay=maximum + rng.randint(-2, 5), certified=bool(rng.randrange(2)), cost=cost)
                      for name, cost in zip(names, costs)]
        forced = rng.choice(candidates)
        forced.update(certified=True, capacity=minimum + rng.randint(0, 10), delay=rng.randint(0, maximum))
        rng.shuffle(candidates)
        rng.shuffle(names)
        return dict(minimum_capacity=minimum, maximum_delay=maximum, candidates=candidates), dict(type="choice", order=names)
    if family == "preservation":
        reserve = rng.randint(0, 150)
        ceiling = reserve + rng.randint(5, 150)
        balance = rng.randint(reserve, ceiling)
        facts = dict(reserve=reserve, ceiling=ceiling, balance=balance,
                     change=rng.randint(reserve, ceiling) - balance, authorized=True, frozen=False)
        if index % 2:
            mode = rng.randrange(4)
            if mode == 0:
                facts["authorized"] = False
            elif mode == 1:
                facts["frozen"] = True
            else:
                end = reserve - rng.randint(1, 20) if mode == 2 else ceiling + rng.randint(1, 20)
                facts["change"] = end - balance
        return facts, dict(type="noul")
    people = list(original._PEOPLE)
    initial = owner = rng.choice(people)
    events = []
    for _ in range(rng.randint(3, 12)):
        sender = owner if rng.random() < .6 else rng.choice(people)
        recipient = rng.choice([p for p in people if p != sender])
        accepted = rng.random() < .7
        events.append(dict(sender=sender, recipient=recipient, accepted=accepted))
        if sender == owner and accepted:
            owner = recipient
    rng.shuffle(people)
    query = dict(type="choice", order=people)
    if index % 2:
        person = owner if (index // 2) % 2 else rng.choice([p for p in people if p != owner])
        query = dict(type="noul", person=person)
    return dict(initial_owner=initial, events=events), query


def ordinal(rng, family, index):
    level = index % len(original._RUBRICS[family])
    version = rng.randint(2, 50)

    def check(passed, current):
        if passed:
            return dict(status="passed", version=current)
        status = rng.choice(["pending", "failed", "passed"])
        return dict(status=status, version=current - rng.randint(1, min(current - 1, 4)) if status == "passed" else current)

    if family == "shipment":
        required = rng.randint(5, 300)
        facts = dict(required=required, packed=required + rng.randint(0, 30), seal_intact=True,
                     inspection="passed", handed_over=True, receipt_validation="passed")
        if level == 0:
            field = rng.choice(["packed", "seal_intact", "inspection"])
            facts[field] = {"packed": required - rng.randint(1, required), "seal_intact": False,
                            "inspection": rng.choice(["pending", "failed"])}[field]
            facts["handed_over"] = bool(rng.randrange(2))
            facts["receipt_validation"] = rng.choice(["passed", "pending", "failed"])
        elif level == 1:
            if rng.randrange(2):
                facts["handed_over"] = False
            else:
                facts["receipt_validation"] = rng.choice(["pending", "failed"])
    elif family == "laboratory":
        required, damaged = rng.randint(5, 200), rng.randint(0, 15)
        low = rng.randint(0, 80)
        high = low + rng.randint(2, 20)
        assay_version = version + rng.randint(1, 10)
        facts = dict(required=required, damaged=damaged, collected=required + damaged + rng.randint(0, 20),
                     batch_version=version, assay_version=assay_version,
                     assay=check(level >= 2, version), review=check(level >= 3, assay_version),
                     control_low=low, control_high=high, control=rng.randint(low, high))
        if level == 0:
            facts["collected"] = damaged + rng.randint(0, required - 1)
            facts["assay"] = check(bool(rng.randrange(2)), version)
        elif level == 1 and rng.randrange(2):
            facts["assay"] = check(True, version)
            facts["control"] = low - rng.randint(1, 10) if rng.randrange(2) else high + rng.randint(1, 10)
        if level < 2:
            facts["review"] = check(bool(rng.randrange(2)), assay_version)
    else:
        tests, zones = rng.randint(5, 200), rng.randint(2, 30)
        facts = dict(current_version=version, tests_required=tests, tests_passed=tests,
                     required_zones=zones, healthy_zones=zones)
        fields = ["build", "tests", "approval", "rollout"]
        for i, field in enumerate(fields):
            facts[field] = check(i < level if i <= level else bool(rng.randrange(2)), version)
        if level in (1, 3) and rng.randrange(2):
            facts[fields[level]] = check(True, version)
            field, required = ("tests_passed", tests) if level == 1 else ("healthy_zones", zones)
            facts[field] = rng.randint(0, required - 1)
    return facts, dict(type="score")


def probability(rng, family, index):
    if family == "finite_draw":
        counts = [rng.randint(0, 60) for _ in original._COLORS]
        if not any(counts):
            counts[0] = 1
        divisor = math.gcd(*counts)
        counts = [c // divisor for c in counts]
        colors = list(original._COLORS)
        rng.shuffle(colors)
        query = dict(type="choice", order=colors) if index % 2 else dict(type="noul", color=rng.choice(colors))
        return dict(counts=dict(zip(original._COLORS, counts))), query
    if family == "latent_mixture":
        mechanisms = []
        for name in ("Copper", "Silver"):
            signals, outcomes = rng.randint(2, 40), rng.randint(2, 60)
            mechanisms.append(dict(name=name, prior_weight=rng.randint(1, 20), signal_total=signals,
                                   signal_count=rng.randint(1, signals), outcome_total=outcomes,
                                   success_count=rng.randint(0, outcomes)))
        rng.shuffle(mechanisms)
        order = ["Success", "Failure"]
        rng.shuffle(order)
        return dict(mechanisms=mechanisms), dict(type="noul") if index % 2 else dict(type="choice", order=order)
    draws = rng.randint(2, 6)
    red, blue = rng.randint(0, 40), rng.randint(0, 40)
    if red + blue < draws:
        blue = draws - red
    return dict(red=red, blue=blue, draws=draws), dict(type="score")


def make_s1_pools(seed, candidates_per_pool):
    """One bounded pass, no repeated seeds or quota-driven retries; S0 stays unchanged."""
    pools, inherited = original.make_synthetic_pools(seed=seed)
    families = {
        "synthetic_constraints": (constraints, ["eligibility", "preservation", "ownership"]),
        "synthetic_ordinal": (ordinal, ["shipment", "laboratory", "deployment"]),
        "synthetic_probability": (probability, ["finite_draw", "latent_mixture", "without_replacement"]),
    }
    report = dict(inherited=inherited, supplement_candidates={}, duplicate_facts_discarded={})
    for pool, (generate, names) in families.items():
        rng = random.Random("s1:%d:%s" % (seed, pool))
        seen = {canonical_facts(row) for row in pools[pool]}
        accepted, duplicates = 0, 0
        for i in range(candidates_per_pool):
            family = names[i % len(names)]
            facts, query = generate(rng, family, i // len(names))
            row = original._record(pool, family, i, 0, facts, query, rng.randrange(2))
            signature = canonical_facts(row)
            if signature in seen:
                duplicates += 1
                continue
            seen.add(signature)
            for field in ("case_id", "group_id", "view_id"):
                row[field] = "supplement:" + row[field]
            row["source"]["record_id"] = row["case_id"]
            row["metadata"]["generation"] = "s1_parameter_supplement_v1"
            original.validate_synthetic_record(row)
            pools[pool].append(row)
            accepted += 1
        report["supplement_candidates"][pool] = dict(attempted=candidates_per_pool, distinct=accepted)
        report["duplicate_facts_discarded"][pool] = duplicates
    return pools, report
