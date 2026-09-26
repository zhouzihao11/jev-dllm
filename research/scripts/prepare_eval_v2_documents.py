"""Build document/forecast v2 suites from locally acquired official snapshots only."""

import argparse
from collections import Counter
import json
from pathlib import Path
from urllib.parse import quote
from zipfile import ZipFile


CONTRACT_REV = "eced6528dd3c1d14d73f9a87df8f7bdbc03126f9"
FORECAST_REV = "87a8d050d05bccde9973d478f594acba0ae2ed53"
QUESTION_FILE = "2024-07-21-llm.json"
RESOLUTION_FILE = "2024-07-21_resolution_set.json"
CONTRACT_ADAPTATION = (
    "Full original contract text and one original hypothesis per independent decision; "
    "no evidence spans, chunking, truncation or length filtering. "
    "Entailment=0, Contradiction=1, NotMentioned=2."
)
FORECAST_ADAPTATION = (
    "Retrospective fixed-round evaluation, not contamination-free prospective forecasting. "
    "Only official resolved=true rows; resolved_to 0/1 is hard false/true truth. "
    "Each source horizon/direction is an independent binary decision. For compound "
    "questions true means the conjunction of the specified component polarities holds "
    "(direction +1 means yes, -1 means no). No invented soft targets. "
    "Input uses only frozen question context; dataset date placeholders are bound to "
    "the original forecast due date and scheduled horizon. Actual market resolution "
    "dates and all resolution values remain outside model input."
)
CONTEXT_FIELDS = (
    "question", "background", "resolution_criteria", "market_info_open_datetime",
    "market_info_close_datetime", "market_info_resolution_criteria", "url",
    "freeze_datetime", "freeze_datetime_value", "freeze_datetime_value_explanation",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, "duplicate JSON key: " + key)
        out[key] = value
    return out


def reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def loads(data):
    return json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)


def source_key(row):
    return row["source"], json.dumps(row["id"], ensure_ascii=True, separators=(",", ":"))


def decision(identity, group, state, qdef, gold, options, metadata):
    return dict(id=identity, group_id=group, state=state, qdef=qdef,
                gold_idx=gold, soft=None, gold_score=None, option_ids=options,
                metadata=metadata)


def descriptor(name, category, rows, source, protocol):
    return dict(name=name, path=f"data/{name}.jsonl", category=category,
                expected_count=len(rows), source=source, protocol=protocol)


def contract_suite(raw):
    with ZipFile(raw / "contract-nli.zip") as archive:
        data = loads(archive.read("contract-nli/test.json"))
    source = dict(dataset="stanfordnlp/contract-nli", revision=CONTRACT_REV,
                  split="test", license="CC-BY-4.0", archive_member="contract-nli/test.json")
    options = ["entailment", "contradiction", "not-mentioned"]
    labels = {"Entailment": 0, "Contradiction": 1, "NotMentioned": 2}
    criteria = dict(zip(options, [
        "The contract entails the hypothesis.",
        "The contract contradicts the hypothesis.",
        "The hypothesis is neither entailed nor contradicted by the contract.",
    ]))
    rows = []
    for document in data["documents"]:
        require(len(document["annotation_sets"]) == 1, "ambiguous contract annotation set")
        annotations = document["annotation_sets"][0]["annotations"]
        require(set(annotations) == set(data["labels"]), "incomplete contract hypothesis group")
        group = f"contractnli/test/{document['id']}"
        for hypothesis_id, label in data["labels"].items():
            gold = labels[annotations[hypothesis_id]["choice"]]
            rows.append(decision(
                f"{group}/{hypothesis_id}", group,
                {"contract": document["text"], "hypothesis": label["hypothesis"]},
                dict(type="choice", instructions="Classify the hypothesis using the full contract.",
                     criteria=criteria), gold, options,
                dict(task_id=hypothesis_id, source=dict(source, row_id=document["id"],
                     hypothesis_id=hypothesis_id, file_name=document["file_name"], url=document["url"]),
                     adaptation=CONTRACT_ADAPTATION)))
    spec = descriptor("contractnli", "document_nli", rows, source,
                      dict(selection="All test contracts in source order, all hypotheses in labels order.",
                           adaptation=CONTRACT_ADAPTATION, raw_contracts=len(data["documents"]),
                           hypotheses_per_contract=len(data["labels"]), excluded=0,
                           overlength="Retained for evaluator unsupported_length reporting."))
    return rows, spec


def forecast_context(question, due, horizon):
    context = {field: question[field] for field in CONTEXT_FIELDS}
    for field, value in context.items():
        require(isinstance(value, str), "unexpected question context type: " + field)
        value = value.replace("{forecast_due_date}", due)
        if horizon is not None:
            value = value.replace("{resolution_date}", horizon)
        require("{resolution_date}" not in value, "unbound resolution date")
        context[field] = value
    require(context["freeze_datetime"][:10] <= due, "post-forecast context freeze")
    return context


def forecast_suite(raw):
    questions = loads((raw / QUESTION_FILE).read_text(encoding="utf-8"))
    resolutions = loads((raw / RESOLUTION_FILE).read_text(encoding="utf-8"))
    due = questions["forecast_due_date"]
    require(resolutions["forecast_due_date"] == due, "mismatched forecast rounds")
    require(questions["question_set"] == resolutions["question_set"] == QUESTION_FILE,
            "mismatched question set")
    indexed = {}
    for question in questions["questions"]:
        key = source_key(question)
        require(key not in indexed, "duplicate forecast question")
        indexed[key] = question
    source = dict(dataset="forecastingresearch/forecastbench-datasets", revision=FORECAST_REV,
                  split="2024-07-21-llm/resolved", license="CC-BY-SA-4.0",
                  question_file=QUESTION_FILE, resolution_file=RESOLUTION_FILE,
                  snapshot_commit_datetime="2026-09-25T06:28:42Z", forecast_due_date=due)
    rows, excluded, seen = [], Counter(), set()
    for index, resolution in enumerate(resolutions["resolutions"]):
        key = source_key(resolution)
        require(key in indexed, "resolution without a frozen question")
        question = indexed[key]
        direction = resolution["direction"]
        unique = (key, resolution["resolution_date"], json.dumps(direction))
        require(unique not in seen, "duplicate resolution decision")
        seen.add(unique)
        require(type(resolution["resolved"]) is bool, "invalid resolved flag")
        if not resolution["resolved"]:
            excluded["official_unresolved"] += 1
            continue
        gold = resolution["resolved_to"]
        require(type(gold) in (int, float) and gold in (0, 1), "resolved target is not hard binary")
        dates = question["resolution_dates"]
        horizon = None
        if isinstance(dates, list):
            horizon = resolution["resolution_date"]
            require(horizon in dates and horizon > due, "invalid scheduled dataset horizon")
        else:
            require(dates == "N/A", "unknown horizon format")
        state = {"forecast_as_of_date": due}
        if horizon is not None:
            state["scheduled_resolution_date"] = horizon
        if isinstance(question["id"], list):
            components = question["combination_of"]
            require(isinstance(direction, list) and len(direction) == len(components)
                    and all(type(d) is int and d in (-1, 1) for d in direction),
                    "invalid compound direction")
            require([part["id"] for part in components] == question["id"], "component order mismatch")
            state["components"] = [forecast_context(part, due, horizon) for part in components]
            state["event"] = {"operator": "and", "required_component_answers": [
                "yes" if d == 1 else "no" for d in direction]}
            instruction = (
                "As of forecast_as_of_date, assess one event: every component question resolves "
                "to its corresponding required_component_answer. True means this entire conjunction "
                "occurs; false means it does not. Use only the supplied frozen context."
            )
        else:
            require(direction is None, "unexpected direction on a single question")
            state["question_context"] = forecast_context(question, due, horizon)
            instruction = (
                "As of forecast_as_of_date, assess whether the supplied question resolves as Yes "
                "under its resolution criteria. True means the event occurs; false means it does "
                "not. Use only the supplied frozen context."
            )
        group = "forecastbench/" + due + "/" + quote(key[0], safe="") + "/" + quote(key[1], safe="")
        suffix = quote(json.dumps([resolution["resolution_date"], direction], separators=(",", ":")), safe="")
        rows.append(decision(
            group + "/" + suffix, group, state,
            dict(type="noul", instructions=instruction, criteria=None), int(gold), ["false", "true"],
            dict(task_id=question["source"], source=dict(source, row_id=question["id"],
                 resolution_row_index=index, direction=direction,
                 resolution_date=resolution["resolution_date"]), adaptation=FORECAST_ADAPTATION)))
    spec = descriptor("forecastbench", "forecasting", rows, source,
                      dict(selection="Every official resolved decision in resolution-file order; no subsampling.",
                           adaptation=FORECAST_ADAPTATION, raw_questions=len(indexed),
                           raw_resolution_rows=len(resolutions["resolutions"]), excluded=dict(excluded),
                           eligible_groups=len({row["group_id"] for row in rows}),
                           absent_resolutions="Unpublished/future/nullified source decisions are not synthesized.",
                           input_context="Question snapshot only, including frozen reference/market values; no retrieval.",
                           group_scope="All eligible published decisions per original question, not unresolved horizons."))
    return rows, spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--include-source", action="append", choices=("contractnli", "forecastbench"))
    args = parser.parse_args()
    selected = args.include_source or ["contractnli", "forecastbench"]
    suites = [builder(args.raw_root) for name, builder in
              (("contractnli", contract_suite), ("forecastbench", forecast_suite)) if name in selected]
    identities = set()
    for rows, spec in suites:
        require(bool(rows), "empty suite: " + spec["name"])
        for row in rows:
            require(row["id"] not in identities, "duplicate output decision")
            identities.add(row["id"])
        for path in (spec["path"], spec["name"] + ".suite.json"):
            require(not (args.output_root / path).exists(), "refusing to overwrite: " + path)
    (args.output_root / "data").mkdir(parents=True, exist_ok=True)
    for rows, spec in suites:
        with (args.output_root / spec["path"]).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        with (args.output_root / (spec["name"] + ".suite.json")).open("x", encoding="utf-8") as stream:
            json.dump(spec, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        print(f"{spec['name']}: {len(rows)} decisions -> {spec['path']}")


if __name__ == "__main__":
    main()
