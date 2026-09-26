"""Validate native transport, then run the unmodified public JevBench CLI."""

import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import urllib.request

from common import load, require, rows, save, write_row


PUBLIC = {"easy": 48, "original": 72, "hard": 111}
SETTINGS = {"protocol": "jevbench::v1.4", "batch_size": 1, "temperature": 1,
            "readout_temperature": 1, "backend": "dllm", "probs_source": "native"}


@contextmanager
def service(model, directory):
    command = [sys.executable, "-u", str(Path(__file__).with_name("serve.py")),
               "--model-name", model["name"], "--port", "0"]
    for key in ("model_path", "device", "dtype", "max_length"):
        command.extend(["--" + key.replace("_", "-"), str(model[key])])
    ready, info = threading.Event(), {}
    with (directory / "server.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)

        def drain():
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get("event") == "ready":
                        info.update(event)
                        ready.set()
            finally:
                ready.set()

        reader = threading.Thread(target=drain, name="service-log")
        reader.start()
        try:
            require(ready.wait(120) and info.get("host") == "127.0.0.1"
                    and type(info.get("port")) is int and 0 < info["port"] < 65536,
                    "Service did not report loopback readiness within 120 seconds; see server.log")
            endpoint = f"http://127.0.0.1:{info['port']}"
            with urllib.request.urlopen(endpoint + "/health", timeout=120) as response:
                health = json.load(response)
            require(health["status"] == "ready" and health["model"] == model["name"],
                    "Health identity mismatch")
            require(all(health["runtime"][key] == SETTINGS[key]
                        for key in ("protocol", "batch_size", "temperature", "backend", "probs_source")),
                    "Health runtime settings mismatch")
            yield endpoint
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            reader.join()
            if process.stdout is not None:
                process.stdout.close()


def native(adapter, state, question, index):
    question = dict(question)
    if question["type"] == "noul":
        question.setdefault("criteria", None)
    records, _ = adapter.predict_batch([{"state": state, "qdef": question}], index)
    require(len(records) == 1, "Partial native batch")
    return records[0]


def mapped(question, record):
    kind = question["type"]
    labels = (["no", "yes"] if kind == "noul" else list(question["criteria"])
              if kind == "choice" else [str(i) for i in range(len(question["criteria"]))])
    require(len(labels) == len(record["probs"]), "Native probability count mismatch")
    return dict(zip(labels, record["probs"]))


def check_answer(question, record, answer):
    expected = mapped(question, record)
    kind = question["type"]
    require(answer["type"] == kind, "Answer primitive mismatch")
    if kind == "noul":
        actual = {"no": 1 - answer["noul"], "yes": answer["noul"]}
    else:
        actual = answer["probabilities"]
        chosen = list(expected)[record["pred_idx"]]
        require(chosen == max(expected, key=expected.__getitem__), "Native argmax mismatch")
        if kind == "choice":
            require(answer["choice"] == chosen, "Native choice field mismatch")
        else:
            require(abs(answer["score"] - sum(int(k) * p for k, p in expected.items()))
                    <= 1e-12, "Score field is not native expected index")
    from jevbench.scoring import validate_probs
    validate_probs(actual, list(expected))
    difference = max(abs(expected[k] - actual[k]) for k in expected)
    require(difference <= 1e-12, "Native transport probability mismatch")
    return difference


def validate(model, directory, endpoint, tasks):
    from adapters.inherited import Adapter
    from jevbench.adapters.typesafe import TypeSafeAdapter
    from jevbench.adapters.base import http_post_json
    from jevbench.scoring import argmax_label, validate_probs

    official = TypeSafeAdapter(endpoint=endpoint, model=model["name"], key_env="")
    state = {"status": "failed", "model": model, "attempted": 0, "counts": {},
             "max_diff": 0.0, "flip_count": 0, "tie_flip_map": {},
             "probes": [{"generated_protocol_probe": "health", "status": 200}]}
    adapter, counts, flips, first = None, Counter(), Counter(), {}
    try:
        adapter = Adapter(argparse.Namespace(**model, backend="dllm", batch_size=1))
        with (directory / "validation.jsonl").open("x", encoding="utf-8") as stream:
            for index, task in enumerate(tasks):
                row = {"task_id": task.id, "type": task.question["type"], "status": "failed"}
                try:
                    request = official.build_request(task)
                    question = request["questions"]["decision"]
                    direct = native(adapter, task.state, question, index)
                    result = official.run(task)
                    row.update(native_status=direct["status"], http_status=result.status)
                    if direct["status"] in ("unsupported_length", "unsupported_source_mask"):
                        require(not result.ok and result.status == 422
                                and result.raw.get("error") == direct["status"], "Rejection mismatch")
                        row["status"] = "agreed_rejection"
                    else:
                        require(direct["status"] == "ok" and result.ok and result.status == 200,
                                "Unexpected native/HTTP failure")
                        require(result.model == result.raw["model"] == model["name"]
                                and result.probs_source == result.raw["runtime"]["probs_source"] == "native",
                                "Model/probability provenance mismatch")
                        require(result.raw["usage"] == {"input_tokens": direct["length"], "output_tokens": 0},
                                "Native token usage mismatch")
                        expected = mapped(question, direct)
                        validate_probs(result.probs, task.labels)
                        require(set(expected) == set(task.labels), "Task label mismatch")
                        diff = check_answer(question, direct, result.raw["answers"]["decision"])
                        require(max(abs(expected[k] - result.probs[k]) for k in expected) <= 1e-12,
                                "Official adapter mapping mismatch")
                        native_label = list(expected)[direct["pred_idx"]]
                        lexical = argmax_label(result.probs)
                        if native_label != lexical:
                            require(expected[native_label] == expected[lexical], "Non-tie label flip")
                            flips[f"{native_label} -> {lexical}"] += 1
                        row.update(status="ok", max_diff=diff, native_label=native_label,
                                   official_label=lexical, tie_flip=native_label != lexical)
                        state["max_diff"] = max(state["max_diff"], diff)
                        first.setdefault(question["type"], (task, question, direct))
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                write_row(stream, row)
                stream.flush()
                counts[row["status"]] += 1
                state["attempted"] += 1
                if (index + 1) % 20 == 0:
                    print(f"validate {model['name']}: {index + 1}/{len(tasks)}", flush=True)
        require(not counts["failed"], "Validation rows failed; see validation.jsonl")
        require(set(first) == {"choice", "noul", "score"}, "Missing successful probe primitive")

        def post(label, payload, status):
            actual, body, _ = http_post_json(endpoint + "/v1/systemone", payload,
                                            {"Content-Type": "application/json"})
            state["probes"].append({"generated_protocol_probe": label, "status": actual})
            require(actual == status, f"Probe {label}: expected HTTP {status}, got {actual}")
            return body

        task, question, direct = next(iter(first.values()))
        payload = official.build_request(task)
        alias = post("alias", dict(payload, model="jev-latest"), 200)
        require(alias["model"] == model["name"], "Alias did not resolve actual model")
        check_answer(question, direct, alias["answers"]["decision"])
        post("wrong_model", dict(payload, model=model["name"] + "-unknown"), 400)
        post("malformed_primitive", dict(payload, questions={"decision": {"type": "invalid"}}), 400)
        post("overlong_state", dict(payload, state=" x" * (model["max_length"] * 4)), 422)
        # A multi-question request has one shared state, not each original task's state.
        questions = {kind: item[1] for kind, item in first.items()}
        captured = {kind: native(adapter, task.state, q, len(tasks) + i)
                    for i, (kind, q) in enumerate(questions.items())}
        require(all(r["status"] == "ok" for r in captured.values()), "Aggregate probe unsupported")
        aggregate = post("independent_multiquestion", dict(payload, questions=questions), 200)
        require(set(aggregate["answers"]) == set(questions) and aggregate["model"] == model["name"],
                "Aggregate answer identity mismatch")
        for kind, q in questions.items():
            check_answer(q, captured[kind], aggregate["answers"][kind])
        require(aggregate["usage"] == {"input_tokens": sum(r["length"] for r in captured.values()),
                                       "output_tokens": 0}, "Aggregate usage mismatch")
        state["status"] = "pass"
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state.update(counts=dict(counts), flip_count=sum(flips.values()), tie_flip_map=dict(flips))
        del adapter
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        save(directory / "validation.json", state)


def official_run(args, model, directory, endpoint, tasks, files, identity):
    task_spec = ",".join(map(str, files))
    env = dict(os.environ, PYTHONPATH=str(args.harness_root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    base = [sys.executable, "-u", "-m", "jevbench.cli"]
    command = base + ["run", "--tasks", task_spec, "--adapter", "typesafe", "--endpoint", endpoint,
                      "--model", model["name"], "--key-env", "", "--cost-basis", "self_hosted_no_billable_api",
                      "--reserve-usd", "0", "--cap-usd", "1"]
    for flag, name in (("results", "results.jsonl"), ("raw-dir", "raw"),
                       ("ledger", "ledger.jsonl"), ("manifest", "manifest.json")):
        command.extend(["--" + flag, str(directory / name)])
    with (directory / "harness.log").open("x") as log:
        completed = subprocess.run(command, cwd=args.harness_root, env=env, stdout=log, stderr=subprocess.STDOUT)
    records = rows(directory / "results.jsonl") if (directory / "results.jsonl").exists() else []
    result = {"status": "failed", "identity": identity, "model": model, "cli_returncode": completed.returncode,
              "planned": len(tasks), "attempted": len(records),
              "failure_count": sum(r["status"] == "failed" for r in records),
              "validity_counts": {key: sum(bool(r.get(key)) for r in records)
                                  for key in ("valid", "strict_valid", "renormalized")},
              "http_status_counts": dict(Counter(str(r.get("status_code")) for r in records)),
              "cost_usd": None, "cost_basis": "self_hosted_no_billable_api",
              "latency_evidence": "results.jsonl:latency_s (official runner wall time)",
              "raw_evidence": "raw/", "official_v1.4_score": None, "official_rank": None,
              "scope": "231 public tasks only; no sealed suite, no official v1.4 score or rank"}
    try:
        require(completed.returncode == 0, "Official CLI failed; inspect harness.log and partial evidence")
        require([r["task_id"] for r in records] == [t.id for t in tasks], "Not all 231 tasks attempted in order")
        with (directory / "summary.log").open("x") as log:
            subprocess.run(base + ["summarize", "--tasks", task_spec, "--results", str(directory / "results.jsonl"),
                                   "--ledger", str(directory / "ledger.jsonl"), "--public-export",
                                   str(directory / "summary.json")], cwd=args.harness_root, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        result.update(status="complete", public_summary="summary.json")
    finally:
        save(directory / "public_result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "run"), required=True)
    parser.add_argument("--harness-root", type=Path, required=True)
    parser.add_argument("--harness-revision", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--models", type=Path, required=True,
                        help="JSON file containing name/model_path/device/dtype/max_length entries")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path)
    args = parser.parse_args()
    require(args.mode != "run" or args.validation_dir is not None, "run requires --validation-dir")
    args.harness_root = args.harness_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    require(not args.output_dir.is_relative_to(args.harness_root), "Output must be outside upstream repository")
    models = load(args.models)
    require(isinstance(models, list) and models, "Expected nonempty model list")
    names = set()
    for model in models:
        require(set(model) == {"name", "model_path", "device", "dtype", "max_length"}, "Unexpected model parameters")
        name = model["name"]
        require(isinstance(name, str) and name and name not in (".", "..")
                and all(c.isascii() and (c.isalnum() or c in "._-") for c in name)
                and name not in names, "Model names must be unique safe directory names")
        names.add(name)
        model["model_path"] = str(Path(model["model_path"]).expanduser().resolve())
        require(Path(model["model_path"]).is_dir() and type(model["max_length"]) is int
                and model["max_length"] > 0 and model["dtype"] in ("bfloat16", "float16", "float32"),
                "Invalid checkpoint, dtype or max_length")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    for key in ("TYPESAFE_PRICE_INPUT_PER_M", "TYPESAFE_PRICE_OUTPUT_PER_M"):
        os.environ.pop(key, None)
    sys.path.insert(0, str(args.harness_root))
    from jevbench.tasks import load_jsonl
    files = [args.harness_root / "datasets" / "public" / (name + ".jsonl") for name in PUBLIC]
    tasks = []
    for path, count in zip(files, PUBLIC.values()):
        group = load_jsonl(str(path))
        require(len(group) == count and all(t.split == "public" for t in group), "Public task count/split mismatch")
        tasks.extend(group)
    require(len({t.id for t in tasks}) == 231, "Duplicate public task IDs")
    identity = {"harness_revision": args.harness_revision, "code_commit": args.code_commit,
                "settings": SETTINGS, "public_counts": PUBLIC, "models": models,
                "harness_root": str(args.harness_root), "public_files": list(map(str, files))}
    if args.mode == "run":
        args.validation_dir = args.validation_dir.expanduser().resolve()
        frozen = load(args.validation_dir / "identity.json")
        validated_matrix = load(args.validation_dir / "matrix.json")
        require(validated_matrix["mode"] == "validate", "Expected validation-mode evidence")
        for key in ("harness_revision", "code_commit", "settings", "public_counts"):
            require(frozen[key] == identity[key], "Validation identity mismatch: " + key)
        for model in models:
            validated = load(args.validation_dir / model["name"] / "validation.json")
            require(model in frozen["models"] and validated["model"] == model
                    and validated["status"] == "pass" and validated["attempted"] == 231
                    and any(m["name"] == model["name"] and m["status"] == "pass"
                            for m in validated_matrix["models"]),
                    "Missing matching successful validation: " + model["name"])
        identity["validation_dir"] = str(args.validation_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save(args.output_dir / "identity.json", identity)
    matrix = {"mode": args.mode, "identity": "identity.json", "models": []}
    try:
        for model in models:
            directory = args.output_dir / model["name"]
            directory.mkdir()
            entry = {"name": model["name"], "status": "failed", "directory": model["name"]}
            matrix["models"].append(entry)
            print(f"{args.mode}: starting {model['name']}", flush=True)
            try:
                with service(model, directory) as endpoint:
                    if args.mode == "validate":
                        validate(model, directory, endpoint, tasks)
                    else:
                        official_run(args, model, directory, endpoint, tasks, files, identity)
                entry["status"] = "pass" if args.mode == "validate" else "complete"
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                evidence = directory / ("validation.json" if args.mode == "validate" else "public_result.json")
                if not evidence.exists():
                    save(evidence, {"status": "failed", "model": model, "error": entry["error"]})
                print(f"{args.mode}: {model['name']} failed: {entry['error']}", flush=True)
            print(f"{args.mode}: {model['name']} {entry['status']}", flush=True)
            save(args.output_dir / "matrix.json", matrix)
    finally:
        save(args.output_dir / "matrix.json", matrix)
    return int(any(m["status"] == "failed" for m in matrix["models"]))


if __name__ == "__main__":
    raise SystemExit(main())
