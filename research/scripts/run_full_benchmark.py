"""Run existing evaluators with prepared local data or a frozen full_eval_v1 bundle."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


SOURCES = ("typed_decisions", "ag_news", "emotion", "banking77",
           "prompt_injection", "sst5", "internal_s0")
LIMITS = ("ag_news_test_limit", "emotion_test_limit", "banking_test_limit",
          "sst5_test_limit")
REFERENCES = ("aligned_sdk_temperature", "sdk_unit", "sdk_native")
BENCHMARK_DEFAULTS = {
    "typed_decisions": ("typed_decisions_test.parquet", "parquet", 400, 2000),
    "ag_news": ("ag_news_test.jsonl", "jsonl", 7600, 7600),
    "emotion": ("emotion_test.jsonl", "jsonl", 2000, 2000),
    "banking77": ("banking77_test.csv", "csv", 3080, 3080),
    "prompt_injection": ("prompt_injection_test.parquet", "parquet", 116, 116),
    "sst5": ("sst5_test.jsonl", "jsonl", 2210, 2210),
    "internal_s0": ("internal_s0_test.jsonl", "jsonl", 1000, 1000),
}


def prepared_config(data_root):
    code = Path(__file__).resolve().parents[2]
    stub = code / "support" / "dllm_stub"
    sources = {name: data_root / "bench" / spec[0]
               for name, spec in BENCHMARK_DEFAULTS.items()}
    for path in sources.values():
        if not path.is_file():
            raise ValueError("missing source file: " + str(path))
    if not stub.is_dir():
        raise ValueError("missing support/dllm_stub: " + str(stub))
    commit = "unknown"
    if (code / ".git").exists():
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code,
                                    capture_output=True, text=True, check=False)
            if result.returncode == 0:
                commit = result.stdout.strip() or "unknown"
        except OSError:
            pass
    config = {
        "profile_id": None,
        "sources": {
            name: dict(path="bench/" + filename, format=fmt, rows=rows, decisions=decisions,
                       split="internal_test" if name == "internal_s0" else "test")
            for name, (filename, fmt, rows, decisions) in BENCHMARK_DEFAULTS.items()
        },
        "limits": dict.fromkeys(LIMITS, 0),
        "protocol": dict(max_length=4096, temperature=1.0, laya_primary="aligned_unit",
                         laya_references=list(REFERENCES)),
        "code": dict(commit=commit),
    }
    return config, sources, code, stub, sys.executable


def load_profile(path):
    with path.open(encoding="utf-8") as handle:
        profile = json.load(handle)
    if not isinstance(profile, dict) or profile.get("schema_version") != "full_eval_v1":
        raise ValueError("profile.schema_version must be full_eval_v1")

    def text(value, field):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(field + " must be a nonempty string")
        return value

    def bundled(value, field):
        relative = Path(text(value, field))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(field + " must be relative to the bundle")
        return (path.parent / relative).resolve()

    try:
        text(profile["profile_id"], "profile_id")
        for section in ("sources", "limits", "protocol", "runtime", "code"):
            if not isinstance(profile[section], dict):
                raise ValueError(section + " must be a JSON object")
        sources = {}
        for name in SOURCES:
            source = profile["sources"][name]
            if not isinstance(source, dict):
                raise ValueError("sources.%s must be a JSON object" % name)
            expected_split = "internal_test" if name == "internal_s0" else "test"
            if source["split"] != expected_split:
                raise ValueError("sources.%s.split must be %s" % (name, expected_split))
            text(source["format"], "sources.%s.format" % name)
            for field in ("rows", "decisions"):
                if type(source[field]) is not int or source[field] < 1:
                    raise ValueError("sources.%s.%s must be a positive integer" % (name, field))
            sources[name] = bundled(source["path"], "sources.%s.path" % name)
            if not sources[name].is_file():
                raise ValueError("missing source file: " + str(sources[name]))
        for name in ("ag_news", "emotion"):
            if profile["sources"][name]["format"].lower() != "jsonl":
                raise ValueError(name + " requires an explicit JSONL fixture")
        if profile["sources"]["banking77"]["format"].lower() != "csv":
            raise ValueError("banking77 requires an explicit CSV fixture")
        for name in LIMITS:
            if type(profile["limits"][name]) is not int or profile["limits"][name] != 0:
                raise ValueError("full profile requires limits.%s = 0" % name)
        protocol = profile["protocol"]
        if type(protocol["max_length"]) is not int or protocol["max_length"] < 1:
            raise ValueError("protocol.max_length must be a positive integer")
        if isinstance(protocol["temperature"], bool) or protocol["temperature"] != 1.0:
            raise ValueError("protocol.temperature must be 1.0")
        if protocol["laya_primary"] != "aligned_unit":
            raise ValueError("protocol.laya_primary must be aligned_unit")
        if (not isinstance(protocol["laya_references"], list)
                or sorted(protocol["laya_references"]) != sorted(REFERENCES)):
            raise ValueError("protocol.laya_references must name all three reference conditions")
        runtime = profile["runtime"]
        for name in ("home", "hf_home"):
            if not Path(text(runtime[name], "runtime." + name)).is_absolute():
                raise ValueError("runtime.%s must be absolute" % name)
        python = runtime.get("python") or sys.executable
        if not Path(text(python, "runtime.python")).is_absolute():
            raise ValueError("runtime.python must be absolute")
        stub = bundled(runtime["dllm_stub"], "runtime.dllm_stub")
        text(profile["code"]["commit"], "code.commit")
        code = bundled(profile["code"]["path"], "code.path")
        if not code.is_dir() or not stub.is_dir():
            raise ValueError("bundle code and dllm_stub directories must exist")
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed full_eval_v1 profile: " + str(exc)) from exc
    return profile, sources, code, stub, python


def report_counts(path, stage):
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") == "failed":
        raise ValueError("evaluator report indicates failure: " + str(path))
    if stage == "internal":
        return {"internal_s0": report["count"]}
    if stage == "external":
        return {name: result["n_decisions"] for name, result in report["suites"].items()}
    return {
        condition: {("internal_s0" if name == "internal" else name): result["n_decisions"]
                    for name, result in data["suites"].items()}
        for condition, data in report["conditions"].items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--profile", type=Path, help="Legacy frozen full_eval_v1 bundle profile")
    inputs.add_argument("--data-root", type=Path,
                        help="Prepared helper output (local_data); uses current source and Python")
    parser.add_argument("--backend", required=True, choices=("dllm", "laya"))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true",
                        help="Three decisions per suite; execution check, NOT a backend comparison")
    parser.add_argument("--laya-max-seqs", type=int, default=16)
    parser.add_argument("--laya-max-tokens", type=int, default=8192)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16",
                        help="DLLM dtype only; Laya retains FP32")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="DLLM external only in data-root mode; internal remains batch 1")
    parser.add_argument("--warmup-batches", type=int, default=0,
                        help="DLLM external only: untimed first-batch repeats per suite (data-root mode)")
    args = parser.parse_args()
    resolved_device = "cuda:0" if args.device == "cuda" else args.device
    profile_path = args.profile.expanduser().resolve() if args.profile else None
    data_root = args.data_root.expanduser().resolve() if args.data_root else None
    model = args.model_path.expanduser().resolve()
    output = args.output_dir.expanduser().absolute()
    try:
        if min(args.laya_max_seqs, args.laya_max_tokens) < 1:
            raise ValueError("Laya resource budgets must be positive")
        if args.batch_size < 1 or args.warmup_batches < 0:
            raise ValueError("batch-size must be positive and warmup-batches nonnegative")
        if args.batch_size != 1 or args.warmup_batches != 0:
            if profile_path is not None:
                raise ValueError("legacy --profile requires batch-size 1 and warmup-batches 0; "
                                 "use --data-root for DLLM external batching/warmup")
            if args.backend == "laya":
                raise ValueError("batch-size and warmup-batches are DLLM-only; "
                                 "use --laya-max-seqs/--laya-max-tokens for Laya")
        profile, sources, code, stub, python = (load_profile(profile_path) if profile_path is not None
                                               else prepared_config(data_root))
        if not model.is_dir():
            raise ValueError("model-path must be an existing local checkpoint directory")
        if os.path.lexists(output):
            raise ValueError("output-dir must be NEW; refusing overwrite: " + str(output))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    env = os.environ.copy()
    if profile_path is not None:
        env.update(HOME=profile["runtime"]["home"], HF_HOME=profile["runtime"]["hf_home"])
        # The legacy Transformers install is in HOME/.local, before child startup.
        env.pop("PYTHONUSERBASE", None)
        env.pop("PYTHONNOUSERSITE", None)
    env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1",
               USE_TF="0", USE_TORCH="1", TOKENIZERS_PARALLELISM="false",
               PYTHONPATH=os.pathsep.join(map(str, (code, code / "research" / "scripts", stub))))
    limit = "3" if args.smoke else "0"
    common = ["--model-path", str(model), "--device", resolved_device,
              "--max-length", str(profile["protocol"]["max_length"])]
    external = ["--typed-data", str(sources["typed_decisions"]),
                "--prompt-data", str(sources["prompt_injection"]),
                "--sst5-data", str(sources["sst5"]),
                "--banking-test-csv", str(sources["banking77"]),
                "--ag-news-test-jsonl", str(sources["ag_news"]),
                "--emotion-test-jsonl", str(sources["emotion"]),
                "--ag-news-test-limit", str(profile["limits"]["ag_news_test_limit"]),
                "--emotion-test-limit", str(profile["limits"]["emotion_test_limit"]),
                "--sst5-test-limit", str(profile["limits"]["sst5_test_limit"]),
                "--limit-per-suite", limit]
    tasks = []

    def task(name, script, flags):
        report = output / (name + ".json")
        predictions = output / (name + "_predictions.jsonl")
        command = [python, str(code / "research" / "scripts" / script)] + common + flags
        command += ["--output", str(report), "--predictions", str(predictions)]
        tasks.append(dict(stage=name, command=command, report=str(report),
                          predictions=str(predictions), status="pending", exit_code=None))

    if args.backend == "dllm":
        batching = (["--batch-size", str(args.batch_size), "--warmup-batches", str(args.warmup_batches)]
                    if data_root is not None else [])
        task("external", "bench_diff_yesno.py", external + ["--dtype", args.dtype,
             "--banking-test-limit", str(profile["limits"]["banking_test_limit"])] + batching)
        task("internal", "shared_yesno_supervised.py", ["--data", str(sources["internal_s0"]),
             "--amp-dtype", args.dtype, "--limit", limit])
    else:
        task("laya", "bench_laya_aligned.py", external + ["--internal-data", str(sources["internal_s0"]),
             "--max-seqs", str(args.laya_max_seqs), "--max-tokens", str(args.laya_max_tokens),
             "--head-math-sdpa"])

    expected = {name: profile["sources"][name]["decisions"] for name in SOURCES}
    planned = {name: min(3, count) if args.smoke else count for name, count in expected.items()}
    planned_stages = {
        "external": {name: count for name, count in planned.items() if name != "internal_s0"},
        "internal": {"internal_s0": planned["internal_s0"]},
        "laya": {condition: dict(planned) for condition in ("aligned_unit",) + REFERENCES},
    }
    for current in tasks:
        current["planned_counts"] = planned_stages[current["stage"]]
    execution = (dict(dtype=args.dtype, external_batch_size=args.batch_size, internal_batch_size=1,
                      external_warmup_batches_per_suite=args.warmup_batches, internal_warmup_batches=0)
                 if args.backend == "dllm" else
                 dict(dtype="float32", batching="native length-sorted token-budget batches",
                      max_seqs=args.laya_max_seqs, max_tokens=args.laya_max_tokens, warmup_batches=0))
    manifest = dict(mode="profile" if profile_path is not None else "data_root",
                    profile_id=profile["profile_id"],
                    profile_path=str(profile_path) if profile_path is not None else None,
                    data_root=str(data_root) if data_root is not None else None,
                    python=python, execution=execution,
                    model_path=str(model), backend=args.backend, code_commit=profile["code"]["commit"],
                    code_path=str(code), protocol=profile["protocol"], sources=profile["sources"],
                    resolved_device=resolved_device,
                    expected_profile_decisions=expected, planned_counts=planned, actual_counts={},
                    counts_unit="decisions per suite per condition (not summed across Laya conditions)",
                    smoke=args.smoke, full_status="not_full_smoke" if args.smoke else "pending",
                    status="pending", tasks=tasks,
                    config={key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
                    environment={key: env.get(key) for key in
                                 ("HOME", "HF_HOME", "PYTHONPATH", "HF_HUB_OFFLINE",
                                  "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE",
                                  "USE_TF", "USE_TORCH", "TOKENIZERS_PARALLELISM")},
                    smoke_note="NOT a comparison: external prefixes; internal DLLM uses helper's "
                               "seed42 whole-group spread, Laya uses first three rows.")
    output.mkdir(parents=True, exist_ok=False)

    def save():
        with (output / "run.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
            handle.write("\n")

    manifest["status"] = "running"
    if not args.smoke:
        manifest["full_status"] = "running"
    save()
    for current in tasks:
        current["status"] = "running"
        save()
        try:
            completed = subprocess.run(current["command"], cwd=code, env=env, check=False)
            current["exit_code"] = completed.returncode
            if completed.returncode:
                raise RuntimeError("child exited with code %d" % completed.returncode)
            manifest["actual_counts"][current["stage"]] = report_counts(
                Path(current["report"]), current["stage"])
            if manifest["actual_counts"][current["stage"]] != current["planned_counts"]:
                raise ValueError("reported counts do not match planned counts for %s: expected %s, got %s"
                                 % (current["stage"], current["planned_counts"],
                                    manifest["actual_counts"][current["stage"]]))
            current["status"] = "complete"
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, KeyboardInterrupt) as exc:
            current.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
            current["error_details"] = dict(type=type(exc).__name__, message=str(exc))
            manifest["status"] = "failed"
            if not args.smoke:
                manifest["full_status"] = "failed"
            save()
            print("FAILED: %s; outputs preserved in %s" % (current["stage"], output), file=sys.stderr)
            return current["exit_code"] if current["exit_code"] and current["exit_code"] > 0 else 1
        save()
    manifest["status"] = "complete"
    manifest["full_status"] = "not_full_smoke" if args.smoke else "complete"
    save()
    print("%s complete: %s" % ("SMOKE (not a comparison)" if args.smoke else "Full evaluation", args.backend))
    print("Manifest: " + str(output / "run.json"))
    for current in tasks:
        print("Report: %s\nPredictions: %s" % (current["report"], current["predictions"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
