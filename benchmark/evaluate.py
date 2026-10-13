"""Run input-only inference, then scoring in a separate process."""

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from common import input_bundle, load, require, save, selected_inputs, write_row


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def infer(args):
    started = time.monotonic()
    manifest, all_rows = input_bundle(args.data_root)
    data = selected_inputs(all_rows, args.smoke, args.smoke_limit, args.suites)
    output = args.output_dir.expanduser().resolve()
    info = dict(status="running", profile_id=manifest["profile_id"],
                expected_ids=[r["id"] for r in data], attempted=0,
                input_fields=["state", "qdef"], opened_bundle_files=["manifest.json", "inputs.jsonl"],
                batch_size=args.batch_size, batches_completed=0, failed_batch=None)
    save(output / "inference.json", info)
    current = []
    successful = unsupported = 0
    try:
        log(f"Loading backend={args.backend} model={args.model_path} device={args.device}")
        if args.backend == "kev":
            from adapters.kev import Adapter
        elif args.backend == "dream":
            from adapters.dream import Adapter
        else:
            from adapters.inherited import Adapter

        adapter = Adapter(args)
        log(f"Model ready after {time.monotonic() - started:.1f}s; "
            f"decisions={len(data)} batch_size={args.batch_size} max_length={args.max_length}")
        info["runtime"] = adapter.runtime
        save(output / "inference.json", info)
        with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream, \
                (output / "batches.jsonl").open("x", encoding="utf-8") as batches:
            batch_id = 0
            for spec in manifest["suites"]:
                if args.suites is not None and spec["name"] not in args.suites:
                    continue
                suite_rows = [r for r in data if r["suite"] == spec["name"]]
                suite_started = time.monotonic()
                suite_ok = suite_unsupported = 0
                log(f"Suite {spec['name']} START decisions={len(suite_rows)}")
                for offset in range(0, len(suite_rows), args.batch_size):
                    current = suite_rows[offset:offset + args.batch_size]
                    info["active_batch"] = dict(batch_id=batch_id, suite=spec["name"],
                                               ids=[r["id"] for r in current])
                    save(output / "inference.json", info)
                    predictions, timing = adapter.predict_batch(
                        [{"state": r["state"], "qdef": r["qdef"]} for r in current], batch_id)
                    for row, prediction in zip(current, predictions):
                        write_row(stream, dict(prediction, id=row["id"], suite=row["suite"]))
                    stream.flush()
                    write_row(batches, dict(batch_id=batch_id, suite=spec["name"],
                                            ids=[r["id"] for r in current],
                                            requested_batch_size=args.batch_size,
                                            window_size=len(current), timing=timing))
                    batches.flush()
                    info["attempted"] += len(current)
                    info["batches_completed"] += 1
                    batch_ok = sum(p["status"] == "ok" for p in predictions)
                    batch_unsupported = len(predictions) - batch_ok
                    successful += batch_ok
                    unsupported += batch_unsupported
                    suite_ok += batch_ok
                    suite_unsupported += batch_unsupported
                    info["active_batch"] = None
                    save(output / "inference.json", info)
                    suite_done = offset + len(current)
                    if (offset == 0 or suite_done == len(suite_rows)
                            or info["batches_completed"] % args.log_every_batches == 0):
                        fraction = info["attempted"] / len(data) if data else 1.0
                        log(f"Suite {spec['name']} {suite_done}/{len(suite_rows)} | "
                            f"overall {info['attempted']}/{len(data)} ({fraction:.1%}) | "
                            f"batches={info['batches_completed']} ok={successful} "
                            f"unsupported={unsupported} elapsed={time.monotonic() - started:.1f}s")
                    current = []
                    batch_id += 1
                log(f"Suite {spec['name']} DONE ok={suite_ok} unsupported={suite_unsupported} "
                    f"elapsed={time.monotonic() - suite_started:.1f}s")
        info["status"] = "complete"
        log(f"Inference DONE attempted={info['attempted']} ok={successful} unsupported={unsupported} "
            f"elapsed={time.monotonic() - started:.1f}s")
    except BaseException as exc:
        info.update(status="failed", error=type(exc).__name__ + ": " + str(exc),
                    failed_batch=info.get("active_batch"),
                    uncompleted_ids=info["expected_ids"][info["attempted"]:])
        save(output / "inference.json", info)
        log(f"Inference FAILED after {info['attempted']}/{len(data)} decisions: "
            f"{type(exc).__name__}: {exc}; evidence={output}")
        raise
    save(output / "inference.json", info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("dllm", "laya", "kev", "dream"))
    parser.add_argument("--backend-python", type=Path, help="Python for inference only; scoring uses this interpreter")
    parser.add_argument("--kev-source", type=Path, help="Local checkout containing the native kev package")
    parser.add_argument("--base-path", type=Path, help="Existing local Kev base weights")
    parser.add_argument("--kev-temperature", type=float, help="Diagnostic override; default is checkpoint calibration")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=3)
    parser.add_argument("--suites", nargs="+", help="Select named suites in bundle order; default is all suites")
    parser.add_argument("--log-every-batches", type=int, default=10,
                        help="Print progress every N completed batches, plus suite boundaries")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--inference-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(min(args.batch_size, args.max_length, args.smoke_limit, args.log_every_batches) > 0,
            "Budgets and log interval must be positive")
    if args.backend == "dream":
        require(args.batch_size == 1, "Dream requires --batch-size 1")
        require(Path(args.model_path).expanduser().is_dir(), "Dream requires an existing local checkpoint directory")
    if args.backend == "kev":
        require(args.backend_python is not None and args.kev_source is not None and args.base_path is not None,
                "Kev requires --backend-python, --kev-source and --base-path")
        require(args.backend_python.expanduser().is_file(), "Missing backend Python executable")
        require((args.kev_source.expanduser() / "kev" / "checkpoint.py").is_file(), "Missing native Kev source")
        require(args.base_path.expanduser().is_dir() and Path(args.model_path).expanduser().is_dir(),
                "Kev requires existing local base and adapter directories")
        require(args.batch_size == 1 and args.dtype == "float32", "Kev requires --batch-size 1 --dtype float32")
        require(args.kev_temperature is None or math.isfinite(args.kev_temperature) and args.kev_temperature > 0,
                "Kev temperature must be finite and positive")
    if args.inference_child:
        infer(args)
        return 0
    manifest, data = input_bundle(args.data_root)
    expected = selected_inputs(data, args.smoke, args.smoke_limit, args.suites)
    output = args.output_dir.expanduser().absolute()
    output.mkdir(exist_ok=False)
    run = dict(schema_version="benchmark_core_run_v1", profile_id=manifest["profile_id"],
               source_profile_id=manifest["source_profile_id"], status="running",
               mode="input_only_inference", smoke=args.smoke, smoke_limit=args.smoke_limit,
               full_status="not_full_smoke" if args.smoke else "running",
               backend=args.backend, model_path=args.model_path, device=args.device,
               requested_dtype=args.dtype, batch_size=args.batch_size, max_length=args.max_length,
               expected_ids=[r["id"] for r in expected], predict_only=args.predict_only,
               phases=dict(inference="pending", scoring="not_requested" if args.predict_only else "pending"))
    selected_suites = None
    if args.suites is not None:
        selected_suites = [spec["name"] for spec in manifest["suites"] if spec["name"] in args.suites]
        run["selected_suites"] = selected_suites
        run["selection"] = "smoke prefix only" if args.smoke else "selected retained suite run"
    save(output / "run.json", run)
    log(f"Evaluation START profile={manifest['profile_id']} decisions={len(expected)} "
        f"smoke={args.smoke} output={output}")
    backend_python = str(args.backend_python.expanduser().absolute()) if args.backend_python else sys.executable
    command = [backend_python, str(Path(__file__).resolve()), "--inference-child",
               "--data-root", str(args.data_root.expanduser().resolve()), "--output-dir", str(output),
               "--backend", args.backend, "--model-path", str(Path(args.model_path).expanduser().resolve()),
               "--device", args.device, "--dtype", args.dtype, "--batch-size", str(args.batch_size),
               "--max-length", str(args.max_length), "--smoke-limit", str(args.smoke_limit),
               "--log-every-batches", str(args.log_every_batches)]
    if selected_suites is not None:
        command.extend(["--suites", *selected_suites])
    if args.backend_python:
        command.extend(["--backend-python", backend_python])
    if args.backend == "kev":
        command.extend(["--kev-source", str(args.kev_source.expanduser().resolve()),
                        "--base-path", str(args.base_path.expanduser().resolve())])
        if args.kev_temperature is not None:
            command.extend(["--kev-temperature", str(args.kev_temperature)])
    child_env = None
    if args.backend_python:
        child_env = dict(os.environ, PYTHONNOUSERSITE="1")
        child_env.pop("PYTHONPATH", None)
    if args.smoke:
        command.append("--smoke")
    phase = "inference"
    try:
        run["phases"][phase] = "running"
        save(output / "run.json", run)
        result = subprocess.run(command, check=False, env=child_env)
        run["inference_exit_code"] = result.returncode
        require(result.returncode == 0, "Inference child failed; partial evidence retained; no scoring")
        info = load(output / "inference.json")
        require(info["status"] == "complete" and info["expected_ids"] == run["expected_ids"]
                and info["attempted"] == len(expected), "Incomplete inference")
        run["runtime"] = info["runtime"]
        run["phases"][phase] = "complete"
        save(output / "run.json", run)
        if not args.predict_only:
            phase = "scoring"
            scoring_started = time.monotonic()
            log("Scoring START (separate process)")
            run["phases"][phase] = "running"
            save(output / "run.json", run)
            command = [sys.executable, str(Path(__file__).with_name("score.py")),
                       "--data-root", str(args.data_root.expanduser().resolve()),
                       "--predictions", str(output / "predictions.jsonl"),
                       "--output-dir", str(output / "scores"), "--run-manifest", str(output / "run.json")]
            if args.smoke:
                command.append("--smoke")
            result = subprocess.run(command, check=False)
            run["scoring_exit_code"] = result.returncode
            require(result.returncode == 0, "Scorer failed; inference evidence retained")
            run["phases"][phase] = "complete"
            run["status"] = load(output / "scores" / "report.json")["status"]
            log(f"Scoring DONE elapsed={time.monotonic() - scoring_started:.1f}s "
                f"report={output / 'scores' / 'report.json'}")
        else:
            run["status"] = "predictions_complete_unscored"
        run["full_status"] = "not_full_smoke" if args.smoke else run["status"]
    except BaseException as exc:
        run["phases"][phase] = "failed"
        run.update(status="failed", full_status="not_full_smoke" if args.smoke else "failed",
                   error=type(exc).__name__ + ": " + str(exc))
        save(output / "run.json", run)
        log(f"Evaluation FAILED phase={phase}: {type(exc).__name__}: {exc}; evidence={output}")
        raise
    save(output / "run.json", run)
    log(f"Evaluation DONE status={run['status']} output={output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
