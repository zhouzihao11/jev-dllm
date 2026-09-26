"""Run the input-only benchmark entry point for a runtime model list."""

import argparse
from pathlib import Path
import subprocess
import sys

from common import load, require, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=3)
    args = parser.parse_args()
    models = load(args.models)
    require(isinstance(models, list) and models, "Expected nonempty model list")
    names = [model["name"] for model in models]
    require(len(set(names)) == len(names) and all(
        isinstance(name, str) and name and all(c.isascii() and (c.isalnum() or c in "_-") for c in name)
        for name in names), "Invalid or duplicate model names")
    output = args.output_dir.expanduser().absolute()
    output.mkdir(exist_ok=False)
    matrix = dict(status="running", smoke=args.smoke, models=models, tasks=[])
    save(output / "matrix.json", matrix)
    for model in models:
        command = [sys.executable, str(Path(__file__).with_name("evaluate.py")),
                   "--data-root", str(args.data_root.expanduser().resolve()),
                   "--backend", model["backend"], "--model-path", model["model_path"],
                   "--device", model.get("device", "cuda:0"),
                   "--dtype", model.get("dtype", "bfloat16"),
                   "--batch-size", str(args.batch_size), "--max-length", str(args.max_length),
                   "--smoke-limit", str(args.smoke_limit), "--output-dir", str(output / model["name"])]
        if args.smoke:
            command.append("--smoke")
        task = dict(model=model["name"], command=command, status="running")
        matrix["tasks"].append(task)
        save(output / "matrix.json", matrix)
        print("Starting " + model["name"], flush=True)
        result = subprocess.run(command, check=False)
        task["exit_code"] = result.returncode
        if result.returncode:
            task["status"] = matrix["status"] = "failed"
            save(output / "matrix.json", matrix)
            return result.returncode
        task["status"] = load(output / model["name"] / "run.json")["status"]
        save(output / "matrix.json", matrix)
    matrix["status"] = ("completed_with_unsupported" if any(
        task["status"] == "completed_with_unsupported" for task in matrix["tasks"]) else "complete")
    save(output / "matrix.json", matrix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
