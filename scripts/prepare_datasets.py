#!/usr/bin/env python3
"""Prepare the private bundled datasets without network or model dependencies."""

import argparse
import csv
import gzip
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def checked_path(value):
    if not isinstance(value, str) or not value or "$" in value or "~" in value:
        raise ValueError("Use concrete paths, not literal $VAR or ~ placeholders")
    path = Path(os.path.abspath(value))
    for part in (*reversed(path.parents), path):
        if part.is_symlink():
            raise ValueError(f"Symlink paths are not supported: {part}")
    return path


def relative_path(root, value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid manifest relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError(f"Unsafe manifest relative path: {value!r}")
    return checked_path(str(root / path))


def reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def reserve_target(target, targets):
    if any(target == other or target in other.parents or other in target.parents
           for other in targets):
        raise ValueError(f"Duplicate or overlapping destination: {target}")
    targets.add(target)


def row_count(path, file_format):
    if file_format == "parquet":
        return None
    with path.open(encoding="utf-8", newline="") as stream:
        if file_format == "jsonl":
            count = 0
            for count, line in enumerate(stream, 1):
                try:
                    json.loads(line, parse_constant=reject_constant)
                except ValueError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{count}: {exc}") from exc
            return count
        reader = csv.reader(stream, strict=True)
        if next(reader, None) is None:
            raise ValueError(f"Missing CSV header: {path}")
        return sum(1 for _ in reader)


def prepare(dataset_dir, output_dir):
    if output_dir.exists():
        raise ValueError(f"Output directory already exists; choose a new path: {output_dir}")
    if (output_dir == dataset_dir or dataset_dir in output_dir.parents
            or output_dir in dataset_dir.parents):
        raise ValueError("Output must not overlap the input dataset directory")
    if output_dir == REPO_ROOT or REPO_ROOT in output_dir.parents:
        local_data = REPO_ROOT / "local_data"
        if output_dir != local_data and local_data not in output_dir.parents:
            raise ValueError("Inside the repository, output must be under ignored local_data/")
    manifest_path = relative_path(dataset_dir, "manifest.json")
    if not manifest_path.is_file():
        raise ValueError(
            f"Missing manifest: {manifest_path}. Clone the data-containing private "
            "repository version or use --dataset-dir. No download fallback is provided."
        )
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Expected manifest schema_version 1")
    entries = manifest.get("files")
    copies = manifest.get("copies", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest files must be a nonempty list")
    if not isinstance(copies, list):
        raise ValueError("Manifest copies must be a list")
    sources = set()
    targets = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each files entry must be an object")
        source = relative_path(dataset_dir, entry.get("path"))
        target = relative_path(output_dir, entry.get("output_path"))
        name = entry["path"]
        if source in sources:
            raise ValueError(f"Duplicate source: {source}")
        sources.add(source)
        reserve_target(target, targets)
        compression = entry.get("compression")
        file_format = entry.get("format")
        if compression not in ("gzip", "none"):
            raise ValueError(f"Unsupported compression for {name}: {compression}")
        if file_format not in ("jsonl", "csv", "parquet"):
            raise ValueError(f"Unsupported format for {name}: {file_format}")
        suffix = f".{file_format}" + (".gz" if compression == "gzip" else "")
        if not name.endswith(suffix) or target.suffix != f".{file_format}":
            raise ValueError(f"File extensions disagree with compression/format: {name}")
        for field in ("rows", "decisions", "bytes", "uncompressed_bytes"):
            if type(entry.get(field)) is not int or entry[field] < 0:
                raise ValueError(f"{name}: {field} must be a nonnegative integer")
        if not source.is_file() or source.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing file or incorrect byte length: {source}")
    for entry in copies:
        if not isinstance(entry, dict):
            raise ValueError("Each copies entry must be an object")
        source = relative_path(output_dir, entry.get("source"))
        target = relative_path(output_dir, entry.get("output_path"))
        if source not in targets:
            raise ValueError(f"Copy source must reference an earlier output: {source}")
        if source in sources:
            raise ValueError(f"Duplicate source: {source}")
        sources.add(source)
        reserve_target(target, targets)

    output_dir.mkdir(parents=True, exist_ok=False)
    prepared = {}
    for entry in entries:
        source = relative_path(dataset_dir, entry["path"])
        target = relative_path(output_dir, entry["output_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if entry["compression"] == "gzip" else open
        with opener(source, "rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if target.stat().st_size != entry["uncompressed_bytes"]:
            raise ValueError(f"Incorrect uncompressed byte length: {target}")
        count = row_count(target, entry["format"])
        if count is not None and count != entry["rows"]:
            raise ValueError(f"Row count mismatch for {target}: {count} != {entry['rows']}")
        prepared[entry["output_path"]] = entry
    for entry in copies:
        source = relative_path(output_dir, entry["source"])
        target = relative_path(output_dir, entry["output_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        prepared[entry["output_path"]] = prepared[entry["source"]]
    print(f"Datasets ready: {output_dir}")
    for name, entry in prepared.items():
        check = "declared, not verified" if entry["format"] == "parquet" else "verified"
        print(f"{output_dir / name}: {entry['rows']} rows ({check}); "
              f"{entry['decisions']} decisions (declared)")
    print("No network or models required for preparation. Parquet row/decision "
          "counts require separate verification; model runtime is not prepared.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="New destination; never overwritten")
    parser.add_argument("--dataset-dir", default=str(REPO_ROOT / "datasets"),
                        help="Asset directory containing manifest.json (default: repo/datasets)")
    args = parser.parse_args()
    try:
        prepare(checked_path(args.dataset_dir), checked_path(args.output_dir))
    except (OSError, ValueError, EOFError, csv.Error) as exc:
        print(f"Dataset preparation failed: {exc}\n"
              "Any partial output is left in place; inspect it and choose a fresh output directory.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
