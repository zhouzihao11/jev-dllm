"""Acquire pinned native Kev source and model files only; no ML imports or installs."""

import argparse
import json
import os
from pathlib import Path
import shutil
from urllib.parse import quote

from core_sources import download_github, download_url, require


SOURCE = dict(repo="jaredpalmer/kev", revision="3e1cd3bb588a388a06827443380befece23e68c7",
              include=["kev/__init__.py", "kev/api.py", "kev/checkpoint.py", "kev/model.py",
                       "LICENSE", "NOTICE", "README.md"],
              required=["kev/__init__.py", "kev/api.py", "kev/checkpoint.py", "kev/model.py",
                        "LICENSE", "README.md"])
BASE = dict(repo="Qwen/Qwen3.5-0.8B-Base", revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
            files=["config.json", "model.safetensors.index.json", "tokenizer.json",
                   "tokenizer_config.json", "vocab.json", "merges.txt", "preprocessor_config.json",
                   "video_preprocessor_config.json", "LICENSE", "README.md"])
ADAPTER = dict(repo="jaredpalmer/kev-0.8b", revision="9a45d25eb2ab761841196625383fa1dff0e56c1e",
               files=["adapter_config.json", "adapter_model.safetensors", "head.pt", "tokenizer.json",
                      "tokenizer_config.json", "provenance.json", "training_config.json", "README.md"])


def acquire_file(spec, filename, cache, destination):
    cached = cache / "models" / spec["repo"] / spec["revision"] / filename
    if not cached.is_file() or cached.stat().st_size == 0:
        print(f"Acquiring {spec['repo']} / {filename}", flush=True)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            token = os.environ.get("HF_TOKEN")
            url = (f"https://huggingface.co/{quote(spec['repo'], safe='/')}/resolve/"
                   f"{spec['revision']}/{quote(filename, safe='/')}")
            download_url(url, cached, headers={"Authorization": f"Bearer {token}"} if token else None)
        else:
            try:
                downloaded = hf_hub_download(
                    repo_id=spec["repo"], repo_type="model", filename=filename,
                    revision=spec["revision"], cache_dir=str(cache / "hf"),
                    token=os.environ.get("HF_TOKEN") or None,
                )
            except Exception:
                raise RuntimeError(f"HF acquisition failed for {spec['repo']} / {filename}; "
                                   "check connectivity, pinned file availability and HF_TOKEN access") from None
            cached.parent.mkdir(parents=True, exist_ok=True)
            temporary = cached.with_name(cached.name + ".partial")
            try:
                shutil.copyfile(downloaded, temporary)
                require(temporary.stat().st_size > 0, "Empty model file")
                temporary.replace(cached)
            finally:
                temporary.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty resource directory outside the checkout")
    parser.add_argument("--cache-dir", type=Path, default=Path("~/.cache/jev-dllm/kev"),
                        help="Reusable download cache outside the checkout")
    args = parser.parse_args()
    output, cache = args.output_dir.expanduser().resolve(), args.cache_dir.expanduser().resolve()
    project = Path(__file__).resolve().parents[1]
    require(not output.is_relative_to(project) and not cache.is_relative_to(project),
            "Models and acquisition cache must stay outside the checkout")
    require(output != cache and not cache.is_relative_to(output), "Cache must be outside output directory")
    require(not output.exists() or output.is_dir() and not any(output.iterdir()),
            "Output must be new or empty; move incomplete output aside before retrying")
    output.mkdir(parents=True, exist_ok=True)
    source_cache = cache / "source" / SOURCE["revision"]
    if not source_cache.exists():
        download_github(SOURCE, source_cache)
    require(all((source_cache / name).is_file() for name in SOURCE["required"]),
            "Incomplete source cache; move it aside before retrying")
    shutil.copytree(source_cache, output / "source")
    for spec, relative in ((BASE, "models/base"), (ADAPTER, "models/kev-0.8b")):
        for filename in spec["files"]:
            acquire_file(spec, filename, cache, output / relative / filename)
    index = json.loads((output / "models/base/model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    require(shards and all(isinstance(name, str) and Path(name).name == name
                          and "/" not in name and "\\" not in name and name.endswith(".safetensors")
                          for name in shards), "Unsafe or empty base weight index")
    for filename in shards:
        acquire_file(BASE, filename, cache, output / "models/base" / filename)
    manifest = dict(
        schema_version="kev_paths_v1",
        paths=dict(kev_source="source", base_path="models/base", model_path="models/kev-0.8b"),
        sources=dict(code=dict(repo=SOURCE["repo"], revision=SOURCE["revision"], files=SOURCE["required"]),
                     base=dict(repo=BASE["repo"], revision=BASE["revision"], files=BASE["files"] + shards),
                     adapter=ADAPTER),
    )
    (output / "kev_paths.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {output}; paths in kev_paths.json are relative to this directory", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
