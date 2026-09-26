"""Private acquisition helpers for prepare_core; never import dataset converters here."""

import os
from http.client import HTTPException
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_path(root, source):
    relative = source["path"]
    if relative.startswith("core/") and (root / "jevbench").is_dir():
        relative = relative.removeprefix("core/")
    return root / relative


def required_files(root, source, manifest):
    destination = source_path(root, source)
    if source["kind"] != "github":
        return [destination]
    paths = [destination / name for name in source["required"]]
    if source["repo"] == "bespokelabsai/nimble":
        subsets = [s["name"].removeprefix("nimble_") for s in manifest["suites"]
                   if s["name"].startswith("nimble_")]
        paths += [destination / "docs/assets/public-benchmarks/subsets" / (name + "-manifest.json")
                  for name in subsets]
        modules = {name.split("-")[0] for name in subsets}
        paths += [destination / "nimble/datasets/public_sources" / (name + ".py")
                  for name in sorted(modules)]
    return paths


def missing_files(root, source, manifest):
    return [path for path in required_files(root, source, manifest)
            if not path.is_file() or path.stat().st_size == 0]


class SourceRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            original, target = urlsplit(req.full_url), urlsplit(newurl)
            if original.netloc != target.netloc or original.scheme != target.scheme:
                redirected.remove_header("Authorization")
        return redirected


def download_url(url, destination, headers=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = build_opener(SourceRedirectHandler())
    request_headers = {"User-Agent": "core-v1-preparation/1", **(headers or {})}
    for attempt in range(3):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".download-", delete=False) as out:
                temporary = Path(out.name)
                with opener.open(Request(url, headers=request_headers), timeout=120) as response:
                    expected = response.headers.get("Content-Length")
                    shutil.copyfileobj(response, out, length=1024 * 1024)
                require(out.tell() > 0, "Empty source download")
                if expected is not None:
                    require(out.tell() == int(expected), "Incomplete source download")
            temporary.replace(destination)
            return
        except HTTPError as error:
            if error.code in (401, 403):
                raise RuntimeError("Source access denied (HTTP 401/403); check upstream terms/access. "
                                   "For gated HF sources authorize access and set HF_TOKEN.") from None
            if attempt == 2:
                raise RuntimeError(f"Source download failed (HTTP {error.code}): {destination.name}") from None
        except (URLError, HTTPException, OSError, ValueError):
            if attempt == 2:
                raise RuntimeError(f"Source download failed after three attempts: {destination.name}") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        time.sleep(2 ** attempt)


def download_github(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), f"Incomplete repository cache; move it aside before retrying: {destination}")
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".github-") as temporary:
        scratch = Path(temporary)
        archive_path = scratch / "source.tar.gz"
        download_url(f"https://codeload.github.com/{source['repo']}/tar.gz/{source['revision']}", archive_path)
        tree = scratch / "tree"
        tree.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                require(not path.is_absolute() and ".." not in path.parts, "Unsafe GitHub archive path")
                if len(path.parts) < 2:
                    continue
                relative = PurePosixPath(*path.parts[1:]).as_posix()
                selected = any(relative.startswith(prefix) if prefix.endswith("/") else relative == prefix
                               for prefix in source["include"])
                if not selected or member.isdir():
                    continue
                require(member.isfile(), "Links and special files are not accepted in source archives")
                target = tree / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                src = archive.extractfile(member)
                if src is None:
                    raise ValueError("Unreadable selected archive member")
                with src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
        require(all((tree / path).is_file() for path in source["required"]),
                "Pinned GitHub archive is missing required files")
        tree.rename(destination)


def download_hf(source, destination, cache):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        url = (f"https://huggingface.co/datasets/{quote(source['repo'], safe='/')}/resolve/"
               f"{quote(source['revision'], safe='')}/{quote(source['file'], safe='/')}")
        token = os.environ.get("HF_TOKEN")
        download_url(url, destination, headers={"Authorization": f"Bearer {token}"} if token else None)
        return
    for attempt in range(3):
        try:
            downloaded = Path(hf_hub_download(
                repo_id=source["repo"], repo_type="dataset", filename=source["file"],
                revision=source["revision"], cache_dir=str(cache / "hf"),
                token=os.environ.get("HF_TOKEN") or None,
            ))
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".download-", delete=False) as out:
                temporary = Path(out.name)
                try:
                    with downloaded.open("rb") as src:
                        shutil.copyfileobj(src, out, length=1024 * 1024)
                    require(out.tell() > 0, "Empty HF source")
                    out.close()
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
            return
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status in (401, 403):
                raise RuntimeError(f"HF access denied (HTTP {status}) for {source['repo']}; "
                                   "accept upstream access terms if required and set HF_TOKEN. "
                                   "No source is skipped.") from None
            if attempt == 2:
                raise RuntimeError(f"HF download failed for {source['repo']} at {source['revision']}; "
                                   "check connectivity, access and pinned file availability. No source is skipped.") from None
        time.sleep(2 ** attempt)


def acquire(root, cache, manifest, offline=False, explicit_raw=False):
    missing = [(source, missing_files(root, source, manifest)) for source in manifest["sources"]]
    missing = [(source, paths) for source, paths in missing if paths]
    if missing and (offline or explicit_raw):
        names = "\n".join(str(path) for _, paths in missing for path in paths)
        raise ValueError("Incomplete raw source tree; no network requested for --offline/--raw-root. "
                         "Acquire the pinned sources with --download-only first. Missing files:\n" + names)
    for source, _ in missing:
        destination = source_path(root, source)
        print(f"Acquiring {source['path']}", flush=True)
        if source["kind"] == "github":
            download_github(source, destination)
        elif source["kind"] == "hf":
            download_hf(source, destination, cache)
        else:
            download_url(source["url"], destination)
        require(not missing_files(root, source, manifest), f"Incomplete acquisition: {source['path']}")
    return {
        "core": root if (root / "jevbench").is_dir() else root / "core",
        "nimble": root / "nimble",
        "documents": root / "document_forecast",
    }
