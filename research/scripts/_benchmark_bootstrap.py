"""Resolve packaged benchmark imports without requiring a custom PYTHONPATH."""

from importlib.util import find_spec
from pathlib import Path
import sys


def bootstrap():
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    # The checkpoint's demo-only __main__ dependency check needs an import marker.
    # Never substitute the marker for an available real dllm package.
    if "dllm" not in sys.modules and find_spec("dllm") is None:
        stub = repo / "support" / "dllm_stub"
        if (stub / "dllm" / "__init__.py").is_file():
            sys.path.append(str(stub))
