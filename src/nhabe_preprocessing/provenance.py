"""Ghi dấu vết môi trường và cấu hình để tái lập pipeline."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, project_path
from .io_utils import atomic_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def record_provenance(config: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(config["_config_path"])
    value = {
        "schema_version": "nhabe-himawari-provenance-2025-08",
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "libraries": {
            name: version(name)
            for name in ("arm-pyart", "satpy", "pyresample", "numpy", "PyYAML", "aiohttp")
        },
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": _sha256(config_path),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_worktree_clean": not bool(_git_value("status", "--porcelain")),
        "expected_results": config["expected_results"],
        "raw_data_policy": "READ_ONLY",
    }
    destination = project_path(config["paths"]["audit_root"]) / "provenance.json"
    atomic_write_json(destination, value)
    return value
