"""Command-line interface for tests, mock runs, and ordered pipeline steps."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

from .config import PROJECT_ROOT, load_config
from .sample_pipeline import run_mock_pipeline


STEP_SCRIPTS = {
    "inventory": "01_inventory_radar.py",
    "match-time": "02_build_time_manifest.py",
    "download-himawari": "03_download_himawari.py",
    "preprocess-day": "04_preprocess_one_day.py",
    "preprocess-month": "05_preprocess_month.py",
    "build-index": "06_build_dataset_index.py",
    "audit": "07_audit_dataset.py",
    "provenance": "08_record_provenance.py",
}


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "MISSING"


def doctor(config_path: str, quiet: bool = False) -> int:
    """Check the runtime and the immutable scientific configuration."""
    config = load_config(config_path)
    report = {
        "status": "PASS",
        "python": sys.version.split()[0],
        "config": config["_config_path"],
        "grid_id": config["grid"]["grid_id"],
        "packages": {
            name: _package_version(name)
            for name in ("arm-pyart", "satpy", "pyresample", "numpy", "PyYAML")
        },
    }
    if any(value == "MISSING" for value in report["packages"].values()):
        report["status"] = "FAIL"
    if not quiet:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


def run_tests() -> int:
    """Run the repository test suite from an absolute project path."""
    command = [
        sys.executable, "-m", "unittest", "discover",
        "-s", str(PROJECT_ROOT / "tests"), "-v",
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def run_step(name: str, config_path: str, arguments: Sequence[str]) -> int:
    """Execute one existing numbered script without depending on the current directory."""
    script = PROJECT_ROOT / "scripts" / STEP_SCRIPTS[name]
    command = [sys.executable, str(script), *arguments, "--config", config_path]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nhabe-pipeline",
        description="Radar–Himawari preprocessing utilities and reproducible smoke workflow.",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "pipeline.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="check configuration and dependencies")
    doctor_parser.add_argument("--quiet", action="store_true")
    subparsers.add_parser("test", help="run all unit and integration tests")

    sample_parser = subparsers.add_parser(
        "sample", help="generate and validate one synthetic contract-level sample"
    )
    sample_parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "output" / "sample"
    )

    step_parser = subparsers.add_parser("step", help="run one numbered production step")
    step_parser.add_argument("name", choices=tuple(STEP_SCRIPTS))
    step_parser.add_argument(
        "step_args", nargs=argparse.REMAINDER,
        help="arguments passed to the numbered script; place them after the step name",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = str(Path(args.config).resolve())
    if args.command == "doctor":
        return doctor(config_path, args.quiet)
    if args.command == "test":
        return run_tests()
    if args.command == "sample":
        print(json.dumps(run_mock_pipeline(args.output, load_config(config_path)), indent=2))
        return 0
    if args.command == "step":
        return run_step(args.name, config_path, args.step_args)
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
