"""Reproducibility: data checksums and per-run provenance, so a run made on one machine can be repeated on another.

configs/data_manifest.json  sha256 of every input data file (committed; the data itself is copied by hand)
<run_dir>/provenance.json   git commit, uncommitted-changes flag, data checksums, Python/torch/GPU, package versions
<run_dir>/git_diff.patch    uncommitted code changes at run time
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

from .data.sites import PROJECT_ROOT, SITES

DATA_MANIFEST = PROJECT_ROOT / "configs" / "data_manifest.json"


def data_files() -> list[Path]:
    return sorted({path for site in SITES.values() for path, _, _ in site.source_columns.values()})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_checksums() -> dict:
    return {p.relative_to(PROJECT_ROOT).as_posix(): (file_sha256(p) if p.exists() else None) for p in data_files()}


def write_data_manifest() -> dict:
    """Record the current data checksums (run after the data change, then commit configs/data_manifest.json)."""
    checksums = data_checksums()
    DATA_MANIFEST.write_text(json.dumps(checksums, indent=1) + "\n")
    return checksums


def check_data_against_manifest() -> list[str]:
    """Problems with the local data compared with the committed manifest ([] = identical)."""
    if not DATA_MANIFEST.exists():
        return ["configs/data_manifest.json missing - run write_data_manifest()"]
    expected, actual = json.loads(DATA_MANIFEST.read_text()), data_checksums()
    problems = []
    for file, checksum in expected.items():
        if actual.get(file) is None:
            problems.append(f"MISSING  {file}")
        elif actual[file] != checksum:
            problems.append(f"DIFFERS  {file}")
    return problems


def git_output(*arguments) -> str:
    try:
        return subprocess.run(["git", *arguments], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - no git or not a repository: provenance just records nothing
        return ""


def record_provenance(run_dir: Path) -> dict:
    import importlib.metadata as metadata

    import torch

    commit, uncommitted_diff = git_output("rev-parse", "HEAD"), git_output("diff", "HEAD")
    if uncommitted_diff:
        (run_dir / "git_diff.patch").write_text(uncommitted_diff + "\n")
    package_versions = {}
    for package in ("torch", "numpy", "pandas", "pvlib", "chronos-forecasting", "timesfm", "tirex-ts", "mambapy"):
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    provenance = {
        "git_commit": commit, "has_uncommitted_changes": bool(uncommitted_diff),
        "data_checksums": data_checksums(), "data_matches_manifest": not check_data_against_manifest(),
        "python": platform.python_version(), "operating_system": platform.platform(),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": package_versions,
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=1))
    return provenance
