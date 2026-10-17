"""Reproducibility: data checksums + per-run provenance, so a run made on one machine can be repeated on another.

configs/data_manifest.json  - sha256 of every input data file (committed; the data itself is transferred by hand)
<run_dir>/provenance.json   - git commit, dirty flag, data hashes, torch/CUDA/GPU, package versions
<run_dir>/git_diff.patch    - uncommitted code changes at run time (so even dirty runs can be reproduced)
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

from .data.sites import ROOT, SITES

MANIFEST = ROOT / "configs" / "data_manifest.json"


def data_files() -> list[Path]:
    return sorted({p for s in SITES.values() for p, _, _ in s.files.values()})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_hashes() -> dict:
    return {p.relative_to(ROOT).as_posix(): (sha256(p) if p.exists() else None) for p in data_files()}


def write_manifest() -> dict:
    h = data_hashes()
    MANIFEST.write_text(json.dumps(h, indent=1) + "\n")
    return h


def check_data() -> list[str]:
    """Problems with local data vs the committed manifest ([] = identical)."""
    if not MANIFEST.exists():
        return ["configs/data_manifest.json missing (run: python -m pvf data-manifest)"]
    want, have = json.loads(MANIFEST.read_text()), data_hashes()
    probs = []
    for f, h in want.items():
        if have.get(f) is None:
            probs.append(f"MISSING  {f}")
        elif have[f] != h:
            probs.append(f"DIFFERS  {f}")
    return probs


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def record(run_dir: Path) -> dict:
    import importlib.metadata as md

    import torch

    commit, diff = _git("rev-parse", "HEAD"), _git("diff", "HEAD")
    if diff:
        (run_dir / "git_diff.patch").write_text(diff + "\n")
    pkgs = {}
    for p in ("torch", "numpy", "pandas", "pvlib", "chronos-forecasting", "timesfm", "tirex-ts", "mambapy"):
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    prov = {
        "git_commit": commit, "git_dirty": bool(diff),
        "data_sha256": data_hashes(), "data_matches_manifest": not check_data(),
        "python": platform.python_version(), "os": platform.platform(),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": pkgs,
    }
    (run_dir / "provenance.json").write_text(json.dumps(prov, indent=1))
    return prov
