"""Machine-level helpers called from the notebook or setup scripts.

    check_environment()          is this machine ready? (Python, GPU, packages, data files present)
"""
from __future__ import annotations

import sys

from .data.sites import SITES


def check_environment() -> bool:
    import importlib.metadata as metadata

    import torch

    checks = []

    def check(passed, what):
        checks.append(bool(passed))
        print(("  OK    " if passed else "  FAIL  ") + what)

    print("environment")
    check(sys.version_info[:2] == (3, 11), f"python {sys.version.split()[0]} (expected 3.11)")
    has_gpu = torch.cuda.is_available()
    check(True, f"torch {torch.__version__}, CUDA runtime {torch.version.cuda}")
    check(True, f"device: {torch.cuda.get_device_name(0) if has_gpu else 'CPU (training will be slower)'}")
    if has_gpu and not torch.cuda.is_bf16_supported():
        print("  NOTE  no bf16 on this GPU -> bf16 requests automatically use fp16")
    for package, required in (("mambapy", True), ("pvlib", True), ("chronos-forecasting", False),
                              ("timesfm", False), ("tirex-ts", False)):
        try:
            check(True, f"{package} {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            check(not required, f"{package} not installed" + ("" if required else " (optional)"))
    print("data (copy the Data/ folder into the project root)")
    data_files = sorted({path for site in SITES.values() for path, _, _ in site.source_columns.values()})
    missing = [path for path in data_files if not path.exists()]
    for path in missing:
        check(False, f"missing {path}")
    if not missing:
        check(True, f"{len(data_files)} data files present")
    from .models.architectures.foundation_models import PRETRAINED_WEIGHTS_DIR, WEIGHTS_REPOSITORIES

    present = [n for n, r in WEIGHTS_REPOSITORIES.items()
               if (PRETRAINED_WEIGHTS_DIR / r.replace("/", "--") / ".complete").exists()]
    print(f"  {'OK   ' if len(present) == len(WEIGHTS_REPOSITORIES) else 'NOTE '} pretrained weights present: "
          f"{present or 'none'} (otherwise downloaded on first use)")
    ready = all(checks)
    print("\nREADY" if ready else "\nNOT READY - fix the FAIL lines above")
    return ready

