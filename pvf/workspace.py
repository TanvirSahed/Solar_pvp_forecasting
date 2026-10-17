"""Machine-level helpers called from the notebook (there is no command line).

    check_environment()          is this machine ready? (Python, GPU, packages, data identical to the manifest)
    smoke_test_models()          every model: build + forward + backward on a real batch
    fetch_pretrained_weights()   foundation-model weights -> Data/pretrained (offline afterwards)
    write_data_manifest()        (owner, after the data change) record data checksums, then commit the file
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from .provenance import check_data_against_manifest, data_files, write_data_manifest  # noqa: F401


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
    check(has_gpu, f"GPU: {torch.cuda.get_device_name(0) if has_gpu else 'none -> CPU only (slow); check NVIDIA driver'}")
    if has_gpu and not torch.cuda.is_bf16_supported():
        print("  NOTE  no bf16 on this GPU -> use TrainingRecipe(mixed_precision='fp16')")
    for package, required in (("mambapy", True), ("pvlib", True), ("chronos-forecasting", False),
                              ("timesfm", False), ("tirex-ts", False)):
        try:
            check(True, f"{package} {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            check(not required, f"{package} not installed" + ("" if required else " (optional)"))
    print("data (copy the Data/ folder into the project root)")
    problems = check_data_against_manifest()
    for problem in problems:
        check(False, problem)
    if not problems:
        check(True, f"{len(data_files())} data files identical to configs/data_manifest.json")
    from .models.architectures.foundation_models import PRETRAINED_WEIGHTS_DIR, WEIGHTS_REPOSITORIES

    present = [n for n, r in WEIGHTS_REPOSITORIES.items()
               if (PRETRAINED_WEIGHTS_DIR / r.replace("/", "--") / ".complete").exists()]
    print(f"  {'OK   ' if len(present) == len(WEIGHTS_REPOSITORIES) else 'NOTE '} pretrained weights present: "
          f"{present or 'none'} (optional: fetch_pretrained_weights())")
    ready = all(checks)
    print("\nREADY" if ready else "\nNOT READY - fix the FAIL lines above")
    return ready


def smoke_test_models(data_bundle, model_names=None, include_pretrained=False, batch_size=16) -> pd.DataFrame:
    """Build every model (or `model_names`), run forward + backward on one real batch. Never raises: failures are
    reported in the table so one broken model does not hide the others."""
    import torch

    from .data.pipeline import make_data_loader
    from .models.base import MODEL_REGISTRY, build_model, count_trainable_parameters
    from .training.losses import get_loss
    from .training.trainer import move_to_device, pick_device

    from .models import architectures  # noqa: F401  (registers every model)

    device = pick_device()
    batch = move_to_device(next(iter(make_data_loader(data_bundle.datasets["train"], batch_size, False))), device)
    if model_names is None:
        model_names = [n for n, c in sorted(MODEL_REGISTRY.items())
                       if include_pretrained or not getattr(c, "needs_weight_download", False)]
    rows = []
    for name in model_names:
        started = time.time()
        try:
            model = build_model(name, data_bundle.spec).to(device).train()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch)
                loss_name = "pinball" if model.quantile_levels else "mse"
                loss = get_loss(loss_name)(output["forecast"].float(), batch["target"], batch["target_mask"],
                                           model.quantile_levels)
            if loss.requires_grad:
                loss.backward()
            rows.append({"model": name, "status": "OK", "parameters": count_trainable_parameters(model),
                         "output_shape": tuple(output["forecast"].shape),
                         "milliseconds": round(1000 * (time.time() - started), 1), "error": ""})
        except Exception as error:  # noqa: BLE001 - report and continue with the next model
            rows.append({"model": name, "status": "FAIL", "parameters": None, "output_shape": None,
                         "milliseconds": None, "error": f"{type(error).__name__}: {error}"})
    return pd.DataFrame(rows)


def fetch_pretrained_weights(model_names=("chronos2", "timesfm", "tirex")) -> pd.DataFrame:
    from .models.architectures.foundation_models import WEIGHTS_REPOSITORIES, local_weights

    rows = []
    for name in model_names:
        folder = local_weights(WEIGHTS_REPOSITORIES[name])
        size_mb = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6
        rows.append({"model": name, "size_mb": round(size_mb), "folder": str(folder)})
    return pd.DataFrame(rows)
