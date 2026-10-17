"""Command line.

  python -m pvf list                                   # registered models
  python -m pvf data   site=nist                       # dataset / split summary
  python -m pvf smoke  [-m patchtst,graph_mamba]       # forward+backward every model on a real batch (GPU)
  python -m pvf train  -m patchtst site=nist model.d_model=64 train.epochs=50
  python -m pvf train  -m graph_mamba -c my_overrides.yaml tag=try1
  python -m pvf bench  -x configs/experiments/round1.yaml site=ssp       # many runs -> results/<site>_<exp>.csv
  python -m pvf doctor                                 # is this machine (GPU, packages, data) ready?
  python -m pvf rerun  runs/ssp/patchtst/20261018-101500                # repeat a run from its saved config
  python -m pvf data-manifest                          # (owner) record data checksums after the data changes

Config precedence: configs/base.yaml < configs/models/<model>.yaml < -c file(s) < experiment run entry < key=value
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from .data.sites import ROOT

CFG = ROOT / "configs"


def compose(model: str | None, cfg_files=(), dotlist=(), run_entry=None):
    cfg = OmegaConf.load(CFG / "base.yaml")
    name = model or (run_entry or {}).get("model", {}).get("name")
    if name and (CFG / "models" / f"{name}.yaml").exists():
        mcfg = OmegaConf.load(CFG / "models" / f"{name}.yaml")
        cfg.model = mcfg.model  # replace (not merge) the model block
        cfg = OmegaConf.merge(cfg, {k: v for k, v in mcfg.items() if k != "model"})
    elif name:
        cfg.model = {"name": name}
    for f in cfg_files:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(f))
    if run_entry:
        cfg = OmegaConf.merge(cfg, run_entry)
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(dotlist)))


def cmd_list(_):
    from .models import zoo  # noqa: F401
    from .models.base import MODELS

    for k, v in sorted(MODELS.items()):
        print(f"{k:18s} {v.__module__.split('.')[-1]:14s} {(v.__doc__ or '').strip().splitlines()[0][:80] if v.__doc__ else ''}")


def cmd_data(a):
    from .data.datamodule import build_data, summarize

    print(summarize(build_data(compose(None, a.config, a.overrides))))


def cmd_smoke(a):
    import torch

    from .data.datamodule import build_data
    from .models import zoo  # noqa: F401
    from .models.base import MODELS, build_model, count_params
    from .train.trainer import masked_loss, to_dev

    cfg = compose(None, a.config, a.overrides)
    data = build_data(cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = to_dev(next(iter(torch.utils.data.DataLoader(data["sets"]["train"], batch_size=16))), dev)
    batch["tgt"] = batch.copy()
    names = a.models.split(",") if a.models else sorted(n for n in MODELS if not getattr(MODELS[n], "external", False))
    ok = True
    for n in names:
        c = compose(n, (), a.overrides)
        try:
            m = build_model(OmegaConf.to_container(c.model, resolve=True), data["spec"]).to(dev).train()
            t0 = time.time()
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o = m(batch)
                loss = masked_loss(o["pred"].float(), batch["y"], batch["mask"], quantiles=m.quantiles) \
                    + o.get("aux_loss", 0.0)
            if loss.requires_grad:
                loss.backward()
            print(f"OK   {n:18s} params={count_params(m):>10,}  out={tuple(o['pred'].shape)}  {1e3 * (time.time() - t0):7.1f} ms")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"FAIL {n:18s} {type(e).__name__}: {e}")
    sys.exit(0 if ok else 1)


def cmd_train(a):
    from .train.trainer import run

    print("run dir:", run(compose(a.model, a.config, a.overrides)))


def cmd_doctor(a):
    """Is this machine ready to reproduce our experiments? Exit code 0 = yes."""
    import importlib.metadata as md

    import torch

    from .provenance import check_data, data_files

    ok = True

    def line(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(("  OK   " if good else "  FAIL ") + msg)

    print("environment")
    line(sys.version_info[:2] == (3, 11), f"python {sys.version.split()[0]} (expected 3.11)")
    line(True, f"torch {torch.__version__}  cuda-runtime {torch.version.cuda}")
    cuda = torch.cuda.is_available()
    line(cuda, f"GPU: {torch.cuda.get_device_name(0) if cuda else 'none -> CPU only (slow); check NVIDIA driver'}")
    if cuda:
        line(torch.cuda.is_bf16_supported(), "bf16 supported (else set train.amp=fp16)")
    for p in ("mambapy", "pvlib", "omegaconf", "chronos-forecasting", "timesfm", "tirex-ts"):
        try:
            line(True, f"{p} {md.version(p)}")
        except md.PackageNotFoundError:
            line(p not in ("mambapy", "pvlib", "omegaconf"), f"{p} not installed" + (" (optional)" if p in ("chronos-forecasting", "timesfm", "tirex-ts") else ""))
    print("data (copy the Data/ folder into the repo root)")
    probs = check_data()
    for p in probs:
        line(False, p)
    if not probs:
        line(True, f"{len(data_files())} files identical to configs/data_manifest.json")
    print("model zoo")
    try:
        from .models import zoo  # noqa: F401
        from .models.base import MODELS

        line(True, f"{len(MODELS)} models registered")
    except Exception as e:  # noqa: BLE001
        line(False, f"model import failed: {e}")
    print("\nREADY" if ok else "\nNOT READY - fix the FAIL lines above")
    sys.exit(0 if ok else 1)


def cmd_rerun(a):
    """Repeat a run exactly from its run directory (or its config.yaml)."""
    import json as _json

    from .train.trainer import run

    src = Path(a.run)
    cfg_path = src / "config.yaml" if src.is_dir() else src
    prov_path = cfg_path.parent / "provenance.json"
    if prov_path.exists():
        prov = _json.loads(prov_path.read_text())
        from .provenance import _git

        head = _git("rev-parse", "HEAD")
        if prov.get("git_commit") and head and prov["git_commit"] != head:
            print(f"note: run was made at commit {prov['git_commit'][:8]}, you are at {head[:8]} "
                  f"(exact code: git checkout {prov['git_commit'][:8]})")
        if prov.get("git_dirty"):
            print(f"note: run had uncommitted changes -> git apply {cfg_path.parent / 'git_diff.patch'}")
    cfg = OmegaConf.merge(OmegaConf.load(cfg_path), OmegaConf.from_dotlist(list(a.overrides)))
    print("run dir:", run(cfg))


def cmd_manifest(_):
    from .provenance import write_manifest

    for f, h in write_manifest().items():
        print(f"{h[:16] if h else 'MISSING':16s}  {f}")
    print("-> configs/data_manifest.json (commit it)")


def cmd_bench(a):
    import pandas as pd

    from .train.trainer import run

    exp = OmegaConf.load(a.experiment)
    rows = []
    for entry in OmegaConf.to_container(exp.runs):
        cfg = compose(None, a.config, a.overrides, run_entry=entry)
        try:
            out = run(cfg)
            m = json.loads((out / "metrics.json").read_text())
            for split in cfg.eval_splits:
                if split in m:
                    r = {k: v for k, v in m[split].items() if k != "per_horizon"}
                    rows.append({"model": cfg.model.name, "tag": cfg.tag, "split": split, "params": m["params"],
                                 "wall_sec": m["wall_sec"], **r, "run": str(out)})
        except Exception as e:  # noqa: BLE001
            print(f"!! run {entry} failed: {e}")
        df = pd.DataFrame(rows)
        dest = ROOT / "results" / f"{cfg.site}_{Path(a.experiment).stem}.csv"
        dest.parent.mkdir(exist_ok=True)
        df.to_csv(dest, index=False)
    if rows:
        print(df[df.split == "test"][["model", "tag", "params", "nMAE_%", "nRMSE_%", "skill_vs_SP_rmse"]].to_string(index=False))
        print("->", dest)


def main(argv=None):
    p = argparse.ArgumentParser(prog="pvf")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("list", cmd_list), ("data", cmd_data), ("smoke", cmd_smoke), ("train", cmd_train),
                     ("bench", cmd_bench), ("doctor", cmd_doctor), ("rerun", cmd_rerun), ("data-manifest", cmd_manifest)]:
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("-c", "--config", action="append", default=[], help="extra YAML override file(s)")
        s.add_argument("overrides", nargs="*", help="key=value OmegaConf dotlist overrides")
        if name == "train":
            s.add_argument("-m", "--model", required=True)
        if name == "smoke":
            s.add_argument("-m", "--models", default="")
        if name == "bench":
            s.add_argument("-x", "--experiment", required=True)
        if name == "rerun":
            s.add_argument("run", help="run directory (runs/<site>/<model>/<ts>) or its config.yaml")
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
