"""One generic training loop for every model. Hooks used by models:
    out["aux_loss"]     added to the loss (DANN, regularisers)
    out["deep_preds"]   list of (pred, weight): extra supervised heads (fusion deep supervision)
    model.on_epoch_start(epoch, max_epochs)   schedules (DANN lambda, curriculum ...)
    batch["tgt"]        unlabelled target-domain batch when data.adapt_months is set
"""
from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from ..data.datamodule import build_data, make_loader, summarize
from ..data.sites import ROOT, get_site
from ..eval.metrics import per_horizon, point_metrics, prob_metrics
from ..models.base import build_model, count_params


def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def masked_loss(pred, y, mask, kind="mse", quantiles=None):
    if quantiles is not None:
        q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
        d = y.unsqueeze(-1) - pred
        l = torch.maximum(q * d, (q - 1) * d).mean(-1)
    elif kind == "mse":
        l = (pred - y) ** 2
    elif kind == "mae":
        l = (pred - y).abs()
    elif kind == "huber":
        l = torch.nn.functional.huber_loss(pred, y, reduction="none", delta=0.05)
    else:
        raise ValueError(kind)
    return (l * mask).sum() / mask.sum().clamp_min(1.0)


def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, dev, amp_dtype):
    model.eval()
    keys = ("pred", "y", "mask", "sp", "t0")
    acc = {k: [] for k in keys}
    for b in loader:
        b = to_dev(b, dev)
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(b)
        acc["pred"].append(out["pred"].float().cpu())
        for k in keys[1:]:
            acc[k].append(b[k].cpu())
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


def evaluate(res: dict, model, site) -> dict:
    pred = res["pred"]
    m = {}
    if model.quantiles:
        qs = list(model.quantiles)
        m.update(prob_metrics(pred, res["y"], res["mask"], qs))
        pred = pred[..., qs.index(0.5)] if 0.5 in qs else pred.mean(-1)
    pred = np.clip(pred, 0.0, None)
    m.update(point_metrics(pred, res["y"], res["mask"], res["sp"], site.rated_kw, site.ramp_theta))
    m["per_horizon"] = per_horizon(pred, res["y"], res["mask"], res["sp"])
    return m


def run(cfg) -> Path:
    t_start = time.time()
    seed_all(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.train.amp) if dev.type == "cuda" else None
    data = build_data(cfg)
    spec, sets = data["spec"], data["sets"]
    site = get_site(cfg.site)
    print(summarize(data))
    model = build_model(OmegaConf.to_container(cfg.model, resolve=True), spec).to(dev)
    name = cfg.model.name + (f"_{cfg.tag}" if cfg.get("tag") else "")
    out = ROOT / cfg.out_dir / cfg.site / name / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml")
    n_par = count_params(model)
    print(f"model={cfg.model.name} params={n_par:,} device={dev} amp={cfg.train.amp} -> {out}")

    T = cfg.train
    bs = T.batch_size
    history = []
    if model.trainable and n_par > 0:
        tl = make_loader(sets["train"], bs, True, T.num_workers, cfg.seed)
        vl = make_loader(sets["val"], 4 * bs, False, T.num_workers)
        adapt = itertools.cycle(make_loader(sets["adapt"], bs, True, T.num_workers, cfg.seed)) if "adapt" in sets else None
        opt = torch.optim.AdamW(model.parameters(), lr=T.lr, weight_decay=T.weight_decay) if T.optimizer == "adamw" \
            else torch.optim.Adam(model.parameters(), lr=T.lr, weight_decay=T.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T.epochs) if T.scheduler == "cosine" else None
        scaler = torch.amp.GradScaler(enabled=amp == torch.float16)
        q = list(model.quantiles) if model.quantiles else None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(str(out / "tb"))
        except Exception:  # noqa: BLE001
            tb = None
        best, bad = float("inf"), 0
        for ep in range(T.epochs):
            if hasattr(model, "on_epoch_start"):
                model.on_epoch_start(ep, T.epochs)
            model.train()
            tr_loss, nb, t0 = 0.0, 0, time.time()
            for b in tl:
                b = to_dev(b, dev)
                if adapt is not None:
                    b["tgt"] = to_dev(next(adapt), dev)
                with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                    o = model(b)
                    loss = masked_loss(o["pred"].float(), b["y"], b["mask"], T.loss, q)
                    for p, w in o.get("deep_preds", []):
                        loss = loss + w * masked_loss(p.float(), b["y"], b["mask"], T.loss)
                    if "aux_loss" in o:
                        loss = loss + o["aux_loss"].float()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if T.grad_clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), T.grad_clip)
                scaler.step(opt)
                scaler.update()
                tr_loss += loss.item()
                nb += 1
            if sched:
                sched.step()
            vr = predict(model, vl, dev, amp)
            v = masked_loss(torch.from_numpy(vr["pred"]), torch.from_numpy(vr["y"]), torch.from_numpy(vr["mask"]),
                            T.loss, q).item()
            history.append({"epoch": ep, "train": tr_loss / max(nb, 1), "val": v, "sec": time.time() - t0})
            if tb:
                tb.add_scalars("loss", {"train": tr_loss / max(nb, 1), "val": v}, ep)
            print(f"ep {ep:3d}  train {tr_loss / max(nb, 1):.5f}  val {v:.5f}  ({time.time() - t0:.1f}s)")
            if v < best - 1e-6:
                best, bad = v, 0
                torch.save(model.state_dict(), out / "best.pt")
            else:
                bad += 1
                if bad >= T.patience:
                    print(f"early stop at epoch {ep}")
                    break
        model.load_state_dict(torch.load(out / "best.pt", map_location=dev))

    results = {"model": cfg.model.name, "site": cfg.site, "params": n_par, "history": history}
    for split in cfg.eval_splits:
        if split not in sets:
            continue
        r = predict(model, make_loader(sets[split], 4 * bs, False, T.num_workers), dev, amp)
        np.savez_compressed(out / f"pred_{split}.npz", **r)
        results[split] = evaluate(r, model, site)
        mm = results[split]
        print(f"[{split}] nMAE {mm['nMAE_%']:.3f}%  nRMSE {mm['nRMSE_%']:.3f}%  skill(SP) {mm['skill_vs_SP_rmse']:.3f}"
              + (f"  CRPS {mm['CRPS']:.4f}" if "CRPS" in mm else ""))
    results["wall_sec"] = time.time() - t_start
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    return out
