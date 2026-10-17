"""One experiment = data config + model + training recipe -> a run folder with standard scores.

    config = ExperimentConfig(model_name="patch_tst", model_hyperparameters={"model_size": 64},
                              data=DataConfig(site="nist"), recipe=TrainingRecipe(scheduler="cosine"))
    result = run_experiment(config)
    result.summary()                         # standard metrics per split

Run folder runs/<site>/<run_name>/<timestamp>/:
    config.yaml                the full ExperimentConfig (what data, model and recipe the run used)
    best_model.pt              best weights (lowest validation loss)
    trained_model.pt           the whole trained model object -> load_trained_model(run_dir)
    metrics.json               training history + metrics per split (overall and per horizon step)
    predictions_<split>.npz    forecast, target, target_mask, smart_persistence, origin_index, origin_time
"""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .data.pipeline import DataBundle, DataConfig, build_data_bundle
from .data.sites import PROJECT_ROOT, get_site
from .data.splits import SplitConfig
from .evaluation.metrics import evaluate_predictions, per_horizon_metrics
from .models.base import build_model, count_trainable_parameters
from .training.trainer import TrainingRecipe, predict, train_model

RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_LOG = PROJECT_ROOT / "results" / "experiment_log.csv"


@dataclass
class ExperimentConfig:
    model_name: str = "lstm"
    model_hyperparameters: dict = field(default_factory=dict)
    residual_on_smart_persistence: bool = False      # forecast = smart persistence + model correction
    data: DataConfig = field(default_factory=DataConfig)
    recipe: TrainingRecipe = field(default_factory=TrainingRecipe)
    run_name: str = ""                               # label in comparisons; defaults to model_name
    evaluation_splits: list = field(default_factory=lambda: ["validation", "test", "holdout"])

    def to_dict(self) -> dict:
        return json.loads(json.dumps(dataclasses.asdict(self)))   # tuples -> lists so YAML can store them

    @classmethod
    def from_dict(cls, values: dict) -> "ExperimentConfig":
        values = dict(values)
        data = dict(values.pop("data", {}))
        data["split"] = SplitConfig(**data.get("split", {}))
        return cls(data=DataConfig(**data), recipe=TrainingRecipe(**values.pop("recipe", {})), **values)

    def save(self, path: Path):
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))


@dataclass
class ExperimentResult:
    run_dir: Path
    metrics: dict
    model: object = field(repr=False)
    data_bundle: DataBundle = field(repr=False)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({split: result["overall"] for split, result in self.metrics["splits"].items()}).T


def run_experiment(config: ExperimentConfig, data_bundle: DataBundle | None = None, model=None,
                   runs_dir: Path = RUNS_DIR, log=print) -> ExperimentResult:
    """Pass `data_bundle` to reuse already-built data; pass `model` to train a model object you built yourself
    (e.g. a new fusion in the notebook) - then config.model_name is only a label."""
    data_bundle = data_bundle or build_data_bundle(config.data)
    if model is None:
        model = build_model(config.model_name, data_bundle.spec, config.residual_on_smart_persistence,
                            **config.model_hyperparameters)
    else:
        log("using the model object passed in")
    run_name = config.run_name or config.model_name
    run_dir = Path(runs_dir) / config.data.site / run_name / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.yaml")
    n_parameters = count_trainable_parameters(model)
    log(f"{run_name}: {type(model).__name__}, {n_parameters:,} trainable parameters -> {run_dir}")

    started = time.time()
    history = train_model(model, data_bundle, config.recipe, run_dir / "best_model.pt", log)
    train_seconds = time.time() - started

    spec, site = data_bundle.spec, get_site(config.data.site)
    metrics = {"run_name": run_name, "model_name": config.model_name, "site": config.data.site,
               "timezone": site.timezone, "resolution": spec.resolution, "n_parameters": n_parameters,
               "quantile_levels": list(model.quantile_levels or []), "train_seconds": train_seconds,
               "history": history, "splits": {}}
    for split in config.evaluation_splits:
        if split not in data_bundle.datasets:
            continue
        predictions = predict(model, data_bundle.datasets[split], config.recipe)
        predictions["origin_time"] = spec.time_index[predictions["origin_index"]].as_unit("ns").asi8   # UTC nanoseconds
        np.savez_compressed(run_dir / f"predictions_{split}.npz", **predictions)
        result = evaluate_predictions(predictions, model.quantile_levels, spec.rated_power_kw,
                                      spec.ramp_threshold_fraction)
        result["per_horizon_smart_persistence"] = per_horizon_metrics(
            predictions["smart_persistence"], predictions["target"], predictions["target_mask"],
            predictions["smart_persistence"])
        metrics["splits"][split] = result
        overall = result["overall"]
        log(f"[{split:10s}] nMAE {overall['nmae_pct']:6.3f}%  nRMSE {overall['nrmse_pct']:6.3f}%  "
            f"skill vs persistence {overall['rmse_skill_vs_persistence']:+.3f}"
            + (f"  CRPS {overall['crps']:.4f}" if "crps" in overall else ""))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    if model.is_trainable:   # whole model object too: reloads even models built by hand in the notebook
        try:
            torch.save(model.cpu(), run_dir / "trained_model.pt")
        except Exception as error:  # noqa: BLE001 - e.g. weight-norm layers cannot be pickled; weights are saved
            (run_dir / "trained_model.pt").unlink(missing_ok=True)
            log(f"note: whole-model file skipped ({type(error).__name__}); load_trained_model rebuilds it from "
                f"config.yaml + best_model.pt")
    _append_to_experiment_log(metrics, run_dir)
    log(f"saved: {run_dir}")
    return ExperimentResult(run_dir, metrics, model, data_bundle)


def _append_to_experiment_log(metrics: dict, run_dir: Path):
    """One row per experiment in results/experiment_log.csv - a quick overview across all runs."""
    row = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "run_name": metrics["run_name"],
           "model": metrics["model_name"], "site": metrics["site"], "parameters": metrics["n_parameters"],
           "epochs": len(metrics["history"]), "train_seconds": round(metrics["train_seconds"], 1)}
    for split, result in metrics["splits"].items():
        for name in ("nmae_pct", "nrmse_pct", "rmse_skill_vs_persistence"):
            row[f"{split}_{name}"] = round(result["overall"][name], 4)
    row["run_dir"] = str(run_dir)
    RESULTS_LOG.parent.mkdir(exist_ok=True)
    log_table = pd.concat([pd.read_csv(RESULTS_LOG), pd.DataFrame([row])]) if RESULTS_LOG.exists() else pd.DataFrame([row])
    log_table.to_csv(RESULTS_LOG, index=False)


def load_trained_model(run_dir, data_bundle: DataBundle | None = None):
    """The trained model of a run. Uses trained_model.pt when present, else rebuilds from config.yaml + weights."""
    run_dir = Path(run_dir)
    if (run_dir / "trained_model.pt").exists():
        return torch.load(run_dir / "trained_model.pt", map_location="cpu", weights_only=False)
    config = ExperimentConfig.load(run_dir / "config.yaml")
    data_bundle = data_bundle or build_data_bundle(config.data)
    model = build_model(config.model_name, data_bundle.spec, config.residual_on_smart_persistence,
                        **config.model_hyperparameters)
    if (run_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    return model

