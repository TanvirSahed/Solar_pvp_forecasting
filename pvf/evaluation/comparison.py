"""Compare and diagnose trained runs (each run folder: metrics.json + predictions_<split>.npz).

    compare_runs(site="ssp", split="test")          table of standard metrics, best first
    plot_per_horizon(run_dirs, metric="nrmse_pct")   error vs steps ahead, one line per run
    plot_training_history(run_dir)                   train / validation loss per epoch
    plot_forecast_examples(run_dir)                  forecast vs actual vs smart persistence for a few origins
    plot_error_by_hour(run_dirs)                     where in the day a model fails
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.sites import PROJECT_ROOT

RUNS_DIR = PROJECT_ROOT / "runs"
# validated categorical palette (dataviz reference instance, light mode), assigned in fixed order
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REFERENCE_COLOR = "#8a8984"          # neutral gray for smart persistence (a reference, not a series)
TEXT_COLOR, MUTED_COLOR = "#0b0b0b", "#52514e"
MAX_SERIES = len(SERIES_COLORS)


def load_run(run_dir) -> dict:
    return json.loads((Path(run_dir) / "metrics.json").read_text())


def load_runs(site: str | None = None, runs_dir=RUNS_DIR) -> pd.DataFrame:
    """One row per (run, split) with every overall metric."""
    rows = []
    for metrics_file in sorted(Path(runs_dir).glob("*/*/*/metrics.json")):
        try:
            run = json.loads(metrics_file.read_text())
        except json.JSONDecodeError:          # e.g. a run interrupted while writing
            print(f"note: skipping unreadable {metrics_file}")
            continue
        if "splits" not in run:
            print(f"note: skipping {metrics_file.parent} (older run format)")
            continue
        if site and run["site"] != site:
            continue
        for split, result in run["splits"].items():
            rows.append({"run_name": run["run_name"], "model": run["model_name"], "site": run["site"], "split": split,
                         "parameters": run["n_parameters"], "epochs": len(run["history"]),
                         "train_seconds": run["train_seconds"], **result["overall"],
                         "run_dir": str(metrics_file.parent)})
    return pd.DataFrame(rows)


def compare_runs(site: str | None = None, split: str = "test", latest_only: bool = True, sort_by: str = "nrmse_pct",
                 columns=("nmae_pct", "nrmse_pct", "nbias_pct", "rmse_skill_vs_persistence", "ramp_mae_pct",
                          "p95_abs_error_pct", "r2", "crps", "picp90", "parameters", "train_seconds")) -> pd.DataFrame:
    runs = load_runs(site)
    if runs.empty:
        return runs
    runs = runs[runs.split == split]
    if latest_only:  # run folders are timestamped, so the last one per run name is the newest
        runs = runs.sort_values("run_dir").groupby("run_name", as_index=False).last()
    shown = [c for c in columns if c in runs.columns]
    return runs.sort_values(sort_by)[["run_name", "model", *shown, "run_dir"]].reset_index(drop=True)


def _style(axis, title, x_label, y_label):
    axis.set_title(title, fontsize=10, color=TEXT_COLOR, loc="left")
    axis.set_xlabel(x_label, fontsize=9, color=MUTED_COLOR)
    axis.set_ylabel(y_label, fontsize=9, color=MUTED_COLOR)
    axis.grid(True, color="#e6e5e1", linewidth=0.6)
    axis.tick_params(colors=MUTED_COLOR, labelsize=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#c9c8c3")


def _check_series_count(run_dirs):
    if len(run_dirs) > MAX_SERIES:
        raise ValueError(f"{len(run_dirs)} runs in one chart is unreadable; plot at most {MAX_SERIES} "
                         f"(e.g. compare_runs(...).run_dir[:{MAX_SERIES}])")


def plot_per_horizon(run_dirs, metric: str = "nrmse_pct", split: str = "test", show_persistence: bool = True):
    import matplotlib.pyplot as plt

    run_dirs = list(run_dirs)
    _check_series_count(run_dirs)
    figure, axis = plt.subplots(figsize=(8, 4))
    for color, run_dir in zip(SERIES_COLORS, run_dirs):
        run = load_run(run_dir)
        curve = run["splits"][split]["per_horizon"][metric]
        axis.plot(np.arange(1, len(curve) + 1), curve, color=color, linewidth=2, label=run["run_name"])
    if show_persistence and metric != "rmse_skill_vs_persistence":
        reference = load_run(run_dirs[0])["splits"][split].get("per_horizon_smart_persistence", {}).get(metric)
        if reference:
            axis.plot(np.arange(1, len(reference) + 1), reference, color=REFERENCE_COLOR, linewidth=1.5,
                      linestyle="--", label="smart persistence")
    _style(axis, f"{metric} by forecast step ({split})", "steps ahead", metric)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return figure


def plot_training_history(run_dir):
    import matplotlib.pyplot as plt

    history = pd.DataFrame(load_run(run_dir)["history"])
    figure, axis = plt.subplots(figsize=(7, 3.5))
    if history.empty:
        axis.text(0.5, 0.5, "not trained (evaluation-only model)", ha="center", va="center", color=MUTED_COLOR)
        axis.axis("off")
        return figure
    epoch_number = history.epoch + 1                       # 1-based, as in the report tables
    axis.plot(epoch_number, history.train_loss, color=SERIES_COLORS[0], linewidth=2, label="train")
    axis.plot(epoch_number, history.validation_loss, color=SERIES_COLORS[1], linewidth=2, label="validation")
    config_file = Path(run_dir) / "config.yaml"
    stop_metric = "validation_loss"
    if config_file.exists():
        import yaml
        stop_metric = yaml.safe_load(config_file.read_text())["recipe"].get("early_stopping_metric", stop_metric)
    best = history[stop_metric].idxmin()                    # the epoch whose weights were kept
    axis.scatter([epoch_number[best]], [history.validation_loss[best]], s=40, color=SERIES_COLORS[1], zorder=3,
                 edgecolor="white", linewidth=1.5)
    axis.annotate(f"best epoch {int(epoch_number[best])} (by {stop_metric})",
                  (epoch_number[best], history.validation_loss[best]), textcoords="offset points", xytext=(6, 8),
                  fontsize=8, color=MUTED_COLOR)
    from matplotlib.ticker import MaxNLocator
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style(axis, f"training history - {load_run(run_dir)['run_name']}", "epoch", "loss")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return figure


def plot_forecast_examples(run_dir, split: str = "test", n_examples: int = 3, seed: int = 0):
    """Full-horizon forecasts from a few random origins, against actual power and smart persistence."""
    import matplotlib.pyplot as plt

    run = load_run(run_dir)
    predictions = np.load(Path(run_dir) / f"predictions_{split}.npz")
    forecast = predictions["forecast"]
    if forecast.ndim == 3:   # quantile model: median line + 10-90 % band
        levels = run["quantile_levels"]
        median = forecast[..., levels.index(0.5)] if 0.5 in levels else forecast.mean(-1)
    else:
        median = forecast
    median = np.clip(median, 0.0, None)          # show what is scored: PV power cannot be negative
    chosen = np.random.default_rng(seed).choice(len(median), size=min(n_examples, len(median)), replace=False)
    figure, axes = plt.subplots(1, len(chosen), figsize=(4.2 * len(chosen), 3.2), squeeze=False)
    for axis, i in zip(axes[0], chosen):
        steps = np.arange(1, median.shape[1] + 1)
        axis.plot(steps, predictions["target"][i], color=TEXT_COLOR, linewidth=2, label="actual")
        axis.plot(steps, predictions["smart_persistence"][i], color=REFERENCE_COLOR, linewidth=1.5, linestyle="--",
                  label="smart persistence")
        axis.plot(steps, median[i], color=SERIES_COLORS[0], linewidth=2, label=run["run_name"])
        if forecast.ndim == 3 and 0.1 in levels and 0.9 in levels:
            axis.fill_between(steps, forecast[i, :, levels.index(0.1)], forecast[i, :, levels.index(0.9)],
                              color=SERIES_COLORS[0], alpha=0.15, linewidth=0, label="10-90 %")
        origin = pd.Timestamp(int(predictions["origin_time"][i]), tz="UTC").tz_convert(run["timezone"])
        _style(axis, f"from {origin:%Y-%m-%d %H:%M}", "steps ahead", "P / rated power")
    axes[0][0].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    return figure


def plot_error_by_hour(run_dirs, split: str = "test", metric: str = "nmae_pct"):
    """Mean absolute error grouped by the local hour of the forecast target."""
    import matplotlib.pyplot as plt

    run_dirs = list(run_dirs)
    _check_series_count(run_dirs)
    figure, axis = plt.subplots(figsize=(8, 3.8))
    for color, run_dir in zip(SERIES_COLORS, run_dirs):
        run = load_run(run_dir)
        predictions = np.load(Path(run_dir) / f"predictions_{split}.npz")
        forecast = predictions["forecast"]
        if forecast.ndim == 3:
            forecast = forecast[..., run["quantile_levels"].index(0.5)]
        step = pd.Timedelta(run["resolution"])
        origin = pd.to_datetime(predictions["origin_time"], utc=True).tz_convert(run["timezone"])
        target_hours = np.stack([(origin + (h + 1) * step).hour for h in range(forecast.shape[1])], 1)
        error = np.where(predictions["target_mask"] > 0, np.abs(np.clip(forecast, 0, None) - predictions["target"]),
                         np.nan)
        by_hour = pd.Series(error.ravel()).groupby(target_hours.ravel()).mean() * 100
        axis.plot(by_hour.index, by_hour.values, color=color, linewidth=2, marker="o", markersize=4,
                  label=run["run_name"])
    _style(axis, f"absolute error by hour of day ({split})", "local hour of target", "MAE, % of rated power")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return figure
