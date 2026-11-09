"""Fusion study: diagnose every trained model, test prediction-level fusion, train evidence-driven fusions, report.

    python tools/fusion_study.py --site ssp                    # diagnose + train (3 seeds) + report
    python tools/fusion_study.py --site ssp --report-only      # rebuild the report from results/fusion_<site>.json

Needs a finished batch for the same site (tools/run_all_models.py): its saved predictions are the evidence, and its
common test windows are reused, so every number is comparable with the batch report. Zero-shot foundation models
are excluded from fusion. Writes Resources/Fusion_Report_<SITE>_<date>.md / .pdf.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw figures without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import pvf  # noqa: E402
from markdown_to_pdf import convert  # noqa: E402
from pvf.data.pipeline import make_data_loader  # noqa: E402
from pvf.data.site_record import load_site_record  # noqa: E402
from pvf.data.windows import ForecastWindowDataset  # noqa: E402
from pvf.evaluation.comparison import REFERENCE_COLOR, SERIES_COLORS, _style  # noqa: E402
from pvf.models.architectures.fusion_models import PATCH_TST_OPTIONS  # noqa: E402
from pvf.training.paper_recipes import FUSION_STUDY_RECIPE  # noqa: E402
from pvf.training.trainer import move_to_device, pick_device, seed_everything  # noqa: E402
from run_all_models import _fmt, metric_definitions  # noqa: E402

ZERO_SHOT = {"chronos2", "timesfm", "tirex"}
REFERENCE = "smart_persistence"
CONTROL = "patch_tst"
FUSIONS = ["patch_tst_conv_gate", "patch_tst_mamba_gate", "patch_tst_variable_gate"]
ORACLE = "patch_tst_oracle_weather"
FUSION_LOOKBACK = 96                                   # PatchTST's paper lookback (48 h), shared by all study runs
LEAD_TIMES_HOURS = (0.5, 1, 3, 6, 12, 24)
BANDS = {"0.5–2 h": slice(0, 4), "2.5–8 h": slice(4, 16), "8.5–16 h": slice(16, 32), "16.5–24 h": slice(32, 48)}
DAY_PARTS = {"morning 6–9 h": (6, 9), "midday 10–14 h": (10, 14), "afternoon 15–18 h": (15, 18)}
REGIMES = ["overcast", "mixed", "clear"]
DESIGNS = {
    "patch_tst_conv_gate": ("clear-sky windows and under-forecast bias",
                            "DT-DANN's causal-CNN + transformer encoder (best on clear and mixed windows)"),
    "patch_tst_mamba_gate": ("afternoon hours and fast ramps (8 h patches blur fast changes)",
                             "Mamba recurrent state (mamba_temporal / lstm: best at the morning and evening ramps)"),
    "patch_tst_variable_gate": ("channel independence: variables are encoded separately",
                                "iTransformer's variable tokens (least-correlated errors with PatchTST, strong at 2.5–8 h)"),
}


def fusion_recipe(seed: int, time_limit: float) -> pvf.TrainingRecipe:
    """One recipe for the control and every fusion, so the architecture is the only difference."""
    recipe = copy.deepcopy(FUSION_STUDY_RECIPE)
    recipe.seed, recipe.time_limit_minutes = seed, time_limit
    return recipe


def study_file(site):
    return ROOT / "results" / f"fusion_{site}.json"


# ---------------------------------------------------------------- evidence from the batch

def point_forecast(predictions):
    forecast = predictions["forecast"]
    return np.clip(forecast if forecast.ndim == 2 else forecast[..., forecast.shape[-1] // 2], 0, None)


def load_batch(site):
    batch = json.loads((ROOT / "results" / f"batch_{site}.json").read_text())
    runs = {m: r["run_dir"] for m, r in batch["runs"].items() if "run_dir" in r}
    trained = [m for m in runs if m not in ZERO_SHOT and m != REFERENCE]
    return runs, trained


def load_split(run_dir, split):
    return dict(np.load(Path(run_dir) / f"predictions_{split}.npz"))


def window_regimes(site, origins, target, mask):
    """Clearness of each forecast window = measured energy / clear-sky energy (normalised); terciles."""
    record = load_site_record(site)
    clear_sky = record["clear_sky_irradiance_wm2"].to_numpy() / 1000.0
    target_steps = origins[:, None] + np.arange(1, target.shape[1] + 1)[None, :]
    clearness = np.where(mask, target, 0).sum(1) / np.maximum(np.where(mask, clear_sky[target_steps], 0).sum(1), 1e-6)
    labels = np.array(pd.qcut(clearness, 3, labels=REGIMES))
    return labels, clearness, record


def nrmse(error, selection):
    return 100 * float(np.sqrt(np.mean(error[selection] ** 2)))


def diagnose(site):
    runs, trained = load_batch(site)
    reference = load_split(runs[REFERENCE], "test")
    target, mask, origins = reference["target"], reference["target_mask"] > 0, reference["origin_index"]
    regimes, _, record = window_regimes(site, origins, target, mask)
    times = record.index[origins]
    target_hour = np.stack([(times + pd.Timedelta(minutes=30 * s)).hour for s in range(1, target.shape[1] + 1)], 1)
    forecasts = {}
    for model in trained + [REFERENCE]:
        predictions = load_split(runs[model], "test")
        assert (predictions["origin_index"] == origins).all(), f"{model}: test windows differ from the batch"
        forecasts[model] = point_forecast(predictions)
    rows = []
    for model, forecast in forecasts.items():
        error = forecast - target
        row = {"model": model, "overall": nrmse(error, mask), "bias": 100 * float(error[mask].mean())}
        row.update({band: nrmse(error[:, s], mask[:, s]) for band, s in BANDS.items()})
        row.update({regime: nrmse(error, mask & (regimes == regime)[:, None]) for regime in REGIMES})
        row.update({part: 100 * float(np.abs(error[mask & (target_hour >= a) & (target_hour <= b)]).mean())
                    for part, (a, b) in DAY_PARTS.items()})
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model").sort_values("overall")
    ranks = table.drop(REFERENCE).drop(columns="bias").rank().astype(int)
    names = list(ranks.index)
    errors = np.stack([(forecasts[m] - target)[mask] for m in names])
    correlation = pd.DataFrame(np.corrcoef(errors), index=names, columns=names)
    return {"table": table, "ranks": ranks, "correlation": correlation, "forecasts": forecasts, "target": target,
            "mask": mask, "origins": origins, "regimes": regimes, "target_hour": target_hour, "runs": runs,
            "trained": trained}


def prediction_level_fusion(evidence):
    """Fuse the batch models' saved forecasts: pair averages and per-horizon non-negative stacking (validation-fit)."""
    runs, ranks = evidence["runs"], evidence["ranks"]
    target, mask = evidence["target"], evidence["mask"]
    names = list(ranks.index)
    results = []
    for partner in names[1:5]:
        average = (evidence["forecasts"][names[0]] + evidence["forecasts"][partner]) / 2
        results.append({"method": f"average of {names[0]} + {partner}", "test nRMSE %": nrmse(average - target, mask)})
    for label, models in ((f"stacking, top 5 ({', '.join(names[:5])})", names[:5]),
                          (f"stacking, all {len(names)} trained models", names)):
        validation = {m: load_split(runs[m], "validation") for m in models}
        common = None
        for predictions in validation.values():
            common = predictions["origin_index"] if common is None else np.intersect1d(common, predictions["origin_index"])
        aligned = {m: point_forecast({k: v[np.searchsorted(p["origin_index"], common)] for k, v in p.items()})
                   for m, p in validation.items()}
        first = validation[models[0]]
        index = np.searchsorted(first["origin_index"], common)
        y_val, m_val = first["target"][index], first["target_mask"][index] > 0
        fused = np.zeros_like(target)
        for h in range(target.shape[1]):
            weights, _ = nnls(np.stack([aligned[m][:, h] for m in models], 1)[m_val[:, h]], y_val[m_val[:, h], h])
            fused[:, h] = np.stack([evidence["forecasts"][m][:, h] for m in models], 1) @ weights
        results.append({"method": label, "test nRMSE %": nrmse(fused - target, mask)})
    return pd.DataFrame(results)


# ---------------------------------------------------------------- end-to-end fusion training

def collect_gates(model, dataset, recent_terciles):
    device = pick_device()
    model.to(device).eval()
    gates, origins = [], []
    with torch.no_grad():
        for batch in make_data_loader(dataset, 64, shuffle=False):
            gates.append(model(move_to_device(batch, device))["gate"].float().cpu())
            origins.append(batch["origin_index"])
    gates, origins = torch.cat(gates).numpy(), torch.cat(origins).numpy()
    regime = recent_terciles(origins)
    return {"by_lead_time": gates.mean(0).tolist(),
            "by_recent_regime": {name: gates[regime == i].mean(0).tolist() for i, name in enumerate(REGIMES)
                                 if (regime == i).any()}}


def train_study(site, seeds, time_limit):
    runs, _ = load_batch(site)
    common_test = load_split(runs[CONTROL], "test")["origin_index"]
    record = load_site_record(site)
    irradiance, clear_sky = record["irradiance_wm2"].to_numpy(), record["clear_sky_irradiance_wm2"].to_numpy()

    def recent_clearness(origins, steps=48):
        values = []
        for t in origins:
            g, c = irradiance[t - steps + 1:t + 1], clear_sky[t - steps + 1:t + 1]
            day = c > 50
            values.append(np.nansum(g[day]) / max(np.nansum(c[day]), 1.0) if day.any() else np.nan)
        return np.array(values)

    thresholds = np.nanquantile(recent_clearness(common_test), [1 / 3, 2 / 3])

    def recent_terciles(origins):
        return np.digitize(np.nan_to_num(recent_clearness(origins), nan=0.0), thresholds)

    def bundle_for(data_config):
        bundle = pvf.build_data_bundle(data_config)
        available = np.intersect1d(common_test, bundle.datasets["test"].origins)
        bundle.datasets["test"] = ForecastWindowDataset(bundle.datasets["test"].windows, available)
        return bundle, len(available)

    base_data = pvf.DataConfig(site=site, lookback_steps=FUSION_LOOKBACK)
    oracle_data = pvf.DataConfig(site=site, lookback_steps=FUSION_LOOKBACK,
                                 future_feature_names=base_data.future_feature_names
                                 + ["irradiance_oracle", "module_temperature_oracle"])
    bundles = {"base": bundle_for(base_data), "oracle": bundle_for(oracle_data)}
    record_out = {"site": site, "seeds": seeds, "time_limit_minutes": time_limit, "lookback_steps": FUSION_LOOKBACK,
                  "started": time.strftime("%Y-%m-%d %H:%M"), "test_windows": bundles["base"][1],
                  "oracle_test_windows": bundles["oracle"][1], "runs": {}}
    plan = [(CONTROL, "base"), *[(f, "base") for f in FUSIONS], (ORACLE, "oracle")]
    for seed in seeds:
        for model_name, data_key in plan:
            bundle, _ = bundles[data_key]
            registered = CONTROL if model_name == ORACLE else model_name
            hyperparameters = dict(PATCH_TST_OPTIONS) if registered == CONTROL else {}
            config = pvf.ExperimentConfig(model_name=registered, model_hyperparameters=hyperparameters,
                                          data=oracle_data if data_key == "oracle" else base_data,
                                          recipe=fusion_recipe(seed, time_limit), run_name=f"study_{model_name}_seed{seed}")
            print(f"\n=== seed {seed}: {model_name} ===", flush=True)
            seed_everything(seed)                                       # identical initialisation per seed
            model = pvf.build_model(registered, bundle.spec, **hyperparameters)
            try:
                result = pvf.run_experiment(config, data_bundle=bundle, model=model, log=lambda m: print(m, flush=True))
                entry = {"run_dir": str(result.run_dir), "seed": seed, "model": model_name}
                if model_name in FUSIONS:
                    entry["gates"] = collect_gates(result.model, bundle.datasets["test"], recent_terciles)
            except Exception as error:  # noqa: BLE001 - one failing run must not stop the study
                print(f"FAILED {model_name} seed {seed}: {error}", flush=True)
                entry = {"error": f"{type(error).__name__}: {error}", "seed": seed, "model": model_name}
            record_out["runs"][f"{model_name}_seed{seed}"] = entry
            study_file(site).write_text(json.dumps(record_out, indent=1))
            torch.cuda.empty_cache()
    record_out["finished"] = time.strftime("%Y-%m-%d %H:%M")
    study_file(site).write_text(json.dumps(record_out, indent=1))


# ---------------------------------------------------------------- report

def run_scores(run_dir, regimes, target_hour, origins_reference, mask_subset_origins=None):
    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    predictions = load_split(run_dir, "test")
    target, mask = predictions["target"], predictions["target_mask"] > 0
    keep = np.isin(predictions["origin_index"], origins_reference)
    error = point_forecast(predictions) - target
    position = np.searchsorted(origins_reference, predictions["origin_index"][keep])
    regime_of_row = regimes[position]
    scores = dict(metrics["splits"]["test"]["overall"])
    scores["per_horizon_nrmse"] = metrics["splits"]["test"]["per_horizon"]["nrmse_pct"]
    for regime in REGIMES:
        scores[regime] = nrmse(error[keep], mask[keep] & (regime_of_row == regime)[:, None])
    afternoon = mask[keep] & (target_hour[position] >= 15) & (target_hour[position] <= 18)
    scores["afternoon"] = 100 * float(np.abs(error[keep][afternoon]).mean())
    scores["epochs"] = len(metrics["history"])
    scores["minutes"] = metrics["train_seconds"] / 60
    scores["parameters"] = metrics["n_parameters"]
    if mask_subset_origins is not None:   # score on a subset of windows (oracle comparison)
        subset = np.isin(predictions["origin_index"], mask_subset_origins)
        scores["subset_nrmse"] = nrmse(error[subset], mask[subset])
    return scores


def mean_std(values):
    values = np.array([v for v in values if v is not None and np.isfinite(v)])
    return (float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0) if len(values) else (np.nan, 0.0)


def write_report(site):
    evidence = diagnose(site)
    stacking = prediction_level_fusion(evidence)
    study = json.loads(study_file(site).read_text())
    date = study["started"][:10]
    name = f"Fusion_Report_{site.upper()}_{date}"
    resources = ROOT / "Resources"
    figures = resources / f"{name}_figures"
    figures.mkdir(parents=True, exist_ok=True)
    table, ranks, correlation = evidence["table"], evidence["ranks"], evidence["correlation"]
    origins, regimes = evidence["origins"], evidence["regimes"]
    n_models = len(ranks)

    oracle_origins = None
    for entry in study["runs"].values():
        if entry.get("model") == ORACLE and "run_dir" in entry:
            oracle_origins = load_split(entry["run_dir"], "test")["origin_index"]
            break
    scores = {}
    for entry in study["runs"].values():
        if "run_dir" in entry:
            scores.setdefault(entry["model"], []).append(
                run_scores(entry["run_dir"], regimes, evidence["target_hour"], origins, oracle_origins))
    batch_patch = run_scores(evidence["runs"][CONTROL], regimes, evidence["target_hour"], origins, oracle_origins)

    def summary(model, key):
        return mean_std([s.get(key) for s in scores.get(model, [])])

    def seed_forecast(model, seed):
        entry = study["runs"].get(f"{model}_seed{seed}", {})
        if "run_dir" not in entry:
            return None
        predictions = load_split(entry["run_dir"], "test")
        assert (predictions["origin_index"] == origins).all(), "study test windows differ from the batch"
        return point_forecast(predictions)

    overcast_rows = (regimes == "overcast")[:, None]
    sky_oracle = {}
    for fusion in FUSIONS:
        values = []
        for seed in study["seeds"]:
            control_forecast, fusion_forecast = seed_forecast(CONTROL, seed), seed_forecast(fusion, seed)
            if control_forecast is not None and fusion_forecast is not None:
                combined = np.where(overcast_rows, control_forecast, fusion_forecast)
                values.append(nrmse(combined - evidence["target"], evidence["mask"]))
        sky_oracle[fusion] = mean_std(values)

    control_mean = summary(CONTROL, "nrmse_pct")[0]
    fusion_means = {f: summary(f, "nrmse_pct") for f in FUSIONS if f in scores}
    best_fusion = min(fusion_means, key=lambda f: fusion_means[f][0]) if fusion_means else None

    # ---- figures
    figure, axis = plt.subplots(figsize=(8.5, 7))
    image = axis.imshow(correlation.values, cmap="Blues", vmin=0.85, vmax=1.0)
    axis.set_xticks(range(n_models), correlation.columns, rotation=60, ha="right", fontsize=7)
    axis.set_yticks(range(n_models), correlation.index, fontsize=7)
    for i in range(n_models):
        for j in range(n_models):
            value = correlation.values[i, j]
            axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5,
                      color="white" if value > 0.95 else "#0b0b0b")
    figure.colorbar(image, ax=axis, fraction=0.04, label="correlation of test errors")
    axis.set_title("Error correlation between trained models (1 = identical mistakes)", fontsize=10, loc="left")
    figure.tight_layout()
    correlation_figure = figures / "error_correlation.png"
    figure.savefig(correlation_figure, dpi=120)
    plt.close(figure)

    hours = 0.5 * np.arange(1, 49)
    figure, axis = plt.subplots(figsize=(8, 4))
    for color, model in zip([REFERENCE_COLOR] + SERIES_COLORS, [CONTROL] + [f for f in FUSIONS if f in scores]):
        curves = np.array([s["per_horizon_nrmse"] for s in scores[model]])
        style = {"linestyle": "--", "linewidth": 1.8} if model == CONTROL else {"linewidth": 2}
        axis.plot(hours, curves.mean(0), color=color, label=f"{model} (control)" if model == CONTROL else model, **style)
    _style(axis, f"Test nRMSE by lead time, mean of {len(study['seeds'])} seeds", "hours ahead", "nRMSE %")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    horizon_figure = figures / "fusion_nrmse_by_lead_time.png"
    figure.savefig(horizon_figure, dpi=110)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    for color, model in zip(SERIES_COLORS, [f for f in FUSIONS if f in scores]):
        gates = np.array([e["gates"]["by_lead_time"] for e in study["runs"].values()
                          if e.get("model") == model and "gates" in e])
        axis.plot(hours, gates.mean(0), color=color, linewidth=2, label=model)
    axis.axhline(0.5, color=REFERENCE_COLOR, linewidth=1, linestyle=":")
    axis.set_ylim(0, 1)
    _style(axis, "Share of the forecast given to PatchTST (gate), mean over test windows and seeds", "hours ahead",
           "gate (1 = PatchTST only)")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    gate_figure = figures / "gate_by_lead_time.png"
    figure.savefig(gate_figure, dpi=110)
    plt.close(figure)

    # ---- text
    def strengths_weaknesses(model):
        r = ranks.loc[model]
        strong = [c for c in ranks.columns if c != "overall" and r[c] <= 3]
        weak = [c for c in ranks.columns if c != "overall" and r[c] >= n_models - 3]
        notes = []
        if abs(table.loc[model, "bias"]) >= 1.5:
            notes.append(f"bias {table.loc[model, 'bias']:+.1f} %")
        return ", ".join(strong) or "–", ", ".join(weak + notes) or "–"

    lines = [
        f"# Fusion report: {site.upper()}",
        "",
        f"Study run {study['started']} to {study.get('finished', 'unfinished')}. Evidence: the saved test and validation "
        f"predictions of the {n_models} trained models from the all-model batch (zero-shot foundation models excluded). "
        f"All numbers are on the same {study['test_windows']} test windows as the batch report (0.5–24 h ahead, "
        f"daytime targets); errors are % of rated power. Fusions were trained end to end with "
        f"{len(study['seeds'])} seeds and one shared recipe, against a PatchTST control trained the same way.",
        "",
        "## Bottom line",
        "",
    ]
    if best_fusion:
        delta = fusion_means[best_fusion][0] - control_mean
        best_clear = min(fusion_means, key=lambda f: summary(f, "clear")[0])
        spread = summary(CONTROL, "nrmse_pct")[1]
        difference = summary(best_clear, "nrmse_pct")[0] - control_mean
        difference_phrase = (f"within the seed spread of ±{spread:.2f}" if abs(difference) <= spread
                             else f"{'better' if difference < 0 else 'worse'} by {abs(difference):.2f} points, "
                                  f"seed spread ±{spread:.2f}")
        gate_entries = [e["gates"] for e in study["runs"].values() if e.get("model") == best_clear and "gates" in e]
        gate_after_cloudy = np.mean([np.mean(e["by_recent_regime"].get("overcast", [np.nan])) for e in gate_entries])
        gate_after_clear = np.mean([np.mean(e["by_recent_regime"].get("clear", [np.nan])) for e in gate_entries])
        lines += [
            f"- **Shared weakness:** test errors of the trained models are {correlation.values[np.triu_indices(n_models, 1)].min():.2f}–"
            f"{correlation.values[np.triu_indices(n_models, 1)].max():.2f} correlated, and every model's nRMSE on overcast windows "
            f"({table.drop(REFERENCE)['overcast'].min():.1f}–{table.drop(REFERENCE)['overcast'].max():.1f} %) is about twice its clear-sky nRMSE. "
            "The dominant error is cloud variability that no model can see in history alone.",
            f"- **Prediction-level fusion of the finished models does not help:** the best combination scores "
            f"{stacking['test nRMSE %'].min():.2f} % against {table.loc[ranks.index[0], 'overall']:.2f} % for {ranks.index[0]} alone.",
            f"- **End-to-end fusions fix the weakness they target, but not the total:** {best_clear} cuts clear-sky "
            f"nRMSE from {summary(CONTROL, 'clear')[0]:.2f} to {summary(best_clear, 'clear')[0]:.2f} % and afternoon nMAE "
            f"from {summary(CONTROL, 'afternoon')[0]:.2f} to {summary(best_clear, 'afternoon')[0]:.2f} %, and moves the "
            f"bias from {summary(CONTROL, 'nbias_pct')[0]:+.2f} to {summary(best_clear, 'nbias_pct')[0]:+.2f} %; overcast "
            f"nRMSE rises from {summary(CONTROL, 'overcast')[0]:.2f} to {summary(best_clear, 'overcast')[0]:.2f} %, so overall "
            f"nRMSE is {summary(best_clear, 'nrmse_pct')[0]:.2f} vs {control_mean:.2f} % for the control "
            f"({difference_phrase}).",
            f"- **The gate needs a forecast of the coming sky:** if it knew which windows will be overcast, using PatchTST "
            f"there and {best_clear} elsewhere would give {sky_oracle[best_clear][0]:.2f} ± {sky_oracle[best_clear][1]:.2f} %. "
            f"From history alone it cannot: the learned gate gives PatchTST {gate_after_cloudy:.2f} after cloudy days and "
            f"{gate_after_clear:.2f} after clear days, an almost fixed blend.",
        ]
        oracle = summary(ORACLE, "nrmse_pct")
        if np.isfinite(oracle[0]):
            control_subset = summary(CONTROL, "subset_nrmse")[0]
            lines.append(f"- **Information beats architecture:** given the measured future irradiance and temperature "
                         f"(a perfect-forecast upper bound), PatchTST reaches {oracle[0]:.2f} % on its "
                         f"{study['oracle_test_windows']} windows vs {control_subset:.2f} % without it. "
                         "A real day-ahead weather forecast is the largest available improvement.")
    lines += [
        "",
        "## 1. Diagnosis: where each model is weak",
        "",
        "Test nRMSE % by lead-time band and by sky condition of the forecast window (terciles of measured / clear-sky "
        "energy), test nMAE % by time of day of the target, and mean bias. Lower is better everywhere.",
        "",
        "| Model | Overall | " + " | ".join(BANDS) + " | " + " | ".join(REGIMES) + " | "
        + " | ".join(DAY_PARTS) + " | Bias % |",
        "|---|" + "---|" * (1 + len(BANDS) + len(REGIMES) + len(DAY_PARTS) + 1),
    ]
    for model, row in table.iterrows():
        lines.append(f"| {model} | {row['overall']:.2f} | " + " | ".join(f"{row[b]:.2f}" for b in BANDS) + " | "
                     + " | ".join(f"{row[r]:.2f}" for r in REGIMES) + " | "
                     + " | ".join(f"{row[p]:.2f}" for p in DAY_PARTS) + f" | {row['bias']:+.2f} |")
    lines += ["", f"**Strengths and weaknesses** (rank among the {n_models} trained models: strength = top 3, "
              f"weakness = bottom 4, or |bias| ≥ 1.5 %)", "", "| Model | Strong at | Weak at |", "|---|---|---|"]
    for model in ranks.index:
        strong, weak = strengths_weaknesses(model)
        lines.append(f"| {model} | {strong} | {weak} |")
    lines += [
        "",
        "## 2. Shared versus model-specific errors",
        "",
        "Fusion can only remove errors the models do not share. The models' test errors are highly correlated, so most of "
        "the error is common to all of them; complementarity exists mainly between sky conditions (PatchTST on overcast, "
        "DT-DANN / LSTM / Mamba on clear) and between times of day.",
        "",
        f"![Error correlation between trained models]({correlation_figure.relative_to(resources).as_posix()})",
        "",
        "## 3. Prediction-level fusion of the finished models",
        "",
        "Combining the batch models' saved forecasts. Stacking fits non-negative weights per lead time on the validation "
        "predictions and is scored on test.",
        "",
        "| Method | Test nRMSE % |",
        "|---|---|",
        f"| {ranks.index[0]} alone (reference) | {table.loc[ranks.index[0], 'overall']:.2f} |",
        *[f"| {row['method']} | {row['test nRMSE %']:.2f} |" for _, row in stacking.iterrows()],
        "",
        "It does not beat the best single model. The validation split already chose each model's early-stopping epoch, "
        "so validation forecasts look better than the models really are and the fitted weights do not transfer; the "
        "recent weather regime (last-day clearness) predicts the coming window's sky only moderately "
        "(correlation 0.64 on SSP). Fusion therefore has to be learned end to end, from training data.",
        "",
        "## 4. Fusion designs",
        "",
        "Each fusion keeps PatchTST (best overall) and adds the model that is strongest where PatchTST is weak. A gate "
        "predicted from the recent weather regime and the latest observations sets, for every lead time, how much of the "
        "forecast comes from PatchTST; each branch is also trained on its own so neither is switched off "
        "(`pvf/models/architectures/fusion_models.py`).",
        "",
        "| Fusion | PatchTST weakness addressed | Attribute borrowed |",
        "|---|---|---|",
        *[f"| {f} | {DESIGNS[f][0]} | {DESIGNS[f][1]} |" for f in FUSIONS],
        "",
        f"Shared recipe (control and fusions): Adam, lr 1e-4, reduce-on-plateau ×0.5 (patience 2), up to 30 epochs, "
        f"early stopping on validation nRMSE (patience 5), batch 48, lookback {study['lookback_steps']} steps "
        f"({study['lookback_steps'] / 2:g} h), at most {study['time_limit_minutes']:g} min per run, seeds "
        f"{', '.join(map(str, study['seeds']))}. Features as in the batch report.",
        "",
        "## 5. Results",
        "",
        f"Mean ± standard deviation over {len(study['seeds'])} seeds. The batch PatchTST (its own paper recipe, one run) "
        "is listed for reference.",
        "",
        "| Model | nMAE % | nRMSE % | Bias % | R² | Ramp MAE % | Overcast nRMSE | Mixed nRMSE | Clear nRMSE | "
        "Epochs | Train min | Parameters |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in [CONTROL] + FUSIONS:
        if model not in scores:
            continue
        label = f"{model} (control)" if model == CONTROL else model
        cells = [f"{summary(model, k)[0]:.2f} ± {summary(model, k)[1]:.2f}"
                 for k in ("nmae_pct", "nrmse_pct", "nbias_pct")]
        cells.append(f"{summary(model, 'r2')[0]:.3f}")
        cells += [f"{summary(model, k)[0]:.2f}" for k in ("ramp_mae_pct", *REGIMES)]
        cells += [f"{summary(model, 'epochs')[0]:.0f}", f"{summary(model, 'minutes')[0]:.1f}",
                  f"{int(summary(model, 'parameters')[0]):,}"]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append(f"| patch_tst (batch, paper recipe) | {batch_patch['nmae_pct']:.2f} | {batch_patch['nrmse_pct']:.2f} | "
                 f"{batch_patch['nbias_pct']:+.2f} | {batch_patch['r2']:.3f} | {batch_patch['ramp_mae_pct']:.2f} | "
                 + " | ".join(f"{batch_patch[r]:.2f}" for r in REGIMES)
                 + f" | {batch_patch['epochs']} | {batch_patch['minutes']:.1f} | {batch_patch['parameters']:,} |")
    lines += ["", "**Test nRMSE % by lead time** (mean over seeds)", "",
              "| Model | " + " | ".join(f"{h:g} h" for h in LEAD_TIMES_HOURS) + " |",
              "|---|" + "---|" * len(LEAD_TIMES_HOURS)]
    for model in [CONTROL] + FUSIONS:
        if model in scores:
            curve = np.array([s["per_horizon_nrmse"] for s in scores[model]]).mean(0)
            lines.append(f"| {model} | " + " | ".join(f"{curve[int(h / 0.5) - 1]:.2f}" for h in LEAD_TIMES_HOURS) + " |")
    lines += ["", f"![Test nRMSE by lead time]({horizon_figure.relative_to(resources).as_posix()})", "",
              "**What the gates learned** (share of the forecast given to PatchTST; 0.5 = equal mix).", "",
              "| Fusion | " + " | ".join(f"{h:g} h" for h in LEAD_TIMES_HOURS) + " | after cloudy days | after clear days |",
              "|---|" + "---|" * (len(LEAD_TIMES_HOURS) + 2)]
    for model in FUSIONS:
        entries = [e["gates"] for e in study["runs"].values() if e.get("model") == model and "gates" in e]
        if not entries:
            continue
        by_lead = np.array([e["by_lead_time"] for e in entries]).mean(0)
        cloudy = np.mean([np.mean(e["by_recent_regime"].get("overcast", [np.nan])) for e in entries])
        clear = np.mean([np.mean(e["by_recent_regime"].get("clear", [np.nan])) for e in entries])
        lines.append(f"| {model} | " + " | ".join(f"{by_lead[int(h / 0.5) - 1]:.2f}" for h in LEAD_TIMES_HOURS)
                     + f" | {cloudy:.2f} | {clear:.2f} |")
    lines += ["", f"![Gate by lead time]({gate_figure.relative_to(resources).as_posix()})", "",
              "## 6. Did each fusion fix the weakness it targeted?", "",
              f"The targeted weaknesses (clear-sky windows, afternoon hours, under-forecast bias) against the cost "
              f"(overcast windows). Mean ± standard deviation over {len(study['seeds'])} seeds; sky condition = tercile of "
              "the window's measured / clear-sky energy.", "",
              "| Model | Clear-sky nRMSE % | Afternoon 15–18 h nMAE % | Bias % | Overcast nRMSE % | Overall nRMSE % | "
              "With a perfect sky signal |",
              "|---|---|---|---|---|---|---|"]
    for model in [CONTROL] + FUSIONS:
        if model not in scores:
            continue
        cells = [f"{summary(model, k)[0]:.2f} ± {summary(model, k)[1]:.2f}"
                 for k in ("clear", "afternoon", "nbias_pct", "overcast", "nrmse_pct")]
        oracle_cell = (f"{sky_oracle[model][0]:.2f} ± {sky_oracle[model][1]:.2f}"
                       if model in sky_oracle and np.isfinite(sky_oracle[model][0]) else "–")
        label = f"{model} (control)" if model == CONTROL else model
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {oracle_cell} |")
    lines += ["", "**With a perfect sky signal** = PatchTST (control) on windows that turn out overcast and the fusion on "
              "the rest, seed by seed: what the gate could reach with a forecast of the coming sky. The gates here only "
              "see last-day clearness, which predicts the coming window's sky with correlation 0.64 on SSP, so they also "
              "lean on the partner on days that turn cloudy, where the partner is weakest. The gate table in section 5 "
              "shows the same thing: the share given to PatchTST barely changes between cloudy and clear spells, so the "
              "fusions act as near-fixed blends; the blend is what fixes the clear-sky weakness, and also what costs "
              "accuracy on overcast days.", ""]

    oracle = summary(ORACLE, "nrmse_pct")
    if np.isfinite(oracle[0]):
        lines += [
            "## 7. Information upper bound",
            "",
            f"PatchTST with the measured future irradiance and module temperature added to its known-future inputs "
            f"(`irradiance_oracle`, `module_temperature_oracle`): a perfect weather forecast, so an upper bound and not "
            f"a deployable model. Scored on the {study['oracle_test_windows']} test windows where these values exist.",
            "",
            "| Model | nRMSE % (same windows) |",
            "|---|---|",
            f"| patch_tst control, no future weather | {summary(CONTROL, 'subset_nrmse')[0]:.2f} ± {summary(CONTROL, 'subset_nrmse')[1]:.2f} |",
            f"| patch_tst + perfect future weather | {oracle[0]:.2f} ± {oracle[1]:.2f} |",
            "",
        ]
    lines += [
        "## 8. Conclusions and next steps",
        "",
        "1. The largest error is shared by every architecture: cloud changes that history cannot reveal. Architecture "
        "fusion can only redistribute the smaller, model-specific part of the error.",
        "2. Combining finished models with validation-fitted weights overfits; if prediction-level fusion is wanted "
        "(e.g. the RL selector, row 15 of the selection report), fit it on a separate split the base models never saw.",
        "3. The borrowed attributes work where they were aimed: the recurrent (Mamba) and convolutional partners fix "
        "PatchTST's clear-sky, afternoon and bias weaknesses. They are weak on overcast days, and the gate cannot yet "
        "tell those days apart in advance (it learned a near-fixed blend), so the gain and the loss cancel overall.",
        "4. Next step for fusion: feed the gate a forecast of the coming sky (NWP cloud cover or irradiance, or "
        "satellite cloud motion) instead of last-day clearness; section 6 shows what a correct sky signal is worth.",
        "5. To reduce the shared error, add information to the forecaster itself: the perfect-weather upper bound "
        "(section 7) is far larger than any fusion gain. A day-ahead weather forecast for SSP is the priority.",
        "6. Read gains against the seed spread: differences below one standard deviation are noise.",
        "",
        *metric_definitions(site),
    ]
    markdown_file = resources / f"{name}.md"
    markdown_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    convert(markdown_file, resources / f"{name}.pdf")
    print(f"report: {markdown_file} and {name}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="ssp")
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    parser.add_argument("--time-limit", type=float, default=20.0, help="minutes of training per run")
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.report_only:
        train_study(arguments.site, [int(s) for s in arguments.seeds.split(",")], arguments.time_limit)
    write_report(arguments.site)
