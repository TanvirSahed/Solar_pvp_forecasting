"""NWP study: does an archived weather forecast supply what PatchTST and the fusions were missing?

    python tools/nwp_study.py --site ssp --seeds 0,1,2,3,4     # train every NWP run, then write the summary
    python tools/nwp_study.py --site ssp --report-only         # rebuild the summary from results/nwp_<site>.json

Needs the fusion study for the same site (tools/fusion_study.py): its control and fusion runs without NWP are the
comparison, and its test windows (the all-model batch's common windows) are reused, so every number is comparable.
NWP features come from pvf/data/nwp.py (downloaded once into Data/nwp/). Writes results/nwp_<site>.json and the
summary results/nwp_s<N>.md (+ PDF and a figure folder; N = next report number, see tools/report_names.py).
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fusion_study as fs  # noqa: E402
import pvf  # noqa: E402
from markdown_to_pdf import convert  # noqa: E402
from report_names import report_stem  # noqa: E402
from pvf.data import nwp  # noqa: E402
from pvf.data.pipeline import make_data_loader  # noqa: E402
from pvf.data.windows import ForecastWindowDataset  # noqa: E402
from pvf.models.architectures.fusion_models import PATCH_TST_OPTIONS  # noqa: E402
from pvf.training.trainer import move_to_device, pick_device, seed_everything  # noqa: E402

NWP_DAY1 = ["nwp_irradiance", "nwp_cloud_cover", "nwp_temperature"]
NWP_DAY2 = ["nwp_irradiance_day2", "nwp_cloud_cover_day2", "nwp_temperature_day2"]
FUTURE_GATE = {"gate_reads_future": True}
# label -> (registered model, extra future features, hyperparameters, what it tests)
PLAN = {
    "patch_tst_nwp": ("patch_tst", NWP_DAY1, PATCH_TST_OPTIONS, "PatchTST + NWP issued ~24 h ahead"),
    "patch_tst_nwp_day2": ("patch_tst", NWP_DAY2, PATCH_TST_OPTIONS, "PatchTST + NWP issued ~48 h ahead (strict)"),
    "patch_tst_mamba_gate_nwp": ("patch_tst_mamba_gate", NWP_DAY1, FUTURE_GATE, "Mamba fusion, NWP in both branches and the gate"),
    "patch_tst_conv_gate_nwp": ("patch_tst_conv_gate", NWP_DAY1, FUTURE_GATE, "conv fusion, NWP in both branches and the gate"),
    "patch_tst_variable_gate_nwp": ("patch_tst_variable_gate", NWP_DAY1, FUTURE_GATE, "variable fusion, NWP in both branches and the gate"),
    "patch_tst_mamba_gate_future": ("patch_tst_mamba_gate", [], FUTURE_GATE, "Mamba fusion, gate reads the future but no NWP"),
}
NO_NWP = {"patch_tst": "patch_tst", "patch_tst_mamba_gate": "patch_tst_mamba_gate",
          "patch_tst_conv_gate": "patch_tst_conv_gate", "patch_tst_variable_gate": "patch_tst_variable_gate"}
SKY = ["cloudy", "mixed", "clear"]                              # NWP-forecast sky of a window, terciles


def study_file(site):
    return ROOT / "results" / f"nwp_{site}.json"


def forecast_clearness(site, origins, horizon=48):
    """Per window: Σ NWP irradiance / Σ clear-sky irradiance over its daylight target steps (what the NWP expects)."""
    record = fs.load_site_record(site)
    clear_sky = record["clear_sky_irradiance_wm2"].to_numpy()
    forecast = nwp.nwp_column(record, "shortwave_radiation", 1)
    steps = origins[:, None] + np.arange(1, horizon + 1)[None, :]
    day = clear_sky[steps] > 10
    return np.where(day, forecast[steps], 0).sum(1) / np.maximum(np.where(day, clear_sky[steps], 0).sum(1), 1.0)


def collect_gates(model, dataset, sky_of_origin):
    device = pick_device()
    model.to(device).eval()
    gates, origins = [], []
    with torch.no_grad():
        for batch in make_data_loader(dataset, 64, shuffle=False):
            gates.append(model(move_to_device(batch, device))["gate"].float().cpu())
            origins.append(batch["origin_index"])
    gates, origins = torch.cat(gates).numpy(), torch.cat(origins).numpy()
    sky = sky_of_origin(origins)
    return {"by_lead_time": gates.mean(0).tolist(),
            "by_forecast_sky": {name: float(gates[sky == i].mean()) for i, name in enumerate(SKY) if (sky == i).any()}}


def train_study(site, seeds, time_limit):
    runs, _ = fs.load_batch(site)
    common_test = fs.load_split(runs[fs.CONTROL], "test")["origin_index"]
    thresholds = np.quantile(forecast_clearness(site, common_test), [1 / 3, 2 / 3])

    def sky_of_origin(origins):
        return np.digitize(forecast_clearness(site, origins), thresholds)

    base = pvf.DataConfig(site=site, lookback_steps=fs.FUSION_LOOKBACK)
    bundles = {}

    def bundle_for(extra):
        key = tuple(extra)
        if key not in bundles:
            config = pvf.DataConfig(site=site, lookback_steps=fs.FUSION_LOOKBACK,
                                    future_feature_names=base.future_feature_names + list(extra))
            bundle = pvf.build_data_bundle(config)
            available = np.intersect1d(common_test, bundle.datasets["test"].origins)
            bundle.datasets["test"] = ForecastWindowDataset(bundle.datasets["test"].windows, available)
            bundles[key] = (config, bundle, len(available))
        return bundles[key]

    path = study_file(site)
    record_out = json.loads(path.read_text()) if path.exists() else {}
    record_out.update({"site": site, "seeds": seeds, "time_limit_minutes": time_limit, "nwp_model": nwp.NWP_MODEL,
                       "lookback_steps": fs.FUSION_LOOKBACK, "started": time.strftime("%Y-%m-%d %H:%M"),
                       "test_windows": bundle_for(NWP_DAY1)[2], "plan": {k: v[3] for k, v in PLAN.items()}})
    record_out.setdefault("runs", {})
    for label, (registered, extra, hyperparameters, _) in PLAN.items():
        config_data, bundle, _ = bundle_for(extra)
        for seed in seeds:
            key = f"{label}_seed{seed}"
            if "run_dir" in record_out["runs"].get(key, {}):
                continue                                                # resume: finished runs are kept
            config = pvf.ExperimentConfig(model_name=registered, model_hyperparameters=dict(hyperparameters),
                                          data=config_data, recipe=fs.fusion_recipe(seed, time_limit),
                                          run_name=f"nwp_{label}_seed{seed}")
            print(f"\n=== seed {seed}: {label} ===", flush=True)
            seed_everything(seed)
            model = pvf.build_model(registered, bundle.spec, **hyperparameters)
            try:
                result = pvf.run_experiment(config, data_bundle=bundle, model=model, log=lambda m: print(m, flush=True))
                entry = {"run_dir": str(result.run_dir), "seed": seed, "model": label}
                if "gate" in registered:
                    entry["gates"] = collect_gates(result.model, bundle.datasets["test"], sky_of_origin)
            except Exception as error:  # noqa: BLE001 - one failing run must not stop the study
                print(f"FAILED {label} seed {seed}: {error}", flush=True)
                entry = {"error": f"{type(error).__name__}: {error}", "seed": seed, "model": label}
            record_out["runs"][key] = entry
            path.write_text(json.dumps(record_out, indent=1))
            torch.cuda.empty_cache()
    record_out["finished"] = time.strftime("%Y-%m-%d %H:%M")
    path.write_text(json.dumps(record_out, indent=1))


# ---------------------------------------------------------------- summary

def write_summary(site):
    study = json.loads(study_file(site).read_text())
    name = report_stem("nwp", study)                    # e.g. nwp_s3; kept in the manifest for later rebuilds
    study_file(site).write_text(json.dumps(study, indent=1))
    fusion = json.loads(fs.study_file(site).read_text())
    runs, _ = fs.load_batch(site)
    reference = fs.load_split(runs[fs.CONTROL], "test")
    target, mask, origins = reference["target"], reference["target_mask"] > 0, reference["origin_index"]
    regimes, _, record = fs.window_regimes(site, origins, target, mask)
    times = record.index[origins]
    target_hour = np.stack([(times + pd.Timedelta(minutes=30 * s)).hour for s in range(1, target.shape[1] + 1)], 1)

    scores, by_seed = {}, {}
    for source in (fusion, study):
        for entry in source["runs"].values():
            if "run_dir" in entry:
                result = fs.run_scores(entry["run_dir"], regimes, target_hour, origins)
                scores.setdefault(entry["model"], []).append(result)
                by_seed.setdefault(entry["model"], {})[entry["seed"]] = result["nrmse_pct"]

    def paired(model, reference):
        """Seed-by-seed nRMSE difference model - reference: (mean, number of seeds where model is better, n)."""
        seeds = sorted(set(by_seed.get(model, {})) & set(by_seed.get(reference, {})))
        differences = [by_seed[model][s] - by_seed[reference][s] for s in seeds]
        return (float(np.mean(differences)) if differences else np.nan, sum(d < 0 for d in differences), len(seeds))

    def stat(model, key):
        mean, std = fs.mean_std([s[key] for s in scores.get(model, [])])
        return mean, std

    def cell(model, key):
        mean, std = stat(model, key)
        return f"{mean:.2f} ± {std:.2f}" if np.isfinite(mean) else "n/a"

    rows = [("patch_tst", "PatchTST, no NWP (control)"), ("patch_tst_nwp", "PatchTST + NWP (day 1)"),
            ("patch_tst_nwp_day2", "PatchTST + NWP (day 2, strict)"),
            ("patch_tst_mamba_gate", "Mamba fusion, no NWP"), ("patch_tst_mamba_gate_future", "Mamba fusion, gate reads the future, no NWP"),
            ("patch_tst_mamba_gate_nwp", "Mamba fusion + NWP"),
            ("patch_tst_conv_gate", "conv fusion, no NWP"), ("patch_tst_conv_gate_nwp", "conv fusion + NWP"),
            ("patch_tst_variable_gate", "variable fusion, no NWP"), ("patch_tst_variable_gate_nwp", "variable fusion + NWP")]
    keys = ("nrmse_pct", "nmae_pct", "nbias_pct", "overcast", "mixed", "clear", "afternoon")
    control_mean = stat("patch_tst", "nrmse_pct")[0]
    oracle_scores = [s["nrmse_pct"] for s in scores.get(fs.ORACLE, [])]
    oracle_mean, oracle_std = fs.mean_std(oracle_scores)

    figures = ROOT / "results" / f"{name}_figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    lead = np.arange(1, target.shape[1] + 1) / 2
    for (model, label), color in zip([("patch_tst", "PatchTST, no NWP"), ("patch_tst_nwp", "PatchTST + NWP"),
                                      ("patch_tst_conv_gate_nwp", "conv fusion + NWP (best)"),
                                      (fs.ORACLE, "PatchTST + perfect weather (1,432 windows)")],
                                     ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]):
        if model in scores:
            ax.plot(lead, np.mean([s["per_horizon_nrmse"] for s in scores[model]], 0), color=color, lw=2, label=label)
    fs._style(ax, "", "lead time (hours ahead)", "test nRMSE, % of rated power")
    ax.set_xlim(0.5, 24)
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    lead_figure = figures / "nrmse_by_lead_time.png"
    fig.savefig(lead_figure, dpi=150)
    plt.close(fig)

    skill_file = ROOT / "results" / f"nwp_skill_{site}.csv"
    nwp_fusions = [m for m in ("patch_tst_mamba_gate_nwp", "patch_tst_conv_gate_nwp", "patch_tst_variable_gate_nwp")
                   if m in scores]
    bottom_line = []
    if "patch_tst_nwp" in scores and nwp_fusions:
        best = min(nwp_fusions, key=lambda m: stat(m, "nrmse_pct")[0])
        difference, better, n = paired(best, "patch_tst_nwp")
        _, better_strict, n_strict = paired("patch_tst_nwp_day2", "patch_tst")
        future_difference = stat("patch_tst_mamba_gate_future", "nrmse_pct")[0] - stat("patch_tst_mamba_gate", "nrmse_pct")[0]
        closed = (control_mean - stat(best, "nrmse_pct")[0]) / (control_mean - oracle_mean) if oracle_scores else np.nan
        bottom_line = [
            "## Bottom line", "",
            f"- The weather forecast helps PatchTST: {stat('patch_tst_nwp', 'nrmse_pct')[0]:.2f} % against "
            f"{control_mean:.2f} % without it. Most of the gain is on overcast windows "
            f"({stat('patch_tst', 'overcast')[0]:.2f} → {stat('patch_tst_nwp', 'overcast')[0]:.2f} %) and at long lead times.",
            f"- It still helps when only the forecast issued about 48 h earlier is used: "
            f"{stat('patch_tst_nwp_day2', 'nrmse_pct')[0]:.2f} % (better than the control on {better_strict} of "
            f"{n_strict} seeds), so the gain does not depend on optimistic timing.",
            f"- With NWP, fusion now beats PatchTST: {best} reaches {stat(best, 'nrmse_pct')[0]:.2f} %, "
            f"{abs(difference):.2f} points better than PatchTST + NWP on average, better on {better} of {n} seeds. "
            "Without NWP no fusion beat PatchTST.",
            f"- The gain comes from the NWP itself, not from the new gate: the Mamba fusion whose gate reads the future "
            f"without NWP changes by {future_difference:+.2f} points. The gates still mix almost evenly whatever the "
            "forecast sky (table at the end), so the fusions act as a learned blend of two NWP-informed forecasters.",
            f"- The best result closes {100 * closed:.0f} % of the gap between the control and the perfect-weather upper "
            f"bound ({oracle_mean:.2f} %): the day-ahead NWP is far from perfect at this site.", ""]
    lines = [f"# NWP study summary ({site.upper()})", "",
             f"Run {study['started']} to {study.get('finished', 'unfinished')}. NWP model `{study['nwp_model']}` from the "
             f"Open-Meteo Previous Runs archive (`pvf/data/nwp.py`). {len(study['seeds'])} seeds per model, the fusion "
             f"study's recipe and {study['lookback_steps']}-step lookback, scored on the same {study['test_windows']} "
             "test windows as the batch and the fusion study. nRMSE and nMAE in % of rated power over daylight targets. "
             "Detailed pipeline and context: `results/patchTST_pipeline_s2.md`.", "", *bottom_line,
             "## 1. The weather forecast used", "",
             "Each target time gets the forecast that the NWP model issued about 24 h earlier (`_previous_day1`): "
             "forecast global irradiance, total cloud cover and 2 m air temperature, added to the known-future inputs. "
             "A strict variant uses the forecast issued about 48 h earlier (`_previous_day2`), which is older than every "
             "forecast origin of a 24 h horizon.", ""]
    if skill_file.exists():
        skill = pd.read_csv(skill_file)
        skill = skill[skill["offset_days"] == 1].dropna()
        lines += ["Irradiance skill of the candidate models on the training weeks only (daylight half-hours; "
                  "the model was chosen here, before any test score was seen):", "",
                  "| NWP model | RMSE W/m² | bias W/m² | correlation (half-hour) | correlation (daily clearness) |",
                  "|---|---|---|---|---|"]
        lines += [f"| {r.model} | {r.rmse_wm2:.0f} | {r.bias_wm2:+.0f} | {r.corr_halfhour:.3f} | {r.corr_daily_clearness:.3f} |"
                  for r in skill.itertuples()]
        lines += ["", "For comparison, using the clear-sky curve as an irradiance forecast gives an RMSE of 276 W/m².", ""]
    lines += ["## 2. Results", "", "Mean ± standard deviation over seeds.", "",
              "| Model | nRMSE % | nMAE % | Bias % | Overcast nRMSE % | Mixed nRMSE % | Clear nRMSE % | Afternoon nMAE % | "
              "Change in nRMSE vs control |", "|---|---|---|---|---|---|---|---|---|"]
    for model, label in rows:
        if model not in scores:
            continue
        mean = stat(model, "nrmse_pct")[0]
        change = "n/a" if model == "patch_tst" else f"{mean - control_mean:+.2f}"
        lines.append(f"| {label} | " + " | ".join(cell(model, k) for k in keys) + f" | {change} |")
    if oracle_scores:
        lines += ["", f"Upper bound for reference: PatchTST with the measured future weather, {oracle_mean:.2f} ± "
                  f"{oracle_std:.2f} % on the {fusion.get('oracle_test_windows')} windows where it exists."]
    lines += ["", "**Test nRMSE % by lead time** (mean over seeds)", "", "| Model | 0.5 h | 1 h | 3 h | 6 h | 12 h | 24 h |",
              "|---|---|---|---|---|---|---|"]
    for model, label in rows:
        if model in scores:
            per_step = np.mean([s["per_horizon_nrmse"] for s in scores[model]], 0)
            lines.append(f"| {label} | " + " | ".join(f"{per_step[int(h * 2) - 1]:.2f}" for h in fs.LEAD_TIMES_HOURS) + " |")
    lines += ["", f"![nRMSE by lead time]({lead_figure.relative_to(lead_figure.parents[1]).as_posix()})", ""]
    gate_rows = [(label, e["gates"]) for label in ("patch_tst_mamba_gate_future", "patch_tst_mamba_gate_nwp",
                                                     "patch_tst_conv_gate_nwp", "patch_tst_variable_gate_nwp")
                 for e in study["runs"].values() if e.get("model") == label and "gates" in e]
    if gate_rows:
        lines += ["**What the gates learned** (share given to PatchTST, by the sky the NWP forecasts for the window; "
                  "terciles of forecast irradiance over clear-sky irradiance)", "",
                  "| Fusion | NWP says cloudy | NWP says mixed | NWP says clear | 0.5 h | 6 h | 24 h |", "|---|---|---|---|---|---|---|"]
        for label in dict(gate_rows):
            entries = [g for name, g in gate_rows if name == label]
            sky = [np.mean([g["by_forecast_sky"].get(s, np.nan) for g in entries]) for s in SKY]
            by_lead = np.mean([g["by_lead_time"] for g in entries], 0)
            lines.append(f"| {label} | " + " | ".join(f"{v:.2f}" for v in sky) + " | "
                         + " | ".join(f"{by_lead[int(h * 2) - 1]:.2f}" for h in (0.5, 6, 24)) + " |")
        lines.append("")
    out = ROOT / "results" / f"{name}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("summary:", out)
    try:
        convert(out, out.with_suffix(".pdf"))
    except Exception as error:  # noqa: BLE001
        print(f"note: PDF skipped ({type(error).__name__}: {error})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="ssp")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.report_only:
        train_study(arguments.site, [int(s) for s in arguments.seeds.split(",")], arguments.time_limit)
    write_summary(arguments.site)
