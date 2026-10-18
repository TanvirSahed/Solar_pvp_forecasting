"""Train every model with its paper recipe on one site, then write a report (Markdown + PDF) to Resources/.

    python tools/run_all_models.py --site ssp --time-limit 15            # train all, then report
    python tools/run_all_models.py --site ssp --report-only              # rebuild the report from the last batch

All models are scored on the SAME test windows (those every lookback can forecast) and ranked by test nRMSE.
Finished runs are recorded in results/batch_<site>.json as they complete, so an interrupted batch keeps them.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw figures without a display
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pvf  # noqa: E402
from pvf.data.windows import ForecastWindowDataset  # noqa: E402
from pvf.data.sites import get_site  # noqa: E402
from pvf.evaluation.comparison import plot_per_horizon, plot_training_history  # noqa: E402
from pvf.training.paper_recipes import PAPER_RECIPES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RANK_METRIC = "nrmse_pct"


def batch_file(site):
    return ROOT / "results" / f"batch_{site}.json"


def run_batch(site: str, time_limit: float, model_names: list[str]):
    presets = {name: pvf.paper_recipe(name) for name in model_names}
    bundles = {}
    for preset in presets.values():
        lookback = preset.lookback_steps("30min")
        if lookback not in bundles:
            print(f"building data: lookback {lookback} steps", flush=True)
            bundles[lookback] = pvf.build_data_bundle(pvf.DataConfig(site=site, lookback_steps=lookback))
    common_test = None
    for bundle in bundles.values():
        origins = bundle.datasets["test"].origins
        common_test = origins if common_test is None else np.intersect1d(common_test, origins)
    for bundle in bundles.values():   # identical test windows for every model -> comparable scores
        bundle.datasets["test"] = ForecastWindowDataset(bundle.datasets["test"].windows, common_test)
    print(f"common test windows: {len(common_test)}", flush=True)

    records = {"site": site, "time_limit_minutes": time_limit, "test_windows": int(len(common_test)),
               "started": time.strftime("%Y-%m-%d %H:%M"), "runs": {}}
    for number, (name, preset) in enumerate(presets.items(), start=1):
        lookback = preset.lookback_steps("30min")
        recipe = preset.recipe
        recipe.time_limit_minutes = time_limit
        config = pvf.ExperimentConfig(model_name=name, model_hyperparameters=preset.model_hyperparameters,
                                      data=pvf.DataConfig(site=site, lookback_steps=lookback), recipe=recipe,
                                      run_name=name)
        print(f"\n=== [{number}/{len(presets)}] {name} (lookback {lookback}) ===", flush=True)
        try:
            result = pvf.run_experiment(config, data_bundle=bundles[lookback], log=lambda m: print(m, flush=True))
            records["runs"][name] = {"run_dir": str(result.run_dir), "source": preset.source, "notes": preset.notes}
            del result
        except Exception as error:  # noqa: BLE001 - one failing model must not stop the batch
            print(f"FAILED {name}: {type(error).__name__}: {error}", flush=True)
            records["runs"][name] = {"error": f"{type(error).__name__}: {error}", "source": preset.source}
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        batch_file(site).parent.mkdir(exist_ok=True)
        batch_file(site).write_text(json.dumps(records, indent=1))
    records["finished"] = time.strftime("%Y-%m-%d %H:%M")
    batch_file(site).write_text(json.dumps(records, indent=1))
    return records


LEAD_TIMES_HOURS = (0.5, 1, 3, 6, 12, 24)
CHART_GROUP_SIZE = 7


def horizon_section(rows, site, figures, resources) -> list[str]:
    """Per-lead-time test nMAE / nRMSE tables + nRMSE-vs-lead-time charts (models in rank groups of 7)."""
    if not rows:
        return []
    step_hours = 0.5
    steps = [int(round(h / step_hours)) - 1 for h in LEAD_TIMES_HOURS]
    header = "| Rank | Model | " + " | ".join(f"{h:g} h" for h in LEAD_TIMES_HOURS) + " |"
    divider = "|---|---|" + "---|" * len(LEAD_TIMES_HOURS)
    lines = ["## Performance by forecast horizon", "",
             "Test scores at selected lead times (each column scores only that step ahead; formulas below). "
             "Smart persistence is listed for reference and left out of the charts: its error grows to several times "
             "the models' and would flatten their differences.", ""]
    for metric, label in (("nrmse_pct", "nRMSE %"), ("nmae_pct", "nMAE %")):
        lines += [f"**Test {label} by lead time**", "", header, divider]
        for rank, r in enumerate(rows, start=1):
            curve = r["per_horizon"][metric]
            lines.append(f"| {rank} | {r['model']} | " + " | ".join(_fmt(curve[s]) for s in steps) + " |")
        lines.append("")
    trained = [r for r in rows if r["model"] != "smart_persistence"]
    for start in range(0, len(trained), CHART_GROUP_SIZE):
        group = trained[start:start + CHART_GROUP_SIZE]
        first_rank, last_rank = rows.index(group[0]) + 1, rows.index(group[-1]) + 1
        figure = plot_per_horizon([r["run_dir"] for r in group], metric="nrmse_pct", split="test",
                                  show_persistence=False,
                                  title=f"Test nRMSE by lead time, ranks {first_rank}-{last_rank}")
        figure_file = figures / f"horizon_nrmse_ranks_{first_rank}_{last_rank}.png"
        figure.savefig(figure_file, dpi=110, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(figure)
        lines += [f"![Test nRMSE by lead time, ranks {first_rank}-{last_rank}]({figure_file.relative_to(resources).as_posix()})", ""]
    return lines


def metric_definitions(site) -> list[str]:
    """The formulas exactly as pvf/evaluation/metrics.py computes them."""
    ramp_threshold = get_site(site).ramp_threshold_fraction
    return [
        "## How the metrics are calculated",
        "",
        "Notation: P_rated = inverter AC rating; y = measured AC power / P_rated at a target time; ŷ = forecast / P_rated "
        "(clipped at 0; for quantile models the 0.5 quantile); S = scored points, i.e. every (test window, step ahead) "
        "pair whose target was measured and in daylight (clear-sky irradiance > 10 W/m²); N = number of points in S. "
        "Overall scores use all 48 steps; a lead-time column uses only that step.",
        "",
        "| Metric | Formula | Meaning |",
        "|---|---|---|",
        "| nMAE % | 100 · (1/N) · Σ_S \\|ŷ − y\\| | typical error, % of rated power |",
        "| nRMSE % | 100 · √[ (1/N) · Σ_S (ŷ − y)² ] | like nMAE but weights large misses more; ranking metric |",
        "| Bias % | 100 · (1/N) · Σ_S (ŷ − y) | above 0 = over-forecast on average |",
        "| R² | 1 − Σ_S (ŷ − y)² / Σ_S (y − ȳ)², with ȳ = mean of y over S | share of variance explained (1 = perfect) |",
        "| Smart persistence s | s(t, h) = P(t) · Gcs(t+h) / Gcs(t), set to 0 when Gcs(t) ≤ 1 W/m², clipped to "
        "[0, 1.2] · P_rated | physics reference: current power scaled by the clear-sky (Ineichen) irradiance change |",
        "| Skill vs persistence | 1 − nRMSE(model) / nRMSE(smart persistence), on the same points S | above 0 beats "
        "the reference |",
        f"| Ramp MAE % | 100 · mean over ramps of \\|Δŷ − Δy\\|, with Δy = y(step k+1) − y(step k) inside one forecast "
        f"window; a ramp is a pair of scored steps with \\|Δy\\| > {ramp_threshold:g} (i.e. {ramp_threshold:.0%} of P_rated "
        f"per 30 min) | how well fast rises and drops are tracked |",
        "| Pinball (quantile q) | L_q = max( q · (y − ŷ_q), (q − 1) · (y − ŷ_q) ) | quantile-forecast loss |",
        "| CRPS | 2 · mean over S and the quantile levels q of L_q (levels 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95) | "
        "accuracy of the whole predictive distribution (quantile models only) |",
        "",
    ]


def _fmt(value, digits=2):
    return "–" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


def write_report(site: str):
    records = json.loads(batch_file(site).read_text())
    date = records["started"][:10]
    name = f"Experiment_Report_{site.upper()}_{date}"
    resources = ROOT / "Resources"
    figures = resources / f"{name}_figures"
    figures.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    for model_name, record in records["runs"].items():
        if "error" in record:
            failures.append((model_name, record["error"]))
            continue
        run_dir = Path(record["run_dir"])
        metrics = json.loads((run_dir / "metrics.json").read_text())
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        history = metrics["history"]
        recipe = config["recipe"]
        stop_metric = recipe["early_stopping_metric"]
        best_epoch = (min(history, key=lambda h: h[stop_metric])["epoch"] + 1) if history else None
        if not history:
            stop_reason = "not trained (zero-shot / reference)"
        elif len(history) >= recipe["n_epochs"]:
            stop_reason = "reached max epochs"
        elif recipe.get("time_limit_minutes") and metrics["train_seconds"] >= 60 * recipe["time_limit_minutes"]:
            stop_reason = f"time limit ({recipe['time_limit_minutes']} min)"
        else:
            stop_reason = f"early stopping ({stop_metric}, patience {recipe['early_stopping_patience']})"
        figure_file = None
        if history:
            figure = plot_training_history(run_dir)
            figure_file = figures / f"{model_name}_loss.png"
            figure.savefig(figure_file, dpi=110, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(figure)
        rows.append({"model": model_name, "source": record["source"], "notes": record.get("notes", []),
                     "test": metrics["splits"]["test"]["overall"], "epochs": len(history), "best_epoch": best_epoch,
                     "minutes": metrics["train_seconds"] / 60, "parameters": metrics["n_parameters"],
                     "config": config, "stop_reason": stop_reason, "figure": figure_file, "run_dir": run_dir,
                     "per_horizon": metrics["splits"]["test"]["per_horizon"],
                     "persistence_per_horizon": metrics["splits"]["test"].get("per_horizon_smart_persistence")})
    rows.sort(key=lambda r: r["test"][RANK_METRIC])

    data = rows[0]["config"]["data"] if rows else {}
    lines = [
        f"# Experiment report: every model on {site.upper()}",
        "",
        f"Run {records['started']} to {records.get('finished', 'unfinished')}. Each model trained with its paper recipe "
        f"(`pvf.paper_recipe`, see `docs/training_recipes.md`), capped at {records['time_limit_minutes']} min of training. "
        f"All models are scored on the same {records['test_windows']} test windows (0.5–24 h ahead, daytime targets) and "
        f"ranked by test nRMSE. Errors are % of rated power.",
        "",
        "## Summary",
        "",
        "| Rank | Model | Recipe source | Epochs run (best) | Train min | Parameters | nMAE % | nRMSE % | Bias % | "
        "Skill vs persistence | R² | Ramp MAE % | CRPS |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, r in enumerate(rows, start=1):
        t = r["test"]
        epochs = f"{r['epochs']} ({r['best_epoch']})" if r["epochs"] else "0"
        lines.append(f"| {rank} | {r['model']} | {r['source']} | {epochs} | {r['minutes']:.1f} | {r['parameters']:,} | "
                     f"{_fmt(t['nmae_pct'])} | {_fmt(t['nrmse_pct'])} | {_fmt(t['nbias_pct'])} | "
                     f"{_fmt(t['rmse_skill_vs_persistence'], 3)} | {_fmt(t['r2'], 3)} | {_fmt(t['ramp_mae_pct'])} | "
                     f"{_fmt(t.get('crps'), 4)} |")
    for model_name, error in failures:
        lines.append(f"| – | {model_name} | failed | – | – | – | – | – | – | – | – | – | – |")
    lines += [
        "",
        "Skill vs persistence = 1 − RMSE / RMSE(smart persistence): above 0 beats the physics reference. Smart persistence"
        " is weak from night origins (it predicts 0), so read skill together with nMAE and nRMSE.",
        "",
        *horizon_section(rows, site, figures, resources),
        *metric_definitions(site),
        "## Data used by every model",
        "",
        f"- **Site:** {site.upper()}; resolution {data.get('resolution')}; horizon {data.get('horizon_steps')} steps "
        f"(0.5–24 h).",
        f"- **History features:** {', '.join(data.get('history_feature_names', []))}.",
        f"- **Future (known‑in‑advance) features:** {', '.join(data.get('future_feature_names', []))}.",
        f"- **Split:** {data.get('split', {}).get('mode')} weekly blocks {data.get('split', {}).get('block_pattern')}; "
        f"lookback differs per model (below).",
        "",
        "## Per model",
    ]
    for rank, r in enumerate(rows, start=1):
        recipe, config = r["config"]["recipe"], r["config"]
        scheduler = recipe["scheduler"] + (f" {recipe['scheduler_options']}"
                                           if recipe["scheduler_options"] and recipe["scheduler"] != "none" else "")
        lines += [
            "",
            f"### {rank}. {r['model']}",
            "",
            "| Setting | Value |",
            "|---|---|",
            f"| Recipe source | {r['source']} |",
            f"| Lookback | {config['data']['lookback_steps']} steps ({config['data']['lookback_steps'] / 2:g} h) |",
            f"| Hyperparameters | {config['model_hyperparameters'] or 'defaults'} |",
            f"| Loss | {'pinball (quantile model)' if r['test'].get('crps') is not None else recipe['loss']} |",
            f"| Optimizer | {recipe['optimizer']}, lr {recipe['learning_rate']:g}, weight decay {recipe['weight_decay']:g} "
            f"{recipe['optimizer_options'] or ''} |",
            f"| Scheduler | {scheduler} |",
            f"| Batch size | {recipe['batch_size']} |",
            f"| Epochs | {r['epochs']} run of max {recipe['n_epochs']}; best epoch {r['best_epoch'] or '–'} |",
            f"| Early stopping | " + (f"{recipe['early_stopping_metric']}, patience {recipe['early_stopping_patience']}"
                                    if recipe['early_stopping_patience'] else "off (fixed number of epochs)") + " |",
            f"| Stopped by | {r['stop_reason']} |",
            f"| Training time | {r['minutes']:.1f} min |",
            f"| Test nMAE / nRMSE | {_fmt(r['test']['nmae_pct'])} % / {_fmt(r['test']['nrmse_pct'])} % |",
        ]
        if r["notes"]:
            lines += ["", "Recipe notes: " + "; ".join(r["notes"]) + "."]
        if r["figure"]:
            lines += ["", f"![{r['model']} loss per epoch]({r['figure'].relative_to(resources).as_posix()})"]
    if failures:
        lines += ["", "## Failed runs", ""] + [f"- **{m}:** {e}" for m, e in failures]
    markdown_file = resources / f"{name}.md"
    markdown_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.path.insert(0, str(ROOT / "tools"))
    from markdown_to_pdf import convert

    convert(markdown_file, resources / f"{name}.pdf")
    print(f"report: {markdown_file} and {name}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="ssp")
    parser.add_argument("--time-limit", type=float, default=15.0, help="minutes of training per model")
    parser.add_argument("--models", default="", help="comma-separated subset (default: every model with a recipe)")
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.report_only:
        names = arguments.models.split(",") if arguments.models else sorted(PAPER_RECIPES)
        run_batch(arguments.site, arguments.time_limit, names)
    write_report(arguments.site)
