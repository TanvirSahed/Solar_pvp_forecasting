"""How the reference papers trained each model, as ready-to-use presets (details: docs/training_recipes.md).

    preset = paper_recipe("hdcf_mamba")
    preset.model_hyperparameters, preset.recipe, preset.lookback_steps("30min"), preset.notes

Values the paper does not report are filled with the cross-paper consensus (AdamW 1e-3, reduce-on-plateau,
early stopping) and listed in `notes` as "not reported -> ...". Change anything after loading it.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd

from .trainer import TrainingRecipe

# R20 (Hong & Lee 2026): AdamW 1e-3, wd 1e-4, OneCycleLR, <= 200 epochs, patience 30 on validation MAPE.
# MAPE is undefined at zero PV output, so validation nMAE is used instead.
R20_RECIPE = TrainingRecipe(optimizer="adamw", learning_rate=1e-3, weight_decay=1e-4, scheduler="one_cycle",
                            n_epochs=200, early_stopping_metric="nmae", early_stopping_patience=30)
# R6 (HDCF-Mamba): Adam 1e-4 (0.9, 0.999), ReduceLROnPlateau x0.5, early stopping patience 4, MSE.
R6_RECIPE = TrainingRecipe(optimizer="adam", learning_rate=1e-4, weight_decay=0.0,
                           optimizer_options={"betas": (0.9, 0.999)}, scheduler="reduce_on_plateau",
                           scheduler_options={"factor": 0.5, "patience": 2}, n_epochs=100,
                           early_stopping_metric="validation_loss", early_stopping_patience=4)
CONSENSUS_RECIPE = TrainingRecipe(optimizer="adamw", learning_rate=1e-3, weight_decay=1e-4,
                                  scheduler="reduce_on_plateau", scheduler_options={"factor": 0.5, "patience": 4},
                                  n_epochs=100, early_stopping_metric="nrmse", early_stopping_patience=10)


@dataclass
class PaperRecipe:
    model_name: str
    source: str
    model_hyperparameters: dict
    recipe: TrainingRecipe
    lookback_hours: float
    notes: list = field(default_factory=list)

    def lookback_steps(self, resolution: str = "30min") -> int:
        return int(round(pd.Timedelta(hours=self.lookback_hours) / pd.Timedelta(resolution)))


def _recipe(base: TrainingRecipe, **changes) -> TrainingRecipe:
    recipe = copy.deepcopy(base)
    for name, value in changes.items():
        setattr(recipe, name, value)
    return recipe


PAPER_RECIPES = {r.model_name: r for r in [
    PaperRecipe("lstm", "R20", {"hidden_size": 256, "n_layers": 2, "bidirectional": True, "pooling": "attention"},
                R20_RECIPE, 240, ["early stopping on nMAE instead of MAPE (zero PV at night)"]),
    PaperRecipe("gru", "R20 (LSTM settings)", {"hidden_size": 256, "n_layers": 2, "bidirectional": True,
                                               "pooling": "attention"}, R20_RECIPE, 240,
                ["R20 tested LSTM only; same settings used for GRU"]),
    PaperRecipe("patch_tst", "R7 (PV, hourly)", {"model_size": 128, "n_heads": 4, "n_layers": 3, "patch_length": 16,
                                                 "patch_stride": 8, "dropout": 0.1},
                _recipe(CONSENSUS_RECIPE, optimizer="adam", learning_rate=1e-4, scheduler="none", batch_size=48,
                        n_epochs=10, early_stopping_patience=None), 48,
                ["optimizer and learning rate not reported -> Adam 1e-4", "n_layers not reported -> 3 (original paper)",
                 "R7 forecast 1 h ahead; here the full horizon"]),
    PaperRecipe("i_transformer", "R20", {"model_size": 512, "n_heads": 8, "n_layers": 3, "feedforward_size": 1024},
                R20_RECIPE, 240, ["n_layers not reported in R20 -> 3 (original paper: 2-4)"]),
    PaperRecipe("time_xer", "F03 (original)", {"patch_length": 24, "model_size": 256, "n_layers": 2},
                _recipe(CONSENSUS_RECIPE, optimizer="adam", learning_rate=1e-4, weight_decay=0.0, scheduler="none",
                        n_epochs=10, early_stopping_metric="validation_loss", early_stopping_patience=3), 96,
                ["patch 24 = the paper's short-term setting", "early-stopping patience not reported -> 3"]),
    PaperRecipe("hdcf_mamba", "R6", {"model_size": 128, "state_size": 32, "expansion_factor": 2, "dropout": 0.1},
                R6_RECIPE, 96, ["plateau-scheduler patience not reported -> 2 (must act before early stop at 4)",
                                "lookback not stated exactly (inputs up to 720 x 15 min) -> 96 h",
                                "paper smooths inputs with a moving average and z-scores them (not done here)"]),
    PaperRecipe("power_mamba", "R19 + R20", {"embedding_length": 64, "state_size": 16}, R20_RECIPE, 240,
                ["R19 grid-searched E and the moving-average kernel (values not published) and trained 50 fixed "
                 "epochs; R20's complete recipe is used instead"]),
    PaperRecipe("s_mamba", "R20", {"model_size": 256, "n_layers": 2}, R20_RECIPE, 240,
                ["n_layers not reported -> 2"]),
    PaperRecipe("mamba_temporal", "R6 recipe (no own paper)", {"model_size": 128, "state_size": 32, "n_layers": 2},
                R6_RECIPE, 96, ["Concat-Mamba ablation root: uses the HDCF-Mamba recipe so the ablation is fair"]),
    PaperRecipe("graph_mamba", "R6 recipe (SET Lab synthesis)", {"model_size": 128}, R6_RECIPE, 96,
                ["no paper; temporal Mamba part follows R6"]),
    PaperRecipe("fusion_itransformer_hdcf", "R6 recipe", {"fusion_size": 128, "state_size": 32}, R6_RECIPE, 96,
                ["worked example; follows its Mamba branch's paper (R6)"]),
    PaperRecipe("rfkan", "R16", {"n_harmonics": 8, "n_layers": 2, "recurrent_window_steps": 12},
                _recipe(CONSENSUS_RECIPE, optimizer="adam", learning_rate=1e-3, scheduler="none", batch_size=32,
                        n_epochs=50), 24,
                ["paper input = 12 steps (1 h at 5 min); here 12 steps of 30 min",
                 "optimizer not reported (pykan default is LBFGS) -> Adam 1e-3 as its LSTM baseline",
                 "grid refinement 3->100 not reproduced; n_harmonics fixed"]),
    PaperRecipe("dt_dann", "R18", {"model_size": 64, "domain_loss_weight": 1.0},
                _recipe(CONSENSUS_RECIPE, optimizer="adam", learning_rate=1e-4, weight_decay=0.0), 24,
                ["learning rate, batch and epochs not reported -> Adam 1e-4, consensus otherwise",
                 "paper loss weights domain loss 1:1 with the forecast losses",
                 "set DataConfig(domain_adaptation_months=[...]) to give it a target domain"]),
    PaperRecipe("mtsa_qrn", "R8", {}, CONSENSUS_RECIPE, 48,
                ["no training settings reported -> consensus recipe; pinball loss is automatic",
                 "paper decomposes the power history with VMD (not done here); calibrate afterwards with CQR"]),
    PaperRecipe("deep_mlp", "no paper (consensus)", {}, CONSENSUS_RECIPE, 64, []),
    PaperRecipe("tcn", "no paper (consensus)", {}, CONSENSUS_RECIPE, 64, []),
    PaperRecipe("dlinear", "no paper (consensus)", {}, CONSENSUS_RECIPE, 64, []),
    PaperRecipe("smart_persistence", "physics reference", {}, CONSENSUS_RECIPE, 64, ["nothing to train"]),
    PaperRecipe("chronos2", "R12 + R3", {"use_covariates": True}, CONSENSUS_RECIPE, 504,
                ["zero-shot; R3: context saturates near 2000 x 15 min (~3 weeks)",
                 "R3 LoRA fine-tune (lr 1e-6, 1000 steps, rank 8) was worse on the median - not implemented"]),
    PaperRecipe("timesfm", "R14 + R2", {}, CONSENSUS_RECIPE, 504, ["zero-shot; R2: longer solar context is better"]),
    PaperRecipe("tirex", "R13 + R3", {}, CONSENSUS_RECIPE, 504,
                ["zero-shot; installed package is TiRex 1 (univariate)"]),
]}


def paper_recipe(model_name: str) -> PaperRecipe:
    if model_name not in PAPER_RECIPES:
        raise KeyError(f"no paper recipe for {model_name!r}. Available: {sorted(PAPER_RECIPES)}")
    return copy.deepcopy(PAPER_RECIPES[model_name])


def describe_paper_recipes() -> pd.DataFrame:
    rows = []
    for name, preset in sorted(PAPER_RECIPES.items()):
        r = preset.recipe
        rows.append({"model_name": name, "source": preset.source, "lookback_hours": preset.lookback_hours,
                     "optimizer": r.optimizer, "learning_rate": r.learning_rate, "scheduler": r.scheduler,
                     "epochs": r.n_epochs, "early_stopping": f"{r.early_stopping_metric} / {r.early_stopping_patience}",
                     "batch_size": r.batch_size, "model_hyperparameters": preset.model_hyperparameters,
                     "notes": "; ".join(preset.notes)})
    return pd.DataFrame(rows)
