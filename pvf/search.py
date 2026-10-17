"""Small hyperparameter search (the papers tuned with Optuna, grid search or Nixtla; see docs/training_recipes.md).

    results = search_hyperparameters(base_config, {
        "model.model_size": [64, 128, 256],
        "recipe.learning_rate": [1e-4, 3e-4, 1e-3],
    }, n_trials=6, data_bundle=data_bundle)

Keys are "model.<hyperparameter>" or "recipe.<TrainingRecipe field>". Every trial is a normal saved run
(run_name "<base>_search<i>"); the table is ranked by the validation score, never the test score.
"""
from __future__ import annotations

import copy
import itertools
import random

import pandas as pd

from .experiment import ExperimentConfig, run_experiment


def search_hyperparameters(base_config: ExperimentConfig, search_space: dict, n_trials: int = 10, data_bundle=None,
                           metric: str = "nrmse_pct", seed: int = 0, log=print) -> pd.DataFrame:
    names = list(search_space)
    for name in names:
        if not name.startswith(("model.", "recipe.")):
            raise ValueError(f"search key {name!r} must start with 'model.' or 'recipe.'")
    combinations = list(itertools.product(*(search_space[n] for n in names)))
    random.Random(seed).shuffle(combinations)
    base_name = base_config.run_name or base_config.model_name
    rows = []
    for trial, values in enumerate(combinations[:n_trials]):
        config = copy.deepcopy(base_config)
        config.run_name = f"{base_name}_search{trial}"
        for name, value in zip(names, values):
            section, key = name.split(".", 1)
            if section == "model":
                config.model_hyperparameters[key] = value
            else:
                setattr(config.recipe, key, value)
        log(f"--- trial {trial + 1}/{min(n_trials, len(combinations))}: {dict(zip(names, values))}")
        try:
            result = run_experiment(config, data_bundle=data_bundle, log=lambda *a: None)
            validation = result.metrics["splits"]["validation"]["overall"]
            rows.append({**dict(zip(names, values)), f"validation_{metric}": validation[metric],
                         "epochs": len(result.metrics["history"]), "run_dir": str(result.run_dir)})
            log(f"    validation {metric} = {validation[metric]:.3f}")
        except Exception as error:  # noqa: BLE001 - a failing combination should not stop the search
            rows.append({**dict(zip(names, values)), f"validation_{metric}": float("nan"),
                         "error": f"{type(error).__name__}: {error}"})
            log(f"    failed: {error}")
    return pd.DataFrame(rows).sort_values(f"validation_{metric}").reset_index(drop=True)
