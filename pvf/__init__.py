"""pvf - PV power forecasting research framework (SET Lab, MIST). Everything is driven from notebooks/."""
from .data.features import describe_features  # noqa: F401
from .data.pipeline import DataConfig, build_data_bundle, describe_data_bundle, sample_batch  # noqa: F401
from .data.splits import SplitConfig  # noqa: F401
from .experiment import ExperimentConfig, load_trained_model, run_experiment  # noqa: F401
from .evaluation.comparison import (compare_runs, plot_error_by_hour, plot_forecast_examples,  # noqa: F401
                                    plot_per_horizon, plot_training_history)
from .models.base import MODEL_REGISTRY, build_model, describe_models  # noqa: F401
from .training.losses import LOSSES  # noqa: F401
from .training.optimizers import OPTIMIZERS, SCHEDULERS  # noqa: F401
from .search import search_hyperparameters  # noqa: F401
from .training.paper_recipes import describe_paper_recipes, paper_recipe  # noqa: F401
from .training.trainer import TrainingRecipe  # noqa: F401
from .workspace import check_environment, fetch_pretrained_weights, smoke_test_models  # noqa: F401

__version__ = "0.2.0"
