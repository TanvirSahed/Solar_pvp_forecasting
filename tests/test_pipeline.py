"""Leakage / correctness tests.  Run in the notebook (!python -m pytest -q tests) or a terminal."""
import numpy as np
import pytest

import pvf
from pvf.data.features import check_future_features
from pvf.data.site_record import compute_smart_persistence
from pvf.evaluation.conformal import adaptive_conformal_intervals
from pvf.experiment import ExperimentConfig


@pytest.fixture(scope="module", params=["ssp", "nist"])
def data_bundle(request):
    return pvf.build_data_bundle(pvf.DataConfig(site=request.param))


def test_targets_in_one_split_and_training_inputs_purged(data_bundle):
    roles, lookback, horizon = data_bundle.split_roles, data_bundle.spec.lookback_steps, data_bundle.spec.horizon_steps
    for split, dataset in data_bundle.datasets.items():
        for t in dataset.origins[:: max(1, len(dataset.origins) // 500)]:
            assert (roles[t + 1:t + horizon + 1] == split).all(), split
            if split == "train":
                assert not np.isin(roles[t - lookback + 1:t + 1], ["validation", "test", "holdout"]).any()


def test_every_split_covers_all_seasons(data_bundle):
    for split in ("train", "validation", "test"):
        months = data_bundle.spec.time_index[data_bundle.datasets[split].origins].month
        assert {(m % 12) // 3 for m in months} == {0, 1, 2, 3}, split


def test_samples_are_finite_and_have_daytime_targets(data_bundle):
    dataset = data_bundle.datasets["test"]
    for i in np.random.default_rng(0).integers(0, len(dataset), 50):
        sample = dataset[int(i)]
        assert np.isfinite(sample["history_features"].numpy()).all()
        assert np.isfinite(sample["future_features"].numpy()).all()
        assert sample["target_mask"].sum() > 0


def test_measured_values_cannot_be_future_features():
    with pytest.raises(ValueError):
        check_future_features(["irradiance"])
    check_future_features(["irradiance_oracle", "clear_sky_irradiance"])     # explicit oracle is allowed


def test_smart_persistence_formula():
    power, clear_sky = np.array([0, 10, 20, 30, 40.0]), np.array([0, 100, 200, 300, 400.0])
    np.testing.assert_allclose(compute_smart_persistence(power, clear_sky, np.array([1]), 2), [[20, 30]])
    assert compute_smart_persistence(power, clear_sky, np.array([0]), 1)[0, 0] == 0   # clear sky ~0 -> 0


def test_adaptive_conformal_intervals_do_not_look_ahead():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(300, 3))
    forecast, mask, origins = np.zeros((300, 3)), np.ones((300, 3)), np.arange(300)
    lower_a, _, _ = adaptive_conformal_intervals(forecast, target, mask, origins)
    changed_future = target.copy()
    changed_future[200:] += 100.0
    lower_b, _, _ = adaptive_conformal_intervals(forecast, changed_future, mask, origins)
    np.testing.assert_allclose(lower_a[:201], lower_b[:201])


def test_experiment_config_round_trip(tmp_path):
    config = ExperimentConfig(model_name="lstm", model_hyperparameters={"hidden_size": 16},
                              data=pvf.DataConfig(site="nist", split=pvf.SplitConfig(holdout_months=[12])),
                              recipe=pvf.TrainingRecipe(scheduler="cosine"))
    config.save(tmp_path / "config.yaml")
    assert ExperimentConfig.load(tmp_path / "config.yaml") == config


def test_every_model_trains_one_step(data_bundle):
    """Every registered model builds, runs forward and backward on a real batch (zero-shot ones forward only)."""
    import torch

    from pvf.models import architectures  # noqa: F401
    from pvf.models.base import MODEL_REGISTRY
    from pvf.training.trainer import loss_function_for, move_to_device, pick_device

    device = pick_device()
    batch = move_to_device(pvf.sample_batch(data_bundle, batch_size=8), device)
    for name, model_class in sorted(MODEL_REGISTRY.items()):
        if getattr(model_class, "needs_weight_download", False):
            continue
        model = pvf.build_model(name, data_bundle.spec).to(device).train()
        loss = loss_function_for(model, pvf.TrainingRecipe())(model(batch)["forecast"], batch)
        assert torch.isfinite(loss), name
        if loss.requires_grad:
            loss.backward()
