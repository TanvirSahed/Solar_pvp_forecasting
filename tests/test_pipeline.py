"""Leakage / correctness tests. Run: python -m pytest -q"""
import numpy as np
import pytest
from omegaconf import OmegaConf

from pvf.cli import compose
from pvf.data.datamodule import build_data
from pvf.data.loader import smart_persistence
from pvf.eval.conformal import aci


@pytest.fixture(scope="module", params=["ssp", "nist"])
def data(request):
    return build_data(compose(None, (), [f"site={request.param}"]))


def test_targets_in_one_role_and_train_purged(data):
    ff, sets, role = data["ff"], data["sets"], data["role"]
    L, H = ff.L, ff.H
    for k, ds in sets.items():
        if k == "adapt":
            continue
        for t in ds.t[:: max(1, len(ds.t) // 500)]:
            assert (role[t + 1:t + H + 1] == k).all(), k
            if k == "train":  # purge: no evaluation timestep inside training inputs
                assert not np.isin(role[t - L + 1:t + 1], ["val", "test", "holdout"]).any()


def test_every_split_covers_all_seasons(data):
    for k in ("train", "val", "test"):
        seasons = {(m % 12) // 3 for m in data["ff"].index[data["sets"][k].t].month}
        assert seasons == {0, 1, 2, 3}, (k, seasons)


def test_inputs_finite_and_targets_daytime(data):
    ds = data["sets"]["test"]
    for i in np.random.default_rng(0).integers(0, len(ds), 50):
        b = ds[int(i)]
        assert np.isfinite(b["x_hist"].numpy()).all() and np.isfinite(b["x_fut"].numpy()).all()
        assert b["mask"].sum() > 0


def test_smart_persistence_formula():
    P = np.array([0, 10, 20, 30, 40.0])
    G = np.array([0, 100, 200, 300, 400.0])
    s = smart_persistence(P, G, np.array([1]), np.array([1, 2]))
    np.testing.assert_allclose(s, [[20, 30]])            # clear-sky scaled
    assert smart_persistence(P, G, np.array([0]), np.array([1]))[0, 0] == 0  # Gcs(t) <= eps -> 0


def test_aci_is_causal():
    rng = np.random.default_rng(0)
    N, H = 300, 3
    y = rng.normal(size=(N, H))
    pred = np.zeros((N, H))
    t0 = np.arange(N)
    lo1, hi1, _ = aci(pred, y, np.ones((N, H)), t0)
    y2 = y.copy()
    y2[200:] += 100.0                                     # change the future only
    lo2, hi2, _ = aci(pred, y2, np.ones((N, H)), t0)
    np.testing.assert_allclose(lo1[:201], lo2[:201])     # intervals before the change are unaffected
