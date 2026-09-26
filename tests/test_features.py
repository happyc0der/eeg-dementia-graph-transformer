"""Shapes and sanity of preprocessing / feature extraction on synthetic signals."""

import numpy as np
import pytest

from eegdementia import features as F
from eegdementia.config import BANDS, CHANNELS, EpochConfig, FeatureConfig
from eegdementia.data import Recording, epoch_recording

SF = 250.0


def synthetic(n_ep=3, seconds=10.0, seed=0, alpha_hz=10.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SF)) / SF
    X = rng.normal(size=(n_ep, len(CHANNELS), len(t))).cumsum(-1) * 0.05  # 1/f-ish
    X += rng.normal(size=X.shape)
    X += 5 * np.sin(2 * np.pi * alpha_hz * t)[None, None, :] * np.linspace(0.2, 1.0, len(CHANNELS))[None, :, None]
    return X.astype(np.float32)


@pytest.fixture(scope="module")
def feats():
    return F.extract_epoch_features(synthetic(), SF, FeatureConfig())


def test_feature_shapes(feats):
    n_ep = 3
    assert feats["X"].shape == (n_ep, len(feats["names"]))
    assert len(set(feats["names"])) == len(feats["names"])
    assert feats["cov"].shape == (n_ep, len(BANDS) + 1, len(CHANNELS), len(CHANNELS))
    assert feats["conn"].shape == (n_ep, len(F.CONN_MEASURES), len(BANDS), len(CHANNELS) * (len(CHANNELS) - 1) // 2)
    assert np.isfinite(feats["X"]).all()


def test_relative_power_sums_to_one(feats):
    names = feats["names"]
    for ch in ("O1", "Fz"):
        idx = [names.index(f"spec__relpow__{b}__{ch}") for b in BANDS]
        np.testing.assert_allclose(feats["X"][:, idx].sum(1), 1.0, atol=1e-4)


def test_alpha_peak_detected(feats):
    names = feats["names"]
    paf = feats["X"][:, names.index("spec__peak_freq__all__Pz")]
    assert np.all(np.abs(paf - 10.0) <= 0.5)


def test_covariances_spd_and_connectivity_range(feats):
    ev = np.linalg.eigvalsh(feats["cov"].astype(np.float64))
    assert (ev > 0).all()
    conn = feats["conn"]
    assert (conn[:, :3] >= -1e-6).all() and (conn[:, :3] <= 1 + 1e-5).all()  # coh, icoh, wpli in [0,1]
    assert (np.abs(conn[:, 3]) <= 1 + 1e-5).all()  # AEC is a correlation


def test_regional_means_consistent(feats):
    names = feats["names"]
    X = feats["X"]
    reg = X[:, names.index("specR__relpow__alpha__occipital")]
    chans = X[:, [names.index("spec__relpow__alpha__O1"), names.index("spec__relpow__alpha__O2")]].mean(1)
    np.testing.assert_allclose(reg, chans, rtol=1e-5)


def test_epoching_rejects_boundaries_and_outliers():
    rng = np.random.default_rng(1)
    n = int(120 * SF)
    data = rng.normal(size=(len(CHANNELS), n)).astype(np.float32)
    data[:, int(100 * SF) : int(101 * SF)] *= 100  # big artefact inside the 95-105 s region
    rec = Recording("sub-x", data, SF, list(CHANNELS), np.array([int(42 * SF)]), 180.0)
    ep = epoch_recording(rec, EpochConfig(length_s=10, step_s=5, boundary_margin_s=0.5, reject_mad_k=5))
    assert ep.data.shape[1:] == (len(CHANNELS), int(10 * SF))
    assert ep.n_candidates == 23
    # windows starting 35 and 40 s contain the boundary at 42 s
    assert ep.n_boundary_rejected == 2
    for o in ep.onsets_s:
        assert not (o <= 42 + 0.5 and o + 10 >= 42 - 0.5)
        assert not (o < 101 and o + 10 > 100), "artefact window kept"
    assert ep.n_amplitude_rejected >= 2
    no_rej = epoch_recording(rec, EpochConfig(length_s=10, step_s=5, drop_boundaries=False, reject_mad_k=None))
    assert len(no_rej.data) == 23
