"""Phase-2 deep-model wrapper and combination helpers (CPU, tiny synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from eegdementia import evaluation as ev
from eegdementia.models import ModelSpec, lr_factory

torch = pytest.importorskip("torch")
deep = pytest.importorskip("eegdementia.deep")

from test_leakage import make_dataset, make_subjects  # noqa: E402


def _raw_dataset(seed=0):
    subj = make_subjects((8, 7, 6), seed=seed)
    ds = make_dataset(subj, (4, 7), seed=seed)
    rng = np.random.default_rng(seed)
    n = len(ds.groups)
    # tiny "raw" epochs: 3 channels x 32 samples; channel 0 carries the subject index
    raw = rng.normal(size=(n, 3, 32)).astype(np.float32) + ds.y[:, None, None]
    sid = np.array([list(subj.index).index(g) for g in ds.groups], dtype=np.float32)
    raw[:, 0, 0] = sid
    ds.inputs["raw"] = raw
    return ds


class TinyNet(torch.nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32, n_classes))

    def forward(self, x):
        return self.f(x)


SEEN = {"train": [], "val": []}


class SpyClassifier(deep.TorchEpochClassifier):
    def _train(self, X, y, groups, tr_idx, n_epochs, va_idx=None, seed=0):
        SEEN["train"].append(set(np.asarray(groups)[tr_idx]))
        SEEN["val"].append(set(np.asarray(groups)[va_idx]) if va_idx is not None else set())
        return super()._train(X, y, groups, tr_idx, n_epochs, va_idx, seed)


def test_torch_wrapper_never_sees_test_subjects_and_bags_partition_training():
    ds = _raw_dataset()
    spec = ModelSpec(
        "tiny", "raw",
        lambda p, n, s: SpyClassifier(lambda: TinyNet(n), n, s, max_epochs=3, patience=2, samples_per_subject=4,
                                      n_bags=3, device="cpu"),
        [{}], sample_weight=False, fit_groups=True,
    )
    for sp in ev.outer_splits(ds.subjects, 3, 1, seed=0):
        SEEN["train"].clear(), SEEN["val"].clear()
        r = ev.fit_predict(spec, ds, sp, seed=1)
        assert r["proba"].shape == (len(r["test_idx"]), 3)
        np.testing.assert_allclose(r["proba"].sum(1), 1, atol=1e-5)
        train, test = set(sp.train), set(sp.test)
        assert len(SEEN["val"]) == 3
        for tr, va in zip(SEEN["train"], SEEN["val"]):
            assert tr <= train and va <= train  # early stopping uses training subjects only
            assert not tr & test and not va & test
            assert not tr & va
        # the validation folds partition the training subjects
        assert set().union(*SEEN["val"]) == train
        assert sum(len(v) for v in SEEN["val"]) == len(train)


def test_torch_wrapper_requires_groups():
    ds = _raw_dataset()
    clf = deep.TorchEpochClassifier(lambda: TinyNet(3), 3, 0, max_epochs=1, device="cpu")
    with pytest.raises(ValueError):
        clf.fit(ds.inputs["raw"], ds.y)


def test_resample_epochs():
    sf = 250.0
    t = np.arange(2500) / sf
    x = np.sin(2 * np.pi * 10 * t)[None, None, :].astype(np.float32).repeat(2, 0)
    y = deep.resample_epochs(x, 250.0, 200.0)
    assert y.shape == (2, 1, 2000)
    t2 = np.arange(2000) / 200.0
    np.testing.assert_allclose(y[0, 0, 200:-200], np.sin(2 * np.pi * 10 * t2)[200:-200], atol=1e-2)
    assert deep.resample_epochs(x, 250.0, 125.0).shape == (2, 1, 1250)


def test_soft_vote_matches_harness_vote():
    subj = make_subjects((8, 7, 6))
    ds = make_dataset(subj, (4, 7))
    cols = np.arange(1, 5)
    m1 = ModelSpec("a", "feat", lr_factory, [{"C": 0.1}], columns=cols[:2], sample_weight_param="clf__sample_weight")
    m2 = ModelSpec("b", "feat", lr_factory, [{"C": 1.0}], columns=cols[2:], sample_weight_param="clf__sample_weight")
    vote = ModelSpec("v", "feat", None, members=[m1, m2], combine="vote")
    splits = ev.outer_splits(ds.subjects, 3, 2, seed=0)
    r1 = ev.run_cv(m1, ds, splits, n_jobs=1, seed=5)
    r2 = ev.run_cv(m2, ds, splits, n_jobs=1, seed=5)
    rv = ev.run_cv(vote, ds, splits, n_jobs=1, seed=5)
    sv = ev.soft_vote([r1["epoch"], r2["epoch"]], ds.class_names, ds.subjects)
    a = sv["epoch"].sort_values(["repeat", "epoch_index"])
    b = rv["epoch"].sort_values(["repeat", "epoch_index"])
    np.testing.assert_allclose(a[["p_AD", "p_CN", "p_FTD"]].to_numpy(), b[["p_AD", "p_CN", "p_FTD"]].to_numpy(), atol=1e-12)
    pa = ev.per_repeat_metrics(sv["subject"], ds.class_names)["balanced_accuracy"].to_numpy()
    pb = ev.per_repeat_metrics(rv["subject"], ds.class_names)["balanced_accuracy"].to_numpy()
    np.testing.assert_allclose(pa, pb)


def test_paired_comparison():
    subj = make_subjects((8, 7, 6))
    ds = make_dataset(subj, (4, 7))
    m1 = ModelSpec("a", "feat", lr_factory, [{"C": 1.0}], columns=np.arange(1, 5), sample_weight_param="clf__sample_weight")
    splits = ev.outer_splits(ds.subjects, 3, 2, seed=0)
    r = ev.run_cv(m1, ds, splits, n_jobs=1, seed=5)
    same = ev.paired_comparison(r["subject"], r["subject"], ds.class_names, n_boot=50)
    assert same["mean_diff"] == 0 and same["boot_ci95"] == [0.0, 0.0]
    # a model that is always right beats it
    perfect = r["subject"].copy()
    for i, c in enumerate(ds.class_names):
        perfect[f"p_{c}"] = (perfect["y"] == i).astype(float)
    d = ev.paired_comparison(perfect, r["subject"], ds.class_names, n_boot=50)
    assert d["mean_diff"] >= 0 and d["boot_ci95"][0] >= 0
    with pytest.raises(ValueError):
        ev.paired_comparison(r["subject"], r["subject"].assign(fold=0), ds.class_names, n_boot=5)


def test_biot_bipolar_input_is_reference_free_and_normalised():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 19, 2000)).astype(np.float32)
    B = deep.biot_input(X)
    assert B.shape == (4, 16, 2000)
    # adding a common signal to all channels (a reference change) does not change the input
    common = rng.normal(size=(4, 1, 2000)).astype(np.float32) * 50
    np.testing.assert_allclose(deep.biot_input(X + common), B, atol=1e-4)
    np.testing.assert_allclose(np.quantile(np.abs(B), 0.95, axis=-1), 1.0, atol=1e-3)
