"""Grouping / leakage guarantees of the evaluation harness."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from eegdementia import evaluation as ev
from eegdementia.config import BIDS_ROOT, LEGACY_TEST_SUBJECTS
from eegdementia.models import ModelSpec


def make_subjects(n_per_class=(12, 10, 8), seed=0):
    rng = np.random.default_rng(seed)
    ids, labels = [], []
    k = 0
    for c, n in enumerate(n_per_class):
        for _ in range(n):
            ids.append(f"sub-{k:03d}")
            labels.append(c)
            k += 1
    df = pd.DataFrame({"label": labels, "age": rng.normal(65, 8, len(ids))}, index=pd.Index(ids, name="subject"))
    return df


def make_dataset(subjects, epochs_per_subject=(3, 9), seed=0):
    rng = np.random.default_rng(seed)
    X, groups = [], []
    for i, s in enumerate(subjects.index):
        n = rng.integers(*epochs_per_subject)
        f = rng.normal(size=(n, 4)) + subjects.loc[s, "label"]
        # column 0 encodes the subject index so spies can see which subjects they get
        X.append(np.column_stack([np.full(n, i), f]))
        groups += [s] * n
    return ev.Dataset({"feat": np.concatenate(X)}, np.array(groups), subjects)


# ------------------------------------------------------------------ splits
@pytest.mark.parametrize("n_splits,n_repeats", [(5, 3), (4, 2)])
def test_outer_splits_partition_subjects(n_splits, n_repeats):
    subj = make_subjects()
    splits = ev.outer_splits(subj, n_splits, n_repeats, seed=1)
    assert len(splits) == n_splits * n_repeats
    for r in range(n_repeats):
        rs = [s for s in splits if s.repeat == r]
        tested = [t for s in rs for t in s.test]
        assert sorted(tested) == sorted(subj.index)  # each subject tested exactly once per repeat
        for s in rs:
            assert not set(s.train) & set(s.test)
            assert set(s.train) | set(s.test) == set(subj.index)
            # stratification: each class present in each test fold in proportion (+-1)
            counts = subj.loc[list(s.test), "label"].value_counts()
            for c, n in subj["label"].value_counts().items():
                assert abs(counts.get(c, 0) - n / n_splits) <= 1
    # different repeats give different partitions
    assert set(splits[0].test) != set(splits[n_splits].test)


def test_loso_and_fixed_split():
    subj = make_subjects()
    lo = ev.loso_splits(subj)
    assert len(lo) == len(subj)
    assert sorted(t for s in lo for t in s.test) == sorted(subj.index)
    for s in lo:
        assert len(s.test) == 1 and s.test[0] not in s.train
    fx = ev.fixed_split(subj, ["sub-000", "sub-015"])
    assert set(fx[0].test) == {"sub-000", "sub-015"} and len(fx[0].train) == len(subj) - 2


def test_check_split_detects_leak():
    with pytest.raises(AssertionError):
        ev.check_split(ev.Split(0, 0, ("a", "b"), ("b", "c")))


# ------------------------------------------------------------------ nested selection never sees test subjects
LOG = {"fit": [], "predict": []}


class Spy(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y, sample_weight=None):
        LOG["fit"].append(set(X[:, 0].astype(int)))
        # every epoch of a subject must carry that subject's label
        self.classes_ = np.unique(y)
        self.means_ = np.stack([X[y == c, 1:].mean(0) for c in self.classes_])
        return self

    def predict_proba(self, X):
        LOG["predict"].append(set(X[:, 0].astype(int)))
        d = -((X[:, None, 1:] - self.means_[None]) ** 2).sum(-1) * self.C
        e = np.exp(d - d.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


def test_nested_selection_never_sees_test_subjects():
    subj = make_subjects()
    ds = make_dataset(subj)
    code = {s: i for i, s in enumerate(subj.index)}
    spec = ModelSpec("spy", "feat", lambda p, n, s: Spy(**p), grid=[{"C": 0.1}, {"C": 1.0}, {"C": 10.0}])
    for split in ev.outer_splits(subj, 5, 2, seed=3):
        LOG["fit"].clear()
        LOG["predict"].clear()
        out = ev.fit_predict(spec, ds, split, inner_splits=4, seed=0)
        test_codes = {code[s] for s in split.test}
        train_codes = {code[s] for s in split.train}
        # 4 inner folds x 3 params + 1 final refit
        assert len(LOG["fit"]) == 4 * 3 + 1
        for seen in LOG["fit"]:
            assert not seen & test_codes, "a fit call saw test subjects"
            assert seen <= train_codes
        # all predict calls but the last are inner validation on training subjects only
        for seen in LOG["predict"][:-1]:
            assert seen <= train_codes and not seen & test_codes
        assert LOG["predict"][-1] == test_codes
        # the final refit uses all training subjects
        assert LOG["fit"][-1] == train_codes
        assert set(ds.groups[out["test_idx"]]) == set(split.test)


def test_run_cv_predicts_every_subject_once_per_repeat():
    subj = make_subjects()
    ds = make_dataset(subj)
    spec = ModelSpec("spy", "feat", lambda p, n, s: Spy(**p), grid=[{"C": 1.0}])
    res = ev.run_cv(spec, ds, ev.outer_splits(subj, 5, 2), n_jobs=1)
    sp = res["subject"]
    assert sp.groupby("repeat")["subject"].apply(lambda s: sorted(s) == sorted(subj.index)).all()
    ep = res["epoch"]
    assert len(ep) == 2 * len(ds.groups)
    # well-separated synthetic classes -> near-perfect, sanity check of the plumbing
    m = ev.per_repeat_metrics(sp, ds.class_names)
    assert (m["balanced_accuracy"] > 0.9).all()


def test_permuted_labels_are_constant_within_subject():
    subj = make_subjects()
    ds = make_dataset(subj)
    p = ds.with_permuted_labels(np.random.default_rng(0))
    for s in subj.index:
        assert len(np.unique(p.y[p.groups == s])) == 1
    assert sorted(p.subjects["label"]) == sorted(subj["label"])


# ------------------------------------------------------------------ weights / aggregation
def test_balanced_subject_weights():
    groups = np.array(["a"] * 10 + ["b"] * 2 + ["c"] * 5)
    y = np.array([0] * 12 + [1] * 5)
    w = ev.balanced_subject_weights(groups, y)
    tot = pd.Series(w).groupby(groups).sum()
    assert np.isclose(tot["a"], tot["b"])  # same class, equal subject weight
    assert np.isclose(tot["a"] + tot["b"], tot["c"])  # classes have equal total weight
    assert np.isclose(w.mean(), 1.0)


def test_aggregate_subjects_geometric_mean():
    P = np.array([[0.9, 0.05, 0.05], [0.5, 0.25, 0.25], [0.1, 0.8, 0.1]])
    g = np.array(["x", "x", "y"])
    ids, S = ev.aggregate_subjects(P, g)
    assert list(ids) == ["x", "y"]
    gm = np.sqrt(P[0] * P[1])
    np.testing.assert_allclose(S[0], gm / gm.sum())
    np.testing.assert_allclose(S.sum(1), 1)


def test_metrics_perfect_and_confusion():
    y = np.array([0, 0, 1, 1, 2, 2])
    P = np.eye(3)[y] * 0.9 + 0.1 / 3
    m = ev.metrics_from_proba(y, P, ["AD", "CN", "FTD"])
    assert m["balanced_accuracy"] == 1.0 and m["macro_auc"] == 1.0
    assert m["confusion"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


# ------------------------------------------------------------------ legacy split
@pytest.mark.skipif(not (BIDS_ROOT / "participants.tsv").exists(), reason="dataset not available")
def test_legacy_split_reproduced():
    import csv

    with open(BIDS_ROOT / "participants.tsv") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    df = pd.DataFrame(rows[1:], columns=rows[0])[["participant_id", "Group"]]
    train = df.sample(frac=0.8, random_state=42)
    test = df.drop(train.index)
    assert sorted(test["participant_id"]) == sorted(LEGACY_TEST_SUBJECTS)


def test_nested_model_selection_and_vote_never_see_test_subjects():
    subj = make_subjects()
    ds = make_dataset(subj)
    code = {s: i for i, s in enumerate(subj.index)}
    a = ModelSpec("a", "feat", lambda p, n, s: Spy(**p), grid=[{"C": 0.1}, {"C": 1.0}])
    b = ModelSpec("b", "feat", lambda p, n, s: Spy(**p), grid=[{"C": 10.0}])
    for combine, n_fits in (("select", 3 * 3 + 1), ("vote", 2 * 3 + 1 + 1)):
        spec = ModelSpec("ens", "feat", None, members=[a, b], combine=combine)
        split = ev.outer_splits(subj, 5, 1, seed=5)[2]
        LOG["fit"].clear()
        LOG["predict"].clear()
        out = ev.fit_predict(spec, ds, split, inner_splits=3, seed=0)
        test_codes = {code[s] for s in split.test}
        assert len(LOG["fit"]) == n_fits
        for seen in LOG["fit"]:
            assert not seen & test_codes
        if combine == "select":
            assert out["params"]["model"] in ("a", "b")
        np.testing.assert_allclose(out["proba"].sum(1), 1)


class GroupSpy(Spy):
    seen_groups = []

    def fit(self, X, y, sample_weight=None, groups=None):
        GroupSpy.seen_groups.append(set(groups))
        assert len(groups) == len(X)
        return super().fit(X, y, sample_weight)


def test_fit_groups_hook_only_passes_training_subjects():
    subj = make_subjects()
    ds = make_dataset(subj)
    spec = ModelSpec("g", "feat", lambda p, n, s: GroupSpy(**p), grid=[{"C": 0.1}, {"C": 1.0}], fit_groups=True)
    split = ev.outer_splits(subj, 5, 1, seed=9)[0]
    GroupSpy.seen_groups.clear()
    ev.fit_predict(spec, ds, split, inner_splits=3)
    assert len(GroupSpy.seen_groups) == 3 * 2 + 1
    for g in GroupSpy.seen_groups:
        assert not g & set(split.test)
    assert GroupSpy.seen_groups[-1] == set(split.train)
