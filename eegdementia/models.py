"""Model zoo for the non-deep baselines.

A ``ModelSpec`` bundles: which input to use (feature columns, covariances, ...), a factory
building a fresh sklearn-compatible estimator for a given hyper-parameter dict, and the grid
searched by the *inner* CV. Every estimator is a full pipeline (imputation, scaling,
tangent-space mapping, ...) so that everything fitted lives inside the training fold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


# --------------------------------------------------------------------------------------
# Helpers / custom estimators
# --------------------------------------------------------------------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class ScoreProba(BaseEstimator, ClassifierMixin):
    """Wraps a margin classifier (SVM) and exposes softmax(decision_function) as scores.

    These are *not* calibrated probabilities (log-loss is not meaningful for them) but they
    are monotone in the margins, so argmax, subject aggregation and AUC are well defined.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        self.estimator_ = clone(self.estimator)
        kw = {}
        if sample_weight is not None:
            last = self.estimator_.steps[-1][0] if isinstance(self.estimator_, Pipeline) else None
            kw = {f"{last}__sample_weight": sample_weight} if last else {"sample_weight": sample_weight}
        self.estimator_.fit(X, y, **kw)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        d = self.estimator_.decision_function(X)
        if d.ndim == 1:
            d = np.stack([-d, d], 1) / 2
        return _softmax(d)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class BandTangentSpace(BaseEstimator, TransformerMixin):
    """Tangent-space mapping per frequency band, concatenated.

    X: (n, n_bands, c, c) SPD matrices. The reference point of every band is the Riemannian
    mean of the *training* matrices (fitted in ``fit`` only).
    """

    def __init__(self, bands=None, metric="riemann"):
        self.bands = bands
        self.metric = metric

    def fit(self, X, y=None, sample_weight=None):
        from pyriemann.tangentspace import TangentSpace

        self.bands_ = list(range(X.shape[1])) if self.bands is None else list(self.bands)
        self.ts_ = [TangentSpace(metric=self.metric).fit(X[:, b].astype(np.float64), sample_weight=sample_weight) for b in self.bands_]
        return self

    def transform(self, X):
        return np.concatenate([ts.transform(X[:, b].astype(np.float64)) for b, ts in zip(self.bands_, self.ts_)], axis=1)


class BandMDM(BaseEstimator, ClassifierMixin):
    """Minimum distance to (Riemannian) class means, summing squared distances over bands."""

    def __init__(self, bands=None, metric="riemann"):
        self.bands = bands
        self.metric = metric

    def fit(self, X, y, sample_weight=None):
        from pyriemann.classification import MDM

        self.bands_ = list(range(X.shape[1])) if self.bands is None else list(self.bands)
        self.mdms_ = [MDM(metric=self.metric).fit(X[:, b].astype(np.float64), y, sample_weight=sample_weight) for b in self.bands_]
        self.classes_ = self.mdms_[0].classes_
        return self

    def predict_proba(self, X):
        d2 = sum(m.transform(X[:, b].astype(np.float64)) ** 2 for b, m in zip(self.bands_, self.mdms_))
        # scale-free softmax: distances are divided by their per-sample mean
        return _softmax(-d2 / d2.mean(1, keepdims=True))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# --------------------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    input: str  # key into Dataset.inputs
    factory: Callable[[dict, int, int], object]  # (params, n_classes, seed) -> estimator
    grid: list[dict] = field(default_factory=lambda: [{}])
    columns: np.ndarray | None = None  # column subset of a 2-D input
    sample_weight: bool = True
    sample_weight_param: str = "sample_weight"
    max_epochs_per_subject: int | None = None
    extra_inputs: tuple[str, ...] = ()  # appended columns (e.g. "demo" for age/sex)
    description: str = ""
    members: list["ModelSpec"] | None = None  # ensemble / model-selection members
    combine: str = "vote"  # "vote" (soft voting) | "select" (nested choice of one member)

    def get_X(self, ds, idx):
        X = ds.inputs[self.input][idx]
        if self.columns is not None:
            X = X[:, self.columns]
        if self.extra_inputs:
            X = np.concatenate([X] + [ds.inputs[k][idx] for k in self.extra_inputs], axis=1)
        return X

    def make(self, params: dict, n_classes: int, seed: int):
        return self.factory(params, n_classes, seed)


def feature_columns(names: list[str], pattern: str) -> np.ndarray:
    rx = re.compile(pattern)
    cols = np.array([i for i, n in enumerate(names) if rx.search(n)], dtype=int)
    if len(cols) == 0:
        raise ValueError(f"no feature matches {pattern!r}")
    return cols


# Feature sets (regular expressions on feature names; see features.py for naming)
FEATURE_SETS = {
    # dataset authors' benchmark feature: relative band power per channel
    "rbp": r"^spec__relpow__",
    # all spectral features: band powers, ratios, peak/IAF, entropy, edge freqs, aperiodic
    "spectral": r"^(spec|aper)__",
    "spectral_region": r"^(specR|aperR)__",
    "complexity": r"^cplx__",
    "connectivity": r"^conn(R|G)?__",
    "all": r"^(spec|aper|cplx|conn|connR|connG)__",
}


# --------------------------------------------------------------------------------------
# Estimator factories
# --------------------------------------------------------------------------------------
def _pre(steps):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())] + steps)


def lr_factory(params, n_classes, seed):
    return _pre([("clf", LogisticRegression(C=params.get("C", 1.0), max_iter=3000, tol=1e-4))])


def lr_en_factory(params, n_classes, seed):
    return _pre(
        [
            (
                "clf",
                LogisticRegression(
                    C=params.get("C", 1.0),
                    l1_ratio=params.get("l1_ratio", 0.5),
                    solver="saga",
                    max_iter=3000,
                    tol=1e-3,
                    random_state=seed,
                ),
            )
        ]
    )


def linsvm_factory(params, n_classes, seed):
    return ScoreProba(_pre([("clf", LinearSVC(C=params.get("C", 1.0), dual="auto", max_iter=20000, random_state=seed))]))


def rbfsvm_factory(params, n_classes, seed):
    return ScoreProba(
        _pre([("clf", SVC(kernel="rbf", C=params.get("C", 1.0), gamma=params.get("gamma", "scale"), decision_function_shape="ovr"))])
    )


def rf_factory(params, n_classes, seed):
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=params.get("n_estimators", 400),
                    min_samples_leaf=params.get("min_samples_leaf", 5),
                    max_features=params.get("max_features", "sqrt"),
                    n_jobs=1,
                    random_state=seed,
                ),
            ),
        ]
    )


def lgbm_factory(params, n_classes, seed):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=params.get("n_estimators", 300),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 15),
        min_child_samples=params.get("min_child_samples", 50),
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=params.get("colsample_bytree", 0.3),
        reg_lambda=params.get("reg_lambda", 1.0),
        n_jobs=1,
        random_state=seed,
        verbose=-1,
    )


def ts_lr_factory(params, n_classes, seed):
    return Pipeline(
        [
            ("ts", BandTangentSpace(bands=params.get("bands"))),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=params.get("C", 1.0), max_iter=3000)),
        ]
    )


def mdm_factory(params, n_classes, seed):
    return BandMDM(bands=params.get("bands"))


def dummy_factory(params, n_classes, seed):
    return DummyClassifier(strategy="prior")


def _pipe_sw(name):
    """sample_weight routing name for a Pipeline whose final step is called ``clf``."""
    return "clf__sample_weight"


C_GRID = [{"C": c} for c in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]


def build_specs(feature_names: list[str]) -> dict[str, ModelSpec]:
    """All phase-1 model specifications, keyed by name."""
    cols = {k: feature_columns(feature_names, p) for k, p in FEATURE_SETS.items()}
    S: dict[str, ModelSpec] = {}
    S["chance_prior"] = ModelSpec(
        "chance_prior", "feat", dummy_factory, sample_weight=False, description="DummyClassifier(prior): chance baseline"
    )
    for fs in ("rbp", "spectral", "spectral_region", "complexity", "connectivity", "all"):
        S[f"{fs}_lr"] = ModelSpec(
            f"{fs}_lr",
            "feat",
            lr_factory,
            C_GRID,
            columns=cols[fs],
            sample_weight_param="clf__sample_weight",
            description=f"{fs} features ({len(cols[fs])}) + L2 logistic regression",
        )
    S["all_enet"] = ModelSpec(
        "all_enet",
        "feat",
        lr_en_factory,
        [{"C": c, "l1_ratio": r} for c in (1e-3, 1e-2, 1e-1) for r in (0.2, 0.8)],
        columns=cols["all"],
        sample_weight_param="clf__sample_weight",
        max_epochs_per_subject=40,
        description="all features + elastic-net logistic regression (saga)",
    )
    S["all_linsvm"] = ModelSpec(
        "all_linsvm",
        "feat",
        linsvm_factory,
        [{"C": c} for c in (1e-5, 1e-4, 1e-3, 1e-2)],
        columns=cols["all"],
        description="all features + linear SVM (one-vs-rest margins)",
    )
    S["all_rbfsvm"] = ModelSpec(
        "all_rbfsvm",
        "feat",
        rbfsvm_factory,
        [{"C": c, "gamma": g} for c in (0.1, 1.0, 10.0) for g in ("scale",)],
        columns=cols["all"],
        max_epochs_per_subject=40,
        description="all features + RBF SVM (<=40 random epochs per training subject)",
    )
    S["spectral_rbfsvm"] = ModelSpec(
        "spectral_rbfsvm",
        "feat",
        rbfsvm_factory,
        [{"C": c, "gamma": "scale"} for c in (0.1, 1.0, 10.0)],
        columns=cols["spectral"],
        max_epochs_per_subject=40,
        description="spectral features + RBF SVM (<=40 random epochs per training subject)",
    )
    S["all_rf"] = ModelSpec(
        "all_rf",
        "feat",
        rf_factory,
        [{"min_samples_leaf": m} for m in (5, 50)],
        columns=cols["all"],
        sample_weight_param="clf__sample_weight",
        description="all features + random forest (400 trees)",
    )
    S["all_lgbm"] = ModelSpec(
        "all_lgbm",
        "feat",
        lgbm_factory,
        [{"num_leaves": 7, "n_estimators": 200}, {"num_leaves": 15, "n_estimators": 400, "learning_rate": 0.03}],
        columns=cols["all"],
        description="all features + LightGBM",
    )
    S["riemann_ts_lr"] = ModelSpec(
        "riemann_ts_lr",
        "cov",
        ts_lr_factory,
        C_GRID,
        sample_weight_param="clf__sample_weight",
        description="band-wise (5 bands + broadband) OAS covariances -> tangent space -> L2 LR",
    )
    S["riemann_mdm"] = ModelSpec(
        "riemann_mdm", "cov", mdm_factory, description="band-wise covariances -> MDM (summed squared Riemannian distances)"
    )
    return S
