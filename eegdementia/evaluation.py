"""Leak-free, subject-level evaluation harness.

Design
------
* **Splits are defined over subjects, never over epochs.** ``outer_splits`` stratifies the
  88 subjects by diagnosis (repeated stratified K-fold on the subject table, which is exactly
  stratified *group* K-fold with one group per subject). Epochs follow their subject.
* **Nested selection.** Every hyper-parameter choice is made by an inner stratified K-fold
  over the *training subjects of the current outer fold only* (``fit_predict``). Every fitted
  object (imputer, scaler, tangent-space reference point, model, class/subject weights) is
  fitted inside that training set. Test subjects are only ever passed to ``predict_proba``.
* **Subject-level aggregation.** Epoch probabilities of a subject are combined by the mean
  log-probability (a normalised geometric mean), which is the primary unit of evaluation.
* **Weights.** Training epochs are weighted so every training subject contributes the same
  total weight and every class the same total weight (subjects have different lengths and
  the classes are imbalanced 36/29/23).

A model is anything with ``fit(X, y, sample_weight=None)`` and ``predict_proba(X)``; see
``models.ModelSpec``. Deep models (phase 2) only need to follow the same interface.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

log = logging.getLogger(__name__)

EPS = 1e-7


# --------------------------------------------------------------------------------------
# Data container
# --------------------------------------------------------------------------------------
@dataclass
class Dataset:
    """Epoch-level data for modelling.

    ``inputs`` maps an input name (e.g. "feat", "cov") to an array whose first axis is the
    epoch axis. ``subjects`` is the subject table (index = subject id, must contain ``label``;
    ``age``/``sex`` optional). ``groups`` gives the subject id of every epoch.
    """

    inputs: dict[str, np.ndarray]
    groups: np.ndarray
    subjects: pd.DataFrame
    feature_names: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=lambda: ["AD", "CN", "FTD"])

    def __post_init__(self):
        n = len(self.groups)
        for k, v in self.inputs.items():
            assert len(v) == n, f"input {k} has {len(v)} rows, expected {n}"
        missing = set(np.unique(self.groups)) - set(self.subjects.index)
        assert not missing, f"epochs of unknown subjects: {sorted(missing)[:5]}"
        self.subjects = self.subjects.loc[[s for s in self.subjects.index if s in set(self.groups)]]
        lab = self.subjects["label"]
        self.y = lab.loc[self.groups].to_numpy()

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def restrict(self, subjects: list[str], relabel: dict[int, int] | None = None, class_names=None) -> "Dataset":
        """Subset of subjects (e.g. AD+CN for a binary task), optionally re-coding labels."""
        m = np.isin(self.groups, subjects)
        st = self.subjects.loc[[s for s in self.subjects.index if s in set(subjects)]].copy()
        if relabel is not None:
            st["label"] = st["label"].map(relabel).astype(int)
        return Dataset(
            {k: v[m] for k, v in self.inputs.items()},
            self.groups[m],
            st,
            self.feature_names,
            class_names or self.class_names,
        )

    def with_permuted_labels(self, rng: np.random.Generator) -> "Dataset":
        st = self.subjects.copy()
        st["label"] = rng.permutation(st["label"].to_numpy())
        return Dataset(self.inputs, self.groups, st, self.feature_names, self.class_names)

    def subject_mean(self) -> "Dataset":
        """One row per subject: mean of every input over that subject's epochs."""
        subs = list(self.subjects.index)
        inputs = {}
        for k, v in self.inputs.items():
            inputs[k] = np.stack([np.nanmean(v[self.groups == s], axis=0) for s in subs])
        return Dataset(inputs, np.array(subs), self.subjects, self.feature_names, self.class_names)


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Split:
    repeat: int
    fold: int
    train: tuple[str, ...]
    test: tuple[str, ...]


def outer_splits(subjects: pd.DataFrame, n_splits: int = 5, n_repeats: int = 10, seed: int = 0) -> list[Split]:
    """Repeated stratified K-fold over *subjects* (stratified by ``label``)."""
    ids = np.asarray(subjects.index)
    y = subjects["label"].to_numpy()
    out = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1000 * r)
        for k, (tr, te) in enumerate(skf.split(ids, y)):
            out.append(Split(r, k, tuple(ids[tr]), tuple(ids[te])))
    return out


def loso_splits(subjects: pd.DataFrame) -> list[Split]:
    ids = list(subjects.index)
    return [Split(0, i, tuple(s for s in ids if s != t), (t,)) for i, t in enumerate(ids)]


def fixed_split(subjects: pd.DataFrame, test: list[str]) -> list[Split]:
    ids = list(subjects.index)
    te = [s for s in ids if s in set(test)]
    assert len(te) == len(test), "some test subjects are missing"
    return [Split(0, 0, tuple(s for s in ids if s not in set(test)), tuple(te))]


def check_split(split: Split) -> None:
    tr, te = set(split.train), set(split.test)
    if tr & te:
        raise AssertionError(f"subject leakage: {sorted(tr & te)}")


# --------------------------------------------------------------------------------------
# Weights, aggregation, metrics
# --------------------------------------------------------------------------------------
def balanced_subject_weights(groups: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each subject gets equal total weight within its class; each class equal total weight."""
    subj, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    subj_label = pd.Series(y).groupby(groups).first()
    n_per_class = subj_label.value_counts()
    cls_w = {c: 1.0 / n for c, n in n_per_class.items()}
    w = w * np.array([cls_w[c] for c in y])
    return w * (len(w) / w.sum())


def aggregate_subjects(proba: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean log-probability per subject, renormalised -> (subject ids, probs)."""
    logp = np.log(np.clip(proba, EPS, 1.0))
    df = pd.DataFrame(logp)
    df["g"] = groups
    m = df.groupby("g", sort=False).mean()
    lp = m.to_numpy()
    lp = lp - lp.max(1, keepdims=True)
    p = np.exp(lp)
    p /= p.sum(1, keepdims=True)
    return m.index.to_numpy(), p


def confusion(y: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(y * n + pred, minlength=n * n).reshape(n, n)


def metrics_from_proba(y: np.ndarray, proba: np.ndarray, class_names: list[str]) -> dict:
    n = len(class_names)
    pred = proba.argmax(1)
    cm = confusion(y, pred, n)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    predicted = cm.sum(0)
    recall = np.divide(tp, support, out=np.full(n, np.nan), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(n), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n), where=(precision + recall) > 0)
    tn = cm.sum() - support - predicted + tp
    spec = tn / (cm.sum() - support)
    out = {
        "balanced_accuracy": float(np.nanmean(recall)),
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "log_loss": float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], EPS, 1)))),
        "n": int(len(y)),
    }
    for i, c in enumerate(class_names):
        out[f"recall_{c}"] = float(recall[i])
        out[f"specificity_{c}"] = float(spec[i])
        out[f"f1_{c}"] = float(f1[i])
    aucs = []
    for i, c in enumerate(class_names):
        yi = (y == i).astype(int)
        if 0 < yi.sum() < len(yi):
            a = roc_auc_score(yi, proba[:, i])
            out[f"auc_{c}"] = float(a)
            aucs.append(a)
        else:
            out[f"auc_{c}"] = float("nan")
    out["macro_auc"] = float(np.mean(aucs)) if n > 2 else out[f"auc_{class_names[1]}"]
    out["confusion"] = cm.tolist()
    return out


def _selection_score(y: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Inner-CV criterion: subject-level balanced accuracy, ties broken by macro OvR AUC."""
    n = proba.shape[1]
    pred = proba.argmax(1)
    cm = confusion(y, pred, n)
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    aucs = []
    for i in range(n):
        yi = (y == i).astype(int)
        if 0 < yi.sum() < len(yi):
            aucs.append(roc_auc_score(yi, proba[:, i]))
    return float(rec.mean()), float(np.mean(aucs)) if aucs else 0.0


# --------------------------------------------------------------------------------------
# Nested fit / predict for one outer split
# --------------------------------------------------------------------------------------
def _subsample_mask(groups: np.ndarray, max_per_subject: int | None, seed: int) -> np.ndarray:
    if not max_per_subject:
        return np.ones(len(groups), bool)
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(groups), bool)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(idx) > max_per_subject:
            idx = rng.choice(idx, max_per_subject, replace=False)
        keep[idx] = True
    return keep


def _fit_and_predict(spec, params: dict, ds: Dataset, tr_idx: np.ndarray, te_idx: np.ndarray, seed: int):
    """Fit ``spec`` with ``params`` on epochs ``tr_idx`` and return probabilities for ``te_idx``."""
    sub = _subsample_mask(ds.groups[tr_idx], spec.max_epochs_per_subject, seed)
    tr_idx = tr_idx[sub]
    Xtr = spec.get_X(ds, tr_idx)
    ytr = ds.y[tr_idx]
    w = balanced_subject_weights(ds.groups[tr_idx], ytr)
    est = spec.make(params, n_classes=ds.n_classes, seed=seed)
    kw = {spec.sample_weight_param: w} if spec.sample_weight else {}
    if getattr(spec, "fit_groups", False):
        # e.g. deep models that carve an early-stopping split out of the *training*
        # subjects; they receive the subject id of every training epoch.
        kw["groups"] = ds.groups[tr_idx]
    est.fit(Xtr, ytr, **kw)
    proba = est.predict_proba(spec.get_X(ds, te_idx))
    # make sure columns follow label order 0..n-1
    classes = getattr(est, "classes_", np.arange(ds.n_classes))
    if len(classes) != ds.n_classes or np.any(np.asarray(classes) != np.arange(ds.n_classes)):
        full = np.full((len(te_idx), ds.n_classes), EPS)
        full[:, np.asarray(classes, int)] = proba
        proba = full / full.sum(1, keepdims=True)
    return proba, est


def fit_predict(
    spec,
    ds: Dataset,
    split: Split,
    inner_splits: int = 5,
    seed: int = 0,
    return_estimator: bool = False,
) -> dict:
    """Nested evaluation of one outer split. Returns test epoch probabilities and the choice.

    * plain spec: its ``grid`` is searched by inner CV on the training subjects;
    * ``spec.members`` with ``combine="vote"``: every member is tuned independently (own
      inner CV) and the members' epoch log-probabilities are averaged (soft voting);
    * ``spec.members`` with ``combine="select"``: the inner CV chooses among *all*
      (member, hyper-parameter) candidates, i.e. model selection itself is nested.
    """
    check_split(split)
    members = getattr(spec, "members", None)
    combine = getattr(spec, "combine", "vote")
    if members and combine == "vote":
        parts = [fit_predict(m, ds, split, inner_splits, seed) for m in members]
        logp = np.mean([np.log(np.clip(p["proba"], EPS, 1)) for p in parts], axis=0)
        p = np.exp(logp - logp.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return {
            "proba": p,
            "test_idx": parts[0]["test_idx"],
            "params": {m.name: q["params"] for m, q in zip(members, parts)},
            "inner": None,
        }

    train_set, test_set = set(split.train), set(split.test)
    tr_idx = np.flatnonzero(np.fromiter((g in train_set for g in ds.groups), bool, len(ds.groups)))
    te_idx = np.flatnonzero(np.fromiter((g in test_set for g in ds.groups), bool, len(ds.groups)))
    assert not set(ds.groups[tr_idx]) & set(ds.groups[te_idx])

    cand_specs = members if (members and combine == "select") else [spec]
    candidates = [(m, p) for m in cand_specs for p in (m.grid or [{}])]
    inner_scores = None
    if len(candidates) > 1:
        tr_subj = ds.subjects.loc[list(split.train)]
        inner = outer_splits(tr_subj, n_splits=inner_splits, n_repeats=1, seed=seed + 7)
        inner_scores = []
        oof = {i: {} for i in range(len(candidates))}
        for isp in inner:
            itr_set = set(isp.train)
            m_tr = np.fromiter((g in itr_set for g in ds.groups[tr_idx]), bool, len(tr_idx))
            itr, iva = tr_idx[m_tr], tr_idx[~m_tr]
            # guard: inner validation subjects are training subjects of the outer split only
            assert set(ds.groups[iva]) <= train_set and not set(ds.groups[iva]) & test_set
            for ci, (cs, params) in enumerate(candidates):
                p, _ = _fit_and_predict(cs, params, ds, itr, iva, seed)
                sids, sp = aggregate_subjects(p, ds.groups[iva])
                oof[ci].update(dict(zip(sids, sp)))
        ys = tr_subj["label"]
        for ci, (cs, params) in enumerate(candidates):
            sids = list(oof[ci])
            P = np.stack([oof[ci][s] for s in sids])
            ba, auc = _selection_score(ys.loc[sids].to_numpy(), P)
            inner_scores.append({"model": cs.name, "params": params, "inner_bal_acc": ba, "inner_macro_auc": auc})
        best = max(
            range(len(candidates)),
            key=lambda i: (round(inner_scores[i]["inner_bal_acc"], 9), inner_scores[i]["inner_macro_auc"]),
        )
        best_spec, params = candidates[best]
    else:
        best_spec, params = candidates[0]
    proba, est = _fit_and_predict(best_spec, params, ds, tr_idx, te_idx, seed)
    chosen = {"model": best_spec.name, **params} if best_spec is not spec else params
    out = {"proba": proba, "test_idx": te_idx, "params": chosen, "inner": inner_scores}
    if return_estimator:
        out["estimator"] = est
        out["spec"] = best_spec
    return out


# --------------------------------------------------------------------------------------
# Running an experiment over many splits
# --------------------------------------------------------------------------------------
def run_cv(
    spec,
    ds: Dataset,
    splits: list[Split],
    inner_splits: int = 5,
    n_jobs: int = 12,
    seed: int = 0,
    verbose: int = 0,
) -> dict:
    """Run nested evaluation over ``splits`` in parallel. Returns epoch- and subject-level predictions."""
    from joblib import Parallel, delayed

    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(fit_predict)(spec, ds, sp, inner_splits, seed + 17 * sp.repeat + sp.fold) for sp in splits
    )
    runtime = time.time() - t0
    ep_rows, subj_rows = [], []
    for sp, r in zip(splits, res):
        te = r["test_idx"]
        g = ds.groups[te]
        ep_rows.append(
            pd.DataFrame(
                {
                    "repeat": sp.repeat,
                    "fold": sp.fold,
                    "subject": g,
                    "epoch_index": te,
                    "y": ds.y[te],
                    **{f"p_{c}": r["proba"][:, i] for i, c in enumerate(ds.class_names)},
                }
            )
        )
        sids, sp_ = aggregate_subjects(r["proba"], g)
        subj_rows.append(
            pd.DataFrame(
                {
                    "repeat": sp.repeat,
                    "fold": sp.fold,
                    "subject": sids,
                    "y": ds.subjects.loc[sids, "label"].to_numpy(),
                    **{f"p_{c}": sp_[:, i] for i, c in enumerate(ds.class_names)},
                    "n_epochs": pd.Series(g).value_counts().loc[sids].to_numpy(),
                    "params": json.dumps(r["params"], default=str),
                }
            )
        )
    return {
        "epoch": pd.concat(ep_rows, ignore_index=True),
        "subject": pd.concat(subj_rows, ignore_index=True),
        "runtime_s": runtime,
        "inner": [r["inner"] for r in res],
        "class_names": ds.class_names,
    }


def _proba_cols(df: pd.DataFrame, class_names) -> np.ndarray:
    return df[[f"p_{c}" for c in class_names]].to_numpy()


def per_repeat_metrics(preds: pd.DataFrame, class_names) -> pd.DataFrame:
    rows = []
    for r, d in preds.groupby("repeat"):
        m = metrics_from_proba(d["y"].to_numpy(), _proba_cols(d, class_names), class_names)
        m.pop("confusion")
        rows.append({"repeat": r, **m})
    return pd.DataFrame(rows)


def per_fold_metrics(preds: pd.DataFrame, class_names) -> pd.DataFrame:
    rows = []
    for (r, k), d in preds.groupby(["repeat", "fold"]):
        m = metrics_from_proba(d["y"].to_numpy(), _proba_cols(d, class_names), class_names)
        m.pop("confusion")
        rows.append({"repeat": r, "fold": k, **m})
    return pd.DataFrame(rows)


def bootstrap_ci(
    preds: pd.DataFrame,
    class_names,
    metrics=("balanced_accuracy", "accuracy", "macro_f1", "macro_auc"),
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """95 % percentile CIs from a (class-stratified) bootstrap over *subjects*.

    For each resample of subjects the metric is computed within every repeat and averaged
    across repeats, so the CI reflects subject sampling uncertainty of the repeat-averaged
    estimate. For epoch-level predictions each drawn subject brings all of its epochs
    (cluster bootstrap).
    """
    rng = np.random.default_rng(seed)
    subj_label = preds.groupby("subject")["y"].first()
    by_class = [subj_label.index[subj_label == c].to_numpy() for c in sorted(subj_label.unique())]
    repeats = sorted(preds["repeat"].unique())
    n = len(class_names)
    # precompute per repeat, per subject rows
    per_rep = {}
    for r in repeats:
        d = preds[preds["repeat"] == r]
        idx_by_subj = d.groupby("subject").indices
        per_rep[r] = (d["y"].to_numpy(), _proba_cols(d, class_names), idx_by_subj)
    vals = {m: [] for m in metrics}
    for _ in range(n_boot):
        draw = np.concatenate([rng.choice(s, len(s), replace=True) for s in by_class])
        acc = {m: [] for m in metrics}
        for r in repeats:
            y, P, ix = per_rep[r]
            rows = np.concatenate([ix[s] for s in draw])
            yy, PP = y[rows], P[rows]
            pred = PP.argmax(1)
            cm = confusion(yy, pred, n)
            tp = np.diag(cm).astype(float)
            sup, prd = cm.sum(1), cm.sum(0)
            rec = tp / np.maximum(sup, 1)
            prec = np.divide(tp, prd, out=np.zeros(n), where=prd > 0)
            f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n), where=(prec + rec) > 0)
            if "balanced_accuracy" in acc:
                acc["balanced_accuracy"].append(rec.mean())
            if "accuracy" in acc:
                acc["accuracy"].append(tp.sum() / cm.sum())
            if "macro_f1" in acc:
                acc["macro_f1"].append(f1.mean())
            aucs = None
            if "macro_auc" in acc or any(m.startswith("auc_") for m in acc):
                aucs = [roc_auc_score(yy == i, PP[:, i]) for i in range(n)]
            if "macro_auc" in acc:
                acc["macro_auc"].append(np.mean(aucs) if n > 2 else aucs[1])
            spec_ = (cm.sum() - sup - prd + tp) / np.maximum(cm.sum() - sup, 1)
            for i, c in enumerate(class_names):
                if f"recall_{c}" in acc:
                    acc[f"recall_{c}"].append(rec[i])
                if f"specificity_{c}" in acc:
                    acc[f"specificity_{c}"].append(spec_[i])
                if f"auc_{c}" in acc:
                    acc[f"auc_{c}"].append(aucs[i])
        for m in metrics:
            vals[m].append(np.mean(acc[m]))
    return {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in vals.items()}


def summarise(result: dict, n_boot: int = 2000, epoch_boot: int = 500, seed: int = 0) -> dict:
    cn = result["class_names"]
    out = {"runtime_s": result["runtime_s"]}
    for level in ("subject", "epoch"):
        pr = per_repeat_metrics(result[level], cn)
        desc = {}
        for col in pr.columns:
            if col in ("repeat", "n"):
                continue
            desc[col] = {"mean": float(pr[col].mean()), "sd": float(pr[col].std(ddof=1)) if len(pr) > 1 else 0.0}
        nb = n_boot if level == "subject" else epoch_boot
        if level == "subject":
            mets = ("balanced_accuracy", "accuracy", "macro_f1", "macro_auc") + tuple(
                f"{k}_{c}" for c in cn for k in ("recall", "specificity", "auc")
            )
        else:
            mets = ("balanced_accuracy", "accuracy", "macro_f1")
        ci = bootstrap_ci(result[level], cn, metrics=mets, n_boot=nb, seed=seed)
        for m, c in ci.items():
            desc[m]["ci95"] = c
        # pooled confusion (summed over repeats)
        d = result[level]
        cm = confusion(d["y"].to_numpy(), _proba_cols(d, cn).argmax(1), len(cn))
        desc["confusion_summed"] = cm.tolist()
        desc["n_repeats"] = int(pr.shape[0])
        out[level] = desc
    return out


def permutation_test(
    spec,
    ds: Dataset,
    observed: float,
    n_perm: int = 100,
    n_splits: int = 5,
    n_repeats: int = 1,
    inner_splits: int = 5,
    n_jobs: int = 12,
    seed: int = 0,
    metric: str = "balanced_accuracy",
) -> dict:
    """Subject-label permutation test of the *whole* nested pipeline.

    Labels are permuted across subjects (all epochs of a subject keep one label), then the
    complete nested CV is rerun. ``observed`` must be computed with the same ``n_repeats``.
    """
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)
    perm_ds = [ds.with_permuted_labels(rng) for _ in range(n_perm)]
    jobs = []
    for pi, pds in enumerate(perm_ds):
        for sp in outer_splits(pds.subjects, n_splits, n_repeats, seed=seed + 1 + pi):
            jobs.append((pi, sp))
    res = Parallel(n_jobs=n_jobs)(
        delayed(fit_predict)(spec, perm_ds[pi], sp, inner_splits, seed + pi) for pi, sp in jobs
    )
    scores = []
    for pi in range(n_perm):
        pds = perm_ds[pi]
        rep_scores = {}
        for (pj, sp), r in zip(jobs, res):
            if pj != pi:
                continue
            sids, P = aggregate_subjects(r["proba"], pds.groups[r["test_idx"]])
            rep_scores.setdefault(sp.repeat, []).append((pds.subjects.loc[sids, "label"].to_numpy(), P))
        vals = []
        for parts in rep_scores.values():
            y = np.concatenate([p[0] for p in parts])
            P = np.concatenate([p[1] for p in parts])
            vals.append(metrics_from_proba(y, P, pds.class_names)[metric])
        scores.append(float(np.mean(vals)))
    scores = np.array(scores)
    return {
        "metric": metric,
        "observed": float(observed),
        "null_mean": float(scores.mean()),
        "null_sd": float(scores.std(ddof=1)),
        "null_95th": float(np.percentile(scores, 95)),
        "p_value": float((1 + np.sum(scores >= observed)) / (1 + n_perm)),
        "n_perm": int(n_perm),
        "null": scores.tolist(),
    }
