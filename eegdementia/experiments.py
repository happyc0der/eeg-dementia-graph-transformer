"""Experiment definitions (tasks x models) and result persistence."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluation as ev
from . import models as M
from .config import CACHE_DIR, LEGACY_TEST_SUBJECTS, NAME_TO_INT, RESULTS_DIR, PipelineConfig
from .data import load_participants
from .features import FeatureStore

OUTER_SEED = 2026


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
def build_dataset(cfg: PipelineConfig = PipelineConfig()) -> ev.Dataset:
    subs = load_participants()
    fs = FeatureStore(list(subs.index), cfg)
    age = subs.loc[fs.subject, "age"].to_numpy(float)
    male = (subs.loc[fs.subject, "sex"] == "M").to_numpy(float)
    ds = ev.Dataset(
        {"feat": fs.X, "cov": fs.cov, "demo": np.column_stack([age, male]).astype(np.float32)},
        fs.subject,
        subs,
        fs.names,
    )
    ds.epoch_stats = fs.epoch_stats
    ds.cfg = cfg
    return ds


# --------------------------------------------------------------------------------------
# Specs (all phase-1 models, incl. confound checks and ensembles)
# --------------------------------------------------------------------------------------
def all_specs(feature_names: list[str]) -> dict[str, M.ModelSpec]:
    S = M.build_specs(feature_names)
    # --- confound checks (secondary; never part of the EEG-only headline)
    S["age_lr"] = M.ModelSpec(
        "age_lr", "demo", M.lr_factory, M.C_GRID, columns=np.array([0]), sample_weight_param="clf__sample_weight",
        description="age only + LR (confound baseline)",
    )
    S["age_sex_lr"] = M.ModelSpec(
        "age_sex_lr", "demo", M.lr_factory, M.C_GRID, columns=np.array([0, 1]), sample_weight_param="clf__sample_weight",
        description="age + sex only + LR (confound baseline)",
    )
    for base in ("spectral_lr", "all_lr"):
        b = S[base]
        S[f"{base}+age_sex"] = M.ModelSpec(
            f"{base}+age_sex", b.input, b.factory, b.grid, columns=b.columns, extra_inputs=("demo",),
            sample_weight_param=b.sample_weight_param, description=b.description + " + age + sex (secondary)",
        )
    # --- ensembles. Members are fixed a priori (three different views of the data), not
    # picked after looking at outer-CV results.
    S["ensemble_vote"] = M.ModelSpec(
        "ensemble_vote", "feat", None, members=[S["spectral_lr"], S["riemann_ts_lr"], S["all_lgbm"]], combine="vote",
        description="soft vote (mean log-prob) of spectral_lr, riemann_ts_lr, all_lgbm; each tuned by its own inner CV",
    )
    S["nested_select"] = M.ModelSpec(
        "nested_select", "feat", None,
        members=[S["rbp_lr"], S["spectral_lr"], S["all_lr"], S["riemann_ts_lr"], S["all_lgbm"]],
        combine="select",
        description="inner CV picks model family AND hyper-parameters among rbp_lr, spectral_lr, all_lr, riemann_ts_lr, all_lgbm",
    )
    return S


# --------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------
def task_data_and_splits(task: str, ds: ev.Dataset, n_repeats: int = 10, n_splits: int = 5):
    if task == "cv3":
        return ds, ev.outer_splits(ds.subjects, n_splits, n_repeats, seed=OUTER_SEED)
    if task == "legacy":
        return ds, ev.fixed_split(ds.subjects, LEGACY_TEST_SUBJECTS)
    if task in ("loso_ad_cn", "loso_ftd_cn", "cv_ad_cn", "cv_ftd_cn", "cv_ad_ftd"):
        pos = {"ad_cn": "AD", "ftd_cn": "FTD", "ad_ftd": "AD"}[task.split("_", 1)[1]]
        neg = {"ad_cn": "CN", "ftd_cn": "CN", "ad_ftd": "FTD"}[task.split("_", 1)[1]]
        subs = ds.subjects[ds.subjects["diagnosis"].isin([pos, neg])]
        sub = ds.restrict(list(subs.index), relabel={NAME_TO_INT[neg]: 0, NAME_TO_INT[pos]: 1}, class_names=[neg, pos])
        if task.startswith("loso"):
            return sub, ev.loso_splits(sub.subjects)
        return sub, ev.outer_splits(sub.subjects, n_splits, n_repeats, seed=OUTER_SEED)
    raise ValueError(task)


# --------------------------------------------------------------------------------------
# Running + saving
# --------------------------------------------------------------------------------------
def run_and_save(
    task: str,
    model: str,
    ds: ev.Dataset,
    specs: dict,
    n_jobs: int = 14,
    n_repeats: int = 10,
    inner_splits: int = 5,
    n_boot: int = 2000,
    out_root: Path = RESULTS_DIR,
    tag: str = "",
) -> dict:
    spec = specs[model]
    tds, splits = task_data_and_splits(task, ds, n_repeats=n_repeats)
    if task.startswith("loso") or task == "legacy":
        inner_splits = inner_splits  # same inner CV on the training subjects
    t0 = time.time()
    res = ev.run_cv(spec, tds, splits, inner_splits=inner_splits, n_jobs=n_jobs, seed=OUTER_SEED)
    summ = ev.summarise(res, n_boot=n_boot, seed=OUTER_SEED)
    outdir = Path(out_root) / (task + (f"__{tag}" if tag else "")) / model
    outdir.mkdir(parents=True, exist_ok=True)
    cn = res["class_names"]
    res["subject"].to_csv(outdir / "predictions_subject.csv", index=False, float_format="%.5f")
    ev.per_repeat_metrics(res["subject"], cn).to_csv(outdir / "metrics_subject_per_repeat.csv", index=False, float_format="%.5f")
    ev.per_repeat_metrics(res["epoch"], cn).to_csv(outdir / "metrics_epoch_per_repeat.csv", index=False, float_format="%.5f")
    if not task.startswith("loso"):
        ev.per_fold_metrics(res["subject"], cn).to_csv(outdir / "metrics_subject_per_fold.csv", index=False, float_format="%.5f")
        ev.per_fold_metrics(res["epoch"], cn).to_csv(outdir / "metrics_epoch_per_fold.csv", index=False, float_format="%.5f")
    # epoch-level predictions are large -> cache dir (not committed)
    pdir = CACHE_DIR / "predictions" / ds.cfg.prep.key() / ds.cfg.epoch.key() / (task + (f"__{tag}" if tag else ""))
    pdir.mkdir(parents=True, exist_ok=True)
    res["epoch"].to_parquet(pdir / f"{model}.parquet", index=False)
    chosen = res["subject"].drop_duplicates(["repeat", "fold"])["params"].value_counts().to_dict()
    meta = {
        "task": task,
        "model": model,
        "description": spec.description,
        "class_names": cn,
        "n_subjects": int(len(tds.subjects)),
        "n_epochs": int(len(tds.groups)),
        "n_outer_splits": len(splits),
        "inner_splits": inner_splits,
        "selection_criterion": "pooled inner-OOF subject-level balanced accuracy, ties -> macro OvR AUC",
        "aggregation": "subject probability = softmax(mean epoch log-probability)",
        "pipeline_config": ds.cfg.to_dict(),
        "chosen_params_counts": chosen,
        "wall_time_s": time.time() - t0,
        "n_jobs": n_jobs,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    summ = {**meta, **summ}
    (outdir / "summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def collect_summaries(task_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(task_dir).glob("*/summary.json")):
        s = json.loads(f.read_text())
        row = {"model": s["model"], "description": s["description"], "runtime_s": s["runtime_s"]}
        for level in ("subject", "epoch"):
            for m in ("balanced_accuracy", "accuracy", "macro_f1", "macro_auc"):
                if m in s[level]:
                    row[f"{level}_{m}_mean"] = s[level][m]["mean"]
                    row[f"{level}_{m}_sd"] = s[level][m]["sd"]
                    if "ci95" in s[level][m]:
                        row[f"{level}_{m}_ci_lo"], row[f"{level}_{m}_ci_hi"] = s[level][m]["ci95"]
            for c in s["class_names"]:
                for k in ("recall", "specificity", "auc"):
                    key = f"{k}_{c}"
                    if key in s[level]:
                        row[f"{level}_{key}_mean"] = s[level][key]["mean"]
        rows.append(row)
    return pd.DataFrame(rows)
