"""Interpretability for phase 1.

A. Descriptive (no model): subject-level group means and effect sizes (Hedges' g) of
   relative band power, peak frequency and aperiodic exponent, as scalp maps, plus
   Kruskal-Wallis tests with Benjamini-Hochberg FDR over all per-channel spectral features.
B. Model-based, held-out: grouped permutation importance of a feature model inside the
   outer CV (features of a group are shuffled across the *test* subjects' epochs; the drop
   in subject-level one-vs-rest AUC per class and in balanced accuracy is recorded).
C. Standardised multinomial LR coefficients (spectral_lr refit on all subjects, C = most
   frequently chosen value in the CV) as scalp maps. Descriptive of the fitted model only.

    uv run python scripts/interpret.py --model spectral_lr --repeats 3
"""

import argparse
import json
import os
import re

from eegdementia.utils import be_nice

be_nice()  # CPU-only, 1 BLAS thread per process, below-normal priority

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import kruskal  # noqa: E402

from eegdementia import evaluation as ev  # noqa: E402
from eegdementia import experiments as E  # noqa: E402
from eegdementia import reporting as R  # noqa: E402
from eegdementia.config import BANDS, CHANNELS, CLASS_NAMES, REGIONS, RESULTS_DIR  # noqa: E402

OUT = RESULTS_DIR / "interpretability"


def hedges_g(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2))
    g = (a.mean(0) - b.mean(0)) / s
    return g * (1 - 3 / (4 * (na + nb) - 9))


def bh_fdr(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def descriptive(ds):
    names = ds.feature_names
    subj = ds.subject_mean()
    X = subj.inputs["feat"]
    lab = subj.subjects["diagnosis"].to_numpy()
    col = {n: i for i, n in enumerate(names)}
    grp = {c: X[lab == c] for c in CLASS_NAMES}

    # group-mean maps of relative power
    means = {c: {b: grp[c][:, [col[f"spec__relpow__{b}__{ch}"] for ch in CHANNELS]].mean(0) for b in BANDS} for c in CLASS_NAMES}
    R.topomap_grid(means, CHANNELS, OUT / "relpow_group_means.png", "Relative band power, group means (average reference)",
                   cmap="viridis", symmetric=False, units="fraction of 0.5-45 Hz power")
    contrasts = [("AD", "CN"), ("FTD", "CN"), ("AD", "FTD")]
    measures = {f"relpow {b}": [f"spec__relpow__{b}__{ch}" for ch in CHANNELS] for b in BANDS}
    measures["peak freq"] = [f"spec__peak_freq__all__{ch}" for ch in CHANNELS]
    measures["aperiodic exp."] = [f"aper__exponent__all__{ch}" for ch in CHANNELS]
    eff = {f"{a} - {b}": {m: hedges_g(grp[a][:, [col[n] for n in cols]], grp[b][:, [col[n] for n in cols]]) for m, cols in measures.items()} for a, b in contrasts}
    R.topomap_grid(eff, CHANNELS, OUT / "effect_sizes_topomaps.png", "Effect sizes (Hedges' g) between groups, subject means",
                   units="Hedges' g")

    # univariate tests over per-channel spectral/aperiodic/complexity features
    rows = []
    for n in names:
        if not re.match(r"^(spec|aper|cplx)(R)?__", n):
            continue
        i = col[n]
        h, p = kruskal(*[grp[c][:, i] for c in CLASS_NAMES])
        rows.append({"feature": n, "H": h, "p": p,
                     **{f"mean_{c}": float(grp[c][:, i].mean()) for c in CLASS_NAMES},
                     "g_AD_CN": float(hedges_g(grp["AD"][:, [i]], grp["CN"][:, [i]])[0]),
                     "g_FTD_CN": float(hedges_g(grp["FTD"][:, [i]], grp["CN"][:, [i]])[0]),
                     "g_AD_FTD": float(hedges_g(grp["AD"][:, [i]], grp["FTD"][:, [i]])[0])})
    df = pd.DataFrame(rows)
    df["q_fdr"] = bh_fdr(df["p"])
    df.sort_values("p").to_csv(OUT / "univariate_kruskal.csv", index=False, float_format="%.5g")
    return df


def feature_groups(names):
    """Groups of columns: by measure x band (all channels) and by region (all measures)."""
    groups = {}
    for i, n in enumerate(names):
        fam, meas, band, loc = n.split("__")
        fam = fam.rstrip("RG") if fam in ("specR", "aperR", "cplxR", "connR", "connG") else fam
        # per-channel features and their regional-mean copies go into the same group, so a
        # shuffled measure/region cannot be recovered from a redundant copy
        key = f"{fam}:{meas}" + (f":{band}" if band != "all" else "")
        groups.setdefault("measure|" + key, []).append(i)
        region = loc if loc in REGIONS else next((r for r, chs in REGIONS.items() if loc in chs), None)
        if region:
            groups.setdefault(f"region|{region}", []).append(i)
    return {k: np.array(v) for k, v in groups.items()}


def _perm_importance_split(spec, ds, split, groups_local, n_draws, seed):
    r = ev.fit_predict(spec, ds, split, inner_splits=5, seed=seed, return_estimator=True)
    est, bspec, te = r["estimator"], r["spec"], r["test_idx"]
    Xte = bspec.get_X(ds, te)
    g = ds.groups[te]
    y_s = ds.subjects.loc[pd.unique(g), "label"]

    def score(P):
        sids, SP = ev.aggregate_subjects(P, g)
        m = ev.metrics_from_proba(y_s.loc[sids].to_numpy(), SP, ds.class_names)
        return np.array([m["balanced_accuracy"], m["macro_auc"]] + [m[f"auc_{c}"] for c in ds.class_names])

    base = score(est.predict_proba(Xte))
    rng = np.random.default_rng(seed)
    out = {}
    for k, cols in groups_local.items():
        drops = []
        for _ in range(n_draws):
            Xp = Xte.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, cols] = Xte[perm][:, cols]
            drops.append(base - score(est.predict_proba(Xp)))
        out[k] = np.mean(drops, 0)
    return out


def permutation_importance(ds, spec, repeats, n_draws=5, n_jobs=12):
    names = [ds.feature_names[i] for i in spec.columns]
    groups = feature_groups(names)
    splits = ev.outer_splits(ds.subjects, 5, repeats, seed=E.OUTER_SEED)
    res = Parallel(n_jobs=n_jobs)(
        delayed(_perm_importance_split)(spec, ds, sp, groups, n_draws, 100 + i) for i, sp in enumerate(splits)
    )
    cols = ["drop_bal_acc", "drop_macro_auc"] + [f"drop_auc_{c}" for c in ds.class_names]
    rows = []
    for k in groups:
        arr = np.array([r[k] for r in res])
        rows.append({"group": k, "n_features": len(groups[k]), **{c: arr[:, j].mean() for j, c in enumerate(cols)},
                     **{c + "_sd": arr[:, j].std(ddof=1) for j, c in enumerate(cols)}})
    return pd.DataFrame(rows).sort_values("drop_macro_auc", ascending=False)


def importance_figure(df, out, title):
    import matplotlib.pyplot as plt

    top = df[df.group.str.startswith("measure|")].head(15).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    y = np.arange(len(top))
    w = 0.26
    for j, c in enumerate(CLASS_NAMES):
        ax.barh(y + (j - 1) * w, top[f"drop_auc_{c}"], height=w, color=R.CLASS_COLORS[c], label=f"{c} vs rest", edgecolor="white", linewidth=0.5)
    ax.set_yticks(y, [g.split("|")[1] for g in top.group])
    ax.axvline(0, color=R.MUTED, lw=0.8)
    ax.set_xlabel("Drop in held-out subject-level AUC when shuffled")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_title("Feature groups (measure x band, all channels)", loc="left", fontsize=9)
    reg = df[df.group.str.startswith("region|")].iloc[::-1]
    ax = axes[1]
    y = np.arange(len(reg))
    for j, c in enumerate(CLASS_NAMES):
        ax.barh(y + (j - 1) * w, reg[f"drop_auc_{c}"], height=w, color=R.CLASS_COLORS[c], edgecolor="white", linewidth=0.5)
    ax.set_yticks(y, [g.split("|")[1] for g in reg.group])
    ax.axvline(0, color=R.MUTED, lw=0.8)
    ax.set_xlabel("Drop in AUC")
    ax.grid(axis="y", visible=False)
    ax.set_title("Regions (all measures)", loc="left", fontsize=9)
    fig.suptitle(title, x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def coefficient_maps(ds, specs, model="spectral_lr"):
    spec = specs[model]
    summ = json.loads((RESULTS_DIR / "cv3" / model / "summary.json").read_text())
    params = json.loads(max(summ["chosen_params_counts"], key=summ["chosen_params_counts"].get))
    idx = np.arange(len(ds.groups))
    X = spec.get_X(ds, idx)
    est = spec.make(params, 3, 0)
    est.fit(X, ds.y, clf__sample_weight=ev.balanced_subject_weights(ds.groups, ds.y))
    coef = est.named_steps["clf"].coef_  # (3, n_feat), standardised features
    names = [ds.feature_names[i] for i in spec.columns]
    col = {n: i for i, n in enumerate(names)}
    vals = {c: {b: coef[k, [col[f"spec__relpow__{b}__{ch}"] for ch in CHANNELS]] for b in BANDS} for k, c in enumerate(CLASS_NAMES)}
    for k, c in enumerate(CLASS_NAMES):
        vals[c]["peak freq"] = coef[k, [col[f"spec__peak_freq__all__{ch}"] for ch in CHANNELS]]
        vals[c]["aper. exp."] = coef[k, [col[f"aper__exponent__all__{ch}"] for ch in CHANNELS]]
    R.topomap_grid(vals, CHANNELS, OUT / f"coefficients_{model}.png",
                   f"{model}: standardised multinomial LR coefficients (refit on all subjects, C={params['C']})",
                   units="coefficient")
    pd.DataFrame(coef.T, index=names, columns=CLASS_NAMES).to_csv(OUT / f"coefficients_{model}.csv", float_format="%.5g")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="spectral_lr")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--skip-perm", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ds = E.build_dataset()
    specs = E.all_specs(ds.feature_names)
    uni = descriptive(ds)
    print(uni.sort_values("p").head(20)[["feature", "H", "p", "q_fdr", "g_AD_CN", "g_FTD_CN", "g_AD_FTD"]].to_string())
    print("features with q<0.05:", int((uni.q_fdr < 0.05).sum()), "of", len(uni))
    coefficient_maps(ds, specs, "spectral_lr")
    if not a.skip_perm:
        imp = permutation_importance(ds, specs[a.model], a.repeats, n_jobs=a.n_jobs)
        imp.to_csv(OUT / f"permutation_importance_{a.model}.csv", index=False, float_format="%.5f")
        importance_figure(imp, OUT / f"permutation_importance_{a.model}.png",
                          f"{a.model}: grouped permutation importance on held-out subjects ({a.repeats}x5-fold)")
        print(imp.head(20).to_string())
