"""Build tables (CSV + markdown) and figures from results/*.

    uv run python scripts/make_figures.py --best ensemble_vote --compare rbp_lr,spectral_lr,all_lgbm,ensemble_vote,nested_select
"""

import argparse
import json

import pandas as pd

from eegdementia import reporting as R
from eegdementia.config import RESULTS_DIR
from eegdementia.reporting import fmt, md_table

FIG = RESULTS_DIR / "figures"
TAB = RESULTS_DIR / "tables"
OLD = R.OLD_MODEL  # the April 2025 graph transformer on the legacy split (chunk level)
LIT = R.LITERATURE  # binary LOSO literature numbers (verified, see phase1_summary.md 6.4)

EEG_ONLY_EXCLUDE = {"age_lr", "age_sex_lr", "spectral_lr+age_sex", "all_lr+age_sex", "chance_prior"}


def load_task(task):
    d = RESULTS_DIR / task
    return {p.parent.name: json.loads(p.read_text()) for p in sorted(d.glob("*/summary.json"))} if d.exists() else {}


def table_multiclass(task, S):
    rows = []
    for m, s in S.items():
        rows.append({
            "model": m,
            "subject bal. acc. % (sd) [95% CI]": fmt(s, "subject", "balanced_accuracy"),
            "subject acc. %": fmt(s, "subject", "accuracy", ci=True),
            "subject macro-F1 %": fmt(s, "subject", "macro_f1", ci=False),
            "subject macro AUC": fmt(s, "subject", "macro_auc", pct=False, ci=True),
            "recall AD/CN/FTD %": "/".join(f"{100 * s['subject'][f'recall_{c}']['mean']:.0f}" for c in s["class_names"]),
            "epoch bal. acc. %": fmt(s, "epoch", "balanced_accuracy", ci=False),
            "epoch acc. %": fmt(s, "epoch", "accuracy", ci=False),
            "runtime (min)": f"{s['wall_time_s'] / 60:.1f}",
            "_key": s["subject"]["balanced_accuracy"]["mean"],
        })
    df = pd.DataFrame(rows).sort_values("_key", ascending=False).drop(columns="_key")
    df.to_csv(TAB / f"{task}.csv", index=False)
    R.write_text(TAB / f"{task}.md", md_table(df))
    return df


def table_binary(task, S):
    rows = []
    for m, s in S.items():
        pos = s["class_names"][1]
        neg = s["class_names"][0]
        rows.append({
            "model": m,
            "subject acc. % [95% CI]": fmt(s, "subject", "accuracy"),
            "subject bal. acc. %": fmt(s, "subject", "balanced_accuracy", ci=False),
            f"sens ({pos}) %": f"{100 * s['subject'][f'recall_{pos}']['mean']:.1f}",
            f"spec ({neg}) %": f"{100 * s['subject'][f'recall_{neg}']['mean']:.1f}",
            "subject AUC [95% CI]": fmt(s, "subject", "macro_auc", pct=False),
            "epoch acc. %": fmt(s, "epoch", "accuracy", ci=True),
            "epoch bal. acc. %": fmt(s, "epoch", "balanced_accuracy", ci=False),
            "runtime (min)": f"{s['wall_time_s'] / 60:.1f}",
            "_key": s["subject"]["accuracy"]["mean"],
        })
    df = pd.DataFrame(rows).sort_values("_key", ascending=False).drop(columns="_key")
    df.to_csv(TAB / f"{task}.csv", index=False)
    R.write_text(TAB / f"{task}.md", md_table(df))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="spectral_lr")
    ap.add_argument("--compare", default="rbp_lr,spectral_lr,all_lr,riemann_ts_lr,ensemble_vote")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    # ---------------- 3-class repeated CV
    S = load_task("cv3")
    table_multiclass("cv3", S)
    rows = []
    for m, s in S.items():
        if m == "chance_prior":
            continue
        d = s["subject"]["balanced_accuracy"]
        rows.append({"label": m, "mean": d["mean"], "ci": d["ci95"],
                     "color": R.BAR_SECONDARY if m in EEG_ONLY_EXCLUDE else R.BAR})
    R.comparison_chart(rows, FIG / "cv3_model_comparison.png",
                       "3-class AD/CN/FTD, 5-fold x 10 repeats, nested (light bars: age/sex confound models)",
                       refs=[("chance 0.333", 1 / 3)])
    best = S[a.best]
    cn = best["class_names"]
    R.confusion_figure(
        [("subject level", best["subject"]["confusion_summed"]), ("epoch level", best["epoch"]["confusion_summed"])],
        cn, FIG / f"cv3_confusion_{a.best}.png", f"{a.best}: confusion (row-normalised, pooled over 10 repeats)")
    preds = pd.read_csv(RESULTS_DIR / "cv3" / a.best / "predictions_subject.csv")
    R.roc_figure(preds, cn, FIG / f"cv3_roc_{a.best}.png", f"{a.best}: subject-level ROC (mean of 10 repeats)")
    comp = {m: S[m] for m in a.compare.split(",") if m in S}
    R.per_class_figure(comp, cn, FIG / "cv3_per_class.png", "Per-class subject-level metrics, mean and 95 % bootstrap CI")

    # ---------------- legacy split vs old model
    L = load_task("legacy")
    if L:
        table_multiclass("legacy", L)
        rows = [{"label": "old graph transformer (chunk level)", "mean": OLD["balanced_accuracy"], "color": R.MUTED}]
        for m, s in L.items():
            if m in EEG_ONLY_EXCLUDE:
                continue
            rows.append({"label": f"{m} (epoch level)", "mean": s["epoch"]["balanced_accuracy"]["mean"], "ci": s["epoch"]["balanced_accuracy"].get("ci95")})
        R.comparison_chart(rows, FIG / "legacy_split_comparison.png",
                           "Legacy 18-subject test split: epoch/chunk-level balanced accuracy", xlabel="Balanced accuracy",
                           refs=[("chance", 1 / 3)])

    L3 = load_task("loso3")
    if L3:
        table_multiclass("loso3", L3)

    # ---------------- binary LOSO vs literature
    for task in ("loso_ad_cn", "loso_ftd_cn"):
        B = load_task(task)
        if not B:
            continue
        table_binary(task, B)
        rows = []
        for m, s in B.items():
            rows.append({"label": f"{m} - subject", "mean": s["subject"]["accuracy"]["mean"], "ci": s["subject"]["accuracy"]["ci95"]})
            rows.append({"label": f"{m} - epoch", "mean": s["epoch"]["accuracy"]["mean"], "ci": s["epoch"]["accuracy"]["ci95"], "color": R.BAR_SECONDARY})
        for lab, v in LIT[task]:
            rows.append({"label": lab, "mean": v, "color": R.MUTED})
        R.comparison_chart(rows, FIG / f"{task}_vs_literature.png",
                           f"{ {'loso_ad_cn': 'AD vs CN', 'loso_ftd_cn': 'FTD vs CN'}[task]} leave-one-subject-out accuracy vs literature (bars: ours, dark = subject level, light = epoch level)",
                           xlabel="Accuracy", refs=[("chance", 0.5)], xlim=(0.3, 1.0))

    # ---------------- other binary CVs / sensitivity
    for task in [p.name for p in RESULTS_DIR.iterdir() if p.is_dir() and (p.name.startswith("cv_") or p.name.startswith("cv3__"))]:
        T = load_task(task)
        if task.startswith("cv_"):
            table_binary(task, T)
        else:
            table_multiclass(task, T)

    # ---------------- permutation tests
    for f in sorted((RESULTS_DIR / "permutation").glob("*.json")) if (RESULTS_DIR / "permutation").exists() else []:
        p = json.loads(f.read_text())
        R.null_histogram(p["null"], p["observed"], FIG / f"permutation_{f.stem}.png",
                         f"{f.stem}: label-permutation null ({p['n_perm']} perms), p = {p['p_value']:.3f}", 1 / 3)
    print("done")


if __name__ == "__main__":
    main()
