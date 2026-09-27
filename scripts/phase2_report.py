"""Phase-2 combinations (pre-registered soft votes), paired comparisons, tables and figures.

    uv run python scripts/phase2_report.py [--posthoc-members labram_lr,shallow]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eegdementia import evaluation as ev
from eegdementia import experiments as E
from eegdementia import reporting as R
from eegdementia.config import CACHE_DIR, RESULTS_DIR
from eegdementia.reporting import fmt, md_table

P2 = RESULTS_DIR / "phase2"
FIG = P2 / "figures"
TAB = P2 / "tables"
PHASE1_MEMBERS = ["spectral_lr", "riemann_ts_lr", "all_lgbm"]
PHASE1_REFS = ["ensemble_vote", "all_lgbm", "spectral_lr", "all_lr", "nested_select", "rbp_lr"]
PHASE2_ORDER = ["cbramod_lr", "labram_lr", "cbramod_spectral_lr", "eegnet", "shallow", "cbramod_ft",
                "ensemble_vote+cbramod_lr", "ensemble_vote+cbramod_ft", "ensemble_vote+labram_lr", "ensemble_vote+shallow",
                "biot_lr"]


def pred_dir(ds, task):
    return CACHE_DIR / "predictions" / ds.cfg.prep.key() / ds.cfg.epoch.key() / task


def summaries(root: Path, task: str) -> dict:
    d = root / task
    return {p.parent.name: json.loads(p.read_text()) for p in sorted(d.glob("*/summary.json"))} if d.exists() else {}


def subject_preds(root: Path, task: str, model: str) -> pd.DataFrame | None:
    f = root / task / model / "predictions_subject.csv"
    return pd.read_csv(f) if f.exists() else None


# --------------------------------------------------------------------------------------
def make_votes(ds, posthoc: list[str]):
    votes = {"ensemble_vote+cbramod_lr": "cbramod_lr", "ensemble_vote+cbramod_ft": "cbramod_ft"}
    for m in posthoc:
        votes[f"ensemble_vote+{m}"] = m
    for task in ("cv3", "legacy"):
        tds, splits = E.task_data_and_splits(task, ds, n_repeats=10)
        pdir = pred_dir(ds, task)
        for name, extra in votes.items():
            files = [pdir / f"{m}.parquet" for m in PHASE1_MEMBERS + [extra]]
            if not all(f.exists() for f in files):
                print(f"[vote] {task}/{name}: missing members, skipped")
                continue
            mem = [pd.read_parquet(f) for f in files]
            reps = sorted(set.intersection(*[set(m["repeat"]) for m in mem]))
            mem = [m[m["repeat"].isin(reps)] for m in mem]
            res = ev.soft_vote(mem, tds.class_names, tds.subjects)
            desc = (f"soft vote (mean epoch log-prob) of {', '.join(PHASE1_MEMBERS + [extra])}; each member tuned by its own "
                    f"inner CV; computed from the members' saved outer-fold predictions"
                    + (" [POST HOC member choice]" if extra in posthoc else ""))
            n_splits = len([s for s in splits if s.repeat in reps])
            E.save_result(task, name, desc, ds, tds, n_splits, res, out_root=P2,
                          extra_meta={"phase": 2, "n_repeats_run": len(reps), "wall_time_s": 0.0,
                                      "posthoc": extra in posthoc})
            print(f"[vote] {task}/{name}: {len(reps)} repeats")


# --------------------------------------------------------------------------------------
def table(task, S2, S1, refs, binary=False):
    rows = []
    for src, S, names in (("phase 2", S2, [m for m in PHASE2_ORDER if m in S2] + [m for m in S2 if m not in PHASE2_ORDER]),
                          ("phase 1", S1, [m for m in refs if m in S1])):
        for m in names:
            s = S[m]
            cn = s["class_names"]
            tag = " (phase 1)" if src == "phase 1" else (" [post hoc]" if s.get("posthoc") else "")
            row = {"model": m + tag,
                   "repeats": s["subject"].get("n_repeats", 1)}
            if binary:
                pos, neg = cn[1], cn[0]
                row.update({
                    "subject bal. acc. % (sd) [95% CI]": fmt(s, "subject", "balanced_accuracy"),
                    "subject acc. %": fmt(s, "subject", "accuracy", ci=False),
                    f"sens {pos} / spec {neg} %": f"{100 * s['subject'][f'recall_{pos}']['mean']:.0f} / {100 * s['subject'][f'recall_{neg}']['mean']:.0f}",
                    "subject AUC [95% CI]": fmt(s, "subject", "macro_auc", pct=False),
                    "epoch bal. acc. %": fmt(s, "epoch", "balanced_accuracy", ci=False),
                })
            else:
                row.update({
                    "subject bal. acc. % (sd) [95% CI]": fmt(s, "subject", "balanced_accuracy"),
                    "subject acc. %": fmt(s, "subject", "accuracy", ci=False),
                    "subject macro-F1 %": fmt(s, "subject", "macro_f1", ci=False),
                    "subject macro AUC [95% CI]": fmt(s, "subject", "macro_auc", pct=False),
                    "recall AD/CN/FTD %": "/".join(f"{100 * s['subject'][f'recall_{c}']['mean']:.0f}" for c in cn),
                    "epoch bal. acc. %": fmt(s, "epoch", "balanced_accuracy", ci=False),
                    "epoch acc. %": fmt(s, "epoch", "accuracy", ci=False),
                })
            rt = s.get("runtime_s", s.get("wall_time_s", 0.0))
            row["compute (min)"] = f"{rt / 60:.1f}"
            rows.append(row)
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{task}.csv", index=False, lineterminator="\n")
    R.write_text(TAB / f"{task}.md", md_table(df))
    return df


def paired(task, S2, comparators):
    rows, full = [], {}
    for m in [x for x in PHASE2_ORDER if x in S2] + [x for x in S2 if x not in PHASE2_ORDER]:
        a = subject_preds(P2, task, m)
        cn = S2[m]["class_names"]
        for c in comparators:
            b = subject_preds(RESULTS_DIR, task, c)
            if b is None:
                continue
            r = ev.paired_comparison(a, b, cn, n_boot=2000, seed=E.OUTER_SEED)
            full[f"{m} vs {c}"] = r
            rows.append({
                "model": m + (" [post hoc]" if S2[m].get("posthoc") else ""), "vs": c, "repeats": r["n_repeats"],
                "mean Δ bal. acc. (pp)": f"{100 * r['mean_diff']:+.1f}",
                "sd of per-repeat Δ (pp)": f"{100 * r['sd_per_repeat_diff']:.1f}",
                "repeats better / equal": f"{r['n_repeats_a_better']} / {r['n_repeats_equal']} of {r['n_repeats']}",
                "paired bootstrap 95% CI (pp)": f"[{100 * r['boot_ci95'][0]:+.1f}, {100 * r['boot_ci95'][1]:+.1f}]",
                "P(Δ ≤ 0)": f"{r['boot_p_a_le_b']:.3f}",
                "verdict": "better" if r["boot_ci95"][0] > 0 else ("worse" if r["boot_ci95"][1] < 0 else "not distinguishable"),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(TAB / f"paired_{task}.csv", index=False, lineterminator="\n")
        R.write_text(TAB / f"paired_{task}.md", md_table(df))
        R.write_text(TAB / f"paired_{task}.json", json.dumps(full, indent=1))
    return df, full


# --------------------------------------------------------------------------------------
def fig_comparison(task, S2, S1, refs, title, extra_refs=()):
    FIG.mkdir(parents=True, exist_ok=True)
    rows = []
    for m, s in S2.items():
        d = s["subject"]["balanced_accuracy"]
        rows.append({"label": m + (" [post hoc]" if s.get("posthoc") else ""), "mean": d["mean"], "ci": d.get("ci95"), "color": R.BAR})
    for m in refs:
        if m in S1:
            d = S1[m]["subject"]["balanced_accuracy"]
            rows.append({"label": f"{m} (phase 1)", "mean": d["mean"], "ci": d.get("ci95"), "color": R.MUTED})
    R.comparison_chart(rows, FIG / f"{task}_model_comparison.png", title,
                       refs=[("chance", 1 / len(next(iter(S2.values()))["class_names"]))] + list(extra_refs))


POSTHOC: set = set()


def fig_paired(task, full, comparator, title):
    items = [(k.split(" vs ")[0] + (" [post hoc]" if k.split(" vs ")[0] in POSTHOC else ""), v)
             for k, v in full.items() if k.endswith(f" vs {comparator}")]
    if not items:
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(items) + 1.2))
    for i, (_m, r) in enumerate(items):
        pr = 100 * np.array(r["per_repeat_diff"])
        ax.scatter(pr, np.full(len(pr), i) + np.linspace(-0.12, 0.12, len(pr)), s=10, color=R.BAR_SECONDARY, zorder=2)
        lo, hi = (100 * np.array(r["boot_ci95"])).tolist()
        ax.plot([lo, hi], [i, i], color=R.INK2, lw=1.4, zorder=3)
        ax.plot([100 * r["mean_diff"]], [i], "o", color=R.BAR, ms=6, zorder=4)
    ax.axvline(0, color=R.MUTED, lw=1, ls="--")
    ax.set_yticks(range(len(items)), [m for m, _ in items])
    ax.set_xlabel(f"Δ subject balanced accuracy vs {comparator} (pp)\n(dots: per repeat; line: paired subject-bootstrap 95% CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=10)
    fig.savefig(FIG / f"{task}_paired_vs_{comparator}.png")
    plt.close(fig)


def fig_per_class(S2, S1, models, out):
    cls = ["AD", "CN", "FTD"]
    models = [(m, S2.get(m) or S1.get(m)) for m in models if (m in S2 or m in S1)]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    w = 0.8 / len(models)
    for j, (m, s) in enumerate(models):
        vals = [s["subject"][f"recall_{c}"]["mean"] for c in cls]
        ax.bar(np.arange(3) + (j - (len(models) - 1) / 2) * w, vals, width=w, label=m,
               color=plt.cm.tab10(j % 10), edgecolor="white")
    ax.set_xticks(range(3), [f"{c} recall" for c in cls])
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax.set_title("Per-class subject-level recall (mean over repeats)", loc="left", fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posthoc-members", default="", help="comma list of post-hoc 4th vote members")
    ap.add_argument("--no-votes", action="store_true")
    args = ap.parse_args()
    if not args.no_votes:
        ds = E.build_dataset()
        make_votes(ds, [m for m in args.posthoc_members.split(",") if m])

    out = {}
    # 3-class cv3
    S2, S1 = summaries(P2, "cv3"), summaries(RESULTS_DIR, "cv3")
    POSTHOC.update(m for m, s in S2.items() if s.get("posthoc"))
    if S2:
        out["cv3"] = table("cv3", S2, S1, PHASE1_REFS)
        _, full = paired("cv3", S2, ["ensemble_vote", "spectral_lr"])
        fig_comparison("cv3", S2, S1, ["ensemble_vote", "spectral_lr", "nested_select", "all_lgbm"],
                       "3-class, 5-fold x 10 repeats, subject-level (phase 2 blue, phase 1 grey)")
        fig_paired("cv3", full, "ensemble_vote", "Paired differences vs phase-1 ensemble_vote (same folds)")
        fig_per_class(S2, S1, [m for m in PHASE2_ORDER if m in S2] + ["ensemble_vote"], FIG / "cv3_per_class.png")
        cms = [(m, np.array(S2[m]["subject"]["confusion_summed"])) for m in PHASE2_ORDER if m in S2][:4]
        if cms:
            R.confusion_figure(cms, ["AD", "CN", "FTD"], FIG / "cv3_confusion.png", "Subject-level confusion (summed over repeats), row-normalised")
    # AD vs FTD
    S2b, S1b = summaries(P2, "cv_ad_ftd"), summaries(RESULTS_DIR, "cv_ad_ftd")
    if S2b:
        out["cv_ad_ftd"] = table("cv_ad_ftd", S2b, S1b, ["rbp_lr", "spectral_lr"], binary=True)
        paired("cv_ad_ftd", S2b, ["rbp_lr", "spectral_lr"])
        fig_comparison("cv_ad_ftd", S2b, S1b, ["rbp_lr", "spectral_lr"], "AD vs FTD, 5-fold x 10 repeats, subject-level")
    # legacy
    S2l, S1l = summaries(P2, "legacy"), summaries(RESULTS_DIR, "legacy")
    if S2l:
        out["legacy"] = table("legacy", S2l, S1l, ["ensemble_vote", "all_lgbm", "spectral_lr", "all_lr", "nested_select", "rbp_lr"])
    # permutation test(s) of phase-2 pipelines (scripts/permutation_test.py --phase2)
    for f in sorted((P2 / "permutation").glob("*.json")):
        p = json.loads(f.read_text())
        FIG.mkdir(parents=True, exist_ok=True)
        R.null_histogram(p["null"], p["observed"], FIG / f"permutation_{f.stem}.png",
                         f"{f.stem}: label-permutation null ({p['n_perm']} perms), p = {p['p_value']:.3f}", 1 / 3)
    for k, v in out.items():
        print(f"\n## {k}\n")
        print(md_table(v))


if __name__ == "__main__":
    main()
