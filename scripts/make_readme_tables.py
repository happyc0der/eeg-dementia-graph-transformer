"""Regenerate every results table in README.md from the files in results/.

Each table sits between ``<!-- BEGIN GENERATED: name -->`` and ``<!-- END GENERATED: name -->``
markers; everything between the markers is overwritten. Nothing is recomputed: the numbers are
read from results/**/summary.json, results/phase2/tables/paired_*.json, results/**/permutation/*.json,
results/interpretability/*.csv and results/tables/epochs__*.csv. Needs neither the data nor a GPU.

    uv run python scripts/make_readme_tables.py           # rewrite README.md
    uv run python scripts/make_readme_tables.py --check   # exit 1 if README.md is out of date
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from eegdementia.config import REPO_ROOT, RESULTS_DIR
from eegdementia.reporting import LITERATURE, OLD_MODEL

README = REPO_ROOT / "README.md"
P2 = RESULTS_DIR / "phase2"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def summary(task: str, model: str, phase: int = 1) -> dict:
    root = RESULTS_DIR if phase == 1 else P2
    return json.loads((root / task / model / "summary.json").read_text(encoding="utf-8"))


def pct(d: dict, sd: bool = True, ci: bool = True) -> str:
    out = f"{100 * d['mean']:.1f}"
    if sd and d.get("sd"):
        out += f" ± {100 * d['sd']:.1f}"
    if ci and "ci95" in d:
        out += f" [{100 * d['ci95'][0]:.1f}, {100 * d['ci95'][1]:.1f}]"
    return out


def num(d: dict, ci: bool = True) -> str:
    out = f"{d['mean']:.3f}"
    if ci and "ci95" in d:
        out += f" [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
    return out


def recalls(s: dict) -> str:
    return " / ".join(f"{100 * s['subject'][f'recall_{c}']['mean']:.0f}" for c in s["class_names"])


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    sep = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(sep) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def link(path: Path, text: str | None = None) -> str:
    rel = path.relative_to(REPO_ROOT).as_posix()
    return f"[{text or rel}]({rel})"


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------
MAIN_ROWS = [
    # (phase, model, label, what)
    (1, "ensemble_vote", "**Soft vote** (phase 1)", "spectral LR + Riemannian LR + LightGBM, pre-specified"),
    (1, "all_lgbm", "LightGBM, all features", "1,213 features"),
    (1, "spectral_lr", "**Spectral LR**", "logistic regression, 399 spectral/aperiodic features"),
    (1, "all_lr", "LR, all features", "1,213 features"),
    (1, "rbp_lr", "LR, relative band power", "the dataset authors' feature (95)"),
    (1, "nested_select", "Nested model selection", "inner CV picks family + hyper-parameters"),
    (2, "labram_lr", "**LaBraM frozen** + LR", "foundation model, 3,800-d embedding"),
    (2, "shallow", "**ShallowFBCSPNet**", "CNN trained from scratch"),
    (2, "cbramod_spectral_lr", "CBraMod frozen + spectral + LR", "hybrid features"),
    (2, "cbramod_lr", "CBraMod frozen + LR", "pre-registered primary foundation model"),
    (2, "cbramod_ft", "CBraMod fine-tuned", "foundation model, end to end"),
    (2, "eegnet", "EEGNet", "CNN trained from scratch"),
    (2, "biot_lr", "BIOT frozen + LR", "exploratory"),
    (2, "ensemble_vote+cbramod_lr", "Soft vote + CBraMod LR", "pre-registered primary hybrid"),
    (2, "ensemble_vote+shallow", "Soft vote + ShallowFBCSPNet", "**post hoc** (member chosen after seeing cv3)"),
    (1, "age_sex_lr", "Age + sex only", "confound baseline"),
    (1, "chance_prior", "Chance (class prior)", "DummyClassifier"),
]


def main_3class() -> str:
    paired = json.loads((P2 / "tables" / "paired_cv3.json").read_text(encoding="utf-8"))
    rows = []
    for phase, m, label, what in MAIN_ROWS:
        s = summary("cv3", m, phase)
        sb = s["subject"]
        p = paired.get(f"{m} vs ensemble_vote")
        delta = "reference" if m == "ensemble_vote" else "–"
        if p:
            lo, hi = p["boot_ci95"]
            delta = f"{100 * p['mean_diff']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        rows.append([label, what, str(phase), pct(sb["balanced_accuracy"]), num(sb["macro_auc"], ci=False), recalls(s), delta])
    return table(
        ["Model", "What", "Phase", "Subject bal. acc. %, mean ± SD [95 % CI]", "Macro AUC", "Recall AD / CN / FTD %",
         "Δ vs soft vote, pp [paired 95 % CI]"],
        rows, "llrrrrr",
    )


def binary_loso() -> str:
    rows = []
    for task, name in (("loso_ad_cn", "AD vs CN"), ("loso_ftd_cn", "FTD vs CN")):
        for m, label in (("all_lr", "LR, all features (ours)"), ("rbp_lr", "LR, relative band power (ours)"), ("age_lr", "Age only (ours)")):
            s = summary(task, m)
            rows.append([name, label, pct(s["subject"]["accuracy"], sd=False), pct(s["subject"]["balanced_accuracy"], sd=False, ci=False),
                         num(s["subject"]["macro_auc"], ci=False), pct(s["epoch"]["accuracy"], sd=False, ci=False)])
        for label, v in LITERATURE[task]:
            rows.append([name, label, "", "", "", f"{100 * v:.1f}"])
    return table(["Task", "Model", "Subject acc. % [95 % CI]", "Subject bal. acc. %", "Subject AUC", "Epoch acc. %"], rows, "llrrrr")


def binary_cv() -> str:
    rows = []
    spec = [
        ("cv_ad_cn", "AD vs CN", [(1, "spectral_lr"), (1, "rbp_lr"), (1, "age_lr")]),
        ("cv_ftd_cn", "FTD vs CN", [(1, "spectral_lr"), (1, "rbp_lr"), (1, "age_lr")]),
        ("cv_ad_ftd", "AD vs FTD", [(1, "rbp_lr"), (1, "spectral_lr"), (2, "shallow"), (2, "cbramod_lr"), (2, "labram_lr"), (1, "age_lr")]),
    ]
    for task, name, models in spec:
        for phase, m in models:
            s = summary(task, m, phase)
            label = m + (" (post hoc)" if (phase, m) == (2, "shallow") else "")
            rows.append([name, label, pct(s["subject"]["balanced_accuracy"]), num(s["subject"]["macro_auc"], ci=False)])
    return table(["Task", "Model", "Subject bal. acc. %, mean ± SD [95 % CI]", "Subject AUC"], rows, "llrr")


def permutation() -> str:
    rows = []
    for f in (RESULTS_DIR / "permutation" / "rbp_lr.json", RESULTS_DIR / "permutation" / "spectral_lr.json",
              P2 / "permutation" / "cbramod_lr.json"):
        p = json.loads(f.read_text(encoding="utf-8"))
        rows.append([link(f, p["model"]), f"{100 * p['observed']:.1f}", f"{100 * p['null_mean']:.1f} ± {100 * p['null_sd']:.1f}",
                     f"{100 * p['null_95th']:.1f}", str(p["n_perm"]), f"{p['p_value']:.3f}", f"1/{p['n_perm'] + 1} = {1 / (p['n_perm'] + 1):.3f}"])
    return table(["Pipeline", "Observed %", "Null mean ± SD %", "Null 95th pct. %", "Permutations", "p", "Smallest possible p"], rows)


def legacy_split() -> str:
    rows = [["Graph transformer, dim=128 (April 2025)", f"{100 * OLD_MODEL['accuracy']:.1f} (chunks)",
             f"{100 * OLD_MODEL['balanced_accuracy']:.1f} (chunks)", "–"]]
    for phase, m in ((1, "all_lgbm"), (1, "spectral_lr"), (1, "ensemble_vote"), (1, "nested_select"), (2, "labram_lr"),
                     (2, "shallow"), (2, "cbramod_lr"), (1, "chance_prior")):
        s = summary("legacy", m, phase)
        rows.append([m, f"{100 * s['epoch']['accuracy']['mean']:.1f}", f"{100 * s['epoch']['balanced_accuracy']['mean']:.1f}",
                     pct(s["subject"]["balanced_accuracy"], sd=False)])
    return table(["Model", "Epoch acc. %", "Epoch bal. acc. %", "Subject bal. acc. % [95 % CI]"], rows)


def sensitivity() -> str:
    conds = [
        ("cv3", "average reference, 10 s / 5 s epochs (main)"),
        ("cv3__native_ref", "native A1-A2 reference, 10 s / 5 s"),
        ("cv3__win4s", "average reference, 4 s / 2 s"),
        ("cv3__win30s", "average reference, 30 s / 15 s, boundary windows kept"),
        ("cv3__subjmean", "one row per subject (features averaged over epochs)"),
    ]
    models = ["rbp_lr", "spectral_lr", "all_lr"]
    rows = []
    for task, label in conds:
        cells, n_ep = [], ""
        for m in models:
            f = RESULTS_DIR / task / m / "summary.json"
            if f.exists():
                s = json.loads(f.read_text(encoding="utf-8"))
                cells.append(pct(s["subject"]["balanced_accuracy"]))
                n_ep = f"{s['n_epochs']:,}"
            else:
                cells.append("–")
        rows.append([label, n_ep, *cells])
    return table(["Condition", "Training rows", *models], rows)


def slowing() -> str:
    u = pd.read_csv(RESULTS_DIR / "interpretability" / "univariate_kruskal.csv").set_index("feature")
    rows = []
    for key, label in (("peak_freq__all", "Peak frequency (5-14 Hz)"), ("ratio__theta_alpha", "log10 theta/alpha power")):
        for region in ("frontal", "temporal", "parietal", "occipital"):
            r = u.loc[f"specR__{key}__{region}"]
            rows.append([f"{label}, {region}", f"{r.mean_AD:.2f}", f"{r.mean_FTD:.2f}", f"{r.mean_CN:.2f}",
                         f"{r.g_AD_CN:+.2f}", f"{r.g_FTD_CN:+.2f}", f"{r.g_AD_FTD:+.2f}", f"{r.q_fdr:.1g}"])
    return table(["Feature (subject means)", "AD", "FTD", "CN", "g AD-CN", "g FTD-CN", "g AD-FTD", "FDR q"], rows)


def epochs() -> str:
    st = pd.read_csv(RESULTS_DIR / "tables" / "epochs__average_10s_5s.csv")
    rows = []
    for dx in ("AD", "CN", "FTD"):
        d = st[st.diagnosis == dx]
        rows.append([dx, str(len(d)), f"{d.n_candidates.sum():,}", f"{d.n_boundary_rejected.sum():,}", f"{d.n_amplitude_rejected.sum():,}",
                     f"{d.n_epochs.sum():,}", f"{d.n_epochs.min()}-{d.n_epochs.max()}", f"{d.duration_s.sum() / 3600:.1f}"])
    rows.append(["**total**", str(len(st)), f"{st.n_candidates.sum():,}", f"{st.n_boundary_rejected.sum():,}",
                 f"{st.n_amplitude_rejected.sum():,}", f"**{st.n_epochs.sum():,}**", f"{st.n_epochs.min()}-{st.n_epochs.max()}",
                 f"{st.duration_s.sum() / 3600:.1f}"])
    return table(["Group", "Subjects", "10 s windows", "Dropped: ASR boundary", "Dropped: amplitude", "Epochs kept",
                  "Per subject", "Hours (after crop)"], rows)


def c_grid() -> str:
    rows = []
    for phase, m in ((2, "cbramod_lr"), (2, "labram_lr"), (2, "cbramod_spectral_lr"), (1, "spectral_lr"), (1, "all_lr"), (1, "rbp_lr")):
        s = summary("cv3", m, phase)
        counts = s["chosen_params_counts"]
        n = sum(counts.values())
        smallest = sum(v for k, v in counts.items() if json.loads(k).get("C") == 1e-4)
        rows.append([m, f"{smallest} / {n}", f"{s['subject']['log_loss']['mean']:.2f}"])
    for phase, m in ((2, "shallow"), (1, "ensemble_vote")):
        s = summary("cv3", m, phase)
        rows.append([m, "–", f"{s['subject']['log_loss']['mean']:.2f}"])
    rows.append(["chance (uniform 1/3)", "–", "1.10"])
    return table(["Model", "Outer folds choosing the smallest C (1e-4)", "Subject-level log-loss"], rows)


BLOCKS = {
    "main-3class": main_3class,
    "binary-loso": binary_loso,
    "binary-cv": binary_cv,
    "permutation": permutation,
    "legacy-split": legacy_split,
    "sensitivity": sensitivity,
    "slowing": slowing,
    "epochs": epochs,
    "c-grid": c_grid,
}


def render(text: str) -> str:
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- BEGIN GENERATED: {re.escape(name)} -->\n).*?(<!-- END GENERATED: {re.escape(name)} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"README.md has no block {name!r}")
        body = fn()
        text = pat.sub(lambda mt, body=body: mt.group(1) + body + "\n" + mt.group(2), text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only check that README.md is up to date")
    a = ap.parse_args()
    old = README.read_text(encoding="utf-8").replace("\r\n", "\n")
    new = render(old)
    if a.check:
        if new != old:
            print("README.md tables are out of date: run scripts/make_readme_tables.py", file=sys.stderr)
            return 1
        print("README.md tables are up to date")
        return 0
    README.write_text(new, encoding="utf-8", newline="\n")
    print(f"updated {len(BLOCKS)} tables in README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
