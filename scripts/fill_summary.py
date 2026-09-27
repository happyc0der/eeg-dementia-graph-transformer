"""Insert the generated markdown tables into results/overhaul/phase1_summary.md.

Placeholders (TABLE_CV3, TABLE_LEGACY, ...) are replaced by the tables written by
scripts/make_figures.py; TABLE_SENS is built here from the sensitivity runs.

    uv run python scripts/fill_summary.py
"""

import json

from eegdementia.config import RESULTS_DIR

TAB = RESULTS_DIR / "tables"
DOC = RESULTS_DIR / "phase1_summary.md"


def tab(name: str) -> str:
    f = TAB / f"{name}.md"
    return f.read_text(encoding="utf-8") if f.exists() else f"*(table {name} not available)*"


def sens_table() -> str:
    conds = [
        ("cv3", "average ref., 10 s / 5 s (main)"),
        ("cv3__native_ref", "native A1-A2 ref., 10 s / 5 s"),
        ("cv3__win4s", "average ref., 4 s / 2 s"),
        ("cv3__win30s", "average ref., 30 s / 15 s, boundary windows kept"),
        ("cv3__subjmean", "subject-mean features (1 row per subject)"),
    ]
    models = ["rbp_lr", "spectral_lr", "all_lr"]
    lines = ["| condition | " + " | ".join(models) + " |", "|---|" + "---:|" * len(models)]
    for task, label in conds:
        cells = []
        for m in models:
            f = RESULTS_DIR / task / m / "summary.json"
            if f.exists():
                s = json.loads(f.read_text())
                d = s["subject"]["balanced_accuracy"]
                n_ep = s["n_epochs"]
                cells.append(f"{100 * d['mean']:.1f} ± {100 * d['sd']:.1f} [{100 * d['ci95'][0]:.1f}, {100 * d['ci95'][1]:.1f}]")
            else:
                cells.append("–")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "Subject-level balanced accuracy, % (mean ± SD over 10 repeats [bootstrap 95 % CI]):\n\n" + "\n".join(lines)


if __name__ == "__main__":
    doc = DOC.read_text(encoding="utf-8")
    loso3 = tab("loso3")
    doc = (
        doc.replace("TABLE_CV3", tab("cv3"))
        .replace("TABLE_LEGACY", tab("legacy"))
        .replace("TABLE_LOSO_AD", tab("loso_ad_cn"))
        .replace("TABLE_LOSO_FTD", tab("loso_ftd_cn"))
        .replace("TABLE_LOSO3", "**3-class LOSO** (88 folds, nested inner 5-fold):\n\n" + loso3)
        .replace("TABLE_SENS", sens_table())
    )
    DOC.write_text(doc, encoding="utf-8")
    print("filled", DOC)
