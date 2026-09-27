"""Figures and tables for the phase-1 results (matplotlib, Agg backend, PNG)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

# Colours: validated categorical slots 1-3 (blue, orange, aqua) for the three diagnoses,
# muted greys for reference marks. Text always stays in ink colours.
CLASS_COLORS = {"AD": "#2a78d6", "CN": "#eb6834", "FTD": "#1baf7a"}
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e4e3df"
BAR = "#2a78d6"
BAR_SECONDARY = "#86b6ef"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "text.color": INK,
    }
)


def load_summary(d: Path) -> dict:
    return json.loads((Path(d) / "summary.json").read_text())


def comparison_chart(rows: list[dict], out: Path, title: str, xlabel: str = "Subject-level balanced accuracy",
                     refs: list[tuple[str, float]] = (), xlim=(0.2, 0.9)):
    """Horizontal bars (mean) with 95 % CI whiskers; reference lines for chance/old model."""
    rows = sorted(rows, key=lambda r: r["mean"])
    fig, ax = plt.subplots(figsize=(6.4, 0.28 * len(rows) + 1.2))
    y = np.arange(len(rows))
    colors = [r.get("color", BAR) for r in rows]
    ax.barh(y, [r["mean"] for r in rows], height=0.62, color=colors, edgecolor="white", linewidth=1)
    for i, r in enumerate(rows):
        if r.get("ci"):
            ax.plot(r["ci"], [i, i], color=INK2, lw=1.1, solid_capstyle="butt")
        ax.text(max(r["ci"][1] if r.get("ci") else r["mean"], r["mean"]) + 0.008, i, f"{r['mean']:.3f}",
                va="center", ha="left", fontsize=7.5, color=INK2)
    for lab, v in refs:
        ax.axvline(v, color=MUTED, lw=1, ls="--")
        ax.text(v, -0.9, lab, va="top", ha="center", fontsize=7, color=INK2)
    ax.set_ylim(-1.3, len(rows) - 0.4)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.savefig(out)
    plt.close(fig)


def confusion_figure(cms: list[tuple[str, np.ndarray]], class_names, out: Path, title: str):
    fig, axes = plt.subplots(1, len(cms), figsize=(3.3 * len(cms), 3.1))
    axes = np.atleast_1d(axes)
    for ax, (lab, cm) in zip(axes, cms):
        cm = np.asarray(cm, float)
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.55 else INK, fontsize=9)
        ax.set_xticks(range(len(class_names)), class_names)
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)
        ax.set_title(lab, fontsize=9, loc="left")
    fig.suptitle(title, x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def roc_figure(preds: pd.DataFrame, class_names, out: Path, title: str):
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for i, c in enumerate(class_names):
        tprs = []
        for _, d in preds.groupby("repeat"):
            fpr, tpr, _ = roc_curve(d["y"] == i, d[f"p_{c}"])
            tprs.append(np.interp(grid, fpr, tpr))
        tprs = np.array(tprs)
        m = tprs.mean(0)
        auc = np.trapezoid(m, grid)
        ax.fill_between(grid, np.percentile(tprs, 2.5, 0), np.percentile(tprs, 97.5, 0), color=CLASS_COLORS.get(c, BAR), alpha=0.15, lw=0)
        ax.plot(grid, m, color=CLASS_COLORS.get(c, BAR), lw=2, label=f"{c} vs rest (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title(title, loc="left", fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def per_class_figure(summaries: dict[str, dict], class_names, out: Path, title: str):
    """Per-class recall / specificity / AUC (subject level) with bootstrap 95 % CIs."""
    metrics = ["recall", "specificity", "auc"]
    models = list(summaries)
    fig, axes = plt.subplots(1, 3, figsize=(10, 0.45 * len(models) * len(class_names) / 3 + 2.2), sharey=True)
    for ax, met in zip(axes, metrics):
        ylabels = []
        k = 0
        for mi, m in enumerate(models):
            for ci, c in enumerate(class_names):
                s = summaries[m]["subject"][f"{met}_{c}"]
                ax.plot(s["ci95"], [k, k], color=CLASS_COLORS[c], lw=1.4)
                ax.plot([s["mean"]], [k], "o", color=CLASS_COLORS[c], ms=5, mec="white", mew=1)
                ylabels.append(f"{m} - {c}")
                k += 1
            k += 0.6
        ax.set_title(met if met != "auc" else "one-vs-rest AUC", fontsize=9, loc="left")
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color=MUTED, lw=0.8, ls="--")
    ypos, k = [], 0
    for m in models:
        for c in class_names:
            ypos.append(k)
            k += 1
        k += 0.6
    axes[0].set_yticks(ypos, ylabels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def null_histogram(null: list[float], observed: float, out: Path, title: str, chance: float):
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.hist(null, bins=25, color=BAR_SECONDARY, edgecolor="white")
    ax.axvline(observed, color=INK, lw=1.5)
    ax.text(observed, ax.get_ylim()[1] * 0.95, " observed", va="top", fontsize=8)
    ax.axvline(chance, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("Subject-level balanced accuracy")
    ax.set_ylabel("Permutations")
    ax.set_title(title, loc="left", fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def topomap_grid(values: dict[str, dict[str, np.ndarray]], ch_names, out: Path, title: str,
                 cmap="RdBu_r", symmetric=True, units=""):
    """values[row][col] -> per-channel vector. Rows e.g. contrasts, columns e.g. bands."""
    import mne

    info = mne.create_info(ch_names, 250.0, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    rows, cols = list(values), list(next(iter(values.values())))
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.7 * len(cols) + 0.8, 1.7 * len(rows) + 0.4))
    axes = np.atleast_2d(axes)
    allv = np.concatenate([np.ravel(values[r][c]) for r in rows for c in cols])
    vmax = np.nanmax(np.abs(allv)) if symmetric else np.nanmax(allv)
    vmin = -vmax if symmetric else np.nanmin(allv)
    im = None
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            im, _ = mne.viz.plot_topomap(values[r][c], info, axes=ax, show=False, cmap=cmap, vlim=(vmin, vmax), contours=0, sensors=True)
            if i == 0:
                ax.set_title(c, fontsize=9)
            if j == 0:
                ax.text(-0.15, 0.5, r, transform=ax.transAxes, ha="right", va="center", fontsize=9)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, fraction=0.03)
    cb.set_label(units, fontsize=8)
    fig.suptitle(title, x=0.02, ha="left", fontsize=10)
    fig.savefig(out)
    plt.close(fig)
