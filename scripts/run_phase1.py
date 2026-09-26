"""Run phase-1 experiments (nested, subject-grouped evaluation of non-deep baselines).

    uv run python scripts/run_phase1.py --task cv3 --models rbp_lr,all_lr
    uv run python scripts/run_phase1.py --task loso_ad_cn --models rbp_lr
    uv run python scripts/run_phase1.py --list

Tasks: cv3 (3-class, 5-fold x N repeats), legacy (old fixed 18-subject test split),
loso_ad_cn / loso_ftd_cn (binary leave-one-subject-out), cv_ad_cn / cv_ftd_cn / cv_ad_ftd.
"""

import argparse
import os
import sys
import time

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from eegdementia import experiments as E  # noqa: E402
from eegdementia.config import EpochConfig, PipelineConfig, PrepConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cv3")
    ap.add_argument("--models", default="")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=14)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--reference", default="average")
    ap.add_argument("--epoch-length", type=float, default=10.0)
    ap.add_argument("--epoch-step", type=float, default=5.0)
    ap.add_argument("--tag", default="", help="suffix for the results folder (sensitivity analyses)")
    ap.add_argument("--subject-mean", action="store_true",
                    help="average every input over each subject's epochs and train on one row per subject")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    cfg = PipelineConfig(
        prep=PrepConfig(reference=a.reference),
        epoch=EpochConfig(length_s=a.epoch_length, step_s=a.epoch_step),
    )
    ds = E.build_dataset(cfg)
    if a.subject_mean:
        stats, c = ds.epoch_stats, ds.cfg
        ds = ds.subject_mean()
        ds.epoch_stats, ds.cfg = stats, c
    specs = E.all_specs(ds.feature_names)
    if a.list:
        for k, s in specs.items():
            print(f"{k:28s} {s.description}")
        return
    models = [m for m in a.models.split(",") if m] or list(specs)
    for m in models:
        out = E.RESULTS_DIR / (a.task + (f"__{a.tag}" if a.tag else "")) / m / "summary.json"
        if a.skip_existing and out.exists():
            print(f"skip {m}")
            continue
        t = time.time()
        s = E.run_and_save(a.task, m, ds, specs, n_jobs=a.n_jobs, n_repeats=a.repeats,
                           inner_splits=a.inner_splits, n_boot=a.n_boot, tag=a.tag)
        sb, ep = s["subject"], s["epoch"]
        print(
            f"[{a.task}{'__' + a.tag if a.tag else ''}] {m:24s} subj balacc {sb['balanced_accuracy']['mean']:.3f}"
            f"+-{sb['balanced_accuracy']['sd']:.3f} CI{[round(x, 3) for x in sb['balanced_accuracy']['ci95']]} "
            f"acc {sb['accuracy']['mean']:.3f} AUC {sb['macro_auc']['mean']:.3f} | epoch balacc "
            f"{ep['balanced_accuracy']['mean']:.3f} acc {ep['accuracy']['mean']:.3f} | {time.time() - t:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
