"""Subject-label permutation test of a whole nested pipeline (3-class task).

The observed statistic is the mean subject-level balanced accuracy over the 10 repeats of
the main cv3 run (read from results/[phase2/]cv3/<model>/summary.json). Each permutation
reruns the complete nested 5-fold CV once (1 repeat) with shuffled subject labels. A
single-repeat null is wider than the null of a 10-repeat mean, so the p-value is
conservative.

    uv run python scripts/permutation_test.py --model spectral_lr --n-perm 100
    uv run python scripts/permutation_test.py --phase2 --model cbramod_lr --n-perm 50
"""

import argparse
import json
import time

from eegdementia.utils import be_nice

be_nice()  # CPU-only, 1 BLAS thread per process, below-normal priority

from eegdementia import evaluation as ev  # noqa: E402
from eegdementia import experiments as E  # noqa: E402
from eegdementia.config import RESULTS_DIR  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="spectral_lr")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--phase2", action="store_true", help="phase-2 model on cached frozen embeddings (CPU)")
    a = ap.parse_args()
    root = RESULTS_DIR
    if a.phase2:
        from eegdementia import phase2 as P2

        specs = P2.phase2_specs()
        if a.model not in specs:
            ap.error(f"unknown phase-2 model {a.model!r}")
        need = P2.inputs_needed(specs, [a.model])
        if need & set(P2.RAW_INPUTS):
            ap.error("end-to-end networks are too expensive to permutation-test; use a frozen-embedding model")
        fms = tuple(m for m in P2.FM_NAMES + P2.EXPLORATORY_FM if f"emb_{m}" in need)
        ds = P2.build_phase2_dataset(raw=False, embeddings=fms, device="cpu")
        root = P2.PHASE2_DIR
    else:
        ds = E.build_dataset()
        specs = E.all_specs(ds.feature_names)
        if a.model not in specs:
            ap.error(f"unknown model {a.model!r}")
    obs = json.loads((root / "cv3" / a.model / "summary.json").read_text())["subject"]["balanced_accuracy"]["mean"]
    t = time.time()
    res = ev.permutation_test(specs[a.model], ds, obs, n_perm=a.n_perm, n_repeats=1, n_jobs=a.n_jobs, seed=E.OUTER_SEED)
    res["runtime_s"] = time.time() - t
    res["model"] = a.model
    out = root / "permutation"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.model}.json").write_text(json.dumps(res, indent=2), newline="\n")
    print({k: v for k, v in res.items() if k != "null"})
