"""Subject-label permutation test of a whole nested pipeline (3-class task).

The observed statistic is the mean subject-level balanced accuracy over the 10 repeats of
the main cv3 run. Each permutation reruns the complete nested 5-fold CV once (1 repeat)
with shuffled subject labels. A single-repeat null is wider than the null of a 10-repeat
mean, so the p-value is conservative.

    uv run python scripts/permutation_test.py --model spectral_lr --n-perm 200
"""

import argparse
import json
import os
import time

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from eegdementia import evaluation as ev  # noqa: E402
from eegdementia import experiments as E  # noqa: E402
from eegdementia.config import RESULTS_DIR  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="spectral_lr")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=14)
    a = ap.parse_args()
    ds = E.build_dataset()
    specs = E.all_specs(ds.feature_names)
    obs = json.loads((RESULTS_DIR / "cv3" / a.model / "summary.json").read_text())["subject"]["balanced_accuracy"]["mean"]
    t = time.time()
    res = ev.permutation_test(specs[a.model], ds, obs, n_perm=a.n_perm, n_repeats=1, n_jobs=a.n_jobs, seed=E.OUTER_SEED)
    res["runtime_s"] = time.time() - t
    res["model"] = a.model
    out = RESULTS_DIR / "permutation"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.model}.json").write_text(json.dumps(res, indent=2))
    print({k: v for k, v in res.items() if k != "null"})
