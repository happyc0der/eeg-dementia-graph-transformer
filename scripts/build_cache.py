"""Preprocess all 88 derivative recordings and cache per-epoch features.

    uv run python scripts/build_cache.py [--epoch-length 10 --epoch-step 5 --n-jobs 14]
"""
import argparse
import os
import time

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from eegdementia import data, features  # noqa: E402
from eegdementia.config import EpochConfig, PipelineConfig  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch-length", type=float, default=10.0)
    ap.add_argument("--epoch-step", type=float, default=5.0)
    ap.add_argument("--n-jobs", type=int, default=14)
    a = ap.parse_args()
    cfg = PipelineConfig(epoch=EpochConfig(length_s=a.epoch_length, step_s=a.epoch_step))
    subs = list(data.load_participants().index)
    t = time.time()
    data.preprocess_all(subs, cfg.prep, n_jobs=a.n_jobs)
    print(f"preprocessing: {time.time() - t:.0f} s")
    t = time.time()
    d = features.compute_all_features(subs, cfg, n_jobs=a.n_jobs)
    print(f"features: {time.time() - t:.0f} s -> {d}")
