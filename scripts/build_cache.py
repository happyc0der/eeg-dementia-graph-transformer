"""Preprocess all 88 derivative recordings and cache per-epoch features.

    uv run python scripts/build_cache.py [--epoch-length 10 --epoch-step 5 --n-jobs 14]
"""
import argparse
import os
import time

from eegdementia.utils import be_nice

be_nice()  # CPU-only, 1 BLAS thread per process, below-normal priority

from eegdementia import data, features  # noqa: E402
from eegdementia.config import EpochConfig, PipelineConfig, PrepConfig  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch-length", type=float, default=10.0)
    ap.add_argument("--epoch-step", type=float, default=5.0)
    ap.add_argument("--keep-boundaries", action="store_true",
                    help="do not drop windows containing ASR boundary events (needed for long windows)")
    ap.add_argument("--reference", default="average", choices=["average", "native"])
    ap.add_argument("--n-jobs", type=int, default=12)
    a = ap.parse_args()
    cfg = PipelineConfig(prep=PrepConfig(reference=a.reference), epoch=EpochConfig(length_s=a.epoch_length, step_s=a.epoch_step, drop_boundaries=not a.keep_boundaries))
    subs = list(data.load_participants().index)
    t = time.time()
    data.preprocess_all(subs, cfg.prep, n_jobs=a.n_jobs)
    print(f"preprocessing: {time.time() - t:.0f} s")
    t = time.time()
    d = features.compute_all_features(subs, cfg, n_jobs=a.n_jobs)
    print(f"features: {time.time() - t:.0f} s -> {d}")
