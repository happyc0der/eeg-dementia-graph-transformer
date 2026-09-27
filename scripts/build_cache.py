"""Preprocess all 88 derivative recordings, cache per-epoch features, and record the epoch yield.

    uv run python scripts/build_cache.py [--epoch-length 10 --epoch-step 5 --reference average --n-jobs 12]

Caches go to $EEG_CACHE_DIR (existing files are reused). The per-subject epoch counts of the
configuration are written to results/tables/epochs__<reference>_<length>s_<step>s[_keepbnd].csv.
"""

import argparse
import time

from eegdementia.utils import be_nice

be_nice()  # CPU-only, 1 BLAS thread per process, below-normal priority

from eegdementia import data, features  # noqa: E402
from eegdementia.config import RESULTS_DIR, EpochConfig, PipelineConfig, PrepConfig  # noqa: E402
from eegdementia.experiments import epoch_counts  # noqa: E402


def counts_file(cfg: PipelineConfig):
    e = cfg.epoch
    tag = f"{cfg.prep.reference}_{e.length_s:g}s_{e.step_s:g}s" + ("" if e.drop_boundaries else "_keepbnd")
    return RESULTS_DIR / "tables" / f"epochs__{tag}.csv"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch-length", type=float, default=10.0)
    ap.add_argument("--epoch-step", type=float, default=5.0)
    ap.add_argument("--keep-boundaries", action="store_true",
                    help="do not drop windows containing ASR boundary events (needed for long windows)")
    ap.add_argument("--reference", default="average", choices=["average", "native"])
    ap.add_argument("--n-jobs", type=int, default=12)
    a = ap.parse_args()
    cfg = PipelineConfig(
        prep=PrepConfig(reference=a.reference),
        epoch=EpochConfig(length_s=a.epoch_length, step_s=a.epoch_step, drop_boundaries=not a.keep_boundaries),
    )
    subs = list(data.load_participants().index)
    t = time.time()
    data.preprocess_all(subs, cfg.prep, n_jobs=a.n_jobs)
    print(f"preprocessing: {time.time() - t:.0f} s")
    t = time.time()
    d = features.compute_all_features(subs, cfg, n_jobs=a.n_jobs)
    print(f"features: {time.time() - t:.0f} s -> {d}")
    st = epoch_counts(cfg)
    out = counts_file(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    st.to_csv(out, lineterminator="\n")
    tot = st[["n_candidates", "n_boundary_rejected", "n_amplitude_rejected", "n_epochs"]].sum()
    print(f"{out.name}: {tot.to_dict()}, per subject {st.n_epochs.min()}-{st.n_epochs.max()}, "
          f"by class {st.groupby('diagnosis').n_epochs.sum().to_dict()}")
