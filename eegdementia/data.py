"""Loading ds004504 (BIDS) derivatives, preprocessing, caching and epoching.

Typical use::

    from eegdementia import data
    subjects = data.load_participants()
    rec = data.load_preprocessed("sub-001")            # cached continuous signal
    ep = data.epoch_recording(rec, EpochConfig())       # (n_epochs, 19, n_times) in uV

Everything here is label-free: the preprocessing of one subject never looks at any other
subject or at any label, so it can be done once, up front, without leaking information.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BIDS_ROOT,
    CACHE_DIR,
    CHANNELS,
    GROUP_TO_NAME,
    NAME_TO_INT,
    EpochConfig,
    PrepConfig,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Participants
# --------------------------------------------------------------------------------------
def load_participants(bids_root: Path = BIDS_ROOT) -> pd.DataFrame:
    """participants.tsv -> DataFrame indexed by subject id with columns
    ``group`` (A/C/F), ``diagnosis`` (AD/CN/FTD), ``label`` (0/1/2), ``age``, ``sex``.

    MMSE is deliberately dropped: it is (close to) the diagnostic label itself.
    """
    f = Path(bids_root) / "participants.tsv"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found: download OpenNeuro ds004504 (README, 'Data') and set EEG_BIDS_ROOT")
    df = pd.read_csv(f, sep="\t")
    out = pd.DataFrame(
        {
            "group": df["Group"].values,
            "diagnosis": df["Group"].map(GROUP_TO_NAME).values,
            "age": df["Age"].astype(float).values,
            "sex": df["Gender"].values,
        },
        index=pd.Index(df["participant_id"].values, name="subject"),
    )
    out["label"] = out["diagnosis"].map(NAME_TO_INT).astype(int)
    return out


def derivative_path(subject: str, bids_root: Path = BIDS_ROOT) -> Path:
    return Path(bids_root) / "derivatives" / subject / "eeg" / f"{subject}_task-eyesclosed_eeg.set"


# --------------------------------------------------------------------------------------
# Continuous preprocessing (+ cache)
# --------------------------------------------------------------------------------------
@dataclass
class Recording:
    subject: str
    data: np.ndarray  # (n_channels, n_times) float32, microvolts
    sfreq: float
    ch_names: list[str]
    boundaries: np.ndarray  # sample indices (in ``data``) of discontinuities
    orig_duration_s: float


def _preprocess_from_file(subject: str, cfg: PrepConfig, bids_root: Path) -> Recording:
    import mne

    raw = mne.io.read_raw_eeglab(derivative_path(subject, bids_root), preload=True, verbose="error")
    if raw.ch_names != CHANNELS:
        raw.reorder_channels(CHANNELS)
    sf0 = raw.info["sfreq"]
    dur = raw.n_times / sf0
    # EEGLAB "boundary" events mark where ASR cut data out (discontinuities).
    ann = raw.annotations
    bnd_t = np.array([o for o, d in zip(ann.onset, ann.description, strict=True) if "boundary" in d.lower()])

    tmin, tmax = cfg.crop_start_s, dur - cfg.crop_end_s
    if tmax - tmin < 60:
        raise ValueError(f"{subject}: only {tmax - tmin:.1f} s left after cropping")
    # crop in *seconds* (the legacy script cropped raw.times[-30], i.e. 30 samples)
    raw.crop(tmin=tmin, tmax=tmax, include_tmax=False)
    if cfg.reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="error")
    elif cfg.reference != "native":
        raise ValueError(f"unknown reference {cfg.reference!r}")
    if cfg.sfreq != sf0:
        raw.resample(cfg.sfreq, verbose="error")
    data = (raw.get_data() * 1e6).astype(np.float32)
    bnd_t = bnd_t[(bnd_t >= tmin) & (bnd_t < tmax)] - tmin
    bnd = np.round(bnd_t * cfg.sfreq).astype(np.int64)
    return Recording(subject, data, float(cfg.sfreq), list(raw.ch_names), bnd, float(dur))


def prep_cache_dir(cfg: PrepConfig, cache_dir: Path = CACHE_DIR) -> Path:
    d = Path(cache_dir) / cfg.key()
    d.mkdir(parents=True, exist_ok=True)
    cfg_file = d / "config.json"
    if not cfg_file.exists():
        cfg_file.write_text(json.dumps(asdict(cfg), indent=2))
    return d


def load_preprocessed(
    subject: str,
    cfg: PrepConfig = PrepConfig(),
    bids_root: Path = BIDS_ROOT,
    cache_dir: Path = CACHE_DIR,
    use_cache: bool = True,
) -> Recording:
    f = prep_cache_dir(cfg, cache_dir) / f"{subject}.npz"
    if use_cache and f.exists():
        z = np.load(f, allow_pickle=False)
        return Recording(
            subject,
            z["data"],
            float(z["sfreq"]),
            [str(c) for c in z["ch_names"]],
            z["boundaries"],
            float(z["orig_duration_s"]),
        )
    rec = _preprocess_from_file(subject, cfg, bids_root)
    if use_cache:
        tmp = f.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            data=rec.data,
            sfreq=rec.sfreq,
            ch_names=np.array(rec.ch_names),
            boundaries=rec.boundaries,
            orig_duration_s=rec.orig_duration_s,
        )
        tmp.replace(f)
    return rec


def preprocess_all(
    subjects: list[str], cfg: PrepConfig = PrepConfig(), n_jobs: int = 8, **kw
) -> None:
    """Fill the preprocessing cache for all subjects (parallel, errors are raised)."""
    from joblib import Parallel, delayed

    def _one(s):
        load_preprocessed(s, cfg, **kw)
        return s

    Parallel(n_jobs=n_jobs)(delayed(_one)(s) for s in subjects)


# --------------------------------------------------------------------------------------
# Epoching
# --------------------------------------------------------------------------------------
@dataclass
class Epochs:
    subject: str
    data: np.ndarray  # (n_epochs, n_channels, n_times) float32 uV
    onsets_s: np.ndarray  # epoch start times (s, relative to cropped signal)
    sfreq: float
    n_candidates: int
    n_boundary_rejected: int
    n_amplitude_rejected: int


def epoch_recording(rec: Recording, cfg: EpochConfig = EpochConfig()) -> Epochs:
    sf = rec.sfreq
    L = int(round(cfg.length_s * sf))
    S = int(round(cfg.step_s * sf))
    n = rec.data.shape[1]
    starts = np.arange(0, n - L + 1, S, dtype=np.int64)
    keep = np.ones(len(starts), bool)
    n_bnd = 0
    if cfg.drop_boundaries and len(rec.boundaries):
        m = int(round(cfg.boundary_margin_s * sf))
        b = rec.boundaries[None, :]
        hit = ((b >= starts[:, None] - m) & (b < starts[:, None] + L + m)).any(1)
        n_bnd = int(hit.sum())
        keep &= ~hit
    idx = starts[:, None] + np.arange(L)[None, :]
    X = rec.data[:, idx].transpose(1, 0, 2)  # (n_epochs, ch, L)
    n_amp = 0
    if cfg.reject_mad_k is not None and keep.sum() > 5:
        ptp = np.log(np.ptp(X, axis=2).max(axis=1))
        ref = ptp[keep]
        med = np.median(ref)
        mad = 1.4826 * np.median(np.abs(ref - med)) + 1e-12
        bad = keep & (ptp > med + cfg.reject_mad_k * mad)
        n_amp = int(bad.sum())
        keep &= ~bad
    return Epochs(
        rec.subject,
        np.ascontiguousarray(X[keep]),
        starts[keep] / sf,
        sf,
        len(starts),
        n_bnd,
        n_amp,
    )


def load_epochs(
    subjects: list[str],
    prep: PrepConfig = PrepConfig(),
    epoch: EpochConfig = EpochConfig(),
    **kw,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate epochs of several subjects -> (X, groups). Convenient for deep models."""
    Xs, gs = [], []
    for s in subjects:
        ep = epoch_recording(load_preprocessed(s, prep, **kw), epoch)
        Xs.append(ep.data)
        gs.append(np.repeat(s, len(ep.data)))
    return np.concatenate(Xs), np.concatenate(gs)
