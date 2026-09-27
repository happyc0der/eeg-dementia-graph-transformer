"""Paths, dataset constants and preprocessing / epoching configuration.

Paths are set with environment variables (read once, at import time):

* ``EEG_BIDS_ROOT``   - root of OpenNeuro ds004504 (default ``<repo>/data/ds004504``)
* ``EEG_CACHE_DIR``   - preprocessed recordings, features, frozen embeddings, epoch-level
  predictions and GPU checkpoints (default ``<parent of EEG_BIDS_ROOT>/cache``)
* ``EEG_WEIGHTS_DIR`` - pretrained foundation-model weights, phase 2 only
  (default ``<repo>/data/weights``)

``<repo>/data/`` is git-ignored; nothing in the repository depends on where these live.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = Path(os.environ.get("EEG_BIDS_ROOT", str(REPO_ROOT / "data" / "ds004504")))
CACHE_DIR = Path(os.environ.get("EEG_CACHE_DIR", str(BIDS_ROOT.parent / "cache")))
WEIGHTS_DIR = Path(os.environ.get("EEG_WEIGHTS_DIR", str(REPO_ROOT / "data" / "weights")))
RESULTS_DIR = REPO_ROOT / "results"

# --------------------------------------------------------------------------------------
# Dataset constants (ds004504)
# --------------------------------------------------------------------------------------
# Channel order as stored in the derivative .set files.
CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]
# Coarse lobar regions used for regional averages / connectivity summaries.
REGIONS: dict[str, list[str]] = {
    "frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "central": ["C3", "Cz", "C4"],
    "temporal": ["T3", "T4", "T5", "T6"],
    "parietal": ["P3", "Pz", "P4"],
    "occipital": ["O1", "O2"],
}
# Frequency bands as defined by the dataset authors (Miltiadous et al., 2023, Data 8(6):95).
# The derivative recordings are band-pass filtered 0.5-45 Hz, so nothing above 45 Hz is used.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 25.0),
    "gamma": (25.0, 45.0),
}
FMIN, FMAX = 0.5, 45.0

# Diagnosis labels. Integer codes are fixed and used everywhere.
GROUP_TO_NAME = {"A": "AD", "C": "CN", "F": "FTD"}
CLASS_NAMES = ["AD", "CN", "FTD"]  # label index 0, 1, 2
NAME_TO_INT = {n: i for i, n in enumerate(CLASS_NAMES)}

# Held-out subjects of the legacy (April 2025) 80/20 split, reproduced from data_prep.py
# (pandas .sample(frac=0.8, random_state=42) on participants.tsv).
LEGACY_TEST_SUBJECTS = [
    "sub-002", "sub-003", "sub-015", "sub-021", "sub-022", "sub-024", "sub-030", "sub-038",
    "sub-052", "sub-053", "sub-060", "sub-061", "sub-064", "sub-072", "sub-075", "sub-076",
    "sub-082", "sub-083",
]


# --------------------------------------------------------------------------------------
# Configuration dataclasses
# --------------------------------------------------------------------------------------
def _short_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:10]


@dataclass(frozen=True)
class PrepConfig:
    """Continuous-signal preprocessing applied to each derivative recording.

    The derivatives are already 0.5-45 Hz band-passed, re-referenced to A1-A2, ASR- and
    ICA-cleaned by the dataset authors. We therefore only (1) crop a fixed number of
    *seconds* at the start and end (settling / movement at recording onset and offset),
    (2) re-reference (average reference by default), (3) resample.
    """

    crop_start_s: float = 30.0
    crop_end_s: float = 30.0
    # "average" (default) or "native" (A1-A2 linked mastoids, as distributed). In the
    # native-reference derivatives ~92 % of every channel's variance is one common-mode,
    # mostly <2 Hz signal (~32 uV SD in every subject, inter-channel r ~0.92, no group
    # difference); the average reference removes it. See results/phase1_summary.md.
    reference: str = "average"
    sfreq: float = 250.0  # resampling target (Hz); 250 Hz keeps everything up to 45 Hz exact

    def key(self) -> str:
        return "prep_" + _short_hash(asdict(self))


@dataclass(frozen=True)
class EpochConfig:
    """Fixed-length windowing of the preprocessed signal plus artefact-based rejection."""

    length_s: float = 10.0
    step_s: float = 5.0  # 50 % overlap
    # Windows that contain (or lie within ``boundary_margin_s`` of) an EEGLAB "boundary"
    # event are dropped: ASR removed data there, so the signal is discontinuous.
    drop_boundaries: bool = True
    boundary_margin_s: float = 0.5
    # Per-subject robust outlier rejection on peak-to-peak amplitude (max over channels):
    # drop a window if log(ptp) > median + k * 1.4826 * MAD of that subject's windows.
    # Uses only the subject's own unlabelled signal, so it cannot leak labels.
    reject_mad_k: float | None = 5.0

    def key(self) -> str:
        return "ep_" + _short_hash(asdict(self))


@dataclass(frozen=True)
class FeatureConfig:
    welch_seg_s: float = 2.0  # 0.5 Hz resolution
    specparam_range: tuple[float, float] = (1.0, 40.0)
    alpha_search: tuple[float, float] = (5.0, 14.0)  # peak (dominant posterior) frequency search
    alpha_cog: tuple[float, float] = (7.0, 13.0)  # centre-of-gravity IAF window
    perm_entropy_order: int = 5
    perm_entropy_delay: int = 2
    sample_entropy_decim: int = 2  # compute SampEn on a 125 Hz copy (speed)
    version: int = 1

    def key(self) -> str:
        return "feat_" + _short_hash(asdict(self))


@dataclass(frozen=True)
class PipelineConfig:
    prep: PrepConfig = field(default_factory=PrepConfig)
    epoch: EpochConfig = field(default_factory=EpochConfig)
    feat: FeatureConfig = field(default_factory=FeatureConfig)

    def to_dict(self) -> dict:
        return {"prep": asdict(self.prep), "epoch": asdict(self.epoch), "feat": asdict(self.feat)}
