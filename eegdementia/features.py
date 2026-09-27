"""Per-epoch feature extraction (spectral, aperiodic, complexity, connectivity, covariances).

All features are computed from a single epoch of a single subject, so feature extraction
is label-free and can be cached once for all experiments. Anything *fitted* (scaling,
selection, Riemannian reference points) happens later, inside the CV pipelines.

Feature names follow ``family__measure__band__location`` where location is a channel,
a region (for ``*R`` families: regional means), ``global`` or a region pair.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from .config import BANDS, CACHE_DIR, CHANNELS, FMAX, FMIN, REGIONS, FeatureConfig, PipelineConfig
from .data import epoch_recording, load_preprocessed

log = logging.getLogger(__name__)

BAND_NAMES = list(BANDS)
REGION_NAMES = list(REGIONS)
_CH_IDX = {c: i for i, c in enumerate(CHANNELS)}
REGION_IDX = {r: np.array([_CH_IDX[c] for c in chs]) for r, chs in REGIONS.items()}
IU = np.triu_indices(len(CHANNELS), k=1)  # 171 channel pairs
REGION_PAIRS = [
    (a, b) for i, a in enumerate(REGION_NAMES) for b in REGION_NAMES[i:]
]  # 15 (incl. within-region)
CONN_MEASURES = ["coh", "icoh", "wpli", "aec"]
COV_BANDS = BAND_NAMES + ["broad"]


# --------------------------------------------------------------------------------------
# Spectral
# --------------------------------------------------------------------------------------
def psd_welch(X: np.ndarray, sf: float, seg_s: float) -> tuple[np.ndarray, np.ndarray]:
    nper = int(round(seg_s * sf))
    f, P = welch(X, fs=sf, nperseg=nper, noverlap=nper // 2, window="hann", axis=-1)
    return f, P


def band_powers(f: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Absolute band power (n_ep, ch, n_bands) and total 0.5-45 Hz power (n_ep, ch)."""
    df = f[1] - f[0]
    out = []
    for lo, hi in BANDS.values():
        m = (f >= lo) & ((f < hi) if hi < FMAX else (f <= hi))
        out.append(P[..., m].sum(-1) * df)
    tot_m = (f >= FMIN) & (f <= FMAX)
    return np.stack(out, -1), P[..., tot_m].sum(-1) * df


def spectral_shape(f: np.ndarray, P: np.ndarray, cfg: FeatureConfig) -> dict[str, np.ndarray]:
    m = (f >= FMIN) & (f <= FMAX)
    fm, Pm = f[m], P[..., m]
    p = Pm / Pm.sum(-1, keepdims=True)
    sent = -(p * np.log(p + 1e-20)).sum(-1) / np.log(p.shape[-1])
    cdf = np.cumsum(p, -1)
    mdf = fm[np.argmax(cdf >= 0.5, axis=-1)]
    sef95 = fm[np.argmax(cdf >= 0.95, axis=-1)]
    a = (f >= cfg.alpha_search[0]) & (f <= cfg.alpha_search[1])
    paf = f[a][np.argmax(P[..., a], axis=-1)]
    c = (f >= cfg.alpha_cog[0]) & (f <= cfg.alpha_cog[1])
    cog = (P[..., c] * f[c]).sum(-1) / P[..., c].sum(-1)
    return {"spec_entropy": sent, "median_freq": mdf, "sef95": sef95, "peak_freq": paf, "alpha_cog": cog}


def aperiodic(f: np.ndarray, P: np.ndarray, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """specparam (FOOOF) fixed-mode aperiodic offset / exponent / R^2 per (epoch, channel)."""
    from specparam import SpectralGroupModel

    shp = P.shape[:-1]
    flat = P.reshape(-1, P.shape[-1])
    fg = SpectralGroupModel(
        peak_width_limits=(1.0, 8.0),
        max_n_peaks=4,
        min_peak_height=0.1,
        aperiodic_mode="fixed",
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fg.fit(f, flat, list(cfg.specparam_range))
    ap = np.asarray(fg.get_params("aperiodic"), dtype=float)
    r2 = np.asarray(fg.get_metrics("gof_rsquared"), dtype=float)
    return ap[:, 0].reshape(shp), ap[:, 1].reshape(shp), r2.reshape(shp)


# --------------------------------------------------------------------------------------
# Time domain / complexity
# --------------------------------------------------------------------------------------
def hjorth(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d1 = np.diff(X, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = X.var(-1), d1.var(-1), d2.var(-1)
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob
    return np.log(v0), mob, comp


def complexity(X: np.ndarray, cfg: FeatureConfig) -> dict[str, np.ndarray]:
    import antropy as ant

    n_ep, n_ch, _ = X.shape
    pe = np.empty((n_ep, n_ch))
    se = np.empty((n_ep, n_ch))
    hfd = np.empty((n_ep, n_ch))
    Xd = X[..., :: cfg.sample_entropy_decim].astype(np.float64)
    X64 = X.astype(np.float64)
    for i in range(n_ep):
        for j in range(n_ch):
            pe[i, j] = ant.perm_entropy(
                X64[i, j], order=cfg.perm_entropy_order, delay=cfg.perm_entropy_delay, normalize=True
            )
            se[i, j] = ant.sample_entropy(Xd[i, j], order=2)
            hfd[i, j] = ant.higuchi_fd(X64[i, j], kmax=10)
    se = np.where(np.isfinite(se), se, np.nan)
    return {"perm_entropy": pe, "sample_entropy": se, "higuchi_fd": hfd}


# --------------------------------------------------------------------------------------
# Connectivity + covariance
# --------------------------------------------------------------------------------------
def _segments_fft(X: np.ndarray, sf: float, seg_s: float) -> tuple[np.ndarray, np.ndarray]:
    nper = int(round(seg_s * sf))
    step = nper // 2
    n = X.shape[-1]
    starts = np.arange(0, n - nper + 1, step)
    win = np.hanning(nper).astype(np.float32)
    seg = np.stack([X[..., s : s + nper] for s in starts], axis=1)  # (ep, seg, ch, nper)
    seg = seg - seg.mean(-1, keepdims=True)
    Z = np.fft.rfft(seg * win, axis=-1)  # (ep, seg, ch, nf)
    f = np.fft.rfftfreq(nper, 1 / sf)
    return f, Z.astype(np.complex64)


def spectral_connectivity(X: np.ndarray, sf: float, seg_s: float) -> np.ndarray:
    """Magnitude-squared coherence, |imaginary coherency| and wPLI, band-averaged.

    Returns (n_ep, 3, n_bands, 171) for the upper-triangle channel pairs.
    """
    f, Z = _segments_fft(X, sf, seg_s)
    n_ep = X.shape[0]
    out = np.empty((n_ep, 3, len(BANDS), len(IU[0])), np.float32)
    for b, (lo, hi) in enumerate(BANDS.values()):
        m = (f >= lo) & ((f < hi) if hi < FMAX else (f <= hi))
        Zb = Z[..., m]  # (ep, seg, ch, nfb)
        for e0 in range(0, n_ep, 16):
            z = Zb[e0 : e0 + 16]
            # cross spectra per segment for the channel pairs: (ep, seg, pairs, nfb)
            cs = z[:, :, IU[0], :] * np.conj(z[:, :, IU[1], :])
            S = cs.mean(1)  # (ep, pairs, nfb)
            pw = (np.abs(z) ** 2).mean(1)  # (ep, ch, nfb)
            denom = pw[:, IU[0], :] * pw[:, IU[1], :]
            coh = (np.abs(S) ** 2 / denom).mean(-1)
            icoh = (np.abs(S.imag) / np.sqrt(denom)).mean(-1)
            im = cs.imag
            wpli = (np.abs(im.mean(1)) / (np.abs(im).mean(1) + 1e-30)).mean(-1)
            out[e0 : e0 + 16, 0, b] = coh
            out[e0 : e0 + 16, 1, b] = icoh
            out[e0 : e0 + 16, 2, b] = wpli
    return out


def band_filter(X: np.ndarray, sf: float) -> np.ndarray:
    """(n_bands, n_ep, ch, n_times) zero-phase Butterworth band-passed copies."""
    out = []
    for lo, hi in BANDS.values():
        sos = butter(4, [lo, min(hi, sf / 2 - 1)], btype="bandpass", fs=sf, output="sos")
        out.append(sosfiltfilt(sos, X, axis=-1, padtype="odd", padlen=int(sf)).astype(np.float32))
    return np.stack(out)


def aec(Xb: np.ndarray) -> np.ndarray:
    """Amplitude-envelope correlation per band: (n_ep, n_bands, 171)."""
    env = np.abs(hilbert(Xb, axis=-1))  # (band, ep, ch, t)
    env = env - env.mean(-1, keepdims=True)
    env /= np.linalg.norm(env, axis=-1, keepdims=True) + 1e-12
    C = np.einsum("beit,bejt->beij", env, env)
    return C[..., IU[0], IU[1]].transpose(1, 0, 2).astype(np.float32)


def covariances(X: np.ndarray, Xb: np.ndarray) -> np.ndarray:
    """OAS-shrunk spatial covariances per band + broadband: (n_ep, 6, ch, ch)."""
    from pyriemann.geometry.covariance import covariances as pcov

    covs = [pcov(Xb[b].astype(np.float64), estimator="oas") for b in range(Xb.shape[0])]
    covs.append(pcov(X.astype(np.float64), estimator="oas"))
    return np.stack(covs, 1).astype(np.float32)


def summarise_connectivity(conn: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """conn (n_ep, n_measures, n_bands, 171) -> node strength, region-pair and global means."""
    n_ep, n_m, n_b, _ = conn.shape
    n_ch = len(CHANNELS)
    full = np.zeros((n_ep, n_m, n_b, n_ch, n_ch), np.float32)
    full[..., IU[0], IU[1]] = conn
    full = full + np.swapaxes(full, -1, -2)
    cols, names = [], []
    for mi, meas in enumerate(CONN_MEASURES[:n_m]):
        for bi, band in enumerate(BAND_NAMES):
            M = full[:, mi, bi]
            strength = M.sum(-1) / (n_ch - 1)
            for ci, ch in enumerate(CHANNELS):
                cols.append(strength[:, ci])
                names.append(f"conn__{meas}__{band}__{ch}")
            for ra, rb in REGION_PAIRS:
                ia, ib = REGION_IDX[ra], REGION_IDX[rb]
                sub = M[:, ia][:, :, ib]
                if ra == rb:
                    k = len(ia)
                    val = sub.sum((1, 2)) / (k * (k - 1)) if k > 1 else np.zeros(n_ep)
                else:
                    val = sub.mean((1, 2))
                cols.append(val)
                names.append(f"connR__{meas}__{band}__{ra}-{rb}")
            cols.append(conn[:, mi, bi].mean(-1))
            names.append(f"connG__{meas}__{band}__global")
    return np.stack(cols, 1), names


# --------------------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------------------
def _per_channel(block: dict[str, np.ndarray], family: str) -> tuple[list[np.ndarray], list[str]]:
    """Per-channel arrays -> columns, plus regional means (family + 'R')."""
    cols, names = [], []
    for key, arr in block.items():  # arr: (n_ep, ch)
        for ci, ch in enumerate(CHANNELS):
            cols.append(arr[:, ci])
            names.append(f"{family}__{key}__{ch}")
    for key, arr in block.items():
        for r in REGION_NAMES:
            cols.append(np.nanmean(arr[:, REGION_IDX[r]], axis=1))
            names.append(f"{family}R__{key}__{r}")
    return cols, names


def extract_epoch_features(X: np.ndarray, sf: float, cfg: FeatureConfig) -> dict:
    """X: (n_ep, 19, n_times) in uV -> dict with 'X' (n_ep, n_feat), 'names', 'cov', 'conn'."""
    f, P = psd_welch(X, sf, cfg.welch_seg_s)
    bp, tot = band_powers(f, P)  # (ep, ch, 5), (ep, ch)
    rel = bp / tot[..., None]
    blocks_spec: dict[str, np.ndarray] = {}
    for bi, band in enumerate(BAND_NAMES):
        blocks_spec[f"abspow__{band}"] = np.log10(bp[..., bi])
    for bi, band in enumerate(BAND_NAMES):
        blocks_spec[f"relpow__{band}"] = rel[..., bi]
    blocks_spec["abspow__total"] = np.log10(tot)
    th, al = bp[..., 1], bp[..., 2]
    blocks_spec["ratio__theta_alpha"] = np.log10(th / al)
    blocks_spec["ratio__slow_fast"] = np.log10((bp[..., 0] + th) / (al + bp[..., 3]))
    shape = spectral_shape(f, P, cfg)
    blocks_spec.update({f"{k}__all": v for k, v in shape.items()})

    off, exp_, r2 = aperiodic(f, P, cfg)
    blocks_ap = {"offset__all": off, "exponent__all": exp_, "r2__all": r2}

    act, mob, comp = hjorth(X)
    blocks_cx = {"hjorth_activity__all": act, "hjorth_mobility__all": mob, "hjorth_complexity__all": comp}
    blocks_cx.update({f"{k}__all": v for k, v in complexity(X, cfg).items()})

    cols, names = [], []
    for fam, blk in (("spec", blocks_spec), ("aper", blocks_ap), ("cplx", blocks_cx)):
        c, n = _per_channel(blk, fam)
        cols += c
        names += n

    Xb = band_filter(X, sf)
    conn = np.concatenate([spectral_connectivity(X, sf, cfg.welch_seg_s), aec(Xb)[:, None]], axis=1)
    cs, cn = summarise_connectivity(conn)
    feats = np.concatenate([np.stack(cols, 1), cs], axis=1).astype(np.float32)
    names += cn
    cov = covariances(X, Xb)
    return {"X": feats, "names": names, "cov": cov, "conn": conn}


def feature_cache_dir(cfg: PipelineConfig, cache_dir: Path = CACHE_DIR) -> Path:
    d = Path(cache_dir) / f"{cfg.prep.key()}__{cfg.epoch.key()}__{cfg.feat.key()}"
    d.mkdir(parents=True, exist_ok=True)
    cf = d / "config.json"
    if not cf.exists():
        cf.write_text(json.dumps(cfg.to_dict(), indent=2))
    return d


def compute_subject_features(subject: str, cfg: PipelineConfig, cache_dir: Path = CACHE_DIR, force: bool = False) -> Path:
    out = feature_cache_dir(cfg, cache_dir) / f"{subject}.npz"
    if out.exists() and not force:
        return out
    rec = load_preprocessed(subject, cfg.prep, cache_dir=cache_dir)
    ep = epoch_recording(rec, cfg.epoch)
    if len(ep.data) == 0:
        raise ValueError(f"{subject}: no epochs left with {cfg.epoch} ({ep.n_candidates} candidates)")
    res = extract_epoch_features(ep.data, ep.sfreq, cfg.feat)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        X=res["X"],
        names=np.array(res["names"]),
        cov=res["cov"],
        conn=res["conn"],
        onsets=ep.onsets_s,
        n_candidates=ep.n_candidates,
        n_boundary_rejected=ep.n_boundary_rejected,
        n_amplitude_rejected=ep.n_amplitude_rejected,
        duration_s=rec.data.shape[1] / rec.sfreq,
    )
    tmp.replace(out)
    return out


def compute_all_features(subjects: list[str], cfg: PipelineConfig, n_jobs: int = 12, **kw) -> Path:
    from joblib import Parallel, delayed

    Parallel(n_jobs=n_jobs, verbose=5)(delayed(compute_subject_features)(s, cfg, **kw) for s in subjects)
    return feature_cache_dir(cfg, kw.get("cache_dir", CACHE_DIR))


# --------------------------------------------------------------------------------------
# Loading the feature table for modelling
# --------------------------------------------------------------------------------------
class FeatureStore:
    """All epochs of all subjects, loaded from the feature cache.

    Attributes: ``X`` (n_epochs, n_features) float32, ``names``, ``cov`` (n_epochs, 6, 19, 19),
    ``subject`` (n_epochs,) subject id per epoch,
    ``epoch_stats`` per-subject epoch counts.
    """

    def __init__(self, subjects: list[str], cfg: PipelineConfig, cache_dir: Path = CACHE_DIR):
        d = feature_cache_dir(cfg, cache_dir)
        Xs, covs, subj, stats = [], [], [], []
        names = None
        missing = [s for s in subjects if not (d / f"{s}.npz").exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} subjects have no cached features in {d} (e.g. {missing[0]}); "
                "build them with scripts/build_cache.py (same --reference / --epoch-* flags)"
            )
        for s in subjects:
            z = np.load(d / f"{s}.npz", allow_pickle=False)
            if names is None:
                names = [str(n) for n in z["names"]]
            Xs.append(z["X"])
            covs.append(z["cov"])
            subj.append(np.repeat(s, len(z["X"])))
            stats.append(
                {
                    "subject": s,
                    "n_epochs": len(z["X"]),
                    "n_candidates": int(z["n_candidates"]),
                    "n_boundary_rejected": int(z["n_boundary_rejected"]),
                    "n_amplitude_rejected": int(z["n_amplitude_rejected"]),
                    "duration_s": float(z["duration_s"]),
                }
            )
        self.cfg = cfg
        self.names = names
        self.X = np.concatenate(Xs)
        self.cov = np.concatenate(covs)
        self.subject = np.concatenate(subj)
        import pandas as pd

        self.epoch_stats = pd.DataFrame(stats).set_index("subject")
