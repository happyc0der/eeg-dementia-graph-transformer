"""Phase-2 dataset inputs and model specifications (pre-registered in
results/phase2_plan.md; do not change settings after results are seen)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import deep
from . import experiments as E
from . import models as M
from .config import CACHE_DIR, CHANNELS, RESULTS_DIR

log = logging.getLogger(__name__)

PHASE2_DIR = RESULTS_DIR / "phase2"
FM_NAMES = ("cbramod", "labram")  # pre-registered frozen foundation models
EXPLORATORY_FM = ("biot",)  # optional extra (plan section 7), reported as exploratory
RAW_INPUTS = ("raw", "raw125", "raw200")


def _cached(f: Path, fn):
    if f.exists():
        return np.load(f)
    X = fn()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npy")
    np.save(tmp, X)
    tmp.replace(f)
    return X


def inputs_needed(specs: dict[str, M.ModelSpec], models: list[str]) -> set[str]:
    """Names of the dataset inputs the given models read (their ``input`` + ``extra_inputs``)."""
    need = set()
    for m in models:
        s = specs[m]
        need |= {s.input, *s.extra_inputs}
        for mem in s.members or []:
            need |= {mem.input, *mem.extra_inputs}
    return need


def build_phase2_dataset(raw: bool = False, embeddings: tuple[str, ...] = FM_NAMES, device: str = "cuda", guard=None):
    """Phase-1 dataset plus phase-2 inputs.

    * ``raw`` (250 Hz), ``raw125`` (EEGNet), ``raw200`` (CBraMod fine-tuning) when ``raw=True``;
    * ``emb_<name>`` for every name in ``embeddings`` (``cbramod``, ``labram``: 19 x 200 per
      epoch; ``biot``: 256): frozen embeddings, read from ``$EEG_CACHE_DIR/embeddings`` or
      computed there on first use (which needs the raw epochs and the pretrained weights);
    * ``spec_feat``: the 399 phase-1 spectral+aperiodic features (for hybrids).
    """
    from .config import PipelineConfig

    cfg = PipelineConfig()
    key = f"{cfg.prep.key()}__{cfg.epoch.key()}"
    d = CACHE_DIR / "phase2" / key
    unknown = set(embeddings) - set(FM_NAMES) - set(EXPLORATORY_FM)
    if unknown:
        raise ValueError(f"unknown foundation model(s): {sorted(unknown)}")
    missing_emb = [m for m in embeddings if not deep.embedding_file(m, key, CACHE_DIR).exists()]
    need_raw = raw or bool(missing_emb)
    ds = E.build_dataset(cfg, include_raw=need_raw)
    X200 = None
    if need_raw:
        X250 = ds.inputs["raw"]
        X200 = _cached(d / "raw200.npy", lambda: deep.resample_epochs(X250, 250.0, 200.0))
    for m in embeddings:
        ds.inputs[f"emb_{m}"] = deep.cached_embeddings(m, X200, key, CACHE_DIR, device=device, guard=guard)
        if len(ds.inputs[f"emb_{m}"]) != len(ds.groups):
            raise RuntimeError(f"cached {m} embeddings do not match the epochs; delete them and recompute")
    if raw:
        ds.inputs["raw200"] = X200
        ds.inputs["raw125"] = _cached(d / "raw125.npy", lambda: deep.resample_epochs(ds.inputs["raw"], 250.0, 125.0))
    elif "raw" in ds.inputs:
        del ds.inputs["raw"]
    cols = M.feature_columns(ds.feature_names, M.FEATURE_SETS["spectral"])
    ds.inputs["spec_feat"] = np.ascontiguousarray(ds.inputs["feat"][:, cols])
    return ds


# --------------------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------------------
def _torch_factory(kind: str, guard=None, history_log=None):
    n_ch = len(CHANNELS)

    def factory(params, n_classes, seed):
        if kind == "eegnet":
            return deep.TorchEpochClassifier(
                lambda: deep.build_eegnet(n_ch, n_classes, 1250), n_classes, seed, scale=0.1,
                lr=1e-3, weight_decay=0.0, batch_size=64, max_epochs=40, patience=8,
                samples_per_subject=32, n_bags=5, guard=guard, name="eegnet", history_log=history_log,
            )
        if kind == "shallow":
            return deep.TorchEpochClassifier(
                lambda: deep.build_shallow(n_ch, n_classes, 2500), n_classes, seed, scale=0.1,
                lr=6.25e-4, weight_decay=0.0, batch_size=64, max_epochs=40, patience=8,
                samples_per_subject=32, n_bags=5, guard=guard, name="shallow", history_log=history_log,
            )
        if kind == "cbramod_ft":
            return deep.TorchEpochClassifier(
                lambda: deep.CBraModClassifier.build(n_classes), n_classes, seed, scale=0.01,
                lr=1e-4, head_lr=5e-4, weight_decay=5e-2, batch_size=64, max_epochs=15, patience=4,
                samples_per_subject=32, n_bags=5, label_smoothing=0.1, clip_grad=1.0, amp=True,
                guard=guard, name="cbramod_ft", history_log=history_log,
            )
        raise ValueError(kind)

    return factory


def phase2_specs(guard=None, history_log=None) -> dict[str, M.ModelSpec]:
    S: dict[str, M.ModelSpec] = {}
    for m in FM_NAMES:
        S[f"{m}_lr"] = M.ModelSpec(
            f"{m}_lr", f"emb_{m}", M.lr_factory, M.C_GRID, sample_weight_param="clf__sample_weight",
            description=f"frozen {m} embedding (19 ch x 200, mean over 1-s patches) + L2 logistic regression (inner-CV C)",
        )
    S["biot_lr"] = M.ModelSpec(
        "biot_lr", "emb_biot", M.lr_factory, M.C_GRID, sample_weight_param="clf__sample_weight",
        description="EXPLORATORY: frozen BIOT (six-datasets weights) on 16 bipolar channels, 256-d mean token + L2 LR",
    )
    S["cbramod_spectral_lr"] = M.ModelSpec(
        "cbramod_spectral_lr", "emb_cbramod", M.lr_factory, M.C_GRID, extra_inputs=("spec_feat",),
        sample_weight_param="clf__sample_weight",
        description="hybrid: frozen CBraMod embedding (3800) + phase-1 spectral/aperiodic features (399) + L2 LR",
    )
    for kind, inp, desc in (
        ("eegnet", "raw125", "EEGNet (braindecode defaults), 125 Hz, trained from scratch"),
        ("shallow", "raw", "ShallowFBCSPNet (braindecode defaults), 250 Hz, trained from scratch"),
        ("cbramod_ft", "raw200", "CBraMod fine-tuned end to end (official weights, avg-pool head), 200 Hz"),
    ):
        S[kind] = M.ModelSpec(
            kind, inp, _torch_factory(kind, guard, history_log), [{}], sample_weight=False, fit_groups=True,
            description=desc,
        )
    return S
