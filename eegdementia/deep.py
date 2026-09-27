"""Phase-2 deep models for the leak-free harness (``evaluation.run_cv``).

Two families:

1. **End-to-end networks** (EEGNet, ShallowFBCSPNet, fine-tuned CBraMod) wrapped by
   ``TorchEpochClassifier``, which follows the harness estimator interface
   (``fit(X, y, sample_weight, groups)`` / ``predict_proba``). Early stopping uses a
   validation split of *subjects* carved (stratified by class) from the training subjects it
   is given, i.e. from the training subjects of the current outer fold only.
2. **Frozen foundation-model embeddings** (CBraMod, LaBraM): a fixed, label-free,
   per-epoch transform with the authors' pretrained weights. They are computed once for
   every epoch and then treated exactly like the phase-1 features (logistic-regression head
   with the inner-CV C grid, all fitted inside the training fold).

Nothing here computes statistics over more than one epoch outside ``fit``: input scaling
is a fixed constant, networks are used in eval mode for prediction (BatchNorm uses running
statistics from training), and the foundation models only contain per-sample
normalisation (LayerNorm / GroupNorm).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from .evaluation import aggregate_subjects, balanced_subject_weights

log = logging.getLogger(__name__)

WEIGHTS_DIR = Path(os.environ.get("EEG_WEIGHTS_DIR", "C:/AI/models/eeg-pretrained"))

# 10-20 channel names of ds004504 in stored order (config.CHANNELS)
from .config import CHANNELS  # noqa: E402

# --------------------------------------------------------------------------------------
# Pretrained weights: provenance (never committed to the repo)
# --------------------------------------------------------------------------------------
PRETRAINED = {
    "cbramod": {
        "source": "https://huggingface.co/weighting666/CBraMod (linked from https://github.com/wjq-learning/CBraMod README)",
        "file": "official/cbramod/pretrained_weights.pth",
        "url": "https://huggingface.co/weighting666/CBraMod/resolve/500543c7e30bda1b22bfd51a49301b238dee21fd/pretrained_weights.pth",
        "revision": "HF commit 500543c7e30bda1b22bfd51a49301b238dee21fd; code github wjq-learning/CBraMod@b9e961003214326972c567eff390e75b0287e32a",
        "sha256": "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178",
        "licence": "MIT (GitHub code repo); HF model card: Apache-2.0",
        "paper": "Wang et al., CBraMod, ICLR 2025, arXiv:2412.07236",
    },
    "labram": {
        "source": "https://github.com/935963004/LaBraM (checkpoints/labram-base.pth, the 'student' encoder)",
        "file": "official/labram/labram-base.pth",
        "url": "https://github.com/935963004/LaBraM/raw/c431221e6cfd23dbfa9950e0180682fb322b0548/checkpoints/labram-base.pth",
        "revision": "github 935963004/LaBraM@c431221e6cfd23dbfa9950e0180682fb322b0548",
        "sha256": "7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c",
        "licence": "MIT",
        "paper": "Jiang et al., LaBraM, ICLR 2024, arXiv:2405.18765",
        "bd_repo": "braindecode/labram-pretrained",
        "bd_revision": "0563b6c626e7b40d9a36653b763715db94d945d7",
    },
    "biot": {
        "source": "https://github.com/ycq091044/BIOT (pretrained-models/EEG-six-datasets-18-channels.ckpt)",
        "file": "official/biot/EEG-six-datasets-18-channels.ckpt",
        "url": "https://github.com/ycq091044/BIOT/raw/d138e32634e52ae9fa6ec98ac9c4087b14ca869a/pretrained-models/EEG-six-datasets-18-channels.ckpt",
        "revision": "github ycq091044/BIOT@d138e32634e52ae9fa6ec98ac9c4087b14ca869a",
        "sha256": "78ff15a1782f194286a97b1abe68b2bf100a39803325e2917337d0a77a228542",
        "licence": "MIT",
        "paper": "Yang et al., BIOT, NeurIPS 2023, arXiv:2305.10351",
    },
}

# BIOT's pretraining montage: the 16 TCP bipolar derivations (T7/P7/T8/P8 = T3/T5/T4/T6)
BIOT_BIPOLAR = [("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"), ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"),
                ("T6", "O2"), ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"), ("Fp2", "F4"), ("F4", "C4"),
                ("C4", "P4"), ("P4", "O2")]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pretrained_path(name: str, verify: bool = True) -> Path:
    info = PRETRAINED[name]
    p = WEIGHTS_DIR / info["file"]
    if not p.exists():
        import urllib.request

        p.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s weights from %s", name, info["url"])
        urllib.request.urlretrieve(info["url"], p)
    if verify:
        got = _sha256(p)
        if got != info["sha256"]:
            raise RuntimeError(f"{name}: checksum mismatch for {p}: {got}")
    return p


# --------------------------------------------------------------------------------------
# Input transforms (fixed, label-free, per epoch)
# --------------------------------------------------------------------------------------
def resample_epochs(X: np.ndarray, sfreq_in: float, sfreq_out: float, batch: int = 512) -> np.ndarray:
    """Polyphase resampling of every epoch (last axis). 250 -> 200 Hz or 250 -> 125 Hz."""
    from fractions import Fraction

    from scipy.signal import resample_poly

    if sfreq_in == sfreq_out:
        return X
    fr = Fraction(sfreq_out / sfreq_in).limit_denominator(1000)
    out = []
    for i in range(0, len(X), batch):
        out.append(resample_poly(X[i : i + batch].astype(np.float64), fr.numerator, fr.denominator, axis=-1).astype(np.float32))
    return np.concatenate(out)


# --------------------------------------------------------------------------------------
# Network builders
# --------------------------------------------------------------------------------------
def _torch():
    import torch

    return torch


def build_eegnet(n_chans, n_classes, n_times):
    from braindecode.models import EEGNet

    return EEGNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)  # library defaults (F1=8, D=2, k=64, p=0.25)


def build_shallow(n_chans, n_classes, n_times):
    from braindecode.models import ShallowFBCSPNet

    return ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times, final_conv_length="auto")


def load_cbramod_backbone():
    """CBraMod encoder with the official pretrained weights, as used downstream by the
    authors (the pretraining reconstruction head ``proj_out`` replaced by Identity)."""
    torch = _torch()
    from braindecode.models import CBraMod

    m = CBraMod(n_outputs=1, n_chans=len(CHANNELS), n_times=2000, return_encoder_output=True)
    sd = torch.load(pretrained_path("cbramod"), map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"CBraMod weights do not match: missing={missing} unexpected={unexpected}")
    m.proj_out = torch.nn.Identity()
    m.final_layer = torch.nn.Identity()
    return m


def load_labram_backbone(n_times: int = 2000):
    """LaBraM-Base encoder with the official pretrained 'student' weights, final LayerNorm kept.

    The parameters are read from braindecode's re-hosted conversion
    (``braindecode/labram-pretrained``, pinned revision) because it uses braindecode's
    parameter names, and every tensor is checked to be bit-identical to a tensor of the
    official ``labram-base.pth`` student encoder before use.
    """
    torch = _torch()
    from braindecode.models import Labram
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    info = PRETRAINED["labram"]
    sd = load_file(hf_hub_download(info["bd_repo"], "model.safetensors", revision=info["bd_revision"],
                                   cache_dir=str(WEIGHTS_DIR / "hf")))
    off = torch.load(pretrained_path("labram"), map_location="cpu", weights_only=False)["model"]

    def h(t):
        return hashlib.sha1(t.float().contiguous().numpy().tobytes()).hexdigest()

    official = {h(v) for k, v in off.items() if k.startswith("student.")}
    bad = [k for k, v in sd.items() if h(v) not in official]
    if bad:
        raise RuntimeError(f"LaBraM conversion differs from the official weights: {bad[:5]}")
    n_patch = n_times // 200
    sd["temporal_embedding"] = sd["temporal_embedding"][:, : n_patch + 1]  # official code uses the first n_patch too
    m = Labram(n_outputs=1, n_chans=len(CHANNELS), n_times=n_times, sfreq=200)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("final_layer")]
    if missing or unexpected:
        raise RuntimeError(f"LaBraM weights do not match: missing={missing} unexpected={unexpected}")
    m.final_layer = torch.nn.Identity()
    return m


def load_biot_backbone():
    """BIOT encoder (official six-datasets 18-channel weights; our 16 bipolar channels use
    channel tokens 0-15, the authors' order). Output: the encoder's mean token (256)."""
    torch = _torch()
    from braindecode.models import BIOT

    sd = torch.load(pretrained_path("biot"), map_location="cpu", weights_only=True)
    m = BIOT(n_outputs=1, n_chans=18, n_times=2000, sfreq=200, return_feature=True)
    missing, unexpected = m.load_state_dict({"encoder." + k: v for k, v in sd.items() if k != "index"}, strict=False)
    missing = [k for k in missing if not k.startswith("final_layer")]
    if missing or unexpected:
        raise RuntimeError(f"BIOT weights do not match: missing={missing} unexpected={unexpected}")
    return m


def biot_input(X200: np.ndarray) -> np.ndarray:
    """16 bipolar derivations, each divided by its own 95th percentile of |x| (per epoch), as
    in the authors' data loaders."""
    idx = {c: i for i, c in enumerate(CHANNELS)}
    B = np.stack([X200[:, idx[a]] - X200[:, idx[b]] for a, b in BIOT_BIPOLAR], axis=1)
    return (B / (np.quantile(np.abs(B), 0.95, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)


# --------------------------------------------------------------------------------------
# Frozen embeddings
# --------------------------------------------------------------------------------------
def extract_embeddings(model_name: str, X200: np.ndarray, device: str = "cuda", batch: int = 64, guard=None) -> np.ndarray:
    """Per-epoch frozen embedding: final-layer token features, averaged over the 1-s time
    patches separately for every channel, concatenated over the 19 channels (19 x 200).

    X200: (n, 19, 2000) microvolts at 200 Hz. Scaled by 1/100 (units of 100 uV), exactly as
    in the authors' downstream code.
    """
    torch = _torch()
    if model_name == "cbramod":
        m = load_cbramod_backbone()
    elif model_name == "labram":
        m = load_labram_backbone(X200.shape[-1])
    elif model_name == "biot":
        m = load_biot_backbone()
    else:
        raise ValueError(model_name)
    m.eval().to(device)
    out = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(X200), batch):
            if guard is not None:
                guard.maybe_wait(on_pause=lambda: (m.to("cpu"), torch.cuda.empty_cache()), on_resume=lambda: m.to(device),
                                 context=f"{model_name} embeddings")
            if model_name == "biot":
                x = torch.from_numpy(biot_input(X200[i : i + batch])).to(device)
                f = m(x)
                f = f[-1] if isinstance(f, (tuple, list)) else f  # (b, 256)
                out.append(f.float().cpu().numpy())
                continue
            x = torch.from_numpy(X200[i : i + batch] / 100.0).float().to(device)
            if model_name == "cbramod":
                f = m(x)  # (b, ch, patches, 200)
            else:
                r = m(x, ch_names=CHANNELS, return_features=True)["features"]  # (b, ch*patches, 200), channel-major
                f = r.reshape(x.shape[0], x.shape[1], -1, r.shape[-1])
            out.append(f.mean(2).reshape(x.shape[0], -1).float().cpu().numpy())
    log.info("%s embeddings for %d epochs in %.0f s", model_name, len(X200), time.time() - t0)
    m.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


def embedding_file(model_name: str, cache_key: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / "embeddings" / f"{model_name}__{cache_key}__v1.npy"


def cached_embeddings(model_name: str, X200, cache_key: str, cache_dir: Path, **kw) -> np.ndarray:
    f = embedding_file(model_name, cache_key, cache_dir)
    if f.exists():
        return np.load(f)
    if X200 is None:
        raise ValueError("embedding cache missing and no raw data given")
    E = extract_embeddings(model_name, X200, **kw)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, E)
    f.with_suffix(".json").write_text(json.dumps({"model": model_name, "weights": PRETRAINED[model_name], "shape": list(E.shape),
                                                  "pooling": ("encoder mean token (256)" if model_name == "biot" else
                                                              "per-channel mean over 1-s patches of final-layer tokens, 19x200 concatenated"),
                                                  "input": ("16 TCP bipolar channels, 200 Hz, 10 s, each / its 95th pct |x|" if model_name == "biot"
                                                            else "average reference, 200 Hz, 10 s, uV/100")}, indent=2))
    return E


# --------------------------------------------------------------------------------------
# End-to-end classifier wrapper
# --------------------------------------------------------------------------------------
class TorchEpochClassifier:
    """Harness estimator for a torch network on raw epochs.

    Training (all settings fixed a priori, see results/phase2_plan.md):

    * ``n_bags`` > 1 (default 5): the *training subjects passed to fit* are split into
      ``n_bags`` class-stratified subject folds; one network is trained per fold on the other
      folds and early-stopped on that fold (subject/class-balanced validation cross-entropy,
      patience ``patience``, best state restored); predictions are the mean log-probability
      of the ``n_bags`` networks. ``n_bags=1``: a single ``val_frac`` validation split
      (optionally followed by ``refit`` on all training subjects for the early-stopped
      number of epochs);
    * each training epoch draws ``samples_per_subject`` x n_subjects windows with
      probability proportional to the balanced subject/class weights (so every subject and
      every class contributes equally in expectation), batches of ``batch_size``;
    * optimiser AdamW with cosine annealing over ``max_epochs``;
    """

    def __init__(
        self,
        build,  # callable () -> nn.Module (fresh, possibly pretrained)
        n_classes: int,
        seed: int,
        scale: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        head_lr: float | None = None,  # different lr for parameters whose name starts with "head."
        batch_size: int = 64,
        max_epochs: int = 40,
        patience: int = 8,
        samples_per_subject: int = 32,
        val_frac: float = 0.2,
        refit: bool = False,
        n_bags: int = 5,
        label_smoothing: float = 0.0,
        clip_grad: float | None = None,
        amp: bool = False,
        device: str | None = None,
        guard=None,
        name: str = "net",
        history_log: Path | None = None,
    ):
        self.build = build
        self.n_classes = n_classes
        self.seed = seed
        self.scale = scale
        self.lr = lr
        self.weight_decay = weight_decay
        self.head_lr = head_lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.samples_per_subject = samples_per_subject
        self.val_frac = val_frac
        self.refit = refit
        self.n_bags = n_bags
        self.label_smoothing = label_smoothing
        self.clip_grad = clip_grad
        self.amp = amp
        torch = _torch()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.guard = guard
        self.name = name
        self.history_log = history_log
        self.classes_ = np.arange(n_classes)

    # ---------------------------------------------------------------- helpers
    def _seed_all(self, s):
        torch = _torch()
        import random

        random.seed(s)
        np.random.seed(s % (2**32 - 1))
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _optimizer(self, model):
        torch = _torch()
        if self.head_lr is None:
            return torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        head = [p for n, p in model.named_parameters() if n.startswith("head.")]
        body = [p for n, p in model.named_parameters() if not n.startswith("head.")]
        return torch.optim.AdamW(
            [{"params": body, "lr": self.lr}, {"params": head, "lr": self.head_lr}], weight_decay=self.weight_decay
        )

    def _batches(self, X, idx):
        torch = _torch()
        for i in range(0, len(idx), self.batch_size):
            b = idx[i : i + self.batch_size]
            yield torch.from_numpy(X[b] * self.scale).float().to(self.device, non_blocking=True), b

    def _guard(self, model, context):
        if self.guard is None:
            return
        torch = _torch()

        def pause():
            model.to("cpu")
            torch.cuda.empty_cache()

        self.guard.maybe_wait(on_pause=pause, on_resume=lambda: model.to(self.device), context=context)

    def _logits(self, model, X, idx):
        torch = _torch()
        model.eval()
        outs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and self.device.startswith("cuda")):
            for xb, _ in self._batches(X, idx):
                outs.append(model(xb).float().cpu())
        return torch.cat(outs)

    def _train(self, X, y, groups, tr_idx, n_epochs, va_idx=None, seed=0):
        """Train a fresh model on tr_idx for up to n_epochs (early stopping if va_idx)."""
        torch = _torch()
        F = torch.nn.functional
        self._seed_all(seed)
        model = self.build().to(self.device)
        opt = self._optimizer(model)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        w = balanced_subject_weights(groups[tr_idx], y[tr_idx])
        p = w / w.sum()
        n_subj = len(np.unique(groups[tr_idx]))
        n_draw = self.samples_per_subject * n_subj
        rng = np.random.default_rng(seed)
        yt = torch.from_numpy(y).long()
        if va_idx is not None:
            wv = torch.from_numpy(balanced_subject_weights(groups[va_idx], y[va_idx])).float()
        best = (np.inf, 0, None)
        hist = []
        bad = 0
        for ep in range(n_epochs):
            self._guard(model, f"{self.name} seed {seed} epoch {ep}")
            model.train()
            draw = rng.choice(tr_idx, size=n_draw, replace=True, p=p)
            tl, nb = 0.0, 0
            t0 = time.time()
            for xb, b in self._batches(X, draw):
                yb = yt[b].to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and self.device.startswith("cuda")):
                    out = model(xb)
                loss = F.cross_entropy(out.float(), yb, label_smoothing=self.label_smoothing)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
                opt.step()
                tl += float(loss.detach())
                nb += 1
            sched.step()
            rec = {"epoch": ep + 1, "train_loss": tl / max(nb, 1), "sec": round(time.time() - t0, 1)}
            if va_idx is not None:
                lo = self._logits(model, X, va_idx)
                ce = F.cross_entropy(lo, yt[va_idx], reduction="none")
                vloss = float((ce * wv).sum() / wv.sum())
                sids, P = aggregate_subjects(torch.softmax(lo, 1).numpy(), groups[va_idx])
                ys = np.array([y[va_idx][groups[va_idx] == s][0] for s in sids])
                rec_s = [np.mean(P[ys == c].argmax(1) == c) for c in np.unique(ys)]
                rec.update(val_loss=vloss, val_subject_bal_acc=float(np.mean(rec_s)))
                if vloss < best[0] - 1e-4:
                    best = (vloss, ep + 1, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}))
                    bad = 0
                else:
                    bad += 1
            hist.append(rec)
            log.debug("%s %s", self.name, rec)
            if va_idx is not None and bad >= self.patience:
                break
        if va_idx is not None:
            model.load_state_dict(best[2])
            return model, best[1], hist
        return model, n_epochs, hist

    # ---------------------------------------------------------------- API
    def _val_splits(self, groups, y):
        """Subject-level validation sets carved from the training subjects given to fit."""
        subs = np.unique(groups)
        sub_y = np.array([y[groups == s][0] for s in subs])
        if self.n_bags > 1:
            from sklearn.model_selection import StratifiedKFold

            skf = StratifiedKFold(n_splits=self.n_bags, shuffle=True, random_state=(self.seed + 991) % (2**31))
            return [set(subs[va]) for _, va in skf.split(subs, sub_y)]
        rng = np.random.default_rng(self.seed + 991)
        val = []
        for c in np.unique(sub_y):
            sc = subs[sub_y == c]
            k = max(1, int(round(self.val_frac * len(sc))))
            val += list(rng.choice(sc, k, replace=False))
        return [set(val)]

    def fit(self, X, y, sample_weight=None, groups=None):
        if groups is None:
            raise ValueError("TorchEpochClassifier needs groups (subject ids) for the subject-level validation split")
        groups = np.asarray(groups)
        y = np.asarray(y)
        torch = _torch()
        t0 = time.time()
        self.models_, self.best_epochs_, logs = [], [], []
        for b, val in enumerate(self._val_splits(groups, y)):
            is_val = np.fromiter((g in val for g in groups), bool, len(groups))
            tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
            assert len(va_idx) and not set(groups[tr_idx]) & set(groups[va_idx])
            model, best_ep, hist = self._train(X, y, groups, tr_idx, self.max_epochs, va_idx, seed=self.seed + 7919 * b)
            if self.refit:
                del model
                torch.cuda.empty_cache()
                model, _, _ = self._train(X, y, groups, np.arange(len(y)), max(best_ep, 1), None, seed=self.seed + 7919 * b + 1)
            model.to("cpu")
            self.models_.append(model)
            self.best_epochs_.append(best_ep)
            logs.append({"bag": b, "n_val_subjects": len(val), "best_epoch": best_ep, "history": hist})
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        self.fit_time_ = time.time() - t0
        if self.history_log is not None:
            with open(self.history_log, "a") as fh:
                fh.write(json.dumps({"name": self.name, "seed": self.seed, "n_train_subjects": int(len(np.unique(groups))),
                                     "fit_time_s": round(self.fit_time_, 1), "refit": self.refit, "n_bags": self.n_bags,
                                     "bags": logs}) + "\n")
        return self

    def predict_proba(self, X):
        """Mean log-probability over the bagged models (normalised geometric mean)."""
        torch = _torch()
        logp = 0.0
        for m in self.models_:
            m.to(self.device)
            logp = logp + torch.log_softmax(self._logits(m, X, np.arange(len(X))), 1)
            m.to("cpu")
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return torch.softmax(logp / len(self.models_), 1).numpy()


class CBraModClassifier:
    """CBraMod backbone (official pretrained weights) + the authors' light
    'avgpooling_patch_reps' head: mean over channels and patches -> Linear(200, n_classes)."""

    @staticmethod
    def build(n_classes):
        torch = _torch()

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = load_cbramod_backbone()
                self.head = torch.nn.Linear(200, n_classes)

            def forward(self, x):
                f = self.backbone(x)  # (b, ch, patches, 200)
                return self.head(f.mean(dim=(1, 2)))

        return Net()
