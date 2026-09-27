"""Small runtime helpers."""

import logging
import os

import psutil

log = logging.getLogger(__name__)


def be_nice(threads: int = 1) -> None:
    """Make a long CPU run polite: one BLAS thread per process, below-normal priority, no GPU.

    * ``OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS`` are set to ``threads`` **only if unset**, so
      parallelism comes from joblib processes rather than from nested BLAS threads.
    * ``CUDA_VISIBLE_DEVICES`` is set to ``-1`` (hide every GPU) **only if unset**. A caller
      that wants the GPU sets it first (``scripts/run_phase2.py --gpu`` sets it to ``0``); to
      use a GPU from any other script, export ``CUDA_VISIBLE_DEVICES`` yourself. It only has
      an effect if CUDA is not initialised yet, so call ``be_nice()`` before importing torch.
    * The process priority is lowered (below-normal on Windows, nice 10 elsewhere).
      joblib/loky worker processes inherit it.
    """
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(v, str(threads))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    try:
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except (OSError, psutil.Error) as e:  # best effort: a normal-priority run is still correct
        log.warning("could not lower the process priority: %s", e)
