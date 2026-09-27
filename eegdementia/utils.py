"""Small runtime helpers."""

import os


def be_nice(threads: int = 1) -> None:
    """Run CPU-only, single-threaded BLAS per process, at below-normal priority.

    Child processes (joblib/loky workers) inherit the below-normal priority class on
    Windows, so interactive use of the machine stays responsive during long runs.
    """
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(v, str(threads))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    try:
        import psutil

        p = psutil.Process()
        if os.name == "nt":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
    except Exception:  # pragma: no cover - best effort only
        pass
