"""Phase-2 runner: deep models through the phase-1 harness (see results/phase2_plan.md).

Examples
--------
CPU (frozen-embedding heads; embeddings are computed on first use, which is slow on CPU):
    uv run python scripts/run_phase2.py --task cv3 --models cbramod_lr,labram_lr --n-jobs 10
GPU (end-to-end networks, sequential, checkpointed per outer fold, GPU guard between folds):
    uv run python scripts/run_phase2.py --task cv3 --models eegnet --gpu

Only the inputs the requested models need are loaded (raw epochs for the end-to-end
networks, one embedding per frozen foundation model). Results go to results/phase2/<task>/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cv3")
    ap.add_argument("--models", required=True, help="comma-separated names from eegdementia.phase2.phase2_specs()")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=10, help="parallel outer folds (CPU runs only)")
    ap.add_argument("--gpu", action="store_true", help="use CUDA device 0 (sequential, checkpointed, GPU guard)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-guard", action="store_true", help="do not yield the GPU to other users' jobs")
    args = ap.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from eegdementia.utils import be_nice

    be_nice(4 if args.gpu else 1)

    from eegdementia import experiments as E
    from eegdementia import phase2 as P2
    from eegdementia.config import CACHE_DIR

    if args.task not in E.TASKS:
        ap.error(f"unknown task {args.task!r}; choose from {list(E.TASKS)}")
    specs = P2.phase2_specs()
    models = [m for m in args.models.split(",") if m]
    unknown = [m for m in models if m not in specs]
    if unknown:
        ap.error(f"unknown model(s) {unknown}; choose from {sorted(specs)}")
    need = P2.inputs_needed(specs, models)
    raw = bool(need & set(P2.RAW_INPUTS))
    fms = tuple(m for m in P2.FM_NAMES + P2.EXPLORATORY_FM if f"emb_{m}" in need)
    device = "cpu"
    if args.gpu:
        import torch

        if not torch.cuda.is_available():
            sys.exit("--gpu given but CUDA is not available (install the cu128 extra)")
        device = "cuda"

    out_root = P2.PHASE2_DIR
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / f"run_{args.task}.log")],
    )
    log = logging.getLogger("run_phase2")
    guard = None
    if args.gpu and not args.no_guard:
        from eegdementia.gpu_guard import GpuGuard

        guard = GpuGuard(pause_log=out_root / "gpu_pauses.jsonl")
    ds = P2.build_phase2_dataset(raw=raw, embeddings=fms, device=device, guard=guard)
    hist = log_dir / f"train_history_{args.task}.jsonl"
    specs = P2.phase2_specs(guard=guard, history_log=hist)
    device_name = "cpu"
    if args.gpu:
        import torch

        device_name = f"cuda:0 ({torch.cuda.get_device_name(0)})"
    for model in models:
        outdir = out_root / (args.task + (f"__{args.tag}" if args.tag else "")) / model
        if args.skip_existing and (outdir / "summary.json").exists():
            log.info("skip %s/%s (exists)", args.task, model)
            continue
        log.info("=== %s / %s (%d repeats) ===", args.task, model, args.repeats)
        t0 = time.time()
        kw = {}
        if args.gpu:
            ck = CACHE_DIR / "phase2_ckpt" / (args.task + (f"__{args.tag}" if args.tag else "")) / model

            def before_fold(sp, model=model):
                if guard is not None:
                    guard.wait_until_free(context=f"{model} r{sp.repeat} f{sp.fold}")

            kw = {"checkpoint_dir": ck, "before_fold": before_fold}
        extra = {"phase": 2, "n_repeats_run": args.repeats, "device": device_name}
        if guard is not None:
            extra["gpu_pauses_so_far"] = guard.pauses
        summ = E.run_and_save(args.task, model, ds, specs, n_jobs=1 if args.gpu else args.n_jobs,
                              n_repeats=args.repeats, out_root=out_root, tag=args.tag, extra_meta=extra, **kw)
        s = summ["subject"]
        log.info("%s/%s: subject bal acc %.3f +- %.3f %s | macro AUC %.3f | %.1f min", args.task, model,
                 s["balanced_accuracy"]["mean"], s["balanced_accuracy"]["sd"], s["balanced_accuracy"].get("ci95"),
                 s["macro_auc"]["mean"], (time.time() - t0) / 60)
    if guard is not None and guard.pauses:
        log.info("GPU pauses: %s", json.dumps(guard.pauses))


if __name__ == "__main__":
    main()
