#!/usr/bin/env bash
# Reproduce every phase-1 result (CPU only; several hours on a 24-thread laptop).
# Data: OpenNeuro ds004504 in $EEG_BIDS_ROOT (default C:/AI/datasets/ds004504).
set -euo pipefail
cd "$(dirname "$0")/.."
R="uv run python scripts/run_phase1.py"

# 1. caches (10 s / 5 s default; 4 s and 30 s windows and native reference for sensitivity)
uv run python scripts/build_cache.py
uv run python scripts/build_cache.py --epoch-length 4 --epoch-step 2
uv run python scripts/build_cache.py --epoch-length 30 --epoch-step 15 --keep-boundaries
uv run python scripts/build_cache.py --reference native

# 2. main 3-class repeated nested CV (all models, incl. confound checks and ensembles)
$R --task cv3 --skip-existing
# 3.-6. subject-mean models, binary LOSO, legacy split, binary CV, sensitivity analyses
bash scripts/queue_phase1.sh
bash scripts/queue_loso.sh
# 7. permutation tests, interpretability, figures
uv run python scripts/permutation_test.py --model spectral_lr --n-perm 100
uv run python scripts/permutation_test.py --model rbp_lr --n-perm 500
uv run python scripts/interpret.py --model spectral_lr --repeats 3
uv run python scripts/make_figures.py --best spectral_lr
