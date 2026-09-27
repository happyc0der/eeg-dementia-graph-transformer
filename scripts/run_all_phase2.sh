#!/usr/bin/env bash
# Reproduce every phase-2 result in results/phase2/ (see results/phase2_plan.md).
# Needs the phase-1 main cache and the phase-1 cv3 / legacy / cv_ad_ftd runs (the votes are built
# from their saved epoch-level predictions in $EEG_CACHE_DIR/predictions), the deep + cu128 extras,
# a CUDA GPU for the end-to-end networks (about 6.5 h on an RTX 3080 Ti Laptop) and the pretrained
# weights (downloaded to $EEG_WEIGHTS_DIR and sha256-checked on first use).
set -euo pipefail
cd "$(dirname "$0")/.."
NJ=${NJ:-10}
P="uv run python scripts/run_phase2.py --skip-existing"

# frozen foundation-model embeddings + LR heads (embeddings are computed once and cached; slow on CPU)
$P --task cv3 --models cbramod_lr,labram_lr,cbramod_spectral_lr,biot_lr --n-jobs "$NJ"
$P --task legacy --models cbramod_lr,labram_lr,cbramod_spectral_lr,biot_lr --n-jobs "$NJ"
$P --task cv_ad_ftd --models cbramod_lr,labram_lr,cbramod_spectral_lr --n-jobs "$NJ"
# end-to-end networks (GPU, sequential, checkpointed per outer fold, GPU guard between folds)
$P --task cv3 --models eegnet,shallow,cbramod_ft --gpu
$P --task legacy --models eegnet,shallow,cbramod_ft --gpu
$P --task cv_ad_ftd --models shallow --gpu     # post hoc: best end-to-end model on cv3
# permutation test of the primary foundation-model pipeline (CPU, about 3 h on 12 workers)
uv run python scripts/permutation_test.py --phase2 --model cbramod_lr --n-perm 50 --n-jobs 12
# soft votes (pre-registered + post hoc members), paired comparisons, tables, figures
uv run python scripts/phase2_report.py --posthoc-members labram_lr,shallow
uv run python scripts/make_readme_tables.py
