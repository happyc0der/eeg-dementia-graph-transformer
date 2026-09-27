#!/usr/bin/env bash
# Reproduce every phase-1 result in results/ (CPU only; about 15-20 h on a 24-thread laptop).
# Needs OpenNeuro ds004504 in $EEG_BIDS_ROOT (see README, "Data"). NJ = parallel processes.
set -euo pipefail
cd "$(dirname "$0")/.."
NJ=${NJ:-12}
R="uv run python scripts/run_phase1.py --n-jobs $NJ --skip-existing"

# 1. caches: 10 s / 5 s epochs (main); 4 s and 30 s windows and the native reference (sensitivity)
uv run python scripts/build_cache.py --n-jobs "$NJ"
uv run python scripts/build_cache.py --n-jobs "$NJ" --epoch-length 4 --epoch-step 2
uv run python scripts/build_cache.py --n-jobs "$NJ" --epoch-length 30 --epoch-step 15 --keep-boundaries
uv run python scripts/build_cache.py --n-jobs "$NJ" --reference native

# 2. main result: 3-class, 5-fold x 10 repeats, nested, every phase-1 model (incl. confound checks, ensembles)
$R --task cv3

# 3. legacy 18-subject split, binary repeated CV, subject-mean models
$R --task legacy --models chance_prior,rbp_lr,spectral_lr,all_lr,all_linsvm,riemann_ts_lr,all_lgbm,age_sex_lr,ensemble_vote,nested_select
for t in cv_ad_cn cv_ftd_cn cv_ad_ftd; do
  $R --task $t --models rbp_lr,spectral_lr,age_lr
done
$R --task cv3 --subject-mean --tag subjmean --models chance_prior,rbp_lr,spectral_lr,spectral_region_lr,all_lr,riemann_ts_lr

# 4. sensitivity analyses: native reference, 4 s and 30 s windows
$R --task cv3 --reference native --tag native_ref --models rbp_lr,spectral_lr,all_lr
$R --task cv3 --epoch-length 4 --epoch-step 2 --tag win4s --models rbp_lr,spectral_lr
$R --task cv3 --epoch-length 30 --epoch-step 15 --keep-boundaries --tag win30s --models rbp_lr,spectral_lr

# 5. leave-one-subject-out: binary (comparable with the literature) and 3-class
for t in loso_ad_cn loso_ftd_cn; do
  $R --task $t --models chance_prior,rbp_lr,spectral_lr,all_lr,riemann_ts_lr,age_lr
done
$R --task loso3 --models chance_prior,rbp_lr,spectral_lr,all_lr

# 6. permutation tests, interpretability, tables and figures
uv run python scripts/permutation_test.py --model spectral_lr --n-perm 100 --n-jobs "$NJ"
uv run python scripts/permutation_test.py --model rbp_lr --n-perm 500 --n-jobs "$NJ"
uv run python scripts/interpret.py --model spectral_lr --repeats 3 --n-jobs "$NJ"
uv run python scripts/make_figures.py --best ensemble_vote --compare rbp_lr,spectral_lr,all_lgbm,ensemble_vote,nested_select
uv run python scripts/make_readme_tables.py
