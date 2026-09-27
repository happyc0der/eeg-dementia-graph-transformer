#!/usr/bin/env bash
# Remaining phase-1 experiments after the main cv3 run (sequential, below-normal priority).
cd "$(dirname "$0")/.."
NJ=${NJ:-20}
R="uv run python scripts/run_phase1.py --n-jobs $NJ"
$R --task cv3 --subject-mean --tag subjmean --models chance_prior,rbp_lr,spectral_lr,spectral_region_lr,all_lr,riemann_ts_lr --skip-existing
$R --task legacy --skip-existing --models chance_prior,rbp_lr,spectral_lr,all_lr,all_linsvm,riemann_ts_lr,all_lgbm,age_sex_lr,ensemble_vote,nested_select
for t in cv_ad_cn cv_ftd_cn cv_ad_ftd; do
  $R --task $t --skip-existing --models rbp_lr,spectral_lr,age_lr
done
$R --task cv3 --reference native --tag native_ref --models rbp_lr,spectral_lr,all_lr --skip-existing
$R --task cv3 --epoch-length 4 --epoch-step 2 --tag win4s --models rbp_lr,spectral_lr --skip-existing
$R --task cv3 --epoch-length 30 --epoch-step 15 --keep-boundaries --tag win30s --models rbp_lr,spectral_lr --skip-existing
echo QUEUE_DONE
