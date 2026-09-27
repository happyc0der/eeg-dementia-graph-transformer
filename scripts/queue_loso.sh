#!/usr/bin/env bash
# Binary leave-one-subject-out experiments (comparable with the dataset authors).
cd "$(dirname "$0")/.."
NJ=${NJ:-8}
for t in loso_ad_cn loso_ftd_cn; do
  uv run python scripts/run_phase1.py --n-jobs $NJ --task $t --skip-existing --models chance_prior,rbp_lr,spectral_lr,all_lr,riemann_ts_lr,age_lr
done
uv run python scripts/run_phase1.py --n-jobs $NJ --task loso3 --skip-existing --models chance_prior,rbp_lr,spectral_lr,all_lr
echo LOSO_DONE
