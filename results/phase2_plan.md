# Phase 2 pre-registration: deep models under the phase-1 harness

**Written and committed before any phase-2 model was evaluated on an outer test fold.**
Branch `overhaul`, parent commit 5ec336c (phase 1). Date 2026-09-27.

What was done before this file was committed, and why it does not leak test information:

- Code and dependencies were set up (`eegdementia/deep.py`, `eegdementia/phase2.py`, `eegdementia/gpu_guard.py`, the `deep` / `cu128` extras).
- Pretrained weights were downloaded and checked (section 3).
- Frozen embeddings were computed for every epoch. They are fixed transforms with pretrained weights and use no labels.
- A **timing smoke test** was run only on the *training* subjects of outer split (repeat 0, fold 0): 56 subjects for training and 14 pseudo-test subjects, none of them in that split's test fold. It measured run time and checked that training runs. Its accuracies were printed but were not used to choose any setting below. After the timing it showed, the early-stopping design was switched from "single 80/20 split + refit" to the 5-fold bagging below; that choice is motivated by variance (early stopping on about 11 subjects picks anywhere from epoch 2 to epoch 14) and by compute, not by accuracy.

## 1. Task, data, splits, metrics (identical to phase 1)

- **Data:** ds004504 derivatives, 88 subjects (36 AD, 29 CN, 23 FTD).
- **Preprocessing:** average reference; 250 Hz; 30 s cropped at each end.
- **Epochs:** 10 s windows with a 5 s step. Boundary and amplitude rejection as in phase 1, which leaves **11,616 epochs**. (The phase-1 summary text says 11,796; the phase-1 code and results used 11,616. See the phase-2 summary.)
- **Outer splits:** `outer_splits(subjects, 5, 10, seed=2026)`. These are exactly phase 1's stratified-group 5-fold splits × 10 repeats. Model seed per outer fold = 2026 + 17·repeat + fold, as `run_cv` sets it.
- **Aggregation:** subject probability = softmax(mean epoch log-probability).
- **Primary metric:** 3-class **subject-level balanced accuracy**, pooled over the 5 test folds of a repeat, then mean ± SD over repeats, with a 95 % class-stratified bootstrap CI over subjects (2,000 resamples).
- **Secondary metrics:** subject accuracy, macro-F1, macro one-vs-rest AUC, per-class recall and AUC, and epoch-level metrics.
- **Primary comparator:** phase 1's best pre-specified model, `ensemble_vote` (61.1 ± 3.0 %), on the same 10 repeats. `spectral_lr` (60.5 %) and `nested_select` (58.3 %) are also shown.

## 2. Inputs (fixed, label-free, per epoch)

| input | used by | definition |
|---|---|---|
| `raw` | ShallowFBCSPNet | phase-1 epochs, 19 × 2500 (250 Hz), µV × 0.1 |
| `raw125` | EEGNet | `scipy.signal.resample_poly` 250→125 Hz, 19 × 1250, µV × 0.1 |
| `raw200` | CBraMod fine-tuning, embeddings | `resample_poly` 250→200 Hz, 19 × 2000 (ten 1-s patches), µV / 100, as in the authors' code |
| `emb_cbramod`, `emb_labram` | frozen-feature models | final-layer token features of the pretrained encoder, averaged over the ten 1-s patches **separately for each channel**, then concatenated: 19 × 200 = 3,800 per epoch |
| `spec_feat` | hybrid | phase 1's 399 spectral and aperiodic features |

- No statistic is computed across epochs or subjects outside `fit`. Scaling is a fixed constant.
- The foundation models contain only per-sample normalisation (LayerNorm and GroupNorm).
- CNN BatchNorm layers use running statistics learned in `fit`.

## 3. Models and hyper-parameters (fixed; no tuning on outer folds)

### 3a. Frozen foundation models + logistic regression (primary deep family)

| name | encoder | weights |
|---|---|---|
| `cbramod_lr` (**primary FM model**) | CBraMod (Wang et al., ICLR 2025), 4.9 M params, criss-cross transformer, 200 Hz, 1-s patches, channel-agnostic positional encoding | official `pretrained_weights.pth` from https://huggingface.co/weighting666/CBraMod (commit 500543c, linked from github.com/wjq-learning/CBraMod@b9e9610). sha256 0792cb80…57178, 19.8 MB. Code MIT; HF card Apache-2.0 |
| `labram_lr` | LaBraM-Base (Jiang et al., ICLR 2024), 5.8 M params, 200 Hz, 1-s patches, learned position embedding per 10-20 electrode (T3/T4/T5/T6 are in its channel list) | official `checkpoints/labram-base.pth` (student encoder) from github.com/935963004/LaBraM@c431221. sha256 7c505838…c57c37c, 96.6 MB. MIT |

- Encoders run through `braindecode` 1.8.1. For CBraMod, the pretraining reconstruction head `proj_out` is replaced by the identity, as in the authors' downstream code. (braindecode's `return_encoder_output=True` still applies `proj_out`.)
- The implementations were checked against the official code on random input: max |difference| = 0.0 for both CBraMod and LaBraM.
- The braindecode Hugging Face copies were checked to be tensor-identical to the official files. LaBraM is loaded from `braindecode/labram-pretrained@0563b6c` because of its parameter names, and every tensor is re-checked against the official student weights at load time.
- **Head:** phase 1's `lr_factory` (median imputer → StandardScaler → L2 logistic regression). C ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1} is chosen by the same inner stratified 5-fold subject CV on the outer-training subjects. Epochs are weighted with balanced subject/class weights.
- **Not run:**
  - EEGPT: the official checkpoint is only on a figshare private link that blocks scripted download, and braindecode's re-hosted `eegpt-pretrained` config (62 channels, 250 Hz, 1000 samples) does not match the official checkpoint (58 channels, 256 Hz, 4 s), so its provenance cannot be verified.
  - BIOT: its public weights expect a bipolar (TCP) or 16/18-channel montage. It is listed as an optional extra (section 7).

### 3b. End-to-end networks (braindecode 1.8.1 implementations)

All three use the same training protocol ("5-fold subject bagging"):

1. Inside `fit`, the *outer-training* subjects (~70) are split into 5 class-stratified subject folds (seeded).
2. For each fold, one network is trained on the other 4 folds (~56 subjects) and early-stopped on that fold (~14 subjects):
   - the stopping criterion is the subject/class-balanced validation cross-entropy over all of the validation subjects' epochs;
   - patience is P epochs, and the best state is restored.
3. Test predictions are the mean log-probability of the 5 networks.

No outer-test subject is ever used for early stopping. The validation subjects are training subjects only, which `tests/test_leakage.py` checks.

- **Sampling:** each training epoch draws 32 × n_subjects windows (with replacement), with probability proportional to phase 1's balanced subject/class weights. Every subject and class therefore contributes equally in expectation.
- **Optimiser:** AdamW with a cosine schedule over `max_epochs` and batch size 64. There is no data augmentation.

| name | network (library defaults) | input | lr | weight decay | max epochs / patience | other |
|---|---|---|---|---|---|---|
| `eegnet` | EEGNet v4 (F1 = 8, D = 2, kernel 64, dropout 0.25) | raw125 | 1e-3 (Lawhern et al. 2018: Adam 1e-3) | 0 | 40 / 8 | fp32 |
| `shallow` | ShallowFBCSPNet (40 temporal filters of length 25, 40 spatial, pool 75/15, dropout 0.5) | raw | 6.25e-4 (braindecode's recommended value) | 0 | 40 / 8 | fp32 |
| `cbramod_ft` | CBraMod (official weights, `proj_out` removed) + the authors' light `avgpooling_patch_reps` head (mean over channels and patches → Linear(200, 3)) | raw200 | backbone 1e-4, head 5e-4 (authors' `multi_lr`) | 5e-2 | 15 / 4 | label smoothing 0.1, grad-clip 1.0, bf16 autocast (authors' defaults; epochs capped because training converges in ≤ 6 short epochs on this data) |

Timing from the smoke test (fold of ~56 training subjects): EEGNet 45 s, ShallowFBCSPNet 33 s and CBraMod fine-tuning 115 s per outer fold.

### 3c. Hybrids

- **`cbramod_spectral_lr`:** the CBraMod embedding (3,800) concatenated with the 399 spectral/aperiodic features, fed to the same LR head (inner-CV C).
- **`ensemble_vote+cbramod_lr`** (**primary hybrid**): phase 1's soft vote with `cbramod_lr` added as a 4th member, i.e. the mean epoch log-probability of spectral_lr, riemann_ts_lr, all_lgbm and cbramod_lr.
  - Each member is tuned by its own inner CV, exactly as in phase 1.
  - It is computed from the members' saved outer-fold predictions. Soft voting of independently tuned members is separable, so this is identical to running the 4-member `ModelSpec(combine="vote")`. Recomputing phase 1's 3-member vote this way reproduces its stored predictions exactly (max |Δp| = 0).
- **`ensemble_vote+cbramod_ft`:** the same with `cbramod_ft` as the 4th member (secondary).
- **Post hoc (flagged as such):** `ensemble_vote + <best end-to-end deep model by cv3 mean>`, if that model is not one of the above.

## 4. Repeats and compute

- **All models use all 10 outer repeats** (50 outer folds), because the smoke test shows this fits the budget: about 45 + 35 + 120 min of GPU time and under 1 h of CPU time for the LR heads.
- If the time budget (finish by about 11:00) runs out, end-to-end models are completed in repeat order and at least 5 repeats are reported. The number of repeats depends only on wall-clock time, and the summary says so.
- Paired comparisons always use the repeats both models have.
- **GPU etiquette:** `GpuGuard` runs before every outer fold and at most every 5 min inside training. If it finds a foreign GPU user (Ollama model loaded, another CUDA process, WSL job, game or emulator, new Docker container, or GPU load ≥ 15 % while we are idle), it moves our model to the CPU, frees VRAM and polls every 10 min. Pauses are logged (`results/overhaul/phase2/gpu_pauses.jsonl`) and reported.

## 5. Analyses

1. **3-class cv3 (5 × 10), all models in section 3.** Same metrics and CI as phase 1.
2. **Paired comparison against `ensemble_vote`** (and `spectral_lr`) on identical folds:
   - per-repeat differences in subject balanced accuracy (mean, SD, number of repeats the deep model wins);
   - a paired class-stratified subject bootstrap (2,000 resamples) of the repeat-averaged difference, reported as a 95 % CI and P(Δ ≤ 0).

   Repeats share the same 88 subjects, so the per-repeat SD is **not** used for a p-value.

   **Decision rule, fixed now:** a deep model is called "better" only if the bootstrap 95 % CI of Δ excludes 0. Given phase 1's ±7 pp subject CI, we expect differences under about 5 pp to be undemonstrable.
3. **AD vs FTD (binary, `cv_ad_ftd`, 5 × 10, 59 subjects):**
   - all frozen-FM models and `cbramod_spectral_lr` (cheap);
   - the best end-to-end model by 3-class cv3 mean (post-hoc choice, flagged), trained as a binary model.

   Comparators are phase 1's `rbp_lr` (62.5 %) and `spectral_lr` (60.0 %).
4. **Legacy 18-subject split** (70 train / 18 test, same subjects as the April 2025 model): every phase-2 model, with epoch- and subject-level metrics, for comparison with the old graph transformer (63.6 % chunk accuracy). Not used for ranking.
5. **Permutation test** (if CPU time allows): `cbramod_lr`, 100 subject-label permutations of the whole nested pipeline (1 × 5-fold, as for phase-1 `spectral_lr`). End-to-end permutation tests are unaffordable and will not be run.
6. **Figures:** a model comparison with phase 1, paired-difference plots, confusion matrices and per-class recall for the primary models.

## 6. What will not be done

- There is no tuning of any setting on outer test folds, and no switching to 4 s windows. The 4 s windows looked better in phase 1, but they are not pre-specified here.
- No model is re-run with different settings after its results are seen. If a bug is found, the fix and both results will be reported.

## 7. Optional extras (only if time remains; reported as exploratory)

- Frozen BIOT (six-datasets 18-channel weights) on a 16-channel bipolar TCP montage derived from the 10-20 channels.
- LaBraM fine-tuning with the same protocol as `cbramod_ft`.
