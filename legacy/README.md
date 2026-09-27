# Legacy: the original Spring-2025 course project

This folder holds the code, figures and logged results of the original team project for the NYU *Neuroinformatics* course (Spring 2025): a **spatial-spectral graph transformer** for three-way AD / FTD / CN classification from resting-state EEG (OpenNeuro ds004504), plus the baselines the team compared it with.

It is **kept for reference and credit, not maintained**. The files were moved here unchanged (with `git mv`, so `git log --follow` shows their history) when the project was re-evaluated in 2026. The maintained code is the [`eegdementia`](../eegdementia/) package, and the current results are in [`../results/`](../results/) and the [top-level README](../README.md).

## Credits

- **Team (NYU Neuroinformatics, Spring 2025):** [Subhrajit Dey (@subro608)](https://github.com/subro608), who wrote most of the model and training code, [Keshav Rajput (@happyc0der)](https://github.com/happyc0der), [Sirish Visweswar (@itsSirish)](https://github.com/itsSirish) and [@terka2610](https://github.com/terka2610). The original shared repository is [subro608/Neuroinformatics](https://github.com/subro608/Neuroinformatics). This repo is a cleaned-up snapshot with a fresh history.
- The EEGNet baseline and the original data-prep script build on [Leofierus/eeg-alzheimers-detection](https://github.com/Leofierus/eeg-alzheimers-detection).

## What the old pipeline reported

The unit of evaluation was a 15-second EEG chunk (95 Hz, 1425 samples). The 88 subjects were split once, 80/20 by subject (pandas `sample(frac=0.8, random_state=42)`), giving **18 held-out subjects** (7 AD, 6 CN, 5 FTD; 873 chunks).

| Model (checkpoint) | Params | Cross-subject acc. | Cross-subject bal. acc. | Cross-subject weighted F1 | Within-subject acc. |
|---|---:|---:|---:|---:|---:|
| Spatial-spectral graph transformer, `dim=128` | 1.67 M | 63.6 % | 62.2 % | 62.7 % | 97.4 % |
| Spatial-spectral graph transformer, `dim=264` | 7.14 M | 60.1 % | 58.5 % | 59.1 % | 98.6 % |
| Multi-view transformer (MVT) baseline¹ | – | 64.5 % | – | – | 89.8 % |
| RBF-SVM on hand-crafted features¹ | – | 48.6 % | – | – | 71.2 % |
| Chance (majority class) | – | 36.5 % | 33.3 % | – | 42.4 % |

- The two graph-transformer rows come from [`results/spatial_spectral_dim128_20250405/reeval_matched_features/`](results/spatial_spectral_dim128_20250405/reeval_matched_features/) and the `dim264` equivalent (a CPU re-evaluation that agrees with the April 2025 GPU logs to within 0.5 pp). Per-class cross-subject ROC-AUC for `dim=128` was AD 0.79, CN 0.84, FTD 0.65.
- *Within-subject* accuracy is measured on held-out chunks of the 70 *training* subjects, so it measures memorisation of subject identity as much as disease.
- ¹ Taken from the team's saved artefacts, not re-run: the SVM from [`results/baselines/svm/`](results/baselines/svm/), the MVT from its confusion matrices in [`results/baselines/figures/`](results/baselines/figures/). The EEGNet figures in that folder could not be traced to the same split and are not reported.

## How these numbers compare with the re-evaluation

The 2026 re-evaluation ([`../results/`](../results/)) kept the dataset and the task but changed the protocol (see the top-level README):

- **On the same 18 test subjects**, a logistic regression on spectral features reaches 67.4 % epoch-level accuracy (10 s epochs) and `all_lgbm` 68.8 %, against the graph transformer's 63.6 % chunk-level accuracy ([`../results/tables/legacy.md`](../results/tables/legacy.md), [`../results/phase2/tables/legacy.md`](../results/phase2/tables/legacy.md)). With 18 subjects the subject-level 95 % CI is about ±20 pp, so this split cannot rank models.
- **Under 10 × repeated, nested, subject-grouped 5-fold CV** on all 88 subjects, the best models reach about 60-62 % subject-level balanced accuracy ([`../results/tables/cv3.md`](../results/tables/cv3.md)). The 18-subject split is easier than average: `spectral_lr` scores 66.8 % subject-level balanced accuracy on it against 60.5 % on average.

## Known problems (why the project was re-evaluated)

1. **Leaky chunk-level cross-validation.** The 5-fold CV inside `train_kfold_multi_spatial_graph_spectral_advanced.py` split *chunks*, not subjects, so chunks of the same recording were in both training and validation folds. Validation accuracy was about 99 %, and the "best" checkpoint was chosen on these leaky folds.
2. **A single 18-subject test split.** One 80/20 split of 88 subjects cannot rank models: the subject-level CI is about ±20 pp, and the chance-level spread of a single 5-fold CV on 88 subjects is already ±6 pp (1 SD).
3. **Native-reference artefact.** In the derivative recordings as distributed (A1-A2 reference), about 92 % of every channel's variance is one common-mode, mostly < 2 Hz signal of about 32 µV SD in every subject, with no group difference. The old pipeline z-scored these native-reference channels, so most of its raw-signal input was this artefact. The re-evaluation uses the average reference, which removes it ([`../results/phase1_summary.md`](../results/phase1_summary.md), section 2).
4. **The end crop removed 30 samples, not 30 s.** `data_prep.py` calls `raw.crop(tmin=30, tmax=raw.times[-30])`: `raw.times[-30]` is the time of the 30th-last *sample*, so only about 60 ms were dropped at the end of each recording (the start crop of 30 s was correct).
5. Smaller issues: classes were balanced by *undersampling* chunks (729 per class) rather than by weighting; the within-subject test set rewards memorisation; ASR `boundary` discontinuities were not excluded from chunks; and the old `baselines/` scripts were never evaluated on a common protocol.

## Running the old code

Everything below runs from **this folder** (`cd legacy`). It uses its own requirements ([`requirements.txt`](requirements.txt)), not the top-level `pyproject.toml`.

```bash
cd legacy
uv venv --python 3.12 && source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt                      # GPU: install a CUDA build of torch first

# 1. chunk the ds004504 derivatives (~540 MB) into model-data/{train,test}/ + labels.json
python data_prep.py --bids-root path/to/ds004504 --output-dir model-data
# 2. train (5-fold chunk-level CV, up to 200 epochs/fold; ~8 h on an RTX 3080 Ti Laptop)
python train_kfold_multi_spatial_graph_spectral_advanced.py --data-dir model-data --dim 128
# 3. evaluate a checkpoint on the cross- and within-subject test sets
python test_multispatial_graph_spectral_advanced.py --model-path spatial_spectral_<timestamp>/models/best_model_overall.pth --dim 128
```

The scripts also read `EEG_DATA_DIR` for the chunked-data folder. Training writes `spatial_spectral_<timestamp>/`, evaluation writes `spatial_results_<timestamp>/` (both git-ignored). The baselines run the same way, e.g. `python baselines/train_kfold_svm.py`, and expect `model-data/` in the working directory.

### Trained weights (v1.0 release)

The weights are attached to the [v1.0 release](https://github.com/happyc0der/eeg-dementia-graph-transformer/releases/tag/v1.0), not tracked in git:

```bash
gh release download v1.0 -R happyc0der/eeg-dementia-graph-transformer -p eeg-graph-transformer-dim128.pth
python test_multispatial_graph_spectral_advanced.py --model-path eeg-graph-transformer-dim128.pth --dim 128
```

| Release asset | SHA-256 |
|---|---|
| `eeg-graph-transformer-dim128.pth` (use `--dim 128`) | `28eded145850e31fd688c1f689bf3b34aba7f6091f0d53c68b764adf99235051` |
| `eeg-graph-transformer-dim264.pth` (use `--dim 264`) | `cf8a7d42af05921237e5861ccfccca2c7b3c95566b7c477c98b88a2b4364f668` |
| `svm-baseline-models.zip` | `712644d214ab0adaf5b6b7f58d6e73538fe89ad1bd9abc61d4954eaeacaba2dc` |

Check a download with `sha256sum <file>` (PowerShell: `Get-FileHash <file>`).

## What is in this folder

```
legacy/
├── data_prep.py                                         # ds004504 -> 15 s chunks + labels.json (80/20 subject split, seed 42)
├── eeg_dataset_multispatialgraph_spectral_advanced.py   # Dataset: raw EEG, band-power graph features, adjacency
├── eeg_multi_spatial_graph_spectral_advanced.py         # MVTSpatialSpectralModel (two-branch graph transformer)
├── train_kfold_multi_spatial_graph_spectral_advanced.py # training (chunk-level k-fold CV)
├── test_multispatial_graph_spectral_advanced.py         # evaluation on held-out subjects / chunks
├── requirements.txt
├── baselines/   # SVM, EEGNet, multi-view transformer, CWT, multiscale / time-graph and earlier variants
├── tools/       # cuda_test.py, data_vis.py (MNE signal browser)
├── docs/        # multi-view transformer architecture diagram
└── results/     # logged metrics, confusion matrices and training logs of the April 2025 models and the baselines
```

| Baseline scripts | Model |
|---|---|
| `train_kfold_svm.py`, `train_kfold_svm_grid.py`, `test_svm.py` | RBF-SVM on per-channel statistical and spectral features |
| `train_kfold.py`, `train_batch.py`, `test.py`, `hyperparameter_tuning.py` | EEGNet-style CNN (Optuna tuning) |
| `train_kfold_mvt.py`, `test_mvt.py` (+ `_mfeat` variants) | Multi-view transformer: time, frequency and spatial views with cross-view attention ([diagram](docs/mvt_architecture.svg)) |
| `train_kfold_multiscale_spectral.py`, `test_multiscale_spectral.py` | Multi-scale spectral graph transformer |
| `train_kfold_multi_time_graph_spectral.py`, `test_multitime_graph_spectral.py` | Multi-scale time-window graph variant |
| `train_kfold_multi_spatial_graph_spectral.py`, `test_multispatial_graph_spectral.py` | First version of the spatial-spectral model |
| `train_kfold_cwt_eeg.py` | Continuous-wavelet (CWT) time-frequency transformer/CNN |
| `train_batch_new_mvt.py`, `eeg_eegpt_mvt.py` | EEGPT-based experiment. Needs the external EEGPT code and pretrained weights (not included), so it is not runnable standalone |
