# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Amyloid core prediction pipeline. Given a protein sequence, predict which residues belong to an amyloid core region. Based on Gonay et al. (Alzheimer's Dement. 2025). Binary labels: `AMYLOID=1`, `NONAMYLOID=0`.

## Environment

```bash
source venvM/bin/activate   # always activate before running anything
```

All scripts must be run from the project root. `src/configuration.py` resolves paths relative to the `config/` directory's grandparent — so the working directory matters.

IUPred3 is an external dependency at `scripts/iupred3/iupred3.py` (separate licence). It must exist for any feature extraction that uses the builtin 540-dim vector.

## Common Commands

```bash
# Train (hyperparameter search, all CPUs, select by MCC)
python scripts/train_ensemble.py --combos 50 --workers 0 --metric MCC

# Train specific algorithms only
python scripts/train_ensemble.py --algorithms extra_trees svm adaboost

# Full pipeline: train → adaptive benchmark
python scripts/run_full_pipeline_adaptive.py --combos 50 --workers 0

# Benchmark only (run after training)
python scripts/run_benchmark_adaptive.py --model outputs/training/run_YYYYMMDD_HHMMSS/soft_ensemble.pkl
python scripts/run_benchmark_new.py      --model outputs/training/run_YYYYMMDD_HHMMSS/weighted_ensemble.pkl

# Sequence-level classification (sets 1–10)
python classify_benchmark_sets.py --run-id run_20260507_221701

# ESM2 embeddings (optional enrichment, separate from training)
python esm2_embedding.py -i dataset.json -o embeddings.pt
python feature_builder.py -i dataset.json -e embeddings.pt -f data/aaindex_features.json -o features.npy
```

## Architecture

### Data flow

```
JSON/FASTA sequences
    → IUPred3 (external, per sequence)        ← 1 float per seq
    → extract_sequence_features()             ← 540-dim vector
    → EnsembleClassifier.predict_proba()      ← P(AMYLOID)
    → threshold 0.5 → binary prediction
```

### Feature vector (540 dims, `src/feature_pipeline.py`)

`1 (length) + 20 (AA freq) + 400 (dipeptide AA→AA) + 17 (group freq) + 101 (intra-class transitions) + 1 (IUPred) = 540`

Additional features (e.g. `data/aaindex_features.json` — 304 AAindex means) are concatenated after the builtin vector when `--features builtin data/aaindex_features.json` is used. ESM2 embeddings (`feature_builder.py`) are appended the same way but require a separate pre-computation step.

### Training (`src/training_pipeline.py`)

- 6 algorithms trained in parallel: AdaBoost, Random Forest, Extra Trees, Gradient Boosting, SVM, Decision Tree
- **k-fold CV** (default 5): all algorithms and all hyperparameter combos use the **same folds** for fair comparison
- **Random hyperparameter search** (always on): `n_combos` random combos sampled per algorithm, best selected by chosen metric
- **Shared seeds across combos**: split RNG is seeded with `base_seed XOR 0x5DEECE66D` to decorrelate from combo sampling
- Final model is retrained on the **full dataset** after combo selection
- Three ensemble variants saved per run: `soft_ensemble.pkl` (avg prob), `weighted_ensemble.pkl` (F1-weighted), `best_model.pkl` (single best algo)

### `EnsembleClassifier` (`src/ensemble.py`)

Thin sklearn-compatible wrapper with `predict(X)` / `predict_proba(X)`. Mode is one of `soft_voting`, `weighted_voting`, `best_model`. Weights are the per-algorithm validation F1 scores.

### Benchmark variants (`src/benchmark_pipeline_new.py`, `src/benchmark_pipeline_adaptive.py`)

Both use a **sliding window** over padded sequences (pad = window_size−1 residues on each side so every residue is covered by exactly T windows). Score per residue = mean of T overlapping window scores; threshold at 0.5.

The **adaptive** variant additionally tries window sizes 11–23 when a position's score is within `confidence_margin` (default 0.15) of 0.5. It picks the size that maximises `|score − 0.5|` (two-directional — confident negative counts equally). IUPred is precomputed once for the maximally-padded sequence to avoid per-window external calls.

### Key src/ modules

| File | Role |
|---|---|
| `configuration.py` | Config loading; resolves all paths relative to the pipeline root |
| `feature_pipeline.py` | `extract_sequence_features()` — builds 540-dim vectors |
| `feature_loader.py` | Dispatch for builtin + lookup-table feature specs; `compute_sequence_feature_matrix()` |
| `training_pipeline.py` | Full training loop, k-fold CV, random search, parallel workers |
| `ensemble.py` | `EnsembleClassifier` — sklearn-compatible wrapper for 3 voting modes |
| `padding_sequences.py` | Pre-built padding lookup by terminal amino acid |
| `benchmark_pipeline_new.py` | Padded sliding-window benchmark |
| `benchmark_pipeline_adaptive.py` | Adaptive multi-scale benchmark |
| `defaults.py` | `AMINO_ACIDS`, `CLASSIFICATIONS`, `GROUP_LABELS` constants |
| `hyperparameter_grids.py` | Random search spaces per algorithm |

### Top-level scripts

| Script | Purpose |
|---|---|
| `esm2_embedding.py` | Standalone: embed sequences with ESM2, save `.pt` |
| `feature_builder.py` | Build combined feature matrix: src/ pipeline + optional ESM2 |
| `run_predict.py` | Predict on new sequences using a saved model |
| `classify_benchmark_sets.py` | Classify `benchmark_classification_set_1-10.json`, output per-sample and per-set CSVs |

## Output Structure

```
outputs/training/run_YYYYMMDD_HHMMSS/
├── soft_ensemble.pkl / weighted_ensemble.pkl / best_model.pkl
├── metadata.json          ← run params, feature_files, feature_dim
├── summary.csv            ← per-algorithm mean/std metrics
└── per_algorithm/<algo>/
    ├── best_model.pkl
    ├── scores.csv         ← per-fold metrics + model_seed
    └── summary.json       ← combos_tried, best params, best_seed

outputs/benchmark/run_YYYYMMDD_HHMMSS/<variant>/
├── results.json           ← per-sequence TP/TN/FP/FN + metrics
├── results_scores.csv     ← per-residue scores
└── results_windows.json   ← per-window scores

outputs/classification/run_YYYYMMDD_HHMMSS/
├── results_per_set.csv
├── results_per_sample.csv
└── summary.json
```

## Dataset Formats

**Training/classification input JSON:**
```json
[{"ID": "2E8D_A", "LABEL": "AMYLOID", "Sequence": "SNFLNCY..."}]
```

**Benchmark input JSON** (`benchmark_known_cores_seed_*.json`): same schema plus `Cluster_ID`, `Uniprot_ID`, `core_cluster_assignment` fields for SOV computation.

`benchmark_classification_set_1-10.json`: 26 samples each, balanced 13 AMYLOID / 13 NONAMYLOID. These test whole-sequence classification, not residue-level core prediction.

## Config

`config/default_config.json` is the single source of truth for paths, algorithm params, and benchmark settings. All CLI flags override config values. The config's `feature_files` list (`["builtin"]` by default) controls which feature specs are loaded; override with `--features` on any training script.

The `n_folds` field in config controls both fixed-param evaluation and hyperparameter search — all combos share the same folds. Setting `random_seed: null` gives a new random run each time; an integer gives reproducible results.
