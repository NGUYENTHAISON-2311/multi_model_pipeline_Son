# Multi-Model Ensemble Pipeline

Multi-algorithm pipeline for amyloid core prediction, based on the
Cross-Beta predictor method (Gonay et al., Alzheimer's Dement. 2025).

Extended from `Chloe_prediction_pipeline` with:
- **Ensemble learning**: combine multiple algorithms, saving 3 ensemble variants per run
- **Random hyperparameter search**: always-on, controlled by per-algorithm `n_combos` in config
- **Parallel training**: train multiple algorithms simultaneously (multi-process)
- **Flexible features**: plug in additional AAindex or custom lookup-table features

## Supported Algorithms

| Algorithm | Config key |
|---|---|
| AdaBoost | `adaboost` |
| Random Forest | `random_forest` |
| Extra Trees | `extra_trees` |
| Gradient Boosting | `gradient_boosting` |
| SVM | `svm` |
| Decision Tree | `decision_tree` |
| CatBoost | `catboost` |

## Directory Structure

```
multi_model_pipeline/
├── README.md                          ← this file
├── requirements.txt
├── config/
│   ├── default_config.json            ← default configuration
│   └── config_reference.md            ← config reference documentation
├── data/
│   └── aaindex_features.json          ← 304 AAindex mean features (inline lookup tables)
├── docs/
│   └── pipeline_explanation.md        ← full pipeline explanation
├── src/
│   ├── __init__.py
│   ├── configuration.py               ← load config, resolve paths
│   ├── defaults.py                    ← default paths & constants
│   ├── feature_loader.py              ← feature spec loading & dispatch
│   ├── feature_pipeline.py            ← extract 540 builtin features
│   ├── padding_sequences.py           ← padding for padded benchmark
│   ├── ensemble.py                    ← EnsembleClassifier wrapper
│   ├── hyperparameter_grids.py        ← search spaces for random search
│   ├── training_pipeline.py           ← training loop, optimization, parallel
│   ├── benchmark_pipeline.py          ← sliding-window benchmark (standard)
│   ├── benchmark_pipeline_new.py      ← padded sliding-window benchmark
│   └── benchmark_pipeline_adaptive.py ← adaptive multi-scale benchmark
└── scripts/
    ├── train_ensemble.py              ← CLI: train ensemble
    ├── run_benchmark.py               ← CLI: standard benchmark
    ├── run_benchmark_new.py           ← CLI: padded benchmark
    ├── run_benchmark_adaptive.py      ← CLI: adaptive benchmark
    ├── run_full_pipeline.py           ← CLI: train → padded benchmark
    └── run_full_pipeline_adaptive.py  ← CLI: train → adaptive benchmark
```

## Installation

```bash
cd multi_model_pipeline
pip install -r requirements.txt
```

IUPred3 must be placed at `scripts/iupred3/` (separate licence, see https://iupred3.elte.hu/).

## Usage

### 1. Training

Train all algorithms (hyperparameter optimization always runs, combos controlled by config):

```bash
python scripts/train_ensemble.py
```

Train only specific algorithm(s):

```bash
python scripts/train_ensemble.py --algorithms extra_trees
python scripts/train_ensemble.py --algorithms extra_trees svm adaboost
```

Override global fallback for number of random combos (each algo can also set `n_combos` in config):

```bash
python scripts/train_ensemble.py --combos 50
```

### 2. Feature specs

By default the pipeline uses the original 540-dim builtin features. You can add or replace
with custom lookup-table features using `--features`:

```bash
# Default: builtin 540-dim features only
python scripts/train_ensemble.py

# AAindex mean features only (304 dims)
python scripts/train_ensemble.py --features data/aaindex_features.json

# Both concatenated (540 + 304 = 844 dims)
python scripts/train_ensemble.py --features builtin data/aaindex_features.json

# Custom feature file
python scripts/train_ensemble.py --features my_features.json
```

A feature spec JSON must contain an `"aggregation"` list and a `"tables"` dict
(multi-scale) or `"table"` dict (single-scale):

```json
{
  "aggregation": ["mean"],
  "tables": {
    "FASG890101": {"A": 1.09, "R": 1.29, "...": "..."}
  }
}
```

### 3. Parallel training

```bash
# Use 4 processes
python scripts/train_ensemble.py --workers 4

# Use all CPUs
python scripts/train_ensemble.py --workers 0
```

### 4. Benchmark

Each training run saves **3 ensemble variants**. Choose the one to benchmark with `--model`:

```bash
# Padded benchmark (sliding window + padding, average score)
python scripts/run_benchmark_new.py --model outputs/training/run_XXXXXXXX_XXXXXX/soft_ensemble.pkl
python scripts/run_benchmark_new.py --model outputs/training/run_XXXXXXXX_XXXXXX/weighted_ensemble.pkl
python scripts/run_benchmark_new.py --model outputs/training/run_XXXXXXXX_XXXXXX/best_model.pkl

# Override window size (default: from config)
python scripts/run_benchmark_new.py --model outputs/training/.../soft_ensemble.pkl --window-size 21

# Full pipeline: train → padded benchmark (uses soft_ensemble.pkl by default)
python scripts/run_full_pipeline.py
python scripts/run_full_pipeline.py --combos 50 --workers 0
```

#### Adaptive benchmark (multi-scale)

The adaptive benchmark addresses the scale mismatch between fixed-size
windows and variable-length amyloid cores (11–25 residues). It starts with
the default window, and when the model is uncertain it tries extending or
shortening the **end** of the window to find the scale where the model is
most confident — in either direction (positive or negative).

```bash
# Adaptive benchmark
python scripts/run_benchmark_adaptive.py --model outputs/training/.../best_model.pkl

# Custom confidence margin (lower = fewer extra evaluations)
python scripts/run_benchmark_adaptive.py --model .../best_model.pkl --confidence-margin 0.1

# Full pipeline: train → adaptive benchmark
python scripts/run_full_pipeline_adaptive.py
python scripts/run_full_pipeline_adaptive.py --combos 50 --workers 0
```

> `run_benchmark.py` (max-score aggregation) is still available but not recommended
> as it produces overly generous results.

## CLI Quick Reference

### `train_ensemble.py`

| Flag | Description | Default |
|---|---|---|
| `--config` | Config JSON file | `config/default_config.json` |
| `--positive` | Override positive sequences file | from config |
| `--negative` | Override negative sequences file | from config |
| `--output` | Output directory | `outputs/training/` |
| `--features` | Feature spec(s): `builtin` and/or path(s) to JSON files | `builtin` |
| `--algorithms` | Train only selected algorithm(s) | all |
| `--combos` | Global fallback for random combos per algorithm | `10` |
| `--folds` | Number of stratified k-fold CV splits | from config (`5`) |
| `--metric` | Metric for best combo selection (`F1_score`, `Accuracy`, `Precision`, `Recall`, `MCC`, `Average`) | `F1_score` |
| `--workers` | Parallel processes (0 = all CPUs) | `1` |

### `run_benchmark_new.py`

| Flag | Description | Default |
|---|---|---|
| `--config` | Config JSON file | `config/default_config.json` |
| `--model` | Path to a model pickle (`soft_ensemble.pkl`, `weighted_ensemble.pkl`, or `best_model.pkl`) | auto-detect |
| `--input` | Benchmark sequences JSON | from config |
| `--output` | Output directory | `outputs/benchmark/` |
| `--threshold` | Average-score threshold k | `0.5` |
| `--window-size` | Sliding window size (overrides config) | from config |
| `--name` | Output filename prefix | `benchmark_new_results` |

### `run_full_pipeline.py`

Combines all `train_ensemble.py` flags + benchmark flags:

| Flag | Description | Default |
|---|---|---|
| `--benchmark-input` | Benchmark sequences JSON | from config |
| `--benchmark-output` | Benchmark output directory | `outputs/benchmark/` |
| `--model` | Skip training, benchmark this model | — |
| `--threshold` | Benchmark threshold k | `0.5` |
| `--window-size` | Sliding window size (overrides config) | from config |
| `--name` | Benchmark output filename prefix | `benchmark_results` |

### `run_benchmark_adaptive.py`

| Flag | Description | Default |
|---|---|---|
| `--config` | Config JSON file | `config/default_config.json` |
| `--model` | Path to a model pickle | auto-detect |
| `--input` | Benchmark sequences JSON | from config |
| `--output` | Output directory | `outputs/benchmark/` |
| `--threshold` | Score threshold k | `0.5` |
| `--window-size` | Default sliding window size | from config (18) |
| `--min-window` | Minimum window size for adaptive search | `11` |
| `--max-window` | Maximum window size for adaptive search | `23` |
| `--confidence-margin` | Only explore other sizes when \|score−0.5\| < margin | `0.15` |
| `--name` | Output filename prefix | `benchmark_adaptive_results` |

### `run_full_pipeline_adaptive.py`

Combines all `train_ensemble.py` flags + adaptive benchmark flags (see above).

## Builtin Features (540)

| Feature group | Count | Description |
|---|---|---|
| Length | 1 | Sequence length |
| AA frequency | 20 | Frequency of 20 standard amino acids |
| Dipeptide AA→AA | 400 | Transition pairs across 20×20 amino acids |
| Group frequency | 17 | Amino acid group frequency (3 classification schemes) |
| Intra-class transitions | 101 | Transitions within classification schemes |
| IUPred score | 1 | Average disorder score |

Additional features (e.g. `data/aaindex_features.json`) are concatenated after the builtin
vector when `--features builtin data/aaindex_features.json` is used.

## Outputs

After training, the directory `outputs/training/run_YYYYMMDD_HHMMSS/` contains:

```
run_YYYYMMDD_HHMMSS/
├── soft_ensemble.pkl      ← soft-voting ensemble (average probability)
├── weighted_ensemble.pkl  ← weighted-voting ensemble (F1-weighted)
├── best_model.pkl         ← single best-performing algorithm
├── metadata.json          ← run info, params, results
├── summary.csv            ← per-algorithm summary table
└── per_algorithm/
    ├── adaboost/
    │   ├── best_model.pkl
    │   ├── scores.csv
    │   └── summary.json
    ├── extra_trees/
    │   └── ...
    └── ...
```

## Recommended Workflow

```bash
# 1. Train with all CPUs, 50 combos per algo, select by MCC
python scripts/train_ensemble.py --combos 50 --workers 0 --metric MCC

# 2. Review metadata.json → copy best params into config for reproducibility

# 3. Benchmark all three variants and compare
python scripts/run_benchmark_adaptive.py --model outputs/training/.../soft_ensemble.pkl     --name results_soft
python scripts/run_benchmark_adaptive.py --model outputs/training/.../weighted_ensemble.pkl --name results_weighted
python scripts/run_benchmark_adaptive.py --model outputs/training/.../best_model.pkl        --name results_best
```

## References

- Gonay V, et al. Developing machine-learning-based amyloidogenicity predictors
  with Cross-Beta DB. Alzheimer's Dement. 2025; 1-7.
  https://doi.org/10.1002/alz.14510
- Source code: https://github.com/Valentin-Gonay/cross-beta-predictor-modelcreation
