"""Multi-model ensemble training pipeline.

Trains multiple algorithm types in a single run, each with its own
iteration count and sampling strategy, then wraps them into an
:class:`~.ensemble.EnsembleClassifier`.

Reuses :func:`feature_pipeline.extract_sequence_features` and
:func:`feature_pipeline.compute_average_iupred_scores_from_sequences`
verbatim from the original Chloe prediction pipeline.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import random
import time as _time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm

from .configuration import ensure_directory, resolve_pipeline_path
from .ensemble import EnsembleClassifier
from .feature_loader import compute_sequence_feature_matrix, load_and_prepare_feature_specs
from .feature_pipeline import (
    compute_average_iupred_scores_from_sequences,
    load_sequences_from_file,
)
from .hyperparameter_grids import SEARCH_SPACES, sample_n_combos


# ── Classifier factory (identical to the original pipeline) ──────────────────

SUPPORTED_MODELS = [
    "adaboost",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "decision_tree",
    "svm",
    "catboost",
]

# Algorithms that support internal multi-threading via n_jobs (OpenMP / joblib).
# These run sequentially so every CPU core goes to the single active algorithm.
_CPU_INTENSIVE = {"random_forest", "extra_trees", "gradient_boosting", "catboost"}

# Algorithms that are inherently single-threaded (n_jobs has no effect on fit).
# These run concurrently in a thread pool — no memory duplication, no OOM risk.
_THREAD_PARALLEL = {"adaboost", "decision_tree", "svm"}


def create_classifier(model_type: str, params: dict, run_seed: int, n_jobs: int = 1):
    """Instantiate a classifier from *model_type* and *params*.

    All algorithm-specific hyperparameters should be set explicitly in the
    config (or via optimization).  This factory only injects the random seed
    and the few invariant settings (e.g. ``probability=True`` for SVM).

    *n_jobs* is forwarded to algorithms that support intra-fit parallelism
    (RandomForest, ExtraTrees).  Set to -1 for all CPUs (recommended in
    sequential mode) or to ``n_cpus // n_parallel_algos`` in parallel mode.
    """
    p = dict(params)

    if model_type == "adaboost":
        p["random_state"] = run_seed
        return AdaBoostClassifier(**p)

    if model_type == "random_forest":
        p["random_state"] = run_seed
        p.setdefault("n_jobs", n_jobs)
        return RandomForestClassifier(**p)

    if model_type == "extra_trees":
        p["random_state"] = run_seed
        p.setdefault("n_jobs", n_jobs)
        return ExtraTreesClassifier(**p)

    if model_type == "gradient_boosting":
        p["random_state"] = run_seed
        return GradientBoostingClassifier(**p)

    if model_type == "decision_tree":
        p["random_state"] = run_seed
        return DecisionTreeClassifier(**p)

    if model_type == "svm":
        p["probability"] = True
        p["random_state"] = run_seed
        return SVC(**p)

    if model_type == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("CatBoost is not installed. Run: pip install catboost") from exc
        p.setdefault("verbose", 0)
        p["random_seed"] = run_seed
        return CatBoostClassifier(**p)

    raise ValueError(f"Unknown model type: {model_type!r}. Supported: {SUPPORTED_MODELS}")


# ── Reused sampling logic ────────────────────────────────────────────────────

def _compute_sample_weights(y_train: list[int]) -> list[float]:
    n = len(y_train)
    n_pos = y_train.count(1)
    n_neg = y_train.count(0)
    w_pos = n / (2 * n_pos) if n_pos else 1.0
    w_neg = n / (2 * n_neg) if n_neg else 1.0
    return [w_pos if y == 1 else w_neg for y in y_train]


def make_stratified_folds(
    positive_features: list[list[float]],
    negative_features: list[list[float]],
    n_folds: int,
    base_seed: int | None,
) -> list[tuple[list[int], list[int]]]:
    """Build *n_folds* stratified fold indices over the combined pos+neg data.

    Returns a list of (train_indices, test_indices) tuples, where indices
    refer to positions in the concatenated [positives + negatives] array.
    All samples appear in the test set exactly once across all folds.
    """
    n_pos = len(positive_features)
    n_neg = len(negative_features)
    n_total = n_pos + n_neg
    y_all = [1] * n_pos + [0] * n_neg

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
    indices = list(range(n_total))
    folds = [
        (list(train_idx), list(test_idx))
        for train_idx, test_idx in skf.split(indices, y_all)
    ]
    return folds


def _fold_train_test(
    positive_features: list[list[float]],
    negative_features: list[list[float]],
    train_idx: list[int],
    test_idx: list[int],
) -> tuple[list, list[int], list, list[int], list[float]]:
    """Extract train/test sets from pre-built fold indices.

    Indices refer to the concatenated [positives + negatives] list.
    Returns (x_train, y_train, x_test, y_test, sample_weights).
    """
    all_x = positive_features + negative_features
    n_pos = len(positive_features)
    all_y = [1] * n_pos + [0] * len(negative_features)

    x_train = [all_x[i] for i in train_idx]
    y_train = [all_y[i] for i in train_idx]
    x_test = [all_x[i] for i in test_idx]
    y_test = [all_y[i] for i in test_idx]
    sample_weights = _compute_sample_weights(y_train)
    return x_train, y_train, x_test, y_test, sample_weights


def prepare_train_test_sets(
    positive_features: list[list[float]],
    negative_features: list[list[float]],
    test_ratio: float,
    rng: random.Random,
    sampling_strategy: str | float = "min",
) -> tuple[list, list[int], list, list[int], list[float] | None]:
    """Split positive/negative features into train/test sets.

    *sampling_strategy* can be:

    - ``"min"`` – downsample both classes to the size of the smaller class,
      then split by *test_ratio*.  Classes are perfectly balanced; no sample
      weights.
    - A **float** (e.g. ``0.7``) – use that fraction of each class for
      training and the remainder for testing.  All data is used.  Sample
      weights compensate for any class imbalance.
    - ``"ratio"`` – legacy alias: equivalent to passing ``1 - test_ratio``
      as a float.
    """
    # Normalise legacy string "ratio" → float
    if isinstance(sampling_strategy, str) and sampling_strategy == "ratio":
        sampling_strategy = 1.0 - test_ratio

    if sampling_strategy == "min":
        min_size = min(len(positive_features), len(negative_features))
        positive_subset = rng.sample(positive_features, min_size)
        negative_subset = rng.sample(negative_features, min_size)
        pos_train_n = int(min_size * (1 - test_ratio))
        neg_train_n = pos_train_n
        pos_test_n = min_size - pos_train_n
        neg_test_n = pos_test_n
        sample_weights: list[float] | None = None
    elif isinstance(sampling_strategy, (int, float)):
        train_ratio = float(sampling_strategy)
        if not 0 < train_ratio < 1:
            raise ValueError(f"sampling_strategy ratio must be between 0 and 1, got {train_ratio}")
        pos_shuffled = rng.sample(positive_features, len(positive_features))
        neg_shuffled = rng.sample(negative_features, len(negative_features))
        pos_train_n = int(len(pos_shuffled) * train_ratio)
        neg_train_n = int(len(neg_shuffled) * train_ratio)
        pos_test_n = len(pos_shuffled) - pos_train_n
        neg_test_n = len(neg_shuffled) - neg_train_n
        positive_subset = pos_shuffled
        negative_subset = neg_shuffled
        sample_weights = None
    else:
        raise ValueError(f"Unknown sampling_strategy: {sampling_strategy!r}  (use 'min' or a float like 0.7)")

    x_train = positive_subset[:pos_train_n] + negative_subset[:neg_train_n]
    y_train = [1] * pos_train_n + [0] * neg_train_n
    x_test = positive_subset[pos_train_n:pos_train_n + pos_test_n] + negative_subset[neg_train_n:neg_train_n + neg_test_n]
    y_test = [1] * pos_test_n + [0] * neg_test_n

    train_pairs = list(zip(x_train, y_train))
    test_pairs = list(zip(x_test, y_test))
    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)
    x_tr, y_tr = zip(*train_pairs)
    x_te, y_te = zip(*test_pairs)
    x_train_l, y_train_l = list(x_tr), list(y_tr)
    x_test_l, y_test_l = list(x_te), list(y_te)

    if sampling_strategy != "min":
        sample_weights = _compute_sample_weights(y_train_l)

    return x_train_l, y_train_l, x_test_l, y_test_l, sample_weights


def _metric_summary(rows: list[dict]) -> dict:
    summary: dict = {"num_folds": len(rows)}
    for metric in ["F1_score", "Accuracy", "Precision", "Recall", "MCC"]:
        vals = [r[metric] for r in rows]
        summary[f"mean_{metric}"] = mean(vals)
        summary[f"std_{metric}"] = pstdev(vals) if len(vals) > 1 else 0.0
    # Average = mean of means across all 5 metrics
    summary["mean_Average"] = mean(
        summary[f"mean_{m}"] for m in ["F1_score", "Accuracy", "Precision", "Recall", "MCC"]
    )
    return summary


# ── K-fold helpers ────────────────────────────────────────────────────────────

def _retrain_on_full_data(
    model_type: str,
    model_params: dict,
    positive_features: list[list[float]],
    negative_features: list[list[float]],
    model_seed: int,
    n_jobs: int = 1,
) -> object:
    """Train one final model on the full combined dataset with class weights."""
    all_x = positive_features + negative_features
    all_y = [1] * len(positive_features) + [0] * len(negative_features)
    sw = _compute_sample_weights(all_y)
    clf = create_classifier(model_type, model_params, model_seed, n_jobs=n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(all_x, all_y, sample_weight=sw)
    return clf

# ── Per-algorithm training (k-fold) ─────────────────────────────────────────

def _train_single_algorithm(
    model_type: str,
    model_params: dict,
    positive_features: list[list[float]],
    negative_features: list[list[float]],
    fold_indices: list[tuple[list[int], list[int]]],
    base_seed: int | None,
    progress_value=None,
    selection_metric: str = "F1_score",
    n_jobs: int = 1,
) -> dict:
    """Evaluate one (model_type, params) combo across the shared k folds.

    All folds are pre-built and shared across all algos and combos so that
    every comparison is fair (same train/test partition).

    After evaluation, the final model is retrained on the **full dataset**
    using the best model seed (from the fold with the highest metric).
    This means the ensemble model has seen all available data.

    Returns a dict with keys compatible with the rest of the pipeline:
    ``best_model``, ``best_f1``, ``best_metric_val``, ``mean_f1``, ``results``,
    ``summary``, ``selection_metric``.
    """
    n_folds = len(fold_indices)
    results: list[dict] = []
    best_metric_val = -float("inf")
    best_f1 = -1.0
    best_model_seed: int | None = None

    use_local_bar = progress_value is None
    iterator = (
        tqdm(range(n_folds), desc=f"  {model_type}", unit="fold", leave=False)
        if use_local_bar
        else range(n_folds)
    )

    # Use a deterministic seed per fold derived from base_seed so that
    # model random_state is reproducible but differs across folds.
    fold_seed_rng = random.Random(base_seed)
    fold_model_seeds = [fold_seed_rng.randint(0, 2**31 - 1) for _ in range(n_folds)]

    for fold_idx in iterator:
        train_idx, test_idx = fold_indices[fold_idx]
        x_train, y_train, x_test, y_test, sw = _fold_train_test(
            positive_features, negative_features, train_idx, test_idx
        )
        model_seed = fold_model_seeds[fold_idx]
        clf = create_classifier(model_type, model_params, model_seed, n_jobs=n_jobs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(x_train, y_train, sample_weight=sw)
            preds = clf.predict(x_test)
        f1 = float(f1_score(y_test, preds, zero_division=0.0))
        results.append({
            "Fold": fold_idx + 1,
            "model_seed": model_seed,
            "F1_score": f1,
            "Accuracy": float(accuracy_score(y_test, preds)),
            "Precision": float(precision_score(y_test, preds, zero_division=0.0)),
            "Recall": float(recall_score(y_test, preds, zero_division=0.0)),
            "MCC": float(matthews_corrcoef(y_test, preds)),
        })
        fold_metric_val = results[-1].get(selection_metric, f1)
        if fold_metric_val > best_metric_val:
            best_metric_val = fold_metric_val
            best_f1 = f1
            best_model_seed = model_seed
        if progress_value is not None:
            progress_value.value += 1

    # Retrain final model on ALL data using the seed from the best fold.
    final_model = _retrain_on_full_data(
        model_type, model_params, positive_features, negative_features,
        model_seed=best_model_seed if best_model_seed is not None else (base_seed or 0),
        n_jobs=n_jobs,
    )

    return {
        "model_type": model_type,
        "best_model": final_model,
        "best_f1": best_f1,
        "best_metric_val": best_metric_val,
        "best_seed": best_model_seed,
        "selection_metric": selection_metric,
        "mean_f1": mean(r["F1_score"] for r in results),
        "results": results,
        "summary": _metric_summary(results),
    }


# ── Hyperparameter optimization ──────────────────────────────────────────────

def _optimize_single_algorithm(
    model_type: str,
    positive_features: list[list[float]],
    negative_features: list[list[float]],
    fold_indices: list[tuple[list[int], list[int]]],
    n_combos: int,
    base_seed: int | None,
    metric: str = "F1_score",
    n_jobs: int = 1,
    combo_workers: int = 1,
) -> dict:
    """Sample *n_combos* random hyperparameter combos and evaluate them in parallel.

    All combos use the **same shared k folds** for fair comparison.
    *combo_workers* combos are evaluated concurrently via a ThreadPoolExecutor —
    sklearn estimators release the GIL during ``fit()``, so threads give true
    parallelism without duplicating the feature matrices in memory.

    *n_jobs* controls intra-combo parallelism (e.g. ``n_estimators`` threads for
    RandomForest).  When *combo_workers* > 1 this should be 1 to avoid
    over-subscription.

    Returns the same dict structure as :func:`_train_single_algorithm`
    with extra keys ``best_params`` and ``combos_tried``.
    """
    combos = sample_n_combos(model_type, n_combos, seed=base_seed)
    n_effective = len(combos)
    workers = min(combo_workers, n_effective)
    tqdm.write(
        f"    {n_effective} combos × {len(fold_indices)} folds  "
        f"({workers} parallel combo thread{'s' if workers != 1 else ''})"
    )

    def _eval_combo(g_idx: int, params: dict) -> tuple[int, dict]:
        return g_idx, _train_single_algorithm(
            model_type=model_type,
            model_params=params,
            positive_features=positive_features,
            negative_features=negative_features,
            fold_indices=fold_indices,
            base_seed=base_seed,
            selection_metric=metric,
            n_jobs=n_jobs,
        )

    all_combo_summaries: list[dict | None] = [None] * n_effective
    all_results:         list[dict | None] = [None] * n_effective

    combo_bar = tqdm(
        total=n_effective, desc=f"  {model_type}", unit="combo", leave=True
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_eval_combo, i, p): i
            for i, p in enumerate(combos)
        }
        for future in as_completed(futures):
            g_idx, result = future.result()
            metric_val = result["summary"].get(f"mean_{metric}", result["mean_f1"])
            all_combo_summaries[g_idx] = {
                "combo_index": g_idx,
                "params": combos[g_idx],
                f"mean_{metric}": metric_val,
                "mean_f1": result["mean_f1"],
                "best_f1": result["best_f1"],
            }
            all_results[g_idx] = result
            combo_bar.update(1)
            combo_bar.set_postfix({metric[:6]: f"{metric_val:.4f}"})
    combo_bar.close()

    best_idx = max(
        range(n_effective),
        key=lambda i: all_combo_summaries[i][f"mean_{metric}"],
    )
    best_result = all_results[best_idx]
    best_metric_val = all_combo_summaries[best_idx][f"mean_{metric}"]
    tqdm.write(
        f"    Best combo #{best_idx + 1}: {_compact_params(combos[best_idx])}\n"
        f"    mean {metric}={best_metric_val:.4f}  "
        f"mean F1={best_result['mean_f1']:.4f}"
    )
    best_result["best_params"] = combos[best_idx]
    best_result["combos_tried"] = [s for s in all_combo_summaries if s is not None]
    return best_result


def _compact_params(params: dict) -> str:
    """One-line string showing key=value pairs."""
    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.3g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


# ── Default per-algorithm configs ────────────────────────────────────────────

DEFAULT_ALGORITHMS: list[dict] = [
    {
        "type": "adaboost",
        "runs": 30,
        "params": {"n_estimators": 50, "algorithm": "SAMME", "learning_rate": 1.0},
        "n_combos": 10,
    },
    {
        "type": "random_forest",
        "runs": 30,
        "params": {"n_estimators": 100, "max_features": "sqrt", "max_depth": None,
                   "min_samples_split": 2, "min_samples_leaf": 1, "bootstrap": True},
        "n_combos": 10,
    },
    {
        "type": "extra_trees",
        "runs": 30,
        "params": {"n_estimators": 100, "max_features": "sqrt", "max_depth": None,
                   "min_samples_split": 2, "min_samples_leaf": 1, "bootstrap": False},
        "n_combos": 10,
    },
    {
        "type": "gradient_boosting",
        "runs": 30,
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3,
                   "min_samples_split": 2, "min_samples_leaf": 1,
                   "max_features": None, "min_impurity_decrease": 0.0},
        "n_combos": 10,
    },
    {
        "type": "svm",
        "runs": 30,
        "params": {"C": 1.0, "kernel": "rbf"},
        "n_combos": 10,
    },
    {
        "type": "decision_tree",
        "runs": 30,
        "params": {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1,
                   "max_features": None, "criterion": "gini", "splitter": "best"},
        "n_combos": 10,
    },
]


# ── Multiprocessing worker (must be top-level for pickling) ──────────────────

def _worker_train_algorithm(args: dict) -> dict:
    """Top-level function called by each ThreadPoolExecutor worker.

    Accepts a single dict so callers can pass it uniformly.
    """
    pv = args.get("progress_value")
    n_jobs = args.get("n_jobs", 1)
    if args.get("optimize"):
        return _optimize_single_algorithm(
            model_type=args["model_type"],
            positive_features=args["positive_features"],
            negative_features=args["negative_features"],
            fold_indices=args["fold_indices"],
            n_combos=args["n_combos"],
            base_seed=args["base_seed"],
            metric=args["metric"],
            progress_value=pv,
            n_jobs=n_jobs,
        )
    else:
        return _train_single_algorithm(
            model_type=args["model_type"],
            model_params=args["model_params"],
            positive_features=args["positive_features"],
            negative_features=args["negative_features"],
            fold_indices=args["fold_indices"],
            base_seed=args["base_seed"],
            selection_metric=args.get("metric", "F1_score"),
            progress_value=pv,
            n_jobs=n_jobs,
        )


# ── Main entry point ────────────────────────────────────────────────────────

def run_ensemble_training(
    config: dict,
    *,
    positive_path: str | Path | None = None,
    negative_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    n_combos: int = 10,
    n_folds: int | None = None,
    optimization_metric: str = "F1_score",
    n_workers: int = 1,
    positive_features_matrix=None,
    negative_features_matrix=None,
    feature_dim_label: str | None = None,
) -> tuple[dict[str, EnsembleClassifier], dict[str, Path]]:
    """Train all configured algorithms and wrap their best models into ensembles.

    Always saves three ensemble variants:
      * ``soft_ensemble.pkl``    — equal average of predict_proba
      * ``weighted_ensemble.pkl``— weighted average by validation metric
      * ``best_model.pkl``       — single model with highest validation metric

    Returns a dict ``{"soft": ..., "weighted": ..., "best": ...}`` and artifacts.

    Parameters
    ----------
    n_combos : int
        Global fallback for number of random hyperparameter combos per algorithm.
        Each algorithm can override via ``n_combos`` in its config block.
    n_folds : int | None
        Number of stratified k-fold splits for cross-validation.
        Overrides the ``"n_folds"`` key in config (default 5).
    optimization_metric : str
        Metric used to select the best combo/model.  Default ``"F1_score"``.
    n_workers : int
        Number of parallel worker processes.  ``1`` = sequential (default).
        ``0`` = use all available CPUs.
    positive_features_matrix : array-like of shape (N, D) | None
        Pre-built feature matrix for positive sequences.  When provided together
        with *negative_features_matrix*, the IUPred and feature-extraction steps
        are skipped entirely.  Rows must be in the same order as the sequences
        in *positive_path*.  Accepts numpy arrays or list-of-lists.
    negative_features_matrix : array-like of shape (N, D) | None
        Pre-built feature matrix for negative sequences.  Must be provided
        together with *positive_features_matrix*.
    feature_dim_label : str | None
        Human-readable description of the feature columns written to metadata.json
        (e.g. "540-builtin + 320-esm2").  Auto-derived when not given.
    """
    import numpy as _np

    training_cfg = config["training"]
    algorithms = training_cfg.get("algorithms", DEFAULT_ALGORITHMS)

    base_output_dir = Path(output_dir) if output_dir else resolve_pipeline_path(config, training_cfg["output_dir"])
    ensure_directory(base_output_dir)

    using_prebuilt = positive_features_matrix is not None and negative_features_matrix is not None

    # ── Data loading ────────────────────────────────────────────────────────
    pos_path = Path(positive_path) if positive_path else resolve_pipeline_path(config, training_cfg["positive_json"])
    neg_path = Path(negative_path) if negative_path else resolve_pipeline_path(config, training_cfg["negative_json"])

    tqdm.write(f"[1/4] Loading positive sequences  ({pos_path.name}) …")
    positive_sequences = load_sequences_from_file(pos_path)
    tqdm.write(f"      Loaded {len(positive_sequences)} positive sequences.")

    tqdm.write(f"[2/4] Loading negative sequences  ({neg_path.name}) …")
    negative_sequences = load_sequences_from_file(neg_path)
    tqdm.write(f"      Loaded {len(negative_sequences)} negative sequences.")

    if using_prebuilt:
        # Convert to plain lists-of-lists so the rest of the pipeline is unchanged.
        pos_arr = _np.asarray(positive_features_matrix, dtype=_np.float32)
        neg_arr = _np.asarray(negative_features_matrix, dtype=_np.float32)
        if pos_arr.shape[0] != len(positive_sequences):
            raise ValueError(
                f"positive_features_matrix has {pos_arr.shape[0]} rows but "
                f"{len(positive_sequences)} positive sequences were loaded."
            )
        if neg_arr.shape[0] != len(negative_sequences):
            raise ValueError(
                f"negative_features_matrix has {neg_arr.shape[0]} rows but "
                f"{len(negative_sequences)} negative sequences were loaded."
            )
        pos_features: list[list[float]] = pos_arr.tolist()
        neg_features: list[list[float]] = neg_arr.tolist()
        total_dim = pos_arr.shape[1]
        dim_label = feature_dim_label or f"{total_dim}-prebuilt"
        tqdm.write(f"[3/4] Using pre-built feature matrices  ({dim_label}) …")
        tqdm.write(f"      Positive: {pos_arr.shape}   Negative: {neg_arr.shape}")
        tqdm.write("[4/4] Skipped (features already provided).")
        feature_files_meta: list[str] = [dim_label]
    else:
        iupred_script = resolve_pipeline_path(config, training_cfg["iupred_script"])
        tqdm.write("[3/4] Computing IUPred scores …")
        pos_iupred = compute_average_iupred_scores_from_sequences(
            positive_sequences, iupred_script, input_type=training_cfg["iupred_input_type"],
        )
        neg_iupred = compute_average_iupred_scores_from_sequences(
            negative_sequences, iupred_script, input_type=training_cfg["iupred_input_type"],
        )

        tqdm.write("[4/4] Extracting features …")
        feature_specs = load_and_prepare_feature_specs(config)
        tqdm.write(f"      Feature sets: {['builtin' if s.get('_builtin') else s.get('_source_path', '?') for s in feature_specs]}")
        def _n_features(s):
            if s.get("_builtin"):
                return 540
            if "_tables" in s:
                return len(s["_tables"]) * len(s.get("aggregation", ["mean"]))
            if "_table" in s:
                return len(s.get("_classes", [])) or len(s.get("aggregation", ["mean"]))
            return 0
        total_dim = sum(_n_features(s) for s in feature_specs)
        tqdm.write(f"      Dimensions  : {' + '.join(str(_n_features(s)) for s in feature_specs)} = {total_dim}")
        pos_features = compute_sequence_feature_matrix(
            positive_sequences, feature_specs, pos_iupred, desc="  Positive"
        )
        neg_features = compute_sequence_feature_matrix(
            negative_sequences, feature_specs, neg_iupred, desc="  Negative"
        )
        feature_files_meta = [
            "builtin" if s.get("_builtin") else s.get("_source_path", "?")
            for s in feature_specs
        ]

    test_ratio = training_cfg.get("test_ratio", 0.3)
    base_seed = training_cfg.get("random_seed")

    # ── Build shared k-fold splits (once for ALL algos and combos) ──────────
    effective_n_folds = n_folds or training_cfg.get("n_folds", 5)
    tqdm.write(
        f"\n[*] Building {effective_n_folds}-fold stratified splits "
        f"(seed={base_seed}) …"
    )
    fold_indices = make_stratified_folds(
        pos_features, neg_features, n_folds=effective_n_folds, base_seed=base_seed
    )
    tqdm.write(
        f"    Folds: {len(fold_indices)} × "
        f"~{len(fold_indices[0][0])} train / ~{len(fold_indices[0][1])} test samples"
    )

    # ── Create run directory BEFORE training so results can be saved early ──
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base_output_dir / run_id
    ensure_directory(run_dir)
    per_algo_dir = run_dir / "per_algorithm"
    ensure_directory(per_algo_dir)
    tqdm.write(f"\n[*] Run directory: {run_dir}")

    # ── Worker count → parallel combo threads per algorithm ────────────────
    #   --workers 0  →  combo_workers = all CPUs  (recommended)
    #   --workers N  →  combo_workers = min(N, all CPUs)
    #
    # Algorithms run sequentially; combos within each algorithm run in
    # parallel threads (n_jobs=1 per combo — thread-level parallelism).
    max_cpus = os.cpu_count() or 1
    combo_workers = max_cpus if n_workers <= 0 else min(n_workers, max_cpus)

    tqdm.write(
        f"\nTraining {len(algorithms)} algorithm(s) sequentially  "
        f"(metric={optimization_metric}, {effective_n_folds} folds, "
        f"{combo_workers} combo threads each) …"
    )

    # ── Incremental per-algo save ────────────────────────────────────────────
    def _save_algo_result(result: dict) -> None:
        m_type = result["model_type"]
        algo_dir = per_algo_dir / m_type
        ensure_directory(algo_dir)
        with (algo_dir / "best_model.pkl").open("wb") as fh:
            pickle.dump(result["best_model"], fh)
        with (algo_dir / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(result["results"][0].keys()))
            writer.writeheader()
            writer.writerows(result["results"])
        with (algo_dir / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(result["summary"], fh, indent=2)

    # Preserve original algorithm order in the final results list.
    algo_order = {a["type"]: i for i, a in enumerate(algorithms)}
    algo_results: list[dict | None] = [None] * len(algorithms)

    def _record(result: dict) -> None:
        """Store result and log a one-line summary."""
        idx = algo_order[result["model_type"]]
        algo_results[idx] = result
        _save_algo_result(result)
        if "best_params" in result:
            tqdm.write(
                f"✓ {result['model_type']}  "
                f"best combo: {_compact_params(result['best_params'])}\n"
                f"  mean F1={result['mean_f1']:.4f}  best F1={result['best_f1']:.4f}"
                f"  (seed={result['best_seed']})"
            )
        else:
            tqdm.write(
                f"✓ {result['model_type']}  "
                f"mean F1={result['mean_f1']:.4f}  best F1={result['best_f1']:.4f}"
                f"  (seed={result['best_seed']})"
            )

    # ── Sequential algorithm loop, parallel combos within each ─────────────
    for algo_cfg in algorithms:
        atype = algo_cfg["type"]
        algo_combos = algo_cfg.get("n_combos", n_combos)
        tqdm.write(
            f"\n── {atype}  "
            f"({algo_combos} combos × {effective_n_folds} folds, "
            f"{min(combo_workers, algo_combos)} combo threads) ──"
        )
        result = _optimize_single_algorithm(
            model_type=atype,
            positive_features=pos_features,
            negative_features=neg_features,
            fold_indices=fold_indices,
            n_combos=algo_combos,
            base_seed=base_seed,
            metric=optimization_metric,
            n_jobs=1,
            combo_workers=combo_workers,
        )
        _record(result)

    # ── Build ensembles (all 3 modes) ───────────────────────────────────────
    models = [r["best_model"] for r in algo_results]
    # Use best_metric_val (the metric that was used for model selection) as weights.
    # Fall back to best_f1 for backward compatibility.
    weights = [r.get("best_metric_val", r["best_f1"]) for r in algo_results]

    soft_ensemble = EnsembleClassifier(models=models, weights=weights, mode="soft_voting")
    weighted_ensemble = EnsembleClassifier(models=models, weights=weights, mode="weighted_voting")
    best_ensemble = EnsembleClassifier(models=models, weights=weights, mode="best_model")
    tqdm.write(f"\nEnsembles created: {len(models)} model(s)")
    tqdm.write(f"  soft_ensemble    : {soft_ensemble}")
    tqdm.write(f"  weighted_ensemble: {weighted_ensemble}")
    tqdm.write(f"  best_model       : {best_ensemble}")

    # ── Save ensemble PKLs (per-algo files already written incrementally) ────
    soft_pkl = run_dir / "soft_ensemble.pkl"
    weighted_pkl = run_dir / "weighted_ensemble.pkl"
    best_pkl = run_dir / "best_model.pkl"
    soft_ensemble.save(soft_pkl)
    weighted_ensemble.save(weighted_pkl)
    best_ensemble.save(best_pkl)

    # Top-level metadata
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "optimized": True,
        "n_algorithms": len(algorithms),
        "algorithms": [
            {
                "type": r["model_type"],
                "n_folds": len(r["results"]),
                "best_f1": r["best_f1"],
                "mean_f1": r["mean_f1"],
                "params": r.get("best_params", next(
                    a.get("params", {}) for a in algorithms if a["type"] == r["model_type"]
                )),
                **({"combos_tried": r["combos_tried"]} if "combos_tried" in r else {}),
            }
            for r in algo_results
        ],
        "data": {
            "positive": str(pos_path),
            "negative": str(neg_path),
            "n_positive": len(positive_sequences),
            "n_negative": len(negative_sequences),
            "feature_dim": len(pos_features[0]) if pos_features else 0,
        },
        "parameters": {
            "n_folds": effective_n_folds,
            "random_seed": base_seed,
            "iupred_input_type": training_cfg.get("iupred_input_type", "long"),
        },
        "feature_files": feature_files_meta,
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    # Global summary CSV (one row per algorithm)
    sel_metric = optimization_metric  # the metric used to pick best model per fold
    summary_rows = []
    for r in algo_results:
        s = r["summary"]
        row = {
            "algorithm": r["model_type"],
            "n_folds": s["num_folds"],
            "mean_F1": s["mean_F1_score"],
            "std_F1": s["std_F1_score"],
            "mean_Accuracy": s["mean_Accuracy"],
            "std_Accuracy": s["std_Accuracy"],
            "mean_MCC": s["mean_MCC"],
            "std_MCC": s["std_MCC"],
            "best_F1": r["best_f1"],
        }
        if sel_metric != "F1_score":
            row[f"best_{sel_metric}"] = r.get("best_metric_val", r["best_f1"])
        summary_rows.append(row)
    summary_csv = run_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    tqdm.write(f"\nArtifacts saved to: {run_dir}")

    ensembles = {"soft": soft_ensemble, "weighted": weighted_ensemble, "best": best_ensemble}
    artifacts = {
        "run_dir": run_dir,
        "soft_pkl": soft_pkl,
        "weighted_pkl": weighted_pkl,
        "best_pkl": best_pkl,
        "metadata_json": run_dir / "metadata.json",
        "summary_csv": summary_csv,
    }
    return ensembles, artifacts


# ── Model loading helpers ────────────────────────────────────────────────────

def load_model(model_path: str | Path):
    """Load an ensemble or single model from a pickle file."""
    with Path(model_path).open("rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, EnsembleClassifier):
        return obj
    if isinstance(obj, dict):
        if "model" in obj:
            return obj["model"]
        if "Model" in obj:
            return obj["Model"]
    return obj
