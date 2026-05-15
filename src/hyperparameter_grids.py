"""Hyperparameter search spaces for each algorithm.

Ranges are based on the Cross-Beta predictor publication
(Gonay et al., Alzheimer's Dement. 2025; Table S3) and the
companion repository: github.com/Valentin-Gonay/cross-beta-predictor-modelcreation
"""

from __future__ import annotations

import random

import numpy as np

# ── Search‑space definitions ─────────────────────────────────────────────────
# Each key maps to a dict of {param_name: list_of_values}.

_n_estimators = [int(x) for x in np.linspace(500, 5000, num=500)]
_max_depth = [int(x) for x in np.linspace(5, 110, num=30)] + [None]
_max_features = ["sqrt", "log2"]
_min_samples_split = list(range(2, 11))           # [2..10]
_min_samples_leaf = [1, 2, 4, 8]
_bootstrap = [True, False]
_criterion = ["gini", "entropy", "log_loss"]
_splitter = ["best", "random"]
_C_value = [float(x) for x in np.linspace(0.01, 5, num=100)]
_kernel = ["linear", "poly", "rbf", "sigmoid"]
_min_impurity_decrease = [float(x) for x in np.linspace(0, 10, num=100)]
_max_depth_cat = list(range(17))                   # [0..16]
_bootstrap_type_cat = ["Bayesian", "Bernoulli", "MVS", "No"]


SEARCH_SPACES: dict[str, dict[str, list]] = {
    "random_forest": {
        "n_estimators": _n_estimators,
        "max_features": _max_features,
        "max_depth": _max_depth,
        "min_samples_split": _min_samples_split,
        "min_samples_leaf": _min_samples_leaf,
        "bootstrap": _bootstrap,
    },
    "extra_trees": {
        "n_estimators": _n_estimators,
        "max_features": _max_features,
        "max_depth": _max_depth,
        "min_samples_split": _min_samples_split,
        "min_samples_leaf": _min_samples_leaf,
        "bootstrap": _bootstrap,
    },
    "catboost": {
        "n_estimators": _n_estimators,
        "max_depth": _max_depth_cat,
        "min_child_samples": _min_samples_split,
        "bootstrap_type": _bootstrap_type_cat,
    },
    "gradient_boosting": {
        "n_estimators": _n_estimators,
        "max_depth": _max_depth,
        "min_samples_split": _min_samples_split,
        "min_samples_leaf": _min_samples_leaf,
        "max_features": _max_features,
        "min_impurity_decrease": _min_impurity_decrease,
    },
    "adaboost": {
        "n_estimators": _n_estimators,
    },
    "decision_tree": {
        "max_depth": _max_depth,
        "min_samples_split": _min_samples_split,
        "min_samples_leaf": _min_samples_leaf,
        "max_features": _max_features,
        "criterion": _criterion,
        "splitter": _splitter,
    },
    "svm": {
        "C": _C_value,
        "kernel": _kernel,
    },
}


def sample_random_params(model_type: str, rng: random.Random | None = None) -> dict:
    """Return one random hyperparameter combination for *model_type*.

    Each parameter is independently sampled uniformly from its possible values.
    """
    if model_type not in SEARCH_SPACES:
        raise ValueError(
            f"No search space defined for {model_type!r}. "
            f"Available: {sorted(SEARCH_SPACES)}"
        )
    rng = rng or random.Random()
    space = SEARCH_SPACES[model_type]
    return {key: rng.choice(values) for key, values in space.items()}


def sample_n_combos(model_type: str, n: int, seed: int | None = None) -> list[dict]:
    """Return *n* distinct random hyperparameter combinations for *model_type*."""
    rng = random.Random(seed)
    seen: set[str] = set()
    combos: list[dict] = []
    attempts = 0
    max_attempts = n * 20  # safety limit
    while len(combos) < n and attempts < max_attempts:
        g = sample_random_params(model_type, rng)
        key = str(sorted(g.items()))
        if key not in seen:
            seen.add(key)
            combos.append(g)
        attempts += 1
    return combos
