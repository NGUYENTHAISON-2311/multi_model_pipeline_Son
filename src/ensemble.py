"""Ensemble classifier that wraps multiple trained models.

Supports three prediction modes selectable at load time:

*  ``soft_voting`` — average ``predict_proba`` across all models.
*  ``weighted_voting`` — weighted average of ``predict_proba``, where each
   model's weight is its validation F1 score.
*  ``best_model`` — use only the single model with the highest validation F1.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class EnsembleClassifier:
    """Thin wrapper that provides the standard sklearn ``predict`` /
    ``predict_proba`` interface over a collection of heterogeneous models.

    Parameters
    ----------
    models : list
        Fitted sklearn-compatible estimators.
    weights : list[float] | None
        Per-model weights (typically validation F1).  Required for
        ``weighted_voting``; ignored otherwise.
    mode : str
        One of ``"soft_voting"``, ``"weighted_voting"``, ``"best_model"``.
    best_index : int | None
        Index of the best model (highest weight).  Computed automatically
        if not supplied.
    """

    def __init__(
        self,
        models: list,
        weights: list[float] | None = None,
        mode: str = "soft_voting",
        best_index: int | None = None,
    ):
        if not models:
            raise ValueError("At least one model is required.")
        self.models = models
        self.weights = weights
        self.mode = mode

        if best_index is not None:
            self.best_index = best_index
        elif weights:
            self.best_index = int(np.argmax(weights))
        else:
            self.best_index = 0

        # Expose n_features_in_ so benchmark auto-detect still works.
        self.n_features_in_ = getattr(models[0], "n_features_in_", None)

    # -- sklearn-compatible interface ----------------------------------------

    def predict_proba(self, X) -> np.ndarray:
        if self.mode == "best_model":
            return self.models[self.best_index].predict_proba(X)

        probas = np.array([m.predict_proba(X) for m in self.models])

        if self.mode == "weighted_voting" and self.weights is not None:
            w = np.array(self.weights, dtype=float)
            w = w / w.sum()  # normalise
            # probas shape: (n_models, n_samples, n_classes)
            avg = np.tensordot(w, probas, axes=([0], [0]))
        else:
            # soft_voting
            avg = probas.mean(axis=0)
        return avg

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    # -- Persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> "EnsembleClassifier":
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, EnsembleClassifier):
            return obj
        raise TypeError(f"Expected EnsembleClassifier, got {type(obj).__name__}")

    # -- Introspection -------------------------------------------------------

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "n_models": len(self.models),
            "model_types": [type(m).__name__ for m in self.models],
            "weights": self.weights,
            "best_index": self.best_index,
        }

    def __repr__(self) -> str:
        types = [type(m).__name__ for m in self.models]
        return (
            f"EnsembleClassifier(mode={self.mode!r}, "
            f"n_models={len(self.models)}, types={types})"
        )
