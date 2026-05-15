"""
Classification benchmark over sets 1-10.

For each benchmark_classification_set_N.json:
  1. Load sequences and ground-truth labels (AMYLOID=1, NONAMYLOID=0).
  2. Compute per-sequence average IUPred disorder scores.
  3. Extract the 540-dim feature vector.
  4. Predict with the trained best_model ensemble.
  5. Report per-set and aggregate metrics + confusion matrix.

Outputs
-------
  outputs/classification/run_<RUN_ID>/results_per_set.csv
  outputs/classification/run_<RUN_ID>/results_per_sample.csv
  outputs/classification/run_<RUN_ID>/summary.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_pipeline import (
    compute_average_iupred_scores_from_sequences,
    extract_sequence_features,
)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_RUN      = "run_20260507_221701"
DEFAULT_SETS_DIR = PROJECT_ROOT / "benchmark_dataset"
DEFAULT_IUPRED   = PROJECT_ROOT / "scripts" / "iupred3" / "iupred3.py"
LABEL_MAP        = {"AMYLOID": 1, "NONAMYLOID": 0}
IDX_TO_LABEL     = {1: "AMYLOID", 0: "NONAMYLOID"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_classification_set(path: Path) -> tuple[list[str], list[str], list[int]]:
    """Return (ids, sequences, integer_labels) from a classification JSON."""
    with path.open() as f:
        records = json.load(f)
    ids   = [r["ID"]       for r in records]
    seqs  = [r["Sequence"] for r in records]
    labels = [LABEL_MAP[r["LABEL"]] for r in records]
    return ids, seqs, labels


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn = int(cm[0, 0]), int(cm[0, 1])
    fp, tn = int(cm[1, 0]), int(cm[1, 1])
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "mcc":       round(matthews_corrcoef(y_true, y_pred), 4),
    }


def print_confusion_matrix(metrics: dict, title: str = "") -> None:
    if title:
        print(f"\n  {title}")
    tp, tn = metrics["TP"], metrics["TN"]
    fp, fn = metrics["FP"], metrics["FN"]
    print(f"  {'':20s} Pred AMYLOID  Pred NONAMYLOID")
    print(f"  {'Actual AMYLOID':20s}   {tp:>6}          {fn:>6}")
    print(f"  {'Actual NONAMYLOID':20s}   {fp:>6}          {tn:>6}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    run_dir    = PROJECT_ROOT / "outputs" / "training" / args.run_id
    model_path = run_dir / "best_model.pkl"
    out_dir    = PROJECT_ROOT / "outputs" / "classification" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {model_path}")
    with model_path.open("rb") as f:
        model = pickle.load(f)
    print(f"  {model}")

    # ── Discover sets ─────────────────────────────────────────────────────────
    set_paths = sorted(
        Path(args.sets_dir).glob("benchmark_classification_set_*.json"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not set_paths:
        sys.exit(f"No classification sets found in {args.sets_dir}")
    print(f"\nFound {len(set_paths)} classification sets")

    per_set_rows: list[dict] = []
    per_sample_rows: list[dict] = []

    all_true:  list[int] = []
    all_pred:  list[int] = []

    # ── Process each set ──────────────────────────────────────────────────────
    for set_path in set_paths:
        set_name = set_path.stem
        set_num  = int(set_path.stem.split("_")[-1])
        print(f"\n{'─'*60}")
        print(f"  Set {set_num:2d}: {set_path.name}")

        ids, sequences, y_true = load_classification_set(set_path)
        n = len(sequences)
        print(f"  {n} samples  (AMYLOID={sum(y_true)}, NONAMYLOID={n - sum(y_true)})")

        # IUPred disorder scores
        print("  Computing IUPred scores …")
        iupred_scores = compute_average_iupred_scores_from_sequences(
            sequences,
            iupred_script=args.iupred,
            input_type=args.iupred_type,
        )

        # Feature extraction
        features = extract_sequence_features(
            sequences, iupred_scores, desc=f"  Set {set_num} features"
        )
        X = np.array(features, dtype=float)

        # Predict
        y_pred = model.predict(X).tolist()
        proba  = model.predict_proba(X)[:, 1].tolist()  # P(AMYLOID)

        # Metrics
        m = compute_metrics(y_true, y_pred)
        print_confusion_matrix(m, title=f"Confusion Matrix — {set_name}")
        print(f"  F1={m['f1']}  Acc={m['accuracy']}  MCC={m['mcc']}  "
              f"Prec={m['precision']}  Rec={m['recall']}")

        per_set_rows.append({"set": set_name, "n_samples": n, **m})

        for sid, seq, yt, yp, prob in zip(ids, sequences, y_true, y_pred, proba):
            per_sample_rows.append({
                "set":        set_name,
                "ID":         sid,
                "sequence":   seq,
                "true_label": IDX_TO_LABEL[yt],
                "pred_label": IDX_TO_LABEL[yp],
                "prob_amyloid": round(prob, 4),
                "correct":    yt == yp,
            })

        all_true.extend(y_true)
        all_pred.extend(y_pred)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  AGGREGATE RESULTS (all sets combined)")
    agg = compute_metrics(all_true, all_pred)
    print_confusion_matrix(agg, title="Aggregate Confusion Matrix")
    print(f"  F1={agg['f1']}  Acc={agg['accuracy']}  MCC={agg['mcc']}  "
          f"Prec={agg['precision']}  Rec={agg['recall']}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    df_sets    = pd.DataFrame(per_set_rows)
    df_samples = pd.DataFrame(per_sample_rows)

    sets_csv    = out_dir / "results_per_set.csv"
    samples_csv = out_dir / "results_per_sample.csv"
    summary_json = out_dir / "summary.json"

    df_sets.to_csv(sets_csv, index=False)
    df_samples.to_csv(samples_csv, index=False)

    summary = {
        "run_id":      args.run_id,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "n_sets":      len(set_paths),
        "n_samples":   len(all_true),
        "aggregate":   agg,
        "per_set":     per_set_rows,
    }
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs saved to: {out_dir}")
    print(f"  {sets_csv.name}")
    print(f"  {samples_csv.name}")
    print(f"  {summary_json.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify benchmark_classification_set_1-10 with a trained ensemble.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-id", default=DEFAULT_RUN,
        help="Training run ID (subfolder under outputs/training/).",
    )
    parser.add_argument(
        "--sets-dir", default=str(DEFAULT_SETS_DIR),
        help="Directory containing benchmark_classification_set_*.json files.",
    )
    parser.add_argument(
        "--iupred", default=str(DEFAULT_IUPRED),
        help="Path to iupred3.py script.",
    )
    parser.add_argument(
        "--iupred-type", default="long", choices=["long", "short"],
        help="IUPred input type.",
    )
    main(parser.parse_args())
