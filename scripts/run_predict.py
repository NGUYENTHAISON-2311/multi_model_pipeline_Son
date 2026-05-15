"""
Sequence-level prediction using a trained ensemble + feature_builder features.

Builds a combined feature matrix (src/ pipeline + ESM2 embeddings) via
feature_builder.py, then passes it to a trained EnsembleClassifier (.pkl)
for sequence-level amyloid/non-amyloid classification.

The model must have been trained on the same feature set (same --features,
same ESM2 model / pool) that is used here.  A dimension mismatch check is
performed at startup.

CLI examples
------------
  # AAindex + ESM2, model trained on the same feature set:
  python run_predict.py \\
      --input  benchmark_dataset/benchmark_set.json \\
      --model  outputs/training/run_YYYYMMDD_HHMMSS/soft_ensemble.pkl \\
      --features data/aaindex_features.json \\
      --output predictions.csv

  # Re-use pre-computed embeddings (faster):
  python esm2_embedding.py -i benchmark_dataset/benchmark_set.json -o embeddings.pt
  python run_predict.py \\
      --input     benchmark_dataset/benchmark_set.json \\
      --model     outputs/training/.../soft_ensemble.pkl \\
      --embeddings embeddings.pt \\
      --features  data/aaindex_features.json \\
      --output    predictions.csv

  # Builtin 540-dim + ESM2 (IUPred scores required):
  python run_predict.py \\
      --input        benchmark_dataset/benchmark_set.json \\
      --model        outputs/training/.../soft_ensemble.pkl \\
      --features     builtin \\
      --iupred-scores scores.json \\
      --output       predictions.csv

  # Pipeline features only, no ESM2:
  python run_predict.py \\
      --input    benchmark_dataset/benchmark_set.json \\
      --model    outputs/training/.../soft_ensemble.pkl \\
      --features data/aaindex_features.json \\
      --no-esm2 \\
      --output   predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Add project root (for src.*) and scripts/ (for feature_builder, esm2_embedding)
_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[1]))
sys.path.insert(0, str(_here.parent))

from src.ensemble import EnsembleClassifier
from src.training_pipeline import load_model
from feature_builder import (
    _load_embeddings_from_pt,
    build_features,
)
from esm2_embedding import load_model as load_esm2, embed_sequences, load_records_from_json
from src.configuration import load_config
from src.feature_loader import load_and_prepare_feature_specs, compute_sequence_feature_matrix


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


def _write_output(path: Path, rows: list[dict]) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        _write_json(path, rows)
    else:
        _write_csv(path, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict amyloid/non-amyloid labels using a trained ensemble + feature_builder features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output columns
--------------
  id         : record ID from the JSON file
  sequence   : amino acid sequence
  label      : ground-truth LABEL (empty if absent in input)
  pred       : predicted class  (1 = amyloid, 0 = non-amyloid)
  prob_pos   : P(amyloid)  from predict_proba
  prob_neg   : P(non-amyloid) from predict_proba
""",
    )

    # ── Input / model ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--input", "-i", required=True, metavar="JSON",
        help='Benchmark dataset JSON: [{"ID": "...", "Sequence": "...", "LABEL": "..."}, ...]',
    )
    parser.add_argument(
        "--model", "-m", required=True, metavar="PKL",
        help="Trained ensemble pickle file (soft_ensemble.pkl / weighted_ensemble.pkl / best_model.pkl).",
    )

    # ── Feature building (mirrors feature_builder.py CLI) ────────────────────
    parser.add_argument(
        "--features", "-f", nargs="+", metavar="PATH", default=None,
        help=(
            'Feature spec paths. Use "builtin" for 540-dim handcrafted features '
            "or a path to a JSON spec (e.g. data/aaindex_features.json). "
            "Must match the feature set used during training."
        ),
    )
    parser.add_argument(
        "--embeddings", "-e", metavar="PT",
        help=(
            "Pre-computed ESM2 embeddings (.pt) from esm2_embedding.py. "
            "Skips re-running the ESM2 model. Ignored when --no-esm2 is set."
        ),
    )
    parser.add_argument(
        "--iupred-scores", metavar="JSON",
        help=(
            'JSON file with per-sequence average IUPred scores: {"<ID>": <score>} '
            "or a flat list of floats. Required when \"builtin\" is in --features."
        ),
    )
    parser.add_argument(
        "--no-esm2", action="store_true",
        help="Skip ESM2 embeddings (pipeline features only). Must match training setup.",
    )
    parser.add_argument(
        "--esm2-model", metavar="NAME", default="facebook/esm2_t6_8M_UR50D",
        help="HuggingFace ESM2 model name (default: facebook/esm2_t6_8M_UR50D).",
    )
    parser.add_argument(
        "--pool", choices=["mean", "cls"], default="mean",
        help="ESM2 pooling strategy (default: mean). Ignored when --embeddings is set.",
    )
    parser.add_argument(
        "--config", metavar="JSON",
        help="Path to pipeline config JSON (default: config/default_config.json).",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", "-o", metavar="FILE", default="predictions.csv",
        help="Output file (.csv or .json). Default: predictions.csv",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, metavar="T",
        help="Decision threshold for P(amyloid) → class 1 (default: 0.5).",
    )

    args = parser.parse_args()

    # ── Load records ──────────────────────────────────────────────────────────
    print(f"Loading sequences from {args.input} ...")
    records = load_records_from_json(args.input)
    ids = [r.get("ID", str(i)) for i, r in enumerate(records)]
    labels = [r.get("LABEL", "") for r in records]
    sequences = [r["Sequence"] for r in records]
    print(f"  {len(sequences)} sequence(s) loaded.")

    # ── IUPred scores ─────────────────────────────────────────────────────────
    iupred_scores: list[float] | None = None
    if args.iupred_scores:
        with open(args.iupred_scores) as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            iupred_scores = [float(raw[sid]) for sid in ids]
        elif isinstance(raw, list):
            if len(raw) != len(sequences):
                parser.error(
                    f"--iupred-scores list has {len(raw)} entries but input has {len(sequences)} sequences."
                )
            iupred_scores = [float(v) for v in raw]
        else:
            parser.error("--iupred-scores must be a JSON dict {ID: score} or a flat list of floats.")

    # ── Build pipeline features ───────────────────────────────────────────────
    config = load_config(args.config)
    specs = load_and_prepare_feature_specs(config, args.features)

    has_builtin = any(s.get("_builtin") for s in specs)
    if has_builtin and iupred_scores is None:
        parser.error(
            '--iupred-scores is required when "builtin" is in --features.'
        )

    scores = iupred_scores or [0.0] * len(sequences)

    print("Computing pipeline features...")
    pipeline_rows = compute_sequence_feature_matrix(sequences, specs, scores)
    pipeline_matrix = np.array(pipeline_rows, dtype=np.float32)
    print(f"  Pipeline features shape: {pipeline_matrix.shape}")

    parts: list[np.ndarray] = [pipeline_matrix]

    # ── ESM2 embeddings ───────────────────────────────────────────────────────
    if not args.no_esm2:
        if args.embeddings:
            print(f"Loading pre-computed embeddings from {args.embeddings} ...")
            esm = _load_embeddings_from_pt(args.embeddings, ids)
        else:
            print(f"Computing ESM2 embeddings ({args.esm2_model})...")
            tokenizer, esm2_model_obj, device = load_esm2(args.esm2_model)
            print(f"  Running on: {device}")
            esm = (
                embed_sequences(sequences, tokenizer, esm2_model_obj, device, pool=args.pool)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        print(f"  ESM2 shape: {esm.shape}")
        parts.append(esm)

    X = np.concatenate(parts, axis=1)
    print(f"Combined feature shape: {X.shape}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model from {args.model} ...")
    ensemble = load_model(args.model)
    print(f"  {ensemble}")

    # Dimension check
    expected_dim = getattr(ensemble, "n_features_in_", None)
    if expected_dim is not None and expected_dim != X.shape[1]:
        print(
            f"\nERROR: Feature dimension mismatch.\n"
            f"  Model expects : {expected_dim} features\n"
            f"  Built matrix  : {X.shape[1]} features\n"
            f"Ensure --features / --no-esm2 / --esm2-model match the training setup.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Predict ───────────────────────────────────────────────────────────────
    print("Running predictions...")
    proba = ensemble.predict_proba(X)        # (N, 2)
    prob_neg = proba[:, 0]
    prob_pos = proba[:, 1]
    preds = (prob_pos >= args.threshold).astype(int)

    # ── Assemble result rows ──────────────────────────────────────────────────
    rows: list[dict] = []
    for i, (sid, seq, lbl, pred, pp, pn) in enumerate(
        zip(ids, sequences, labels, preds, prob_pos, prob_neg)
    ):
        rows.append({
            "id":       sid,
            "sequence": seq,
            "label":    lbl,
            "pred":     int(pred),
            "prob_pos": round(float(pp), 6),
            "prob_neg": round(float(pn), 6),
        })

    # ── Save output ───────────────────────────────────────────────────────────
    out_path = Path(args.output)
    _write_output(out_path, rows)
    print(f"\nPredictions saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pos_pred = int(preds.sum())
    n_neg_pred = len(preds) - n_pos_pred
    print(f"\nPrediction summary (threshold={args.threshold}):")
    print(f"  Amyloid     (pred=1) : {n_pos_pred}")
    print(f"  Non-amyloid (pred=0) : {n_neg_pred}")

    # If ground-truth labels are present, print quick accuracy metrics
    known = [(lbl, int(p)) for lbl, p in zip(labels, preds) if lbl in ("AMYLOID", "NON-AMYLOID")]
    if known:
        y_true = [1 if lbl == "AMYLOID" else 0 for lbl, _ in known]
        y_pred = [p for _, p in known]
        tp = sum(yt == 1 and yp == 1 for yt, yp in zip(y_true, y_pred))
        tn = sum(yt == 0 and yp == 0 for yt, yp in zip(y_true, y_pred))
        fp = sum(yt == 0 and yp == 1 for yt, yp in zip(y_true, y_pred))
        fn = sum(yt == 1 and yp == 0 for yt, yp in zip(y_true, y_pred))
        acc = (tp + tn) / len(y_true) if y_true else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        mcc_denom = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
        mcc  = (tp*tn - fp*fn) / mcc_denom if mcc_denom else 0.0
        print(f"\nMetrics vs ground-truth labels ({len(known)} labeled sequences):")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(f"  MCC       : {mcc:.4f}")

    # Feature breakdown
    print(f"\nFeature breakdown:")
    print(f"  Pipeline features : {pipeline_matrix.shape[1]}")
    if not args.no_esm2:
        print(f"  ESM2 embeddings   : {parts[-1].shape[1]}")
    print(f"  Total             : {X.shape[1]}")


if __name__ == "__main__":
    main()
