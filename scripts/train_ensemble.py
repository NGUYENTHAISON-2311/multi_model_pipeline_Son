#!/usr/bin/env python3
"""Train a multi-model ensemble and save soft_ensemble.pkl, weighted_ensemble.pkl, best_model.pkl.

Supports two feature modes:

1. Pipeline-only (default):
   Built-in 540-dim handcrafted features and/or AAindex lookup-table features.
   IUPred is run internally.

2. With ESM2 embeddings (--esm2 or --embeddings-pos / --embeddings-neg):
   feature_builder.py builds the combined matrix (pipeline + ESM2) for both
   positive and negative sets, and the pre-built matrices are passed directly
   to the training loop, skipping the internal IUPred + feature-extraction.

Examples
--------
  # Pipeline only (builtin 540-dim):
  python scripts/train_ensemble.py --positive pos.json --negative neg.json

  # AAindex + ESM2 (compute embeddings on the fly):
  python scripts/train_ensemble.py \\
      --positive pos.json --negative neg.json \\
      --features data/aaindex_features.json \\
      --esm2

  # AAindex + pre-computed embeddings (faster):
  python esm2_embedding.py -i pos.json -o pos_emb.pt
  python esm2_embedding.py -i neg.json -o neg_emb.pt
  python scripts/train_ensemble.py \\
      --positive pos.json --negative neg.json \\
      --features data/aaindex_features.json \\
      --embeddings-pos pos_emb.pt \\
      --embeddings-neg neg_emb.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configuration import load_config
from src.training_pipeline import SUPPORTED_MODELS, run_ensemble_training


def _load_iupred_scores(path: str, ids: list[str]) -> list[float]:
    with open(path) as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        return [float(raw[sid]) for sid in ids]
    return [float(v) for v in raw]


def _build_matrix(
    sequences: list[str],
    ids: list[str],
    feature_paths: list[str] | None,
    iupred_scores: list[float] | None,
    embeddings_pt: str | None,
    esm2_model: str,
    pool: str,
    config_path: str | None,
    no_esm2: bool,
    label: str,
) -> np.ndarray:
    """Build the combined feature matrix for one set of sequences."""
    from feature_builder import _load_embeddings_from_pt
    from esm2_embedding import load_model as load_esm2, embed_sequences
    from src.configuration import load_config
    from src.feature_loader import load_and_prepare_feature_specs, compute_sequence_feature_matrix

    config = load_config(config_path)
    specs = load_and_prepare_feature_specs(config, feature_paths)

    has_builtin = any(s.get("_builtin") for s in specs)
    if has_builtin and iupred_scores is None:
        raise ValueError(
            f'--iupred-scores-{label} is required when "builtin" is in --features.'
        )
    scores = iupred_scores or [0.0] * len(sequences)

    print(f"  Computing pipeline features ({label})...")
    rows = compute_sequence_feature_matrix(sequences, specs, scores)
    parts: list[np.ndarray] = [np.array(rows, dtype=np.float32)]
    print(f"    Pipeline shape: {parts[0].shape}")

    if not no_esm2:
        if embeddings_pt:
            print(f"  Loading pre-computed ESM2 embeddings ({label}) from {embeddings_pt} ...")
            esm = _load_embeddings_from_pt(embeddings_pt, ids)
        else:
            print(f"  Computing ESM2 embeddings ({label}, model={esm2_model})...")
            tokenizer, esm2_obj, device = load_esm2(esm2_model)
            print(f"    Running on: {device}")
            esm = (
                embed_sequences(sequences, tokenizer, esm2_obj, device, pool=pool)
                .cpu().numpy().astype(np.float32)
            )
        print(f"    ESM2 shape  : {esm.shape}")
        parts.append(esm)

    matrix = np.concatenate(parts, axis=1)
    print(f"  Combined shape ({label}): {matrix.shape}")
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train multiple algorithms and build ensemble classifiers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--config", help="Path to a pipeline config JSON file.")
    parser.add_argument("--positive", help="Override: positive sequences file (JSON or FASTA).")
    parser.add_argument("--negative", help="Override: negative sequences file (JSON or FASTA).")
    parser.add_argument("--output", help="Directory for training outputs. Default: outputs/training/.")

    # ── Training knobs ────────────────────────────────────────────────────────
    parser.add_argument(
        "--combos", type=int, default=10,
        help="Random hyperparameter combos per algorithm (default: 10).",
    )
    parser.add_argument(
        "--folds", type=int, default=None,
        help="Stratified k-fold CV splits (overrides config n_folds, default 5).",
    )
    parser.add_argument(
        "--metric", default="F1_score",
        choices=["F1_score", "Accuracy", "Precision", "Recall", "MCC", "Average"],
        help="Metric for combo selection (default: F1_score).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel worker processes. 1 = sequential (default). 0 = all CPUs.",
    )
    parser.add_argument(
        "--algorithms", nargs="+", metavar="ALGO",
        help=f"Restrict to these algorithm(s). Choices: {', '.join(SUPPORTED_MODELS)}.",
    )

    # ── Feature pipeline ──────────────────────────────────────────────────────
    parser.add_argument(
        "--features", nargs="+", metavar="SPEC", default=None,
        help=(
            'Feature spec(s): "builtin" or path to a JSON spec. '
            "Concatenated when multiple are given. Overrides config feature_files."
        ),
    )
    parser.add_argument(
        "--iupred-scores-pos", metavar="JSON",
        help='IUPred scores for positive set: {"ID": score} or flat list. Required when "builtin" in --features.',
    )
    parser.add_argument(
        "--iupred-scores-neg", metavar="JSON",
        help='IUPred scores for negative set: {"ID": score} or flat list. Required when "builtin" in --features.',
    )

    # ── ESM2 / embeddings ─────────────────────────────────────────────────────
    parser.add_argument(
        "--esm2", action="store_true",
        help=(
            "Append ESM2 embeddings to the pipeline features. "
            "Embeddings are computed on the fly unless --embeddings-pos/neg are given."
        ),
    )
    parser.add_argument(
        "--embeddings-pos", metavar="PT",
        help="Pre-computed ESM2 embeddings (.pt) for positive sequences (implies --esm2).",
    )
    parser.add_argument(
        "--embeddings-neg", metavar="PT",
        help="Pre-computed ESM2 embeddings (.pt) for negative sequences (implies --esm2).",
    )
    parser.add_argument(
        "--esm2-model", metavar="NAME", default="facebook/esm2_t6_8M_UR50D",
        help="HuggingFace ESM2 model name (default: facebook/esm2_t6_8M_UR50D).",
    )
    parser.add_argument(
        "--pool", choices=["mean", "cls"], default="mean",
        help="ESM2 pooling strategy (default: mean).",
    )

    args = parser.parse_args()

    use_esm2 = args.esm2 or bool(args.embeddings_pos or args.embeddings_neg)

    config = load_config(args.config)
    if args.features:
        config["feature_files"] = args.features

    # Filter algorithms if --algorithms is specified
    if args.algorithms:
        valid = {a.lower() for a in args.algorithms}
        unknown = valid - set(SUPPORTED_MODELS)
        if unknown:
            parser.error(f"Unknown algorithm(s): {', '.join(unknown)}. Choices: {', '.join(SUPPORTED_MODELS)}")
        all_algos = config["training"].get("algorithms", [])
        config["training"]["algorithms"] = [a for a in all_algos if a["type"] in valid]
        if not config["training"]["algorithms"]:
            parser.error(f"No matching algorithms found in config for: {', '.join(valid)}")

    _root = Path(__file__).resolve().parents[1]
    output_dir = args.output or str(_root / "outputs" / "training")

    # ── Build combined feature matrices (if ESM2 is requested) ───────────────
    pos_matrix: np.ndarray | None = None
    neg_matrix: np.ndarray | None = None
    dim_label: str | None = None

    if use_esm2:
        from src.configuration import resolve_pipeline_path
        from src.feature_pipeline import load_sequences_from_file
        from esm2_embedding import load_records_from_json

        training_cfg = config["training"]
        pos_path = Path(args.positive) if args.positive else resolve_pipeline_path(config, training_cfg["positive_json"])
        neg_path = Path(args.negative) if args.negative else resolve_pipeline_path(config, training_cfg["negative_json"])

        # Load sequences + IDs
        pos_records = load_records_from_json(str(pos_path)) if str(pos_path).endswith(".json") else None
        neg_records = load_records_from_json(str(neg_path)) if str(neg_path).endswith(".json") else None

        pos_seqs = [r["Sequence"] for r in pos_records] if pos_records else load_sequences_from_file(pos_path)
        neg_seqs = [r["Sequence"] for r in neg_records] if neg_records else load_sequences_from_file(neg_path)
        pos_ids = [r.get("ID", str(i)) for i, r in enumerate(pos_records)] if pos_records else [str(i) for i in range(len(pos_seqs))]
        neg_ids = [r.get("ID", str(i)) for i, r in enumerate(neg_records)] if neg_records else [str(i) for i in range(len(neg_seqs))]

        pos_iupred = _load_iupred_scores(args.iupred_scores_pos, pos_ids) if args.iupred_scores_pos else None
        neg_iupred = _load_iupred_scores(args.iupred_scores_neg, neg_ids) if args.iupred_scores_neg else None

        print("\n[Feature builder] Building combined feature matrices …")
        pos_matrix = _build_matrix(
            pos_seqs, pos_ids,
            feature_paths=args.features,
            iupred_scores=pos_iupred,
            embeddings_pt=args.embeddings_pos,
            esm2_model=args.esm2_model,
            pool=args.pool,
            config_path=args.config,
            no_esm2=False,
            label="positive",
        )
        neg_matrix = _build_matrix(
            neg_seqs, neg_ids,
            feature_paths=args.features,
            iupred_scores=neg_iupred,
            embeddings_pt=args.embeddings_neg,
            esm2_model=args.esm2_model,
            pool=args.pool,
            config_path=args.config,
            no_esm2=False,
            label="negative",
        )

        feat_label = "+".join(args.features) if args.features else "builtin"
        dim_label = f"{pos_matrix.shape[1]}-dim ({feat_label} + esm2:{args.esm2_model})"
        print(f"\nFeature matrix ready: {pos_matrix.shape[1]} total dims")

    ensembles, artifacts = run_ensemble_training(
        config,
        positive_path=args.positive,
        negative_path=args.negative,
        output_dir=output_dir,
        n_combos=args.combos,
        n_folds=args.folds,
        optimization_metric=args.metric,
        n_workers=args.workers,
        positive_features_matrix=pos_matrix,
        negative_features_matrix=neg_matrix,
        feature_dim_label=dim_label,
    )

    print(f"\nTraining completed — {len(ensembles)} ensemble variants")
    print(f"Run directory:        {artifacts['run_dir']}")
    print(f"soft_ensemble.pkl:    {artifacts['soft_pkl']}")
    print(f"weighted_ensemble.pkl:{artifacts['weighted_pkl']}")
    print(f"best_model.pkl:       {artifacts['best_pkl']}")
    print(f"Summary CSV:          {artifacts['summary_csv']}")
    print(f"Metadata:             {artifacts['metadata_json']}")
    print(f"\nTo predict with run_predict.py (e.g. soft ensemble):")
    print(f'  python run_predict.py --model "{artifacts["soft_pkl"]}" --input <dataset.json> ...')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
