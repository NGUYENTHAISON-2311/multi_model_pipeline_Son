#!/usr/bin/env python3
"""Train ensemble + run padded benchmark in one go."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_pipeline_new import run_benchmark_new
from src.configuration import load_config
from src.training_pipeline import SUPPORTED_MODELS, run_ensemble_training


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full pipeline: train ensemble → padded benchmark.",
    )
    # Training args
    parser.add_argument("--config", help="Config JSON file.")
    parser.add_argument("--positive", help="Positive sequences file.")
    parser.add_argument("--negative", help="Negative sequences file.")
    parser.add_argument("--training-output", dest="train_out", help="Training output directory.")
    parser.add_argument("--combos", type=int, default=10, help="Global fallback for random combos per algorithm (overridable per-algo in config). Default: 10.")
    parser.add_argument("--folds", type=int, default=None, help="Number of k-fold CV splits (overrides config n_folds, default 5).")
    parser.add_argument(
        "--metric", default="F1_score",
        choices=["F1_score", "Accuracy", "Precision", "Recall", "MCC", "Average"],
        help=(
            "Metric used to select the best hyperparameter combo. Default: F1_score. "
            "MCC is least affected by class imbalance. "
            "Average = mean of F1_score + Accuracy + Precision + Recall + MCC."
        ),
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers. 0 = all CPUs.")
    parser.add_argument(
        "--algorithms", nargs="+", metavar="ALGO",
        help=f"Only train these algorithm(s). Choices: {', '.join(SUPPORTED_MODELS)}.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        metavar="SPEC",
        default=None,
        help=(
            "Feature spec(s) to use. Each entry is either the keyword 'builtin' "
            "(original 540-dim features) or a path to a lookup_table JSON file. "
            "Multiple entries are concatenated. Overrides config feature_files. "
            "Default: builtin. "
            "Examples: --features builtin | "
            "--features data/aaindex_features.json | "
            "--features builtin data/aaindex_features.json"
        ),
    )
    # Benchmark args
    parser.add_argument("--benchmark-input", dest="bench_input", help="Benchmark sequences JSON.")
    parser.add_argument("--benchmark-output", dest="bench_out", help="Benchmark output directory.")
    parser.add_argument("--model", help="Skip training — benchmark with this existing model.")
    parser.add_argument("--threshold", type=float, default=None, help="Padded benchmark threshold k.")
    parser.add_argument("--window-size", type=int, default=None, dest="window_size", help="Sliding window size (overrides config).")
    parser.add_argument("--name", default="benchmark_results", help="Benchmark output filename prefix.")
    parser.add_argument(
        "--classifier",
        action="store_true",
        help=(
            "Sequence-level classification mode. Instead of per-residue position metrics, "
            "evaluate each sequence as positive (if any residue is predicted as core) or "
            "negative. Ground truth comes from the LABEL field in the input JSON."
        ),
    )
    parser.add_argument(
        "--positive-label",
        default=None,
        dest="positive_label",
        help="Label string that counts as the positive class (case-insensitive). Default: AMYLOID.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.features:
        config["feature_files"] = args.features
    _root = Path(__file__).resolve().parents[1]

    # Filter algorithms if requested
    if args.algorithms:
        valid = {a.lower() for a in args.algorithms}
        unknown = valid - set(SUPPORTED_MODELS)
        if unknown:
            parser.error(f"Unknown algorithm(s): {', '.join(unknown)}. Choices: {', '.join(SUPPORTED_MODELS)}")
        all_algos = config["training"].get("algorithms", [])
        config["training"]["algorithms"] = [a for a in all_algos if a["type"] in valid]
        if not config["training"]["algorithms"]:
            parser.error(f"No matching algorithms found in config for: {', '.join(valid)}")

    if args.model:
        ensemble_path = Path(args.model)
    else:
        train_out = args.train_out or str(_root / "outputs" / "training")
        ensembles, train_artifacts = run_ensemble_training(
            config,
            positive_path=args.positive,
            negative_path=args.negative,
            output_dir=train_out,
            n_combos=args.combos,
            n_folds=args.folds,
            optimization_metric=args.metric,
            n_workers=args.workers,
        )
        ensemble_path = train_artifacts["soft_pkl"]

    # Benchmark (padded avg)
    bench_out = args.bench_out or str(_root / "outputs" / "benchmark")
    bench_arts = run_benchmark_new(
        config, model_path=ensemble_path, input_json=args.bench_input,
        output_path=bench_out, threshold=args.threshold,
        window_size=args.window_size, output_name=args.name,
        cli_feature_paths=args.features if args.model else None,
        classifier_mode=args.classifier, positive_label=args.positive_label,
    )

    print("\nFull pipeline completed.")
    print(f"Ensemble:            {ensemble_path}")
    csv_out = bench_arts.get('results_csv') or bench_arts.get('predictions_csv')
    print(f"Benchmark CSV:       {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
