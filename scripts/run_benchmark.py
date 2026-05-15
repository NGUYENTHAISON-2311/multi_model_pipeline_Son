#!/usr/bin/env python3
"""Run the sliding-window benchmark with an ensemble (or single) model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_pipeline import run_benchmark
from src.configuration import load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sliding-window benchmark using an ensemble model.",
    )
    parser.add_argument("--config", help="Path to a pipeline config JSON file.")
    parser.add_argument(
        "--model",
        help="Path to an ensemble.pkl (or any model pickle). "
             "Defaults to the latest ensemble in the training output.",
    )
    parser.add_argument("--input", help="Benchmark sequences JSON.")
    parser.add_argument("--output", help="Output directory. Default: outputs/benchmark/.")
    parser.add_argument("--name", default="benchmark_results", help="Output filename prefix. Default: benchmark_results.")
    parser.add_argument(
        "--features",
        nargs="+",
        metavar="SPEC",
        default=None,
        help=(
            "Feature spec(s) to use. 'builtin' or path(s) to JSON files. "
            "Overrides config and model metadata. Default: auto-detect from model metadata."
        ),
    )
    parser.add_argument(
        "--aggregation-method",
        choices=["max", "mean", "vote"],
        help="Aggregation method for per-residue scores. Overrides config value.",
    )
    parser.add_argument(
        "--vote-fraction",
        type=float,
        help="Minimum fraction of windows that must score >= threshold to predict positive (only used with --aggregation-method vote). Overrides config value.",
    )
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
    if args.aggregation_method:
        config["benchmark"]["aggregation_method"] = args.aggregation_method
    if args.vote_fraction is not None:
        config["benchmark"]["vote_fraction"] = args.vote_fraction
    _root = Path(__file__).resolve().parents[1]
    output_dir = args.output or str(_root / "outputs" / "benchmark")

    artifacts = run_benchmark(
        config, model_path=args.model, input_json=args.input, output_path=output_dir,
        output_name=args.name, cli_feature_paths=args.features,
        classifier_mode=args.classifier, positive_label=args.positive_label,
    )
    print("Benchmark completed.")
    print(f"Results CSV:   {artifacts['results_csv']}")
    print(f"Results JSON:  {artifacts['results_json']}")
    print(f"Windows JSON:  {artifacts['windows_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
