#!/usr/bin/env python3
"""Adaptive multi-scale sliding-window benchmark."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_pipeline_adaptive import run_benchmark_adaptive
from src.configuration import load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adaptive benchmark: uses multiple window sizes where the "
                    "model is uncertain, selecting the most confident scale.",
    )
    parser.add_argument("--config", help="Path to a pipeline config JSON file.")
    parser.add_argument("--model", help="Path to an ensemble.pkl or model pickle.")
    parser.add_argument("--input", help="Benchmark sequences JSON.")
    parser.add_argument("--output", help="Output directory. Default: outputs/benchmark/.")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Score threshold k. Default: 0.5.",
    )
    parser.add_argument(
        "--window-size", type=int, default=None, dest="default_window",
        help="Default sliding window size. Default: from config (18).",
    )
    parser.add_argument(
        "--min-window", type=int, default=None, dest="min_window",
        help="Minimum window size for adaptive search. Default: 11.",
    )
    parser.add_argument(
        "--max-window", type=int, default=None, dest="max_window",
        help="Maximum window size for adaptive search. Default: 23.",
    )
    parser.add_argument(
        "--confidence-margin", type=float, default=None, dest="confidence_margin",
        help="Confidence margin: only try other sizes when |score-0.5| < margin. "
             "Default: 0.15. Lower = fewer extra evaluations, higher = more exploration.",
    )
    parser.add_argument(
        "--name", default="benchmark_adaptive_results",
        help="Output filename prefix. Default: benchmark_adaptive_results.",
    )
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
    _root = Path(__file__).resolve().parents[1]
    output_dir = args.output or str(_root / "outputs" / "benchmark")

    artifacts = run_benchmark_adaptive(
        config,
        model_path=args.model,
        input_json=args.input,
        output_path=output_dir,
        threshold=args.threshold,
        default_window=args.default_window,
        min_window=args.min_window,
        max_window=args.max_window,
        confidence_margin=args.confidence_margin,
        output_name=args.name,
        cli_feature_paths=args.features,
        classifier_mode=args.classifier,
        positive_label=args.positive_label,
    )
    print("Adaptive benchmark completed.")
    print(f"Results CSV:   {artifacts['results_csv']}")
    print(f"Results JSON:  {artifacts['results_json']}")
    print(f"Windows JSON:  {artifacts['windows_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
