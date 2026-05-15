#!/usr/bin/env python3
"""Run the ESM2 sliding-window benchmark on a trained run.

For every residue the feature vector is derived entirely from ESM2:

  sequence
      → pad (window_size - 1 on each side)
      → ESM2 forward pass  → (L_pad, H) per-residue embeddings
      → Cython window pooling → (N_windows, H) in one C pass
      → PCA                → (N_windows, D_pca)
      → batch predict_proba → (N_windows,) scores
      → Cython accumulate  → (L,) per-residue mean scores
      → threshold          → binary labels

The ESM2 model, pooling strategy, and PCA reducer are loaded automatically
from <run_dir>/esm2_pca_reducer.pkl saved by scripts/train_ensemble_esm2.py.

Speed notes
-----------
  * One ESM2 forward pass per sequence (not per window).
  * Window pooling and score accumulation are C loops (Cython).
  * Model scoring is one batch call over the entire window matrix.
  * Build the Cython extension first:
        python setup_cython.py build_ext --inplace

Usage examples
--------------
  # Auto-detect latest ESM2-trained run:
  python scripts/run_benchmark_esm2.py

  # Specific run:
  python scripts/run_benchmark_esm2.py \\
      --run-dir outputs/training/run_20260507_221701

  # Specific model variant (soft / weighted / best):
  python scripts/run_benchmark_esm2.py \\
      --run-dir outputs/training/run_20260507_221701 \\
      --variant weighted

  # Custom dataset + override window size:
  python scripts/run_benchmark_esm2.py \\
      --input benchmark_dataset/my_set.json \\
      --window-size 20

  # With extra aaindex features (must match training):
  python scripts/run_benchmark_esm2.py \\
      --features data/aaindex_features.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_esm2 import run_benchmark_esm2
from src.configuration import load_config

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BENCHMARK = _ROOT / "benchmark_dataset" / "benchmark_known_cores_seed_20.json"
_DEFAULT_TRAIN_OUT = _ROOT / "outputs" / "training"
_DEFAULT_BENCH_OUT = _ROOT / "outputs" / "benchmark_esm2"

_VARIANT_PKLS = {
    "soft":     "soft_ensemble.pkl",
    "weighted": "weighted_ensemble.pkl",
    "best":     "best_model.pkl",
}


def _find_latest_esm2_run(training_output: Path) -> Path:
    for run_dir in sorted(training_output.glob("run_*"), reverse=True):
        if (run_dir / "esm2_pca_reducer.pkl").exists():
            return run_dir
    raise FileNotFoundError(
        f"No ESM2-trained run found in {training_output}.\n"
        "Train with:  python scripts/train_ensemble_esm2.py --features ... esm2"
    )


def _read_global_metrics(results_json: Path) -> dict:
    with results_json.open(encoding="utf-8") as fh:
        rows = json.load(fh)
    for row in rows:
        if str(row.get("id", "")).upper() == "GLOBAL":
            return {
                "F1":        round(float(row.get("f1_score",  0.0)), 4),
                "Accuracy":  round(float(row.get("accuracy",  0.0)), 4),
                "Precision": round(float(row.get("precision", 0.0)), 4),
                "Recall":    round(float(row.get("recall",    0.0)), 4),
                "MCC":       round(float(row.get("mcc",       0.0)), 4),
                "SOV":       round(float(row.get("sov",       0.0)), 4),
            }
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ESM2 sliding-window benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-dir", metavar="DIR",
        help="Training run directory.  Auto-detects latest ESM2 run when omitted.",
    )
    parser.add_argument(
        "--variant", choices=list(_VARIANT_PKLS), default=None, metavar="V",
        help="Ensemble variant to benchmark: soft | weighted | best (default: all three).",
    )
    parser.add_argument(
        "--input", "-i", default=str(_DEFAULT_BENCHMARK), metavar="JSON",
        help=f"Benchmark dataset JSON (default: {_DEFAULT_BENCHMARK.name}).",
    )
    parser.add_argument(
        "--output", "-o", default=str(_DEFAULT_BENCH_OUT), metavar="DIR",
        help=f"Root output directory (default: {_DEFAULT_BENCH_OUT}).",
    )
    parser.add_argument(
        "--config", metavar="JSON",
        help="Pipeline config JSON (default: config/default_config.json).",
    )
    parser.add_argument(
        "--features", nargs="+", metavar="SPEC", default=None,
        help=(
            "Extra feature specs (e.g. data/aaindex_features.json) appended after "
            "the ESM2 embedding.  Must match what was used during training.  "
            "Auto-detected from model metadata.json when omitted."
        ),
    )
    parser.add_argument(
        "--window-size", type=int, default=None, dest="window_size",
        help="Sliding window width (default: from config).",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold (default: 0.5).",
    )
    parser.add_argument(
        "--esm2-model", metavar="MODEL", dest="esm2_model",
        help="Override the ESM2 model name stored in esm2_pca_reducer.pkl.",
    )
    parser.add_argument(
        "--esm2-pool", choices=["mean", "max", "cls"], dest="esm2_pool",
        help="Override the ESM2 pooling strategy stored in esm2_pca_reducer.pkl.",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    # ── Locate run directory ──────────────────────────────────────────────────
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_dir():
            parser.error(f"--run-dir does not exist: {run_dir}")
    else:
        print(f"No --run-dir given. Searching for latest ESM2 run in {_DEFAULT_TRAIN_OUT} …")
        run_dir = _find_latest_esm2_run(_DEFAULT_TRAIN_OUT)

    print(f"\nTraining run : {run_dir}")

    bench_input = Path(args.input)
    if not bench_input.exists():
        parser.error(f"Benchmark input not found: {bench_input}")
    print(f"Dataset      : {bench_input}")

    # ── Choose variants ───────────────────────────────────────────────────────
    variants = (
        [(args.variant, _VARIANT_PKLS[args.variant])]
        if args.variant
        else list(_VARIANT_PKLS.items())
    )

    suite_out = Path(args.output) / run_dir.name
    suite_out.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[dict] = []
    t0 = time.monotonic()

    for variant_name, pkl_name in variants:
        model_path = run_dir / pkl_name
        if not model_path.exists():
            print(f"  [skip] {pkl_name} not found in {run_dir}")
            continue

        out_subdir = suite_out / variant_name
        out_subdir.mkdir(parents=True, exist_ok=True)
        output_stem = str(out_subdir / "results")

        print(f"\n{'─' * 60}")
        print(f"  Variant : {variant_name}")
        print(f"  Model   : {model_path}")
        print(f"  Output  : {out_subdir}")
        print(f"{'─' * 60}")

        try:
            artifacts = run_benchmark_esm2(
                config,
                model_path=str(model_path),
                input_json=str(bench_input),
                output_path=str(out_subdir),
                output_name=output_stem,
                window_size=args.window_size,
                threshold=args.threshold,
                cli_feature_paths=args.features,
                esm2_model_override=args.esm2_model,
                esm2_pool_override=args.esm2_pool,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            comparison_rows.append({"variant": variant_name, "F1": "ERROR",
                                    "Accuracy": "", "Precision": "", "Recall": "",
                                    "MCC": "", "SOV": ""})
            continue

        results_json = artifacts.get("results_json")
        metrics = _read_global_metrics(Path(results_json)) if results_json and Path(results_json).exists() else {}
        comparison_rows.append({"variant": variant_name, **metrics})

    elapsed = time.monotonic() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("ESM2 BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"Run       : {run_dir.name}")
    print(f"Dataset   : {bench_input.name}")
    print(f"Elapsed   : {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")

    if comparison_rows:
        cols = ["variant", "F1", "Accuracy", "Precision", "Recall", "MCC", "SOV"]
        col_w = {c: max(len(c), max(len(str(r.get(c, ""))) for r in comparison_rows)) for c in cols}
        sep    = "  ".join("-" * col_w[c] for c in cols)
        header = "  ".join(c.ljust(col_w[c]) for c in cols)
        print(f"\n{sep}\n{header}\n{sep}")
        for r in comparison_rows:
            print("  ".join(str(r.get(c, "")).ljust(col_w[c]) for c in cols))
        print(sep)

    # Save summary CSV
    summary_csv = suite_out / "summary_esm2.csv"
    if comparison_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)
        print(f"\nSummary → {summary_csv}")

    # Save runtime log
    runtime_log = suite_out / "runtime_esm2.json"
    runtime_log.write_text(json.dumps({
        "run_id":     run_dir.name,
        "dataset":    str(bench_input),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s":  round(elapsed, 1),
    }, indent=2))
    print(f"Runtime  → {runtime_log}")
    print(f"\nAll outputs under: {suite_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
