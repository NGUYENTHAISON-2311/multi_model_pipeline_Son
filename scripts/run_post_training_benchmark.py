#!/usr/bin/env python3
"""Run the full benchmark suite automatically after training.

Discovers all three ensemble variants (soft_ensemble.pkl, weighted_ensemble.pkl,
best_model.pkl) inside a training run directory, then benchmarks each one with
both the padded sliding-window and the adaptive multi-scale methods against
benchmark_known_cores_seed_20.json.

Outputs
-------
Per-variant / per-mode results land in:
  <output_dir>/<run_id>/<variant>_<mode>/

A final comparison table is printed to stdout and saved to:
  <output_dir>/<run_id>/summary_comparison.csv

Usage examples
--------------
  # Watch mode: run in a second terminal BEFORE or DURING training.
  # The script waits until training completes, records total runtime,
  # then runs all benchmarks automatically:
  python scripts/run_post_training_benchmark.py --watch

  # Watch a specific run that is still in progress:
  python scripts/run_post_training_benchmark.py \\
      --watch --run-dir outputs/training/run_20260506_201459

  # Benchmark a finished run immediately (no watching):
  python scripts/run_post_training_benchmark.py \\
      --run-dir outputs/training/run_20260506_201459

  # Auto-detect latest finished run:
  python scripts/run_post_training_benchmark.py

  # Custom benchmark dataset:
  python scripts/run_post_training_benchmark.py \\
      --input benchmark_dataset/benchmark_known_cores_seed_20.json

  # With extra features (must match training setup):
  python scripts/run_post_training_benchmark.py \\
      --features data/aaindex_features.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_pipeline import run_benchmark
from src.benchmark_pipeline_adaptive import run_benchmark_adaptive
from src.configuration import load_config

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BENCHMARK = _ROOT / "benchmark_dataset" / "benchmark_known_cores_seed_20.json"
_DEFAULT_TRAIN_OUT  = _ROOT / "outputs" / "training"
_DEFAULT_BENCH_OUT  = _ROOT / "outputs" / "benchmark"

_VARIANTS = [
    ("soft",     "soft_ensemble.pkl"),
    ("weighted", "weighted_ensemble.pkl"),
    ("best",     "best_model.pkl"),
]

_MODES = ["padded", "adaptive"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s   = divmod(rem, 60)
    if td.days:
        h += td.days * 24
    return f"{h:02d}:{m:02d}:{s:02d}"


def _run_is_complete(run_dir: Path) -> bool:
    """A run is complete when metadata.json exists (written last by training)."""
    return (run_dir / "metadata.json").exists()


def _find_latest_run(training_output: Path) -> Path:
    runs = sorted(training_output.glob("run_*"), reverse=True)
    for run_dir in runs:
        if any((run_dir / pkl).exists() for _, pkl in _VARIANTS):
            return run_dir
    raise FileNotFoundError(
        f"No training run with ensemble pickles found in {training_output}.\n"
        "Run train_ensemble.py first, or pass --run-dir explicitly."
    )


def _watch_for_completion(
    training_output: Path,
    run_dir: Path | None,
    poll_interval: int = 10,
) -> tuple[Path, float]:
    """Block until a training run appears and is complete.

    If *run_dir* is given, watch that specific directory.
    Otherwise, watch *training_output* for any new run_* directory.

    Returns (completed_run_dir, elapsed_seconds).
    """
    watch_start = time.monotonic()

    if run_dir is not None:
        # Watch a known directory that may still be running.
        print(f"Watching run directory: {run_dir}")
        print(f"Waiting for metadata.json (written when training finishes) …\n")
        while not _run_is_complete(run_dir):
            elapsed = time.monotonic() - watch_start
            print(f"\r  Elapsed: {_fmt_duration(elapsed)}  still running …", end="", flush=True)
            time.sleep(poll_interval)
        elapsed = time.monotonic() - watch_start
        print(f"\r  Elapsed: {_fmt_duration(elapsed)}  ✓ Training complete!      ")
        return run_dir, elapsed

    # Watch the training output directory for a NEW completed run.
    seen_runs: set[Path] = set(training_output.glob("run_*"))
    print(f"Watching {training_output} for a new completed training run …")
    print(f"(Start training now in another terminal.)\n")
    target_dir: Path | None = None

    while True:
        elapsed = time.monotonic() - watch_start
        current_runs = set(training_output.glob("run_*"))
        new_runs = current_runs - seen_runs

        # First: check if a newly-seen run is already complete.
        for r in sorted(new_runs, reverse=True):
            if _run_is_complete(r):
                target_dir = r
                break
        if target_dir:
            break

        # If we spotted a new (still in-progress) run, lock onto it.
        if new_runs and target_dir is None:
            candidate = sorted(new_runs, reverse=True)[0]
            print(f"\n  New run detected: {candidate.name} — waiting for completion …")
            run_dir = candidate  # fall through to the "known dir" path

        if run_dir is not None and _run_is_complete(run_dir):
            target_dir = run_dir
            break

        print(f"\r  Elapsed: {_fmt_duration(elapsed)}  waiting …", end="", flush=True)
        time.sleep(poll_interval)

    elapsed = time.monotonic() - watch_start
    print(f"\r  Elapsed: {_fmt_duration(elapsed)}  ✓ Training complete!      ")
    return target_dir, elapsed


def _read_global_metrics(results_json: Path) -> dict[str, float]:
    """Pull the GLOBAL row from a benchmark results JSON."""
    with results_json.open(encoding="utf-8") as fh:
        rows = json.load(fh)
    for row in rows:
        if str(row.get("id", "")).upper() == "GLOBAL":
            return {
                "F1":        round(float(row.get("f1_score",  row.get("F1_score",  0.0))), 4),
                "Accuracy":  round(float(row.get("accuracy",  row.get("Accuracy",  0.0))), 4),
                "Precision": round(float(row.get("precision", row.get("Precision", 0.0))), 4),
                "Recall":    round(float(row.get("recall",    row.get("Recall",    0.0))), 4),
                "MCC":       round(float(row.get("mcc",       row.get("MCC",       0.0))), 4),
                "SOV":       round(float(row.get("sov",       row.get("SOV",       0.0))), 4),
            }
    return {}


def _print_comparison(rows: list[dict]) -> None:
    if not rows:
        return
    cols = ["variant", "mode", "F1", "Accuracy", "Precision", "Recall", "MCC", "SOV"]
    col_w = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(col_w[c]) for c in cols)
    sep    = "  ".join("-" * col_w[c] for c in cols)
    print("\n" + sep)
    print(header)
    print(sep)
    prev_variant = None
    for r in rows:
        if prev_variant and r["variant"] != prev_variant:
            print()
        print("  ".join(str(r.get(c, "")).ljust(col_w[c]) for c in cols))
        prev_variant = r["variant"]
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark all ensemble variants from a training run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--watch", action="store_true",
        help=(
            "Watch mode: block until training finishes, then benchmark automatically. "
            "Measures and records total training runtime. "
            "Run this in a second terminal before or during training."
        ),
    )
    parser.add_argument(
        "--poll", type=int, default=10, metavar="SEC",
        help="Polling interval in seconds for --watch mode (default: 10).",
    )
    parser.add_argument(
        "--run-dir", metavar="DIR",
        help=(
            "Training run directory (e.g. outputs/training/run_20260506_201459). "
            "In --watch mode, watches this specific dir for completion. "
            "Otherwise auto-detects the latest finished run."
        ),
    )
    parser.add_argument(
        "--input", "-i",
        default=str(_DEFAULT_BENCHMARK),
        metavar="JSON",
        help=f"Benchmark dataset JSON (default: {_DEFAULT_BENCHMARK.name}).",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(_DEFAULT_BENCH_OUT),
        metavar="DIR",
        help=f"Root output directory (default: {_DEFAULT_BENCH_OUT}).",
    )
    parser.add_argument(
        "--config", metavar="JSON",
        help="Pipeline config JSON (default: config/default_config.json).",
    )
    parser.add_argument(
        "--features", nargs="+", metavar="SPEC", default=None,
        help=(
            'Feature spec(s): "builtin" or path to a JSON spec. '
            "Must match the feature set used during training. "
            "Auto-detected from model metadata.json when omitted."
        ),
    )
    parser.add_argument(
        "--variants", nargs="+",
        choices=["soft", "weighted", "best"], default=None,
        metavar="V",
        help="Which ensemble variants to benchmark (default: all three).",
    )
    parser.add_argument(
        "--modes", nargs="+",
        choices=["padded", "adaptive"], default=None,
        metavar="M",
        help="Which benchmark modes to run (default: both).",
    )
    # Padded benchmark options
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: 0.5).")
    parser.add_argument("--window-size", type=int, default=None, dest="window_size",
                        help="Sliding window size (default: from config).")
    # Adaptive benchmark options
    parser.add_argument("--min-window",        type=int,   default=None, dest="min_window")
    parser.add_argument("--max-window",        type=int,   default=None, dest="max_window")
    parser.add_argument("--confidence-margin", type=float, default=None, dest="confidence_margin",
                        help="Adaptive confidence margin (default: 0.15).")

    args = parser.parse_args()

    config = load_config(args.config)

    train_out   = _ROOT / "outputs" / "training"
    training_elapsed: float | None = None
    benchmark_start = time.monotonic()

    # ── Locate / wait for run directory ──────────────────────────────────────
    if args.watch:
        known_dir = Path(args.run_dir) if args.run_dir else None
        if known_dir and not known_dir.exists():
            # Directory doesn't exist yet — that's fine, we'll wait.
            known_dir.mkdir(parents=True, exist_ok=True)
        run_dir, training_elapsed = _watch_for_completion(
            train_out, known_dir, poll_interval=args.poll
        )
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_dir():
            parser.error(f"--run-dir does not exist: {run_dir}")
    else:
        print(f"No --run-dir given. Searching for latest run in {train_out} …")
        run_dir = _find_latest_run(train_out)

    print(f"\nTraining run : {run_dir}")
    if training_elapsed is not None:
        print(f"Training time: {_fmt_duration(training_elapsed)}")

    # ── Verify benchmark input ────────────────────────────────────────────────
    bench_input = Path(args.input)
    if not bench_input.exists():
        parser.error(f"Benchmark input not found: {bench_input}")
    print(f"Benchmark dataset: {bench_input}")

    # ── Decide which variants / modes to run ──────────────────────────────────
    requested_variants = set(args.variants) if args.variants else {v for v, _ in _VARIANTS}
    requested_modes    = set(args.modes)    if args.modes    else set(_MODES)

    variants_to_run = [(name, pkl) for name, pkl in _VARIANTS if name in requested_variants]
    modes_to_run    = [m for m in _MODES if m in requested_modes]

    # ── Output root for this suite run ────────────────────────────────────────
    suite_out = Path(args.output) / run_dir.name
    suite_out.mkdir(parents=True, exist_ok=True)

    # ── Run benchmarks ────────────────────────────────────────────────────────
    comparison_rows: list[dict] = []
    missing: list[str] = []

    for variant_name, pkl_name in variants_to_run:
        model_path = run_dir / pkl_name
        if not model_path.exists():
            print(f"  [skip] {pkl_name} not found in {run_dir}")
            missing.append(variant_name)
            continue

        for mode in modes_to_run:
            tag        = f"{variant_name}_{mode}"
            out_subdir = suite_out / tag
            out_subdir.mkdir(parents=True, exist_ok=True)
            output_name = str(out_subdir / "results")

            print(f"\n{'─'*60}")
            print(f"  Variant : {variant_name}  |  Mode : {mode}")
            print(f"  Model   : {model_path}")
            print(f"  Output  : {out_subdir}")
            print(f"{'─'*60}")

            try:
                if mode == "padded":
                    artifacts = run_benchmark(
                        config,
                        model_path=str(model_path),
                        input_json=str(bench_input),
                        output_path=str(out_subdir),
                        output_name=output_name,
                        cli_feature_paths=args.features,
                    )
                else:  # adaptive
                    artifacts = run_benchmark_adaptive(
                        config,
                        model_path=str(model_path),
                        input_json=str(bench_input),
                        output_path=str(out_subdir),
                        output_name=output_name,
                        threshold=args.threshold,
                        default_window=args.window_size,
                        min_window=args.min_window,
                        max_window=args.max_window,
                        confidence_margin=args.confidence_margin,
                        cli_feature_paths=args.features,
                    )
            except Exception as exc:
                print(f"  ERROR during {tag}: {exc}")
                comparison_rows.append({
                    "variant": variant_name, "mode": mode,
                    "F1": "ERROR", "Accuracy": "", "Precision": "",
                    "Recall": "", "MCC": "", "SOV": "",
                })
                continue

            # Pull global metrics from the JSON results file
            results_json = artifacts.get("results_json")
            if results_json and Path(results_json).exists():
                metrics = _read_global_metrics(Path(results_json))
            else:
                metrics = {}

            comparison_rows.append({
                "variant":   variant_name,
                "mode":      mode,
                **metrics,
            })

    # ── Timing ────────────────────────────────────────────────────────────────
    benchmark_elapsed = time.monotonic() - benchmark_start
    total_elapsed     = (training_elapsed or 0.0) + benchmark_elapsed

    # ── Summary comparison ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Run:              {run_dir.name}")
    print(f"Dataset:          {bench_input.name}")
    if training_elapsed is not None:
        print(f"Training time:    {_fmt_duration(training_elapsed)}")
    print(f"Benchmark time:   {_fmt_duration(benchmark_elapsed)}")
    if training_elapsed is not None:
        print(f"Total time:       {_fmt_duration(total_elapsed)}")

    _print_comparison(comparison_rows)

    # Save summary CSV (with runtime columns)
    summary_csv = suite_out / "summary_comparison.csv"
    if comparison_rows:
        fieldnames = list(comparison_rows[0].keys())
        with summary_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)
        print(f"\nSummary saved → {summary_csv}")

    # Save runtime log
    runtime_log = suite_out / "runtime.json"
    runtime_data: dict = {
        "run_id":            run_dir.name,
        "benchmark_dataset": str(bench_input),
        "finished_at":       datetime.now().isoformat(timespec="seconds"),
        "benchmark_time_s":  round(benchmark_elapsed, 1),
        "benchmark_time":    _fmt_duration(benchmark_elapsed),
    }
    if training_elapsed is not None:
        runtime_data["training_time_s"] = round(training_elapsed, 1)
        runtime_data["training_time"]   = _fmt_duration(training_elapsed)
        runtime_data["total_time_s"]    = round(total_elapsed, 1)
        runtime_data["total_time"]      = _fmt_duration(total_elapsed)
    runtime_log.write_text(json.dumps(runtime_data, indent=2))
    print(f"Runtime log   → {runtime_log}")

    if missing:
        print(f"\nSkipped (pkl not found): {', '.join(missing)}")

    print(f"\nAll outputs under: {suite_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
