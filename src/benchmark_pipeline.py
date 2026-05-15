"""Sliding-window benchmark pipeline for ensemble models.

Reuses scoring / SOV helpers from the original pipeline.  The only structural
difference is the model-loading path which now also accepts
:class:`~.ensemble.EnsembleClassifier` objects.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

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

from .configuration import ensure_directory, resolve_pipeline_path
from .feature_loader import compute_window_feature_vector_live, load_and_prepare_feature_specs, resolve_feature_paths_for_model
from .feature_pipeline import (
    load_json_records,
)
from .training_pipeline import load_model


# ── Label / SOV helpers (identical to the original pipeline) ─────────────────

def create_true_labels(sequence_length: int, core_regions: Iterable[dict]) -> list[int]:
    labels = [0] * sequence_length
    for region in core_regions:
        start = int(region["start"])
        end = int(region["end"])
        for index in range(start, end + 1):
            if 0 <= index < sequence_length:
                labels[index] = 1
    return labels


def get_segments(labels: list[int], target: int = 1) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, label in enumerate(labels):
        if label == target:
            if start is None:
                start = index
        elif start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(labels) - 1))
    return segments


def segment_overlap(observed: tuple[int, int], predicted: tuple[int, int]) -> int:
    start = max(observed[0], predicted[0])
    end = min(observed[1], predicted[1])
    return max(0, end - start + 1)


def sov_score(true_labels: list[int], predicted_labels: list[int], target: int = 1) -> float:
    true_segments = get_segments(true_labels, target)
    predicted_segments = get_segments(predicted_labels, target)
    denominator = sum(end - start + 1 for start, end in true_segments)
    numerator = 0.0
    for obs_s, obs_e in true_segments:
        max_sov = 0.0
        for pred_s, pred_e in predicted_segments:
            if pred_e < obs_s or pred_s > obs_e:
                continue
            ov = segment_overlap((obs_s, obs_e), (pred_s, pred_e))
            un = max(obs_e, pred_e) - min(obs_s, pred_s) + 1
            delta = min(un - ov, ov, (obs_e - obs_s + 1) // 2, (pred_e - pred_s + 1) // 2)
            sov = ((ov + delta) / un) * (obs_e - obs_s + 1)
            max_sov = max(max_sov, sov)
        numerator += max_sov
    return (numerator / denominator) * 100 if denominator > 0 else 0.0


# ── Classifier (sequence-level) helpers ──────────────────────────────────────

def classifier_sequence_result(
    seq_id: str,
    sequence: str,
    raw_label: str,
    pred_labels: list[int],
    positive_label: str,
) -> dict:
    """Per-sequence result dict for classifier (sequence-level) mode.

    A sequence is predicted positive if at least one residue is labelled as
    core (any element of *pred_labels* equals 1).  Ground truth comes from the
    ``LABEL`` field in the input record, compared case-insensitively against
    *positive_label*.
    """
    true_seq = 1 if raw_label.strip().lower() == positive_label.strip().lower() else 0
    pred_seq = 1 if any(pred_labels) else 0
    tp = 1 if true_seq == 1 and pred_seq == 1 else 0
    tn = 1 if true_seq == 0 and pred_seq == 0 else 0
    fp = 1 if true_seq == 0 and pred_seq == 1 else 0
    fn = 1 if true_seq == 1 and pred_seq == 0 else 0
    return {
        "id": seq_id,
        "sequence_length": len(sequence),
        "true_label": raw_label,
        "true_seq": true_seq,
        "pred_seq": pred_seq,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def classifier_global_row(per_seq_results: list[dict]) -> dict:
    """Compute global classification metrics from per-sequence classifier results."""
    tp = sum(r["tp"] for r in per_seq_results)
    tn = sum(r["tn"] for r in per_seq_results)
    fp = sum(r["fp"] for r in per_seq_results)
    fn = sum(r["fn"] for r in per_seq_results)
    n = len(per_seq_results)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return {
        "id": "GLOBAL",
        "sequence_length": sum(r["sequence_length"] for r in per_seq_results) / n if n else 0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "mcc": mcc,
    }


def is_prediction_only_dataset(records: list[dict]) -> bool:
    """Return True if every record lacks both a non-empty LABEL and non-empty
    matched_core_regions — meaning there is no ground truth to evaluate against."""
    for r in records:
        if r.get("LABEL", "").strip():
            return False
        if r.get("matched_core_regions"):
            return False
    return len(records) > 0


# ── Window analysis ─────────────────────────────────────────────────────────


def analyse_long_sequence_with_prediction(
    model,
    long_sequence: str,
    iupred_script: str | Path,
    window_size: int,
    input_type: str = "long",
    sequence_id: str = "",
    feature_specs: list[dict] | None = None,
) -> tuple[dict[int, list[float]], list[dict]]:
    if feature_specs is None:
        feature_specs = []
    seq_len = len(long_sequence)
    scores_per_residue: dict[int, list[float]] = {i: [] for i in range(seq_len)}
    window_records: list[dict] = []
    num_windows = seq_len - window_size + 1
    use_v2 = getattr(model, "n_features_in_", None) == 539

    for win_idx, start in enumerate(range(num_windows), start=1):
        window = long_sequence[start : start + window_size]
        raw = compute_window_feature_vector_live(
            window, feature_specs, iupred_script,
            input_type=input_type, use_v2=use_v2,
        )
        features = np.array(raw).reshape(1, -1)
        proba = model.predict_proba(features)
        score = float(proba[0][1])
        for offset in range(window_size):
            scores_per_residue[start + offset].append(score)
        window_records.append({
            "number": win_idx,
            "score": score,
            "length": window_size,
            "start": start,
            "end": start + window_size - 1,
            "sequence": window,
        })
    return scores_per_residue, window_records


def calculate_prediction_stats(
    scores_per_residue: dict[int, list[float]],
    threshold: float = 0.51,
) -> dict[int, dict]:
    stats: dict[int, dict] = {}
    for idx, scores in scores_per_residue.items():
        if scores:
            vf = sum(1 for s in scores if s >= threshold) / len(scores)
        else:
            vf = None
        stats[idx] = {
            "max": max(scores) if scores else None,
            "min": min(scores) if scores else None,
            "mean": (sum(scores) / len(scores)) if scores else None,
            "vote_fraction": vf,
        }
    return stats


def get_predicted_labels(
    stats_per_residue: dict[int, dict],
    threshold: float,
    aggregation_method: str = "max",
    vote_fraction: float = 0.5,
) -> list[int]:
    result = []
    for idx in sorted(stats_per_residue.keys()):
        st = stats_per_residue[idx]
        if aggregation_method == "vote":
            vf = st.get("vote_fraction")
            label = 1 if vf is not None and vf >= vote_fraction else 0
        elif aggregation_method == "mean":
            avg = st.get("mean")
            label = 1 if avg is not None and avg >= threshold else 0
        else:  # "max" (default)
            label = 1 if st["max"] is not None and st["max"] >= threshold else 0
        result.append(label)
    return result


# ── Full benchmark driver ───────────────────────────────────────────────────

def run_benchmark(
    config: dict,
    model_path: str | Path | None = None,
    input_json: str | Path | None = None,
    output_path: str | Path | None = None,
    output_name: str = "benchmark_results",
    cli_feature_paths: list[str] | None = None,
    classifier_mode: bool = False,
    positive_label: str | None = None,
) -> dict[str, Path]:
    bench_cfg = config["benchmark"]
    train_cfg = config["training"]

    bench_json = Path(input_json) if input_json else resolve_pipeline_path(config, bench_cfg["input_json"])
    iupred_script = resolve_pipeline_path(config, train_cfg["iupred_script"])

    if output_path:
        out_dir = Path(output_path)
    else:
        out_dir = resolve_pipeline_path(config, bench_cfg["output_dir"])
    _name_path = Path(output_name)
    if _name_path.parent != Path("."):
        output_stem = _name_path
    else:
        ensure_directory(out_dir)
        output_stem = out_dir / output_name
    ensure_directory(output_stem.parent)

    if model_path:
        resolved = Path(model_path)
    else:
        # Look for latest ensemble.pkl in the training output
        out_dir = resolve_pipeline_path(config, train_cfg["output_dir"])
        runs = sorted(out_dir.glob("run_*"), reverse=True)
        resolved = None
        for run_dir in runs:
            candidate = run_dir / "ensemble.pkl"
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            raise FileNotFoundError(
                f"No ensemble.pkl found in {out_dir}. Train first or specify --model."
            )

    tqdm.write(f"Loading model from: {resolved}")
    model = load_model(resolved)

    # ── Feature specs ─────────────────────────────────────────────────────────
    feature_specs = load_and_prepare_feature_specs(
        config, resolve_feature_paths_for_model(resolved, cli_feature_paths)
    )
    tqdm.write(f"[Features] {['builtin' if s.get('_builtin') else s.get('_source_path', '?') for s in feature_specs]}")

    bench_records = load_json_records(bench_json)
    per_seq_results: list[dict] = []
    all_window_records: list[dict] = []
    prediction_rows: list[dict] = []

    _pos_label = positive_label or bench_cfg.get("classifier_positive_label", "AMYLOID")
    prediction_only = is_prediction_only_dataset(bench_records)
    if prediction_only:
        tqdm.write("[Prediction-only mode] No LABEL or matched_core_regions found — skipping metric computation.")
    elif classifier_mode:
        tqdm.write(f"[Classifier mode] positive_label='{_pos_label}' (sequence-level evaluation)")

    for record in tqdm(bench_records, desc="Benchmarking sequences", unit="seq"):
        seq_id = record.get("ID", "unknown")
        sequence = record["Sequence"]
        scores_per_residue, window_records = analyse_long_sequence_with_prediction(
            model, sequence, iupred_script=iupred_script,
            window_size=bench_cfg["window_size"],
            input_type=train_cfg["iupred_input_type"],
            sequence_id=seq_id,
            feature_specs=feature_specs,
        )
        win_out = {k: v for k, v in record.items() if k != "matched_core_regions"}
        win_out["windows"] = window_records
        all_window_records.append(win_out)
        stats = calculate_prediction_stats(scores_per_residue, bench_cfg["threshold"])
        pred_labels = get_predicted_labels(
            stats, bench_cfg["threshold"],
            aggregation_method=bench_cfg.get("aggregation_method", "max"),
            vote_fraction=bench_cfg.get("vote_fraction", 0.5),
        )
        agg = bench_cfg.get("aggregation_method", "max")
        if agg == "mean":
            residue_scores = [stats[i]["mean"] or 0.0 for i in range(len(sequence))]
        elif agg == "vote":
            residue_scores = [stats[i]["vote_fraction"] or 0.0 for i in range(len(sequence))]
        else:  # max
            residue_scores = [stats[i]["max"] or 0.0 for i in range(len(sequence))]
        prediction_rows.append({"ID": seq_id, "Sequence": sequence, "Score_residues": residue_scores})
        if not prediction_only:
            if classifier_mode:
                per_seq_results.append(classifier_sequence_result(
                    seq_id, sequence, record.get("LABEL", ""), pred_labels, _pos_label,
                ))
            else:
                true_labels = create_true_labels(len(sequence), record.get("matched_core_regions", []))
                tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels, labels=[0, 1]).ravel()
                per_seq_results.append({
                    "id": seq_id,
                    "sequence_length": len(sequence),
                    "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
                    "precision": float(precision_score(true_labels, pred_labels, zero_division=0)),
                    "recall": float(recall_score(true_labels, pred_labels, zero_division=0)),
                    "f1_score": float(f1_score(true_labels, pred_labels, zero_division=0)),
                    "accuracy": float(accuracy_score(true_labels, pred_labels)),
                    "mcc": float(matthews_corrcoef(true_labels, pred_labels)),
                    "sov": float(sov_score(true_labels, pred_labels)),
                })

    csv_path = output_stem.with_suffix(".csv")
    json_path = output_stem.with_suffix(".json")
    win_path = output_stem.parent / (output_stem.name + "_windows.json")
    scores_path = output_stem.parent / (output_stem.name + "_scores.csv")
    with win_path.open("w", encoding="utf-8") as fh:
        json.dump(all_window_records, fh, indent=2)

    if prediction_only:
        pd.DataFrame(prediction_rows).to_csv(scores_path, index=False)
        return {"predictions_csv": scores_path, "windows_json": win_path}

    pd.DataFrame(prediction_rows).to_csv(scores_path, index=False)

    # Global row
    if classifier_mode:
        global_row = classifier_global_row(per_seq_results)
    else:
        _metrics = ["precision", "recall", "f1_score", "accuracy", "mcc", "sov"]
        n = len(per_seq_results)
        global_row = {
            "id": "GLOBAL",
            "sequence_length": sum(r["sequence_length"] for r in per_seq_results) / n if n else 0,
            "tp": sum(r["tp"] for r in per_seq_results),
            "tn": sum(r["tn"] for r in per_seq_results),
            "fp": sum(r["fp"] for r in per_seq_results),
            "fn": sum(r["fn"] for r in per_seq_results),
        }
        for col in _metrics:
            global_row[col] = sum(r[col] for r in per_seq_results) / n if n else 0.0
    all_rows = per_seq_results + [global_row]

    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(all_rows, fh, indent=2)

    return {"results_csv": csv_path, "results_json": json_path, "scores_csv": scores_path, "windows_json": win_path}
