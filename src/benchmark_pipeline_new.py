"""Padded sliding-window benchmark with average-score aggregation.

Every original residue is covered by exactly T windows (via padding).
The per-residue prediction is the average of those T window scores,
compared against threshold k.
"""

from __future__ import annotations

import json
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

from .benchmark_pipeline import classifier_global_row, classifier_sequence_result, create_true_labels, is_prediction_only_dataset, sov_score
from .configuration import ensure_directory, resolve_pipeline_path
from .feature_loader import compute_window_feature_vector_live, load_and_prepare_feature_specs, resolve_feature_paths_for_model
from .feature_pipeline import load_json_records
from .padding_sequences import build_padding
from .training_pipeline import load_model


# ── Padding helpers ──────────────────────────────────────────────────────────

def pad_sequence(sequence: str, window_size: int) -> tuple[str, int]:
    pad_length = window_size - 1
    left = build_padding(sequence[0], pad_length, side="left")
    right = build_padding(sequence[-1], pad_length, side="right")
    return left + sequence + right, pad_length


# ── Padded analysis ──────────────────────────────────────────────────────────

def analyse_padded_sequence(
    model,
    sequence: str,
    iupred_script: str | Path,
    window_size: int,
    input_type: str = "long",
    sequence_id: str = "",
    feature_specs: list[dict] | None = None,
) -> tuple[list[float], list[float], list[dict]]:
    """Return (average_scores, window_scores, window_records).

    *average_scores* — per-residue mean of overlapping window scores.
    *window_scores*  — raw score for each padded window position.
    *window_records* — diagnostic records for each window.
    """
    if feature_specs is None:
        feature_specs = []
    original_length = len(sequence)
    padded_sequence, pad_length = pad_sequence(sequence, window_size)

    use_v2 = getattr(model, "n_features_in_", None) == 539

    num_windows = len(padded_sequence) - window_size + 1
    window_scores: list[float] = []
    window_records: list[dict] = []

    for start_index in range(num_windows):
        window = padded_sequence[start_index : start_index + window_size]
        raw = compute_window_feature_vector_live(
            window, feature_specs, iupred_script,
            input_type=input_type, use_v2=use_v2,
        )
        features = np.array(raw).reshape(1, -1)
        proba = model.predict_proba(features)
        score = float(proba[0][1])
        window_scores.append(score)
        window_records.append({
            "number": start_index + 1,
            "score": score,
            "length": window_size,
            "start": start_index - pad_length,
            "end": start_index - pad_length + window_size - 1,
            "sequence": window,
        })

    average_scores: list[float] = []
    for i in range(original_length):
        relevant = window_scores[i : i + window_size]
        average_scores.append(sum(relevant) / len(relevant))

    return average_scores, window_scores, window_records


# ── Full benchmark driver ────────────────────────────────────────────────────

def run_benchmark_new(
    config: dict,
    model_path: str | Path | None = None,
    input_json: str | Path | None = None,
    output_path: str | Path | None = None,
    threshold: float | None = None,
    window_size: int | None = None,
    output_name: str = "benchmark_new_results",
    cli_feature_paths: list[str] | None = None,
    classifier_mode: bool = False,
    positive_label: str | None = None,
) -> dict[str, Path]:
    bench_cfg = config["benchmark"]
    train_cfg = config["training"]

    bench_json = Path(input_json) if input_json else resolve_pipeline_path(config, bench_cfg["input_json"])
    iupred_script = resolve_pipeline_path(config, train_cfg["iupred_script"])
    k = threshold if threshold is not None else bench_cfg.get("threshold", 0.5)

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
    tqdm.write(f"Threshold k = {k}")

    feature_specs = load_and_prepare_feature_specs(
        config, resolve_feature_paths_for_model(resolved, cli_feature_paths)
    )
    tqdm.write(f"[Features] {['builtin' if s.get('_builtin') else s.get('_source_path', '?') for s in feature_specs]}")

    bench_records = load_json_records(bench_json)
    window_size = window_size if window_size is not None else bench_cfg["window_size"]

    per_seq_results: list[dict] = []
    all_window_records: list[dict] = []
    prediction_rows: list[dict] = []

    _pos_label = positive_label or bench_cfg.get("classifier_positive_label", "AMYLOID")
    prediction_only = is_prediction_only_dataset(bench_records)
    if prediction_only:
        tqdm.write("[Prediction-only mode] No LABEL or matched_core_regions found — skipping metric computation.")
    elif classifier_mode:
        tqdm.write(f"[Classifier mode] positive_label='{_pos_label}' (sequence-level evaluation)")

    for record in tqdm(bench_records, desc="Benchmark (padded avg)", unit="seq"):
        seq_id = record.get("ID", "unknown")
        sequence = record["Sequence"]
        avg_scores, window_scores, win_records = analyse_padded_sequence(
            model, sequence, iupred_script=iupred_script,
            window_size=window_size, input_type=train_cfg["iupred_input_type"],
            sequence_id=seq_id,
            feature_specs=feature_specs,
        )

        agg_method = bench_cfg.get("aggregation_method", "mean")
        vote_frac = bench_cfg.get("vote_fraction", 0.5)
        if agg_method == "max":
            pred_labels = [
                1 if max(window_scores[i : i + window_size]) >= k else 0
                for i in range(len(avg_scores))
            ]
        elif agg_method == "vote":
            pred_labels = [
                1 if sum(s >= k for s in window_scores[i : i + window_size]) / window_size >= vote_frac else 0
                for i in range(len(avg_scores))
            ]
        else:  # "mean" (default)
            pred_labels = [1 if s >= k else 0 for s in avg_scores]

        win_out = {key: val for key, val in record.items() if key != "matched_core_regions"}
        win_out["windows"] = win_records
        win_out["residue_avg_scores"] = avg_scores
        all_window_records.append(win_out)

        if agg_method == "max":
            residue_scores = [
                max(window_scores[i : i + window_size])
                for i in range(len(avg_scores))
            ]
        elif agg_method == "vote":
            residue_scores = [
                sum(s >= k for s in window_scores[i : i + window_size]) / window_size
                for i in range(len(avg_scores))
            ]
        else:
            residue_scores = avg_scores
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
