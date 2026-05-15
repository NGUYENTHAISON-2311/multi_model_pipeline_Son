"""Adaptive multi-scale sliding-window benchmark.

Instead of a single fixed window size, this benchmark:
1. Slides the window (start moves +1 each step) with the default size.
2. If the model is *confident* (|score − 0.5| ≥ margin), keeps that score.
3. If *uncertain*, keeps the start position fixed and tries extending or
   shortening the **end** of the window (min_size..max_size).  Selects the
   window length where the model is most confident — which may be
   confidently *negative* or *positive* (two-directional, fair).
4. Each residue may be covered by a different number of overlapping windows
   (variable-size), so scores are normalised per residue by overlap count.

IUPred disorder scores are precomputed **once** for the maximally-padded
sequence and reused across all window sizes, eliminating the main bottleneck.
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
from .feature_loader import compute_extra_for_window, load_and_prepare_feature_specs, resolve_feature_paths_for_model
from .feature_pipeline import (
    _load_iupred_library,
    compute_amino_acid_frequencies,
    compute_cross_classification_transitions,
    compute_dipeptide_aa_transitions,
    compute_group_frequencies,
    encode_sequence,
    extract_window_features,
    extract_window_features_v2,
    load_json_records,
)
from .defaults import AMINO_ACIDS, CLASSIFICATIONS, GROUP_LABELS
from .padding_sequences import build_padding
from .training_pipeline import load_model


# ── IUPred precompute ────────────────────────────────────────────────────────

def precompute_iupred_scores(
    sequence: str,
    iupred_script: str | Path,
    input_type: str = "long",
) -> list[float]:
    """Run IUPred once on a full sequence, return per-residue scores."""
    lib = _load_iupred_library(iupred_script)
    scores, _ = lib.iupred(sequence, input_type, smoothing=False)
    return list(scores)


def _mean_iupred_from_cache(
    per_residue_scores: list[float],
    start: int,
    end: int,
) -> float:
    """Mean IUPred over a slice, using precomputed per-residue scores."""
    seg = per_residue_scores[start:end]
    return sum(seg) / len(seg) if seg else 0.0


# ── Feature extraction with cached IUPred ────────────────────────────────────

def _extract_window_features_cached(
    window: str,
    iupred_mean: float,
    use_v2: bool,
) -> list[float]:
    """Build feature vector without calling IUPred (score provided)."""
    row: list[float] = []

    if use_v2:
        # v2: AA freq(20) + group freq(17) + intra-class(101) + dipeptide(400) + iupred(1)
        row.extend(compute_amino_acid_frequencies(window, AMINO_ACIDS))
        encoded = [encode_sequence(window, cls) for cls in CLASSIFICATIONS]
        for enc, groups in zip(encoded, GROUP_LABELS):
            row.extend(compute_group_frequencies(enc, groups))
        for ci in range(len(CLASSIFICATIONS)):
            pairs = [(a, b) for a in GROUP_LABELS[ci] for b in GROUP_LABELS[ci]]
            row.extend(compute_cross_classification_transitions(
                window, (encoded[ci], encoded[ci]), pairs,
            ))
        row.extend(compute_dipeptide_aa_transitions(window, AMINO_ACIDS))
    else:
        # v1: length(1) + AA freq(20) + dipeptide(400) + group freq(17) + intra-class(101)
        row.append(float(len(window)))
        row.extend(compute_amino_acid_frequencies(window, AMINO_ACIDS))
        row.extend(compute_dipeptide_aa_transitions(window, AMINO_ACIDS))
        encoded = [encode_sequence(window, cls) for cls in CLASSIFICATIONS]
        for enc, groups in zip(encoded, GROUP_LABELS):
            row.extend(compute_group_frequencies(enc, groups))
        for ci in range(len(CLASSIFICATIONS)):
            pairs = [(a, b) for a in GROUP_LABELS[ci] for b in GROUP_LABELS[ci]]
            row.extend(compute_cross_classification_transitions(
                window, (encoded[ci], encoded[ci]), pairs,
            ))

    row.append(iupred_mean)
    return row


# ── Padding for adaptive ─────────────────────────────────────────────────────

def pad_sequence_adaptive(
    sequence: str,
    default_window: int,
    max_window: int,
) -> tuple[str, int, int]:
    """Pad for adaptive benchmark.

    Left pad  = default_window - 1  (start positions same as default scheme)
    Right pad = max_window - 1      (end can extend up to max_window)

    Returns (padded_sequence, left_pad, right_pad).
    """
    left_pad = default_window - 1
    right_pad = max_window - 1
    left = build_padding(sequence[0], left_pad, side="left")
    right = build_padding(sequence[-1], right_pad, side="right")
    return left + sequence + right, left_pad, right_pad


# ── Core: analyse one sequence adaptively ────────────────────────────────────

def analyse_adaptive_sequence(
    model,
    sequence: str,
    iupred_script: str | Path,
    default_window: int = 18,
    min_window: int = 11,
    max_window: int = 25,
    confidence_margin: float = 0.15,
    input_type: str = "long",
    feature_specs: list[dict] | None = None,
) -> tuple[list[float], list[dict]]:
    """Score every residue using adaptive window sizing.

    For each start position (sliding by 1, same as padded benchmark):
      1. Score with default_window.
      2. If confident → keep that score + window size.
      3. If uncertain → keep start fixed, try all sizes from min_window to
         max_window (only the end position changes).  Pick the size where
         the model is most confident (|score − 0.5| largest).
      4. Accumulate the chosen score for each residue covered by the window.

    Each residue's final score = mean of all overlapping window scores
    (windows may have different sizes → different overlap counts per residue).

    Returns
    -------
    average_scores : per-residue scores (len = len(sequence))
    window_records : list of every window evaluated (for diagnostics)
    """
    if feature_specs is None:
        feature_specs = []
    original_length = len(sequence)
    padded, left_pad, right_pad = pad_sequence_adaptive(sequence, default_window, max_window)
    padded_length = len(padded)

    # Precompute IUPred once for the full padded sequence
    iupred_per_residue = precompute_iupred_scores(padded, iupred_script, input_type)

    _n_features = getattr(model, "n_features_in_", None)
    use_v2 = _n_features == 539
    has_builtin = any(s.get("_builtin") for s in feature_specs)
    extra_specs = [s for s in feature_specs if not s.get("_builtin")]

    # Accumulators per original residue
    score_accum = np.zeros(original_length, dtype=np.float64)
    count_accum = np.zeros(original_length, dtype=np.float64)
    max_accum = np.full(original_length, -np.inf, dtype=np.float64)
    window_records: list[dict] = []
    window_counter = 0

    def _score_window(p_start: int, wsize: int) -> float:
        window_seq = padded[p_start : p_start + wsize]
        iup = _mean_iupred_from_cache(iupred_per_residue, p_start, p_start + wsize)
        row = _extract_window_features_cached(window_seq, iup, use_v2) if has_builtin else []
        row = row + compute_extra_for_window(window_seq, extra_specs)
        feats = np.array(row).reshape(1, -1)
        return float(model.predict_proba(feats)[0][1])

    def _accumulate(p_start: int, wsize: int, score: float) -> None:
        """Add score to every original residue covered by this window."""
        orig_left = max(0, p_start - left_pad)
        orig_right = min(original_length - 1, p_start - left_pad + wsize - 1)
        for ri in range(orig_left, orig_right + 1):
            score_accum[ri] += score
            count_accum[ri] += 1
            if score > max_accum[ri]:
                max_accum[ri] = score

    # Number of start positions = same as default padded benchmark
    # First start in padded coords: 0  (left_pad = default_window - 1)
    # Last start: left_pad + original_length - 1
    # Total: original_length + default_window - 1
    n_positions = original_length + default_window - 1

    other_sizes = [s for s in range(min_window, max_window + 1) if s != default_window]

    for wi in range(n_positions):
        p_start = wi  # padded start index

        # Bounds check for default window
        if p_start + default_window > padded_length:
            continue

        # Phase 1: score with default window size
        default_score = _score_window(p_start, default_window)
        default_conf = abs(default_score - 0.5)

        if default_conf >= confidence_margin:
            # Confident → use default score directly
            window_counter += 1
            window_records.append({
                "number": window_counter,
                "score": default_score,
                "length": default_window,
                "start": p_start - left_pad,
                "end": p_start - left_pad + default_window - 1,
                "sequence": padded[p_start : p_start + default_window],
                "phase": "default",
            })
            _accumulate(p_start, default_window, default_score)
            continue

        # Phase 2: uncertain → try other sizes (same start, different end)
        best_score = default_score
        best_conf = default_conf
        best_size = default_window

        for wsize in other_sizes:
            if p_start + wsize > padded_length:
                continue

            score = _score_window(p_start, wsize)
            conf = abs(score - 0.5)

            if conf > best_conf:
                best_conf = conf
                best_score = score
                best_size = wsize

        # Record only the selected window
        window_counter += 1
        window_records.append({
            "number": window_counter,
            "score": best_score,
            "length": best_size,
            "start": p_start - left_pad,
            "end": p_start - left_pad + best_size - 1,
            "sequence": padded[p_start : p_start + best_size],
            "phase": "adaptive" if best_size != default_window else "default",
        })

        # Accumulate the most-confident score with its actual window size
        _accumulate(p_start, best_size, best_score)

    # Per-residue average and max
    average_scores: list[float] = []
    max_scores: list[float] = []
    for i in range(original_length):
        if count_accum[i] > 0:
            average_scores.append(float(score_accum[i] / count_accum[i]))
            max_scores.append(float(max_accum[i]))
        else:
            average_scores.append(0.5)  # fallback: no coverage → neutral
            max_scores.append(0.5)

    return average_scores, max_scores, window_records


# ── Full benchmark driver ────────────────────────────────────────────────────

def run_benchmark_adaptive(
    config: dict,
    model_path: str | Path | None = None,
    input_json: str | Path | None = None,
    output_path: str | Path | None = None,
    threshold: float | None = None,
    default_window: int | None = None,
    min_window: int | None = None,
    max_window: int | None = None,
    confidence_margin: float | None = None,
    output_name: str = "benchmark_adaptive_results",
    cli_feature_paths: list[str] | None = None,
    classifier_mode: bool = False,
    positive_label: str | None = None,
) -> dict[str, Path]:
    """Run the adaptive multi-scale padded benchmark."""
    bench_cfg = config["benchmark"]
    train_cfg = config["training"]
    adaptive_cfg = bench_cfg.get("adaptive", {})

    # Resolve parameters: CLI arg > config.benchmark.adaptive > defaults
    _default_window = default_window or adaptive_cfg.get("default_window", bench_cfg.get("window_size", 18))
    _min_window = min_window or adaptive_cfg.get("min_window", 11)
    _max_window = max_window or adaptive_cfg.get("max_window", 25)
    _confidence = confidence_margin if confidence_margin is not None else adaptive_cfg.get("confidence_margin", 0.15)
    k = threshold if threshold is not None else bench_cfg.get("threshold", 0.5)

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

    # Locate model
    if model_path:
        resolved = Path(model_path)
    else:
        train_out = resolve_pipeline_path(config, train_cfg["output_dir"])
        runs = sorted(train_out.glob("run_*"), reverse=True)
        resolved = None
        for run_dir in runs:
            candidate = run_dir / "ensemble.pkl"
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            raise FileNotFoundError(
                f"No ensemble.pkl found in {train_out}. Train first or specify --model."
            )

    tqdm.write(f"Loading model from: {resolved}")
    model = load_model(resolved)
    tqdm.write(
        f"Adaptive benchmark: default_window={_default_window}, "
        f"range=[{_min_window}, {_max_window}], "
        f"confidence_margin={_confidence}, threshold={k}"
    )

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

    for record in tqdm(bench_records, desc="Benchmark (adaptive)", unit="seq"):
        seq_id = record.get("ID", "unknown")
        sequence = record["Sequence"]

        avg_scores, max_scores, win_records = analyse_adaptive_sequence(
            model, sequence,
            iupred_script=iupred_script,
            default_window=_default_window,
            min_window=_min_window,
            max_window=_max_window,
            confidence_margin=_confidence,
            input_type=train_cfg["iupred_input_type"],
            feature_specs=feature_specs,
        )

        agg_method = bench_cfg.get("aggregation_method", "mean")
        vote_frac = bench_cfg.get("vote_fraction", 0.5)
        if agg_method == "max":
            pred_labels = [1 if s >= k else 0 for s in max_scores]
        elif agg_method == "vote":
            # Count windows covering each residue with score >= k
            seq_len = len(sequence)
            vote_count = [0] * seq_len
            total_count = [0] * seq_len
            for wr in win_records:
                orig_start = max(0, wr["start"])
                orig_end = min(seq_len - 1, wr["end"])
                for ri in range(orig_start, orig_end + 1):
                    total_count[ri] += 1
                    if wr["score"] >= k:
                        vote_count[ri] += 1
            pred_labels = [
                1 if (total_count[i] > 0 and vote_count[i] / total_count[i] >= vote_frac) else 0
                for i in range(seq_len)
            ]
        else:  # "mean" (default)
            pred_labels = [1 if s >= k else 0 for s in avg_scores]

        residue_scores = avg_scores if agg_method != "max" else max_scores

        win_out = {key: val for key, val in record.items() if key != "matched_core_regions"}
        win_out["windows"] = win_records
        win_out["residue_scores"] = residue_scores
        all_window_records.append(win_out)

        if agg_method == "vote":
            score_residues = [
                vote_count[i] / total_count[i] if total_count[i] > 0 else 0.0
                for i in range(len(sequence))
            ]
        else:
            score_residues = residue_scores
        prediction_rows.append({"ID": seq_id, "Sequence": sequence, "Score_residues": score_residues})
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
    with win_path.open("w", encoding="utf-8") as fh:
        json.dump(all_window_records, fh, indent=2)

    scores_path = output_stem.parent / (output_stem.name + "_scores.csv")

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
