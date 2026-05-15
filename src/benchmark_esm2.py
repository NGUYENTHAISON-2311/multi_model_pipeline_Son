"""ESM2-powered sliding-window benchmark.

Feature extraction strategy
-----------------------------
For each window position the feature vector comes entirely from ESM2:

  full (padded) sequence
      → ESM2 → (L_pad, H) per-residue hidden states      [one forward pass]
      → Cython pool_windows_{mean|max}                    [C loop, O(N·H)]
      → (N_windows, H) pooled embeddings
      → PCA transform (if trained with PCA)               [numpy batch op]
      → optional aaindex lookup-table features per window [appended]
      → model.predict_proba(full_batch)                   [one sklearn call]
      → (N_windows,) scores
      → Cython accumulate_scores                          [C loop, O(N)]
      → (L,) per-residue mean scores
      → threshold → binary labels

No IUPred call is needed; the full feature matrix for a sequence is built
in two C passes and one batch model call.

The ESM2 model name, pooling strategy, and PCA reducer are loaded from
``<run_dir>/esm2_pca_reducer.pkl`` written by scripts/train_ensemble_esm2.py.

Requires
--------
    python setup_cython.py build_ext --inplace   # compile _esm2_bench_fast
"""

from __future__ import annotations

import json
import pickle
import sys
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

from .benchmark_pipeline import (
    classifier_global_row,
    classifier_sequence_result,
    create_true_labels,
    is_prediction_only_dataset,
    sov_score,
)
from .configuration import ensure_directory, resolve_pipeline_path
from .feature_loader import compute_extra_for_window, load_and_prepare_feature_specs, resolve_feature_paths_for_model
from .feature_pipeline import load_json_records
from .padding_sequences import build_padding
from .training_pipeline import load_model

# Try Cython extension; fall back to numpy if not compiled.
try:
    from ._esm2_bench_fast import (
        pool_windows_mean,
        pool_windows_max,
        accumulate_scores as _accumulate_scores_c,
    )
    _CYTHON = True
except ImportError:
    _CYTHON = False


# ── ESM2 session ──────────────────────────────────────────────────────────────

def load_esm2_session(
    run_dir: Path,
    model_override: str | None = None,
    pool_override: str | None = None,
) -> dict:
    """Load ESM2 tokenizer + model + PCA from <run_dir>/esm2_pca_reducer.pkl.

    Returns a dict with keys: tokenizer, model, device, pca, pool, model_name.
    """
    pkl_path = run_dir / "esm2_pca_reducer.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"esm2_pca_reducer.pkl not found in {run_dir}.\n"
            "This run was not trained with ESM2 features "
            "(use scripts/train_ensemble_esm2.py --features ... esm2)."
        )
    with pkl_path.open("rb") as fh:
        saved = pickle.load(fh)

    model_name = model_override or saved.get("esm2_model", "facebook/esm2_t12_35M_UR50D")
    pool       = pool_override  or saved.get("esm2_pool",  "mean")
    pca        = saved.get("pca")

    # esm2_embedding.py lives in scripts/
    _scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    from esm2_embedding import load_model as _load_esm2
    tokenizer, model, device = _load_esm2(model_name)

    tqdm.write(
        f"[ESM2] model={model_name}  pool={pool}"
        + (f"  pca={pca.n_components_} dims" if pca is not None else "  no PCA")
        + f"  device={device}"
    )
    return {"tokenizer": tokenizer, "model": model, "device": device,
            "pca": pca, "pool": pool, "model_name": model_name}


# ── Sequence padding ──────────────────────────────────────────────────────────

def _pad_sequence(sequence: str, window_size: int) -> tuple[str, int]:
    """Pad with window_size-1 residues on each side.

    Every original residue is then covered by exactly *window_size* windows,
    matching the guarantee of the standard padded benchmark.

    Returns (padded_sequence, left_pad_length).
    """
    pad_len = window_size - 1
    left    = build_padding(sequence[0],  pad_len, side="left")
    right   = build_padding(sequence[-1], pad_len, side="right")
    return left + sequence + right, pad_len


# ── ESM2 inference ────────────────────────────────────────────────────────────

def _embed_sequence(sequence: str, session: dict) -> np.ndarray:
    """Run one ESM2 forward pass; return (L, H) float32 per-residue hidden states.

    Strips [CLS] (pos 0) and [EOS] (pos L+1) so the array aligns with
    amino-acid positions.
    """
    import torch
    L      = len(sequence)
    inputs = session["tokenizer"](
        [sequence], return_tensors="pt", padding=True, truncation=True
    )
    inputs = {k: v.to(session["device"]) for k, v in inputs.items()}
    with torch.no_grad():
        hidden = session["model"](**inputs).last_hidden_state
    return hidden[0, 1 : L + 1, :].cpu().numpy().astype(np.float32)


# ── Pure-numpy fallbacks (used when Cython .so is absent) ────────────────────

def _pool_numpy(embs: np.ndarray, window_size: int, pool: str) -> np.ndarray:
    N = embs.shape[0] - window_size + 1
    if N <= 0:
        return np.zeros((0, embs.shape[1]), dtype=np.float32)
    # Use stride_tricks for a zero-copy view of windows
    shape   = (N, window_size, embs.shape[1])
    strides = (embs.strides[0], embs.strides[0], embs.strides[1])
    windows = np.lib.stride_tricks.as_strided(embs, shape=shape, strides=strides)
    if pool == "max":
        return windows.max(axis=1).astype(np.float32)
    return windows.mean(axis=1).astype(np.float32)


def _accumulate_scores_numpy(
    window_scores: np.ndarray,
    window_size: int,
    seq_len: int,
    left_pad: int,
) -> np.ndarray:
    score_sum = np.zeros(seq_len, dtype=np.float64)
    count     = np.zeros(seq_len, dtype=np.int32)
    for i, s in enumerate(window_scores):
        orig_start = max(0, i - left_pad)
        orig_end   = min(seq_len - 1, i - left_pad + window_size - 1)
        score_sum[orig_start : orig_end + 1] += s
        count[orig_start : orig_end + 1]     += 1
    return np.where(count > 0, score_sum / count, 0.5)


# ── Window feature matrix ─────────────────────────────────────────────────────

def _build_window_matrix(
    residue_embs: np.ndarray,
    window_size: int,
    pca,
    pool: str,
    extra_specs: list[dict],
    padded_sequence: str,
) -> np.ndarray:
    """Build (N_windows, D) feature matrix for all windows.

    1. Pool ESM2 residues per window       → (N, H) via Cython or numpy
    2. Apply PCA                           → (N, D_pca)
    3. Append aaindex / extra features     → (N, D_pca + D_extra)
    """
    if _CYTHON:
        pooled = (pool_windows_max if pool == "max" else pool_windows_mean)(
            residue_embs, window_size
        )
    else:
        pooled = _pool_numpy(residue_embs, window_size, pool)

    if pca is not None:
        pooled = pca.transform(pooled).astype(np.float32)

    if not extra_specs:
        return pooled

    N = pooled.shape[0]
    extra_rows = [
        compute_extra_for_window(padded_sequence[i : i + window_size], extra_specs)
        for i in range(N)
    ]
    return np.concatenate([pooled, np.array(extra_rows, dtype=np.float32)], axis=1)


# ── Per-sequence scoring ──────────────────────────────────────────────────────

def analyse_sequence_esm2(
    model,
    sequence: str,
    esm2_session: dict,
    window_size: int = 18,
    extra_specs: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every residue in *sequence* using ESM2 window embeddings.

    Returns
    -------
    avg_scores   : (L,) float64 — per-residue mean window score
    window_scores : (N_windows,) float64 — raw per-window scores
    """
    if extra_specs is None:
        extra_specs = []
    L = len(sequence)

    # 1. Pad → one ESM2 forward pass
    padded, left_pad = _pad_sequence(sequence, window_size)
    residue_embs = _embed_sequence(padded, esm2_session)  # (L_pad, H)

    # 2. Build feature matrix for all windows (Cython pool + optional PCA)
    feat_matrix = _build_window_matrix(
        residue_embs, window_size,
        pca=esm2_session.get("pca"),
        pool=esm2_session["pool"],
        extra_specs=extra_specs,
        padded_sequence=padded,
    )  # (N_windows, D)

    # 3. Single batch model call
    window_scores = model.predict_proba(feat_matrix)[:, 1].astype(np.float64)

    # 4. Accumulate per-residue mean scores (Cython or numpy)
    if _CYTHON:
        avg_scores = _accumulate_scores_c(window_scores, window_size, L, left_pad)
    else:
        avg_scores = _accumulate_scores_numpy(window_scores, window_size, L, left_pad)

    return avg_scores, window_scores


# ── Full benchmark driver ─────────────────────────────────────────────────────

def run_benchmark_esm2(
    config: dict,
    model_path: str | Path | None = None,
    input_json: str | Path | None = None,
    output_path: str | Path | None = None,
    output_name: str = "benchmark_esm2_results",
    window_size: int | None = None,
    threshold: float | None = None,
    cli_feature_paths: list[str] | None = None,
    esm2_model_override: str | None = None,
    esm2_pool_override: str | None = None,
    classifier_mode: bool = False,
    positive_label: str | None = None,
) -> dict[str, Path]:
    """Run the ESM2 sliding-window benchmark on *input_json*.

    Parameters
    ----------
    config              : pipeline config dict (from load_config)
    model_path          : path to a .pkl ensemble file; auto-detected when None
    input_json          : benchmark dataset; uses config default when None
    output_path         : output directory
    output_name         : stem for output files
    window_size         : sliding window width (default from config)
    threshold           : decision threshold (default 0.5)
    cli_feature_paths   : extra feature specs (e.g. aaindex); must match training
    esm2_model_override : override the ESM2 model name stored in the pkl
    esm2_pool_override  : override the pooling strategy stored in the pkl
    classifier_mode     : evaluate at sequence level instead of residue level
    positive_label      : positive class label (default "AMYLOID")
    """
    bench_cfg = config["benchmark"]
    train_cfg = config["training"]

    _window_size = window_size or bench_cfg.get("window_size", 18)
    _threshold   = threshold if threshold is not None else bench_cfg.get("threshold", 0.5)

    bench_json = (
        Path(input_json) if input_json
        else resolve_pipeline_path(config, bench_cfg["input_json"])
    )

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

    # ── Locate model ──────────────────────────────────────────────────────────
    if model_path:
        resolved = Path(model_path)
    else:
        train_out = resolve_pipeline_path(config, train_cfg["output_dir"])
        resolved  = None
        for run_dir in sorted(train_out.glob("run_*"), reverse=True):
            if (run_dir / "esm2_pca_reducer.pkl").exists():
                resolved = run_dir / "soft_ensemble.pkl"
                break
        if resolved is None:
            raise FileNotFoundError(
                f"No ESM2-trained run found in {train_out}.\n"
                "Train with:  python scripts/train_ensemble_esm2.py --features ... esm2"
            )

    tqdm.write(f"Loading model   : {resolved}")
    model = load_model(resolved)

    # ── ESM2 session ──────────────────────────────────────────────────────────
    run_dir = resolved.parent
    esm2_session = load_esm2_session(run_dir, esm2_model_override, esm2_pool_override)
    tqdm.write(f"Cython accel    : {'ON  (_esm2_bench_fast)' if _CYTHON else 'OFF (numpy fallback)'}")
    tqdm.write(f"Window size     : {_window_size}   threshold={_threshold}")

    # ── Extra feature specs (aaindex, etc.) ───────────────────────────────────
    feature_specs = load_and_prepare_feature_specs(
        config, resolve_feature_paths_for_model(resolved, cli_feature_paths)
    )
    extra_specs = [s for s in feature_specs if not s.get("_builtin")]
    if any(s.get("_builtin") for s in feature_specs):
        tqdm.write(
            "[Warning] 'builtin' features (IUPred-dependent) are skipped in ESM2 "
            "benchmark mode. Verify the model was not trained with 'builtin'."
        )

    # ── Main loop ─────────────────────────────────────────────────────────────
    bench_records    = load_json_records(bench_json)
    per_seq_results: list[dict] = []
    all_window_records: list[dict] = []
    prediction_rows: list[dict] = []

    _pos_label     = positive_label or bench_cfg.get("classifier_positive_label", "AMYLOID")
    prediction_only = is_prediction_only_dataset(bench_records)
    if prediction_only:
        tqdm.write("[Prediction-only] No ground-truth labels found — skipping metrics.")
    elif classifier_mode:
        tqdm.write(f"[Classifier mode] positive_label='{_pos_label}'")

    for record in tqdm(bench_records, desc="Benchmark (ESM2)", unit="seq"):
        seq_id   = record.get("ID", "unknown")
        sequence = record["Sequence"]

        avg_scores, win_scores = analyse_sequence_esm2(
            model, sequence, esm2_session,
            window_size=_window_size,
            extra_specs=extra_specs,
        )

        pred_labels = [1 if s >= _threshold else 0 for s in avg_scores]

        win_out = {k: v for k, v in record.items() if k != "matched_core_regions"}
        win_out["n_windows"]     = int(len(win_scores))
        win_out["residue_scores"] = avg_scores.tolist()
        all_window_records.append(win_out)

        prediction_rows.append({
            "ID": seq_id,
            "Sequence": sequence,
            "Score_residues": avg_scores.tolist(),
        })

        if not prediction_only:
            if classifier_mode:
                per_seq_results.append(classifier_sequence_result(
                    seq_id, sequence, record.get("LABEL", ""), pred_labels, _pos_label,
                ))
            else:
                true_labels = create_true_labels(
                    len(sequence), record.get("matched_core_regions", [])
                )
                tn, fp, fn, tp = confusion_matrix(
                    true_labels, pred_labels, labels=[0, 1]
                ).ravel()
                per_seq_results.append({
                    "id": seq_id,
                    "sequence_length": len(sequence),
                    "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
                    "precision": float(precision_score(true_labels, pred_labels, zero_division=0)),
                    "recall":    float(recall_score(true_labels, pred_labels, zero_division=0)),
                    "f1_score":  float(f1_score(true_labels, pred_labels, zero_division=0)),
                    "accuracy":  float(accuracy_score(true_labels, pred_labels)),
                    "mcc":       float(matthews_corrcoef(true_labels, pred_labels)),
                    "sov":       float(sov_score(true_labels, pred_labels)),
                })

    # ── Save outputs ──────────────────────────────────────────────────────────
    csv_path     = output_stem.with_suffix(".csv")
    json_path    = output_stem.with_suffix(".json")
    scores_path  = output_stem.parent / (output_stem.name + "_scores.csv")
    windows_path = output_stem.parent / (output_stem.name + "_windows.json")

    pd.DataFrame(prediction_rows).to_csv(scores_path, index=False)
    with windows_path.open("w", encoding="utf-8") as fh:
        json.dump(all_window_records, fh, indent=2)

    if prediction_only:
        return {"predictions_csv": scores_path, "windows_json": windows_path}

    if classifier_mode:
        global_row = classifier_global_row(per_seq_results)
        all_rows   = per_seq_results + [global_row]
    else:
        _metrics = ["precision", "recall", "f1_score", "accuracy", "mcc", "sov"]
        n = len(per_seq_results)
        global_row: dict = {
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

    return {
        "results_csv": csv_path,
        "results_json": json_path,
        "scores_csv":   scores_path,
        "windows_json": windows_path,
    }
