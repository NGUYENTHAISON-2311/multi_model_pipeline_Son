#!/usr/bin/env python3
"""Train ensemble with ESM2 as a first-class --features token.

ESM2 embeddings are pooled over residue tokens, optionally reduced with PCA,
then concatenated with any other requested features before the classifier sees
the data.

Feature tokens accepted in --features
--------------------------------------
  builtin        540-dim handcrafted vector (IUPred required)
  aaindex        304 AAindex mean features  (data/aaindex_features.json)
  esm2           ESM2 embeddings, default model facebook/esm2_t12_35M_UR50D
  <path.json>    Any custom lookup-table spec

Multiple tokens are concatenated left-to-right.

ESM2 processing pipeline
--------------------------
  sequence (variable length)
      → ESM2 → (seq_len, hidden_size) per-residue hidden states
      → pool  → (hidden_size,)  [mean | max | cls, default: mean]
      → PCA   → (esm2_dim,)    [default: 64; set 0 to skip PCA]
      → concat with other features → classifier

The PCA reducer is fitted on the combined positive + negative training set and
saved to <run_dir>/esm2_pca_reducer.pkl so it can be reused at inference time.

Examples
--------
  # Builtin + ESM2 (default 35M model, PCA → 64 dims):
  python scripts/train_ensemble_esm2.py \\
      --positive cores_07/train_core_set_seed_20.json \\
      --negative cores_07/len_matching_filtered_disordered_regions_clustered.json \\
      --features builtin esm2

  # AAindex + ESM2 (larger model, no PCA, max pooling):
  python scripts/train_ensemble_esm2.py \\
      --features aaindex esm2 \\
      --esm2-model facebook/esm2_t33_650M_UR50D \\
      --esm2-pool max \\
      --esm2-dim 0

  # ESM2 only (PCA → 128):
  python scripts/train_ensemble_esm2.py \\
      --features esm2 --esm2-dim 128

  # Use pre-computed embeddings (skip ESM2 inference):
  python esm2_embedding.py -i pos.json -o pos_emb.pt
  python esm2_embedding.py -i neg.json -o neg_emb.pt
  python scripts/train_ensemble_esm2.py \\
      --features builtin esm2 \\
      --embeddings-pos pos_emb.pt \\
      --embeddings-neg neg_emb.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[1]))   # project root → src.*
sys.path.insert(0, str(_here.parent))       # scripts/     → esm2_embedding, feature_builder

from src.configuration import load_config, resolve_pipeline_path
from src.training_pipeline import SUPPORTED_MODELS, run_ensemble_training

AAINDEX_SPEC      = "data/aaindex_features.json"
DEFAULT_ESM2_MODEL = "facebook/esm2_t12_35M_UR50D"
DEFAULT_ESM2_DIM   = 64


# ── ESM2 helpers ──────────────────────────────────────────────────────────────

def _pool(hidden_states, attention_mask, strategy: str):
    """Reduce (N, seq_len, H) → (N, H) with mean / max / cls pooling."""
    import torch
    if strategy == "cls":
        return hidden_states[:, 0, :]
    if strategy == "max":
        mask = attention_mask.unsqueeze(-1).bool()
        hidden_states = hidden_states.masked_fill(~mask, float("-inf"))
        return hidden_states.max(dim=1).values
    # mean (default) — average over non-padding token positions
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)


def _embed_sequences(sequences: list[str], esm2_model: str, pool: str) -> np.ndarray:
    """Run ESM2 inference and return a pooled (N, hidden_size) numpy array."""
    import torch
    from esm2_embedding import load_model
    tokenizer, model, device = load_model(esm2_model)
    print(f"    device: {device}  hidden_size: {model.config.hidden_size}")
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pooled = _pool(outputs.last_hidden_state, inputs["attention_mask"], pool)
    return pooled.cpu().numpy().astype(np.float32)


def _load_precomputed(pt_path: str, ids: list[str]) -> np.ndarray:
    from feature_builder import _load_embeddings_from_pt
    return _load_embeddings_from_pt(pt_path, ids)


# ── Pipeline-feature helper ───────────────────────────────────────────────────

def _pipeline_features(
    sequences: list[str],
    specs: list[str],
    iupred_scores: list[float] | None,
    config: dict,
) -> np.ndarray:
    from src.configuration import resolve_pipeline_path
    from src.feature_loader import load_and_prepare_feature_specs, compute_sequence_feature_matrix
    from src.feature_pipeline import compute_average_iupred_scores_from_sequences
    prepared = load_and_prepare_feature_specs(config, specs)
    has_builtin = any(s.get("_builtin") for s in prepared)
    if has_builtin and iupred_scores is None:
        iupred_script = resolve_pipeline_path(config, config["training"]["iupred_script"])
        iupred_type   = config["training"].get("iupred_input_type", "long")
        print(f"    Running IUPred ({iupred_type}) on {len(sequences)} sequences …")
        iupred_scores = compute_average_iupred_scores_from_sequences(
            sequences, iupred_script, iupred_type
        )
    scores = iupred_scores or [0.0] * len(sequences)
    rows = compute_sequence_feature_matrix(sequences, prepared, scores)
    return np.array(rows, dtype=np.float32)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_iupred_scores(path: str, ids: list[str]) -> list[float]:
    with open(path) as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        return [float(raw[sid]) for sid in ids]
    return [float(v) for v in raw]


def _load_sequences(path: Path) -> tuple[list[str], list[str]]:
    """Return (ids, sequences) from a JSON or FASTA file.

    Accepts both benchmark-style records (keys "ID", "Sequence") and
    training-set records (keys "core_id", "sequence").
    """
    if path.suffix.lower() == ".json":
        with path.open() as f:
            records = json.load(f)
        seqs, ids = [], []
        for i, r in enumerate(records):
            seq = r.get("Sequence") or r.get("sequence")
            if seq is None:
                raise KeyError(f"Record {i} in {path} has no 'Sequence' or 'sequence' field.")
            seqs.append(seq)
            ids.append(r.get("ID") or r.get("core_id") or str(i))
    else:
        from src.feature_pipeline import load_sequences_from_file
        seqs = load_sequences_from_file(path)
        ids  = [str(i) for i in range(len(seqs))]
    return ids, seqs


# ── TensorBoard logging ───────────────────────────────────────────────────────

def _write_tb_logs(
    run_dir: Path,
    feature_info: dict,
    pca_reducer,
    esm2_model: str,
    esm2_pool: str,
    esm2_dim: int,
    title: str | None = None,
) -> None:
    """Write all training metrics to TensorBoard after run_ensemble_training completes.

    Reads back the per-algorithm CSVs and metadata.json saved by the training
    pipeline and emits:
      - Feature dimensions and ESM2 PCA explained variance
      - Per-fold F1 / Accuracy / Precision / Recall / MCC for every algorithm
      - Per-combo hyperparameter search scores (mean metric per combo)
      - Per-algorithm summary (mean ± std of each metric)
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard not available (pip install tensorboard). Skipping TB logs.")
        return

    tb_dir = run_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))

    # ── Run title ─────────────────────────────────────────────────────────────
    if title:
        writer.add_text("run/title", title, 0)

    # ── Feature info ──────────────────────────────────────────────────────────
    writer.add_scalar("features/total_dim",     feature_info["total_dim"],     0)
    if feature_info.get("pipeline_dim"):
        writer.add_scalar("features/pipeline_dim", feature_info["pipeline_dim"], 0)
    if feature_info.get("esm2_raw_dim"):
        writer.add_scalar("features/esm2_raw_dim", feature_info["esm2_raw_dim"], 0)
    if pca_reducer is not None:
        var = float(pca_reducer.explained_variance_ratio_.sum())
        writer.add_scalar("features/esm2_pca_explained_variance", var, 0)
        writer.add_scalar("features/esm2_pca_components", int(pca_reducer.n_components_), 0)
    writer.add_text("features/config", (
        f"esm2_model={esm2_model}  pool={esm2_pool}  pca_dim={esm2_dim}"
    ), 0)

    # ── Per-algorithm metrics ─────────────────────────────────────────────────
    per_algo_dir = run_dir / "per_algorithm"
    metrics = ["F1_score", "Accuracy", "Precision", "Recall", "MCC"]

    for algo_dir in sorted(per_algo_dir.iterdir()):
        if not algo_dir.is_dir():
            continue
        algo = algo_dir.name

        # Per-fold scalars (step = fold index, 0-based)
        scores_csv = algo_dir / "scores.csv"
        if scores_csv.exists():
            with scores_csv.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    fold = int(row["Fold"]) - 1
                    for m in metrics:
                        writer.add_scalar(f"{algo}/fold/{m}", float(row[m]), fold)

        # Summary (mean ± std)
        summary_json = algo_dir / "summary.json"
        if summary_json.exists():
            with summary_json.open() as fh:
                s = json.load(fh)
            for m in metrics:
                writer.add_scalar(f"{algo}/summary/mean_{m}", s[f"mean_{m}"], 0)
                writer.add_scalar(f"{algo}/summary/std_{m}",  s[f"std_{m}"],  0)

    # ── Hyperparameter combo search (from metadata.json) ─────────────────────
    metadata_json = run_dir / "metadata.json"
    if metadata_json.exists():
        with metadata_json.open() as fh:
            metadata = json.load(fh)
        for algo_meta in metadata.get("algorithms", []):
            algo = algo_meta["type"]
            for combo in algo_meta.get("combos_tried", []):
                idx = int(combo["combo_index"])
                for key, val in combo.items():
                    if key.startswith("mean_") and isinstance(val, (int, float)):
                        tag = key[len("mean_"):]
                        writer.add_scalar(f"{algo}/combo/{tag}", float(val), idx)

    writer.close()
    print(f"TensorBoard logs:     {tb_dir}")
    print(f"  tensorboard --logdir {tb_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train ensemble — builtin / aaindex / ESM2 features with optional PCA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--config",   help="Config JSON file.")
    parser.add_argument("--positive", help="Positive sequences (JSON or FASTA).")
    parser.add_argument("--negative", help="Negative sequences (JSON or FASTA).")
    parser.add_argument("--output",   help="Training output directory.")
    parser.add_argument("--title",    help="Human-readable label for this run (shown in terminal and TensorBoard).")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--combos", type=int, default=10,
                        help="Random hyperparameter combos per algorithm (default: 10).")
    parser.add_argument("--folds",  type=int, default=None,
                        help="k-fold CV splits (overrides config, default 5).")
    parser.add_argument(
        "--metric", default="F1_score",
        choices=["F1_score", "Accuracy", "Precision", "Recall", "MCC", "Average"],
        help="Combo selection metric (default: F1_score).",
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes; 0 = all CPUs (default: 1).")
    parser.add_argument("--algorithms", nargs="+", metavar="ALGO",
                        help=f"Restrict to these algorithms. Choices: {', '.join(SUPPORTED_MODELS)}.")

    # ── Features ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--features", nargs="+", metavar="SPEC", default=["builtin"],
        help=(
            'Feature tokens (concatenated in order): '
            '"builtin" (540-dim, needs IUPred), '
            '"aaindex" (304-dim AAindex), '
            '"esm2" (ESM2 embeddings + optional PCA), '
            'or path to a JSON spec. Default: builtin.'
        ),
    )
    parser.add_argument("--iupred-scores-pos", metavar="JSON",
                        help='IUPred scores for positives: {"ID": score} or flat list.')
    parser.add_argument("--iupred-scores-neg", metavar="JSON",
                        help='IUPred scores for negatives: {"ID": score} or flat list.')

    # ── ESM2 ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--esm2-model", default=DEFAULT_ESM2_MODEL,
        help=f"HuggingFace ESM2 model name (default: {DEFAULT_ESM2_MODEL}).",
    )
    parser.add_argument(
        "--esm2-pool", choices=["mean", "max", "cls"], default="mean",
        help="Residue-level pooling strategy (default: mean).",
    )
    parser.add_argument(
        "--esm2-dim", type=int, default=DEFAULT_ESM2_DIM,
        help=(
            f"PCA target dimensions applied after pooling (default: {DEFAULT_ESM2_DIM}). "
            "Set to 0 to skip PCA and keep the full embedding."
        ),
    )
    parser.add_argument("--embeddings-pos", metavar="PT",
                        help="Pre-computed ESM2 .pt file for positives (skips inference).")
    parser.add_argument("--embeddings-neg", metavar="PT",
                        help="Pre-computed ESM2 .pt file for negatives (skips inference).")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging.")
    parser.add_argument(
        "--cache-dir", metavar="DIR",
        help=(
            "Directory for caching the built feature matrices (pos_matrix.npy, "
            "neg_matrix.npy, pca_reducer.pkl).  If the cache exists it is reloaded "
            "and ESM2/IUPred/PCA are skipped.  Pass the same dir on retry to avoid "
            "recomputing after a crash."
        ),
    )

    args = parser.parse_args()

    # ── Run banner ────────────────────────────────────────────────────────────
    if args.title:
        width = max(60, len(args.title) + 4)
        bar = "═" * width
        print(f"\n{bar}")
        print(f"  Run: {args.title}")
        print(f"{bar}\n")

    # ── Resolve feature tokens ────────────────────────────────────────────────
    use_esm2 = "esm2" in args.features
    pipeline_specs = []
    for tok in args.features:
        if tok == "esm2":
            continue
        elif tok == "aaindex":
            pipeline_specs.append(AAINDEX_SPEC)
        else:
            pipeline_specs.append(tok)

    if not use_esm2 and not pipeline_specs:
        parser.error("--features must include at least one token.")

    # ── Config & paths ────────────────────────────────────────────────────────
    config = load_config(args.config)
    training_cfg = config["training"]

    pos_path = Path(args.positive) if args.positive else resolve_pipeline_path(config, training_cfg["positive_json"])
    neg_path = Path(args.negative) if args.negative else resolve_pipeline_path(config, training_cfg["negative_json"])

    pos_ids, pos_seqs = _load_sequences(pos_path)
    neg_ids, neg_seqs = _load_sequences(neg_path)
    print(f"Loaded {len(pos_seqs)} positive, {len(neg_seqs)} negative sequences.")

    pos_iupred = _load_iupred_scores(args.iupred_scores_pos, pos_ids) if args.iupred_scores_pos else None
    neg_iupred = _load_iupred_scores(args.iupred_scores_neg, neg_ids) if args.iupred_scores_neg else None

    # ── Algorithm filter ──────────────────────────────────────────────────────
    if args.algorithms:
        valid = {a.lower() for a in args.algorithms}
        unknown = valid - set(SUPPORTED_MODELS)
        if unknown:
            parser.error(f"Unknown algorithm(s): {', '.join(unknown)}. "
                         f"Choices: {', '.join(SUPPORTED_MODELS)}")
        all_algos = config["training"].get("algorithms", [])
        config["training"]["algorithms"] = [a for a in all_algos if a["type"] in valid]
        if not config["training"]["algorithms"]:
            parser.error(f"No matching algorithms found in config for: {', '.join(valid)}")

    # ── Feature matrix cache (reload if available, save after building) ──────
    cache_dir   = Path(args.cache_dir) if args.cache_dir else None
    _cache_pos  = cache_dir / "pos_matrix.npy"  if cache_dir else None
    _cache_neg  = cache_dir / "neg_matrix.npy"  if cache_dir else None
    _cache_pca  = cache_dir / "pca_reducer.pkl" if cache_dir else None
    _cache_info = cache_dir / "feature_info.json" if cache_dir else None

    if cache_dir and _cache_pos.exists() and _cache_neg.exists():
        print(f"\n[Cache] Loading feature matrices from {cache_dir}")
        pos_matrix   = np.load(_cache_pos)
        neg_matrix   = np.load(_cache_neg)
        feature_info = json.loads(_cache_info.read_text()) if _cache_info.exists() else {}
        pca_reducer  = None
        if _cache_pca and _cache_pca.exists():
            with _cache_pca.open("rb") as fh:
                saved = pickle.load(fh)
            pca_reducer = saved.get("pca")
        dim_label = feature_info.get("dim_label", f"{pos_matrix.shape[1]}-dim (from cache)")
        print(f"    pos {pos_matrix.shape}  neg {neg_matrix.shape}")
        print(f"    Feature breakdown: {dim_label}")
        # Skip feature building entirely
        _skip_build = True
    else:
        _skip_build = False

    # ── Build feature matrices ────────────────────────────────────────────────
    parts_pos: list[np.ndarray] = []
    parts_neg: list[np.ndarray] = []
    dim_parts:  list[str]        = []
    feature_info: dict           = {}

    # 1) Pipeline features (builtin / aaindex / custom JSON)
    if not _skip_build and pipeline_specs:
        print(f"\n[1/2] Pipeline features: {pipeline_specs}")
        pf_pos = _pipeline_features(pos_seqs, pipeline_specs, pos_iupred, config)
        pf_neg = _pipeline_features(neg_seqs, pipeline_specs, neg_iupred, config)
        parts_pos.append(pf_pos)
        parts_neg.append(pf_neg)
        dim_parts.append(f"pipeline:{pf_pos.shape[1]}")
        feature_info["pipeline_dim"] = int(pf_pos.shape[1])
        print(f"    pos {pf_pos.shape}  neg {pf_neg.shape}")

    # 2) ESM2 embeddings → pool → optional PCA
    if not _skip_build:
        pca_reducer = None
    if not _skip_build and use_esm2:
        step = "2/2" if pipeline_specs else "1/1"
        print(f"\n[{step}] ESM2  model={args.esm2_model}  pool={args.esm2_pool}")

        if args.embeddings_pos:
            print(f"    Loading pre-computed positives from {args.embeddings_pos}")
            esm_pos = _load_precomputed(args.embeddings_pos, pos_ids)
        else:
            print("    Running ESM2 inference on positives …")
            esm_pos = _embed_sequences(pos_seqs, args.esm2_model, args.esm2_pool)

        if args.embeddings_neg:
            print(f"    Loading pre-computed negatives from {args.embeddings_neg}")
            esm_neg = _load_precomputed(args.embeddings_neg, neg_ids)
        else:
            print("    Running ESM2 inference on negatives …")
            esm_neg = _embed_sequences(neg_seqs, args.esm2_model, args.esm2_pool)

        print(f"    Raw embedding  pos {esm_pos.shape}  neg {esm_neg.shape}")
        feature_info["esm2_raw_dim"] = int(esm_pos.shape[1])

        if args.esm2_dim and args.esm2_dim > 0:
            from sklearn.decomposition import PCA
            n_components = min(args.esm2_dim, esm_pos.shape[1],
                               esm_pos.shape[0] + esm_neg.shape[0])
            print(f"\n    PCA: {esm_pos.shape[1]} → {n_components} dims "
                  f"(fitted on combined pos+neg)")
            combined = np.concatenate([esm_pos, esm_neg], axis=0)
            pca_reducer = PCA(n_components=n_components, random_state=42)
            pca_reducer.fit(combined)
            var_explained = pca_reducer.explained_variance_ratio_.sum()
            print(f"    Explained variance: {var_explained:.3f}")
            esm_pos = pca_reducer.transform(esm_pos).astype(np.float32)
            esm_neg = pca_reducer.transform(esm_neg).astype(np.float32)
            dim_parts.append(f"esm2-pca:{esm_pos.shape[1]}")
        else:
            dim_parts.append(f"esm2:{esm_pos.shape[1]}")

        parts_pos.append(esm_pos)
        parts_neg.append(esm_neg)
        print(f"    After reduction   pos {esm_pos.shape}  neg {esm_neg.shape}")

    # 3) Concatenate all parts (skipped when loading from cache)
    if not _skip_build:
        pos_matrix = np.concatenate(parts_pos, axis=1)
        neg_matrix = np.concatenate(parts_neg, axis=1)
        dim_label  = " + ".join(dim_parts) + f"  →  {pos_matrix.shape[1]} total dims"
        feature_info["total_dim"] = int(pos_matrix.shape[1])
        feature_info["dim_label"] = dim_label
        print(f"\nFinal feature matrix: pos {pos_matrix.shape}  neg {neg_matrix.shape}")
        print(f"Feature breakdown:    {dim_label}")

        # Save to cache if requested
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(_cache_pos, pos_matrix)
            np.save(_cache_neg, neg_matrix)
            _cache_info.write_text(json.dumps(feature_info, indent=2))
            if pca_reducer is not None:
                with _cache_pca.open("wb") as fh:
                    pickle.dump({"pca": pca_reducer, "esm2_model": args.esm2_model,
                                 "esm2_pool": args.esm2_pool}, fh)
            print(f"[Cache] Saved feature matrices → {cache_dir}")

    # ── Train ─────────────────────────────────────────────────────────────────
    _root = Path(__file__).resolve().parents[1]
    output_dir = args.output or str(_root / "outputs" / "training")

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

    # ── Save PCA reducer for inference ────────────────────────────────────────
    if pca_reducer is not None:
        pca_path = Path(artifacts["run_dir"]) / "esm2_pca_reducer.pkl"
        with open(pca_path, "wb") as fh:
            pickle.dump({
                "pca":          pca_reducer,
                "esm2_model":   args.esm2_model,
                "esm2_pool":    args.esm2_pool,
                "n_components": int(pca_reducer.n_components_),
                "var_explained": float(pca_reducer.explained_variance_ratio_.sum()),
            }, fh)
        print(f"ESM2 PCA reducer:     {pca_path}")

    run_id = Path(artifacts["run_dir"]).name
    lines = [f"  Run ID : {run_id}"]
    if args.title:
        lines.insert(0, f"  Title  : {args.title}")
    width = max(60, max(len(l) for l in lines) + 4)
    bar = "═" * width
    print(f"\n{bar}")
    print(f"  Training completed — {len(ensembles)} ensemble variant(s)")
    for l in lines:
        print(l)
    print(bar)
    print(f"\nRun directory:        {artifacts['run_dir']}")
    print(f"soft_ensemble.pkl:    {artifacts['soft_pkl']}")
    print(f"weighted_ensemble.pkl:{artifacts['weighted_pkl']}")
    print(f"best_model.pkl:       {artifacts['best_pkl']}")
    print(f"Summary CSV:          {artifacts['summary_csv']}")
    print(f"Metadata:             {artifacts['metadata_json']}")

    # ── TensorBoard logs ──────────────────────────────────────────────────────
    if not args.no_tensorboard:
        _write_tb_logs(
            run_dir=Path(artifacts["run_dir"]),
            feature_info=feature_info,
            pca_reducer=pca_reducer,
            esm2_model=args.esm2_model,
            esm2_pool=args.esm2_pool,
            esm2_dim=args.esm2_dim,
            title=args.title,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
