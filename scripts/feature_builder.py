"""
Feature builder: concatenate ESM2 embeddings with the existing src/ feature pipeline.

The src/ pipeline produces features via feature_loader.py:
  - "builtin"  → 540-dim: length + AA freq(20) + dipeptide(400) +
                           group freq(17) + intra-class transitions(101) + IUPred(1)
  - JSON spec  → AAindex or other lookup-table features (e.g. data/aaindex_features.json)

This module adds ESM2 embeddings on top by simple concatenation.

CLI usage
---------
  # Compute everything from scratch (ESM2 + AAindex, no IUPred):
  python feature_builder.py -i dataset.json -f data/aaindex_features.json -o features.npy

  # Use pre-computed embeddings from esm2_embedding.py:
  python esm2_embedding.py -i dataset.json -o embeddings.pt
  python feature_builder.py -i dataset.json -e embeddings.pt -f data/aaindex_features.json -o features.npy

  # Builtin 540-dim + ESM2 (requires --iupred-scores):
  python feature_builder.py -i dataset.json --iupred-scores scores.json -f builtin -o features.npy

Install:
    pip install torch transformers numpy tqdm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root (for src.*) and scripts/ (for esm2_embedding, etc.)
_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[1]))
sys.path.insert(0, str(_here.parent))

from src.configuration import load_config
from src.feature_loader import load_and_prepare_feature_specs, compute_sequence_feature_matrix
from esm2_embedding import load_model, embed_sequences, load_records_from_json

# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_features(
    sequences: list[str],
    iupred_scores: list[float] | None = None,
    feature_paths: list[str] | None = None,
    config_path: str | Path | None = None,
    esm2_model: str = "facebook/esm2_t6_8M_UR50D",
    esm2_pool: str = "mean",
    use_esm2: bool = True,
) -> np.ndarray:
    """
    Build a combined feature matrix for a list of protein sequences.

    The existing feature pipeline (src/) runs first; ESM2 embeddings are
    appended as extra columns.

    Parameters
    ----------
    sequences:
        List of amino acid sequences.
    iupred_scores:
        Pre-computed average IUPred disorder scores, one per sequence.
        Required when "builtin" is in feature_paths (or by default).
        Omit if using non-builtin features only.
    feature_paths:
        Feature spec paths passed to load_and_prepare_feature_specs().
        Examples:
          ["builtin"]                               → 540-dim handcrafted only
          ["data/aaindex_features.json"]            → AAindex only
          ["builtin", "data/aaindex_features.json"] → handcrafted + AAindex
        Defaults to config["feature_files"] or ["builtin"].
    config_path:
        Path to config JSON. Defaults to config/default_config.json.
    esm2_model:
        HuggingFace ESM2 model name.
    esm2_pool:
        "mean" or "cls" pooling for ESM2.
    use_esm2:
        Set to False to run only the src/ pipeline without ESM2.

    Returns
    -------
    numpy array of shape (N, total_features).
    """
    config = load_config(config_path)
    specs = load_and_prepare_feature_specs(config, feature_paths)

    has_builtin = any(s.get("_builtin") for s in specs)
    if has_builtin and not iupred_scores:
        raise ValueError(
            'iupred_scores is required when "builtin" features are enabled. '
            'Provide pre-computed scores or use feature_paths=["data/aaindex_features.json"] '
            "to skip the builtin 540-dim features."
        )

    scores = iupred_scores or [0.0] * len(sequences)

    parts: list[np.ndarray] = []

    # --- src/ pipeline features ---
    print("Computing pipeline features...")
    pipeline_rows = compute_sequence_feature_matrix(sequences, specs, scores)
    pipeline_matrix = np.array(pipeline_rows, dtype=np.float32)
    parts.append(pipeline_matrix)
    print(f"  Pipeline features shape: {pipeline_matrix.shape}")

    # --- ESM2 ---
    if use_esm2:
        print(f"Computing ESM2 embeddings ({esm2_model})...")
        tokenizer, model, device = load_model(esm2_model)
        esm = embed_sequences(sequences, tokenizer, model, device, pool=esm2_pool).cpu().numpy()
        parts.append(esm)
        print(f"  ESM2 shape: {esm.shape}")

    combined = np.concatenate(parts, axis=1)
    print(f"Combined feature shape: {combined.shape}")
    return combined


def build_features_from_json(
    json_path: str | Path,
    iupred_scores: list[float] | None = None,
    feature_paths: list[str] | None = None,
    config_path: str | Path | None = None,
    esm2_model: str = "facebook/esm2_t6_8M_UR50D",
    esm2_pool: str = "mean",
    use_esm2: bool = True,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load a benchmark dataset JSON file and build the combined feature matrix.

    Expected JSON format:
        [{"ID": "2E8D_A", "LABEL": "AMYLOID", "Sequence": "SNFLNCY...", ...}, ...]

    Returns
    -------
    features  — numpy array of shape (N, total_features)
    ids       — list of record IDs
    labels    — list of LABEL strings (empty string if absent)
    """
    records = load_records_from_json(json_path)
    ids = [r.get("ID", "") for r in records]
    labels = [r.get("LABEL", "") for r in records]
    sequences = [r["Sequence"] for r in records]

    features = build_features(
        sequences,
        iupred_scores=iupred_scores,
        feature_paths=feature_paths,
        config_path=config_path,
        esm2_model=esm2_model,
        esm2_pool=esm2_pool,
        use_esm2=use_esm2,
    )
    return features, ids, labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_embeddings_from_pt(pt_path: str | Path, ids: list[str]) -> np.ndarray:
    """Load pre-computed ESM2 embeddings saved by esm2_embedding.py.

    The .pt file contains {"ids": [...], "sequences": [...], "embeddings": Tensor}.
    Rows are reordered to match *ids* if the order differs.
    """
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    saved_ids: list[str] = checkpoint["ids"]
    saved_emb: torch.Tensor = checkpoint["embeddings"]

    if saved_ids == ids:
        return saved_emb.numpy().astype(np.float32)

    # Build index from saved_ids → row position
    id_to_row = {sid: i for i, sid in enumerate(saved_ids)}
    missing = [sid for sid in ids if sid not in id_to_row]
    if missing:
        raise ValueError(
            f"The following IDs are in the input JSON but missing from the embeddings file: {missing}"
        )
    indices = [id_to_row[sid] for sid in ids]
    return saved_emb[indices].numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Build combined feature matrix: src/ pipeline features + ESM2 embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # AAindex + ESM2 (embeddings computed on the fly):
  python feature_builder.py -i dataset.json -f data/aaindex_features.json -o features.npy

  # Use pre-computed embeddings from esm2_embedding.py:
  python esm2_embedding.py -i dataset.json -o embeddings.pt
  python feature_builder.py -i dataset.json -e embeddings.pt -f data/aaindex_features.json -o features.npy

  # Builtin 540-dim + ESM2 (IUPred scores required):
  python feature_builder.py -i dataset.json --iupred-scores scores.json -f builtin -o features.npy

  # Pipeline features only (no ESM2):
  python feature_builder.py -i dataset.json -f data/aaindex_features.json --no-esm2 -o features.npy
""",
    )
    parser.add_argument(
        "--input", "-i", required=True, metavar="JSON",
        help='Benchmark dataset JSON: [{"ID": "...", "Sequence": "...", "LABEL": "..."}, ...]',
    )
    parser.add_argument(
        "--embeddings", "-e", metavar="PT",
        help=(
            "Pre-computed ESM2 embeddings file (.pt) saved by esm2_embedding.py. "
            "When provided, ESM2 is not re-run. Ignored when --no-esm2 is set."
        ),
    )
    parser.add_argument(
        "--features", "-f", nargs="+", metavar="PATH",
        default=None,
        help=(
            'Feature spec paths. Use "builtin" for the 540-dim handcrafted features '
            "or a path to a JSON spec (e.g. data/aaindex_features.json). "
            "Multiple specs are concatenated. Defaults to config[\"feature_files\"] or [\"builtin\"]."
        ),
    )
    parser.add_argument(
        "--iupred-scores", metavar="JSON",
        help=(
            'JSON file with per-sequence average IUPred scores: {"<ID>": <score>, ...} '
            "or a flat list of floats matching the input order. "
            'Required when "builtin" is in --features.'
        ),
    )
    parser.add_argument(
        "--no-esm2", action="store_true",
        help="Skip ESM2 embeddings entirely; output only the src/ pipeline features.",
    )
    parser.add_argument(
        "--esm2-model", metavar="NAME", default="facebook/esm2_t6_8M_UR50D",
        help="HuggingFace ESM2 model name (default: facebook/esm2_t6_8M_UR50D).",
    )
    parser.add_argument(
        "--pool", choices=["mean", "cls"], default="mean",
        help="ESM2 pooling strategy (default: mean). Ignored when --embeddings is set.",
    )
    parser.add_argument(
        "--config", metavar="JSON",
        help="Path to pipeline config JSON (default: config/default_config.json).",
    )
    parser.add_argument(
        "--output", "-o", metavar="FILE", default="features.npy",
        help="Output .npy file for the combined feature matrix (default: features.npy).",
    )
    parser.add_argument(
        "--save-ids", metavar="FILE",
        help="Optional: save record IDs as a .npy or .json file.",
    )
    parser.add_argument(
        "--save-labels", metavar="FILE",
        help="Optional: save LABEL values as a .npy or .json file.",
    )
    args = parser.parse_args()

    # ── Load records ─────────────────────────────────────────────────────────
    records = load_records_from_json(args.input)
    ids = [r.get("ID", str(i)) for i, r in enumerate(records)]
    labels = [r.get("LABEL", "") for r in records]
    sequences = [r["Sequence"] for r in records]

    # ── IUPred scores ─────────────────────────────────────────────────────────
    iupred_scores: list[float] | None = None
    if args.iupred_scores:
        with open(args.iupred_scores) as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            iupred_scores = [float(raw[sid]) for sid in ids]
        elif isinstance(raw, list):
            if len(raw) != len(sequences):
                parser.error(
                    f"--iupred-scores list has {len(raw)} entries but input has {len(sequences)} sequences."
                )
            iupred_scores = [float(v) for v in raw]
        else:
            parser.error("--iupred-scores must be a JSON dict {ID: score} or a flat list of floats.")

    # ── Pipeline features ─────────────────────────────────────────────────────
    config = load_config(args.config)
    specs = load_and_prepare_feature_specs(config, args.features)

    has_builtin = any(s.get("_builtin") for s in specs)
    if has_builtin and iupred_scores is None:
        parser.error(
            '--iupred-scores is required when "builtin" is in --features. '
            "Provide a scores file or switch to a non-builtin feature spec."
        )

    scores = iupred_scores or [0.0] * len(sequences)

    print("Computing pipeline features...")
    pipeline_rows = compute_sequence_feature_matrix(sequences, specs, scores)
    pipeline_matrix = np.array(pipeline_rows, dtype=np.float32)
    print(f"  Pipeline features shape: {pipeline_matrix.shape}")

    parts: list[np.ndarray] = [pipeline_matrix]

    # ── ESM2 embeddings ───────────────────────────────────────────────────────
    if not args.no_esm2:
        if args.embeddings:
            print(f"Loading pre-computed embeddings from {args.embeddings} ...")
            esm = _load_embeddings_from_pt(args.embeddings, ids)
            print(f"  ESM2 shape: {esm.shape}")
        else:
            print(f"Computing ESM2 embeddings ({args.esm2_model})...")
            tokenizer, model, device = load_model(args.esm2_model)
            print(f"  Running on: {device}")
            esm = embed_sequences(sequences, tokenizer, model, device, pool=args.pool).cpu().numpy().astype(np.float32)
            print(f"  ESM2 shape: {esm.shape}")
        parts.append(esm)

    # ── Concatenate & save ────────────────────────────────────────────────────
    combined = np.concatenate(parts, axis=1)
    print(f"\nCombined feature shape: {combined.shape}")

    np.save(args.output, combined)
    print(f"Saved features → {args.output}")

    if args.save_ids:
        p = Path(args.save_ids)
        if p.suffix == ".json":
            p.write_text(json.dumps(ids, indent=2))
        else:
            np.save(p, np.array(ids))
        print(f"Saved IDs      → {args.save_ids}")

    if args.save_labels:
        p = Path(args.save_labels)
        if p.suffix == ".json":
            p.write_text(json.dumps(labels, indent=2))
        else:
            np.save(p, np.array(labels))
        print(f"Saved labels   → {args.save_labels}")

    # Summary
    print("\nFeature breakdown:")
    print(f"  Pipeline features : {pipeline_matrix.shape[1]}")
    if not args.no_esm2:
        print(f"  ESM2 embeddings   : {parts[-1].shape[1]}")
    print(f"  Total             : {combined.shape[1]}")


if __name__ == "__main__":
    main()
