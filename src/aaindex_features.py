"""AAindex feature extraction for the multi-model pipeline.

Provides functions to:
  - parse the aaindex1 flat file
  - load a pruned list of selected accession IDs
  - compute per-sequence mean AAindex features (one float per accession)
  - merge those features with the built-in 540-dim features

Feature-mode semantics (set via config["extra_features"]["mode"]):
  "builtin"    — use only the original 540-dim features (no AAindex)
  "combine"    — concatenate built-in + AAindex features
  "extra_only" — use only AAindex features (for ablation studies)

Note: only **mean** over residues is computed (not std).  This keeps the
feature vector identical between training (full short sequences) and
inference (sliding windows of 18 residues), where std is unreliable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

# Canonical 20-letter alphabet used by aaindex1
_AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")

VALID_MODES = ("builtin", "combine", "extra_only")


# ── Parsing ───────────────────────────────────────────────────────────────────

def load_aaindex1(path: str | Path) -> dict[str, dict[str, float | None]]:
    """Parse the aaindex1 flat file.

    Returns ``{accession: {AA: value | None}}`` for every entry in the file.
    Values listed as ``NA`` are stored as ``None``.
    """
    aaindex: dict[str, dict[str, float | None]] = {}
    current_acc: str | None = None
    values_line: str | None = None

    with Path(path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip()
            if line.startswith("H "):
                current_acc = line[2:].strip()
                values_line = None
            elif line.startswith("I ") and current_acc is not None:
                # Next non-blank line contains the second row of values
                values_line = line[2:]
            elif values_line is not None and current_acc is not None:
                # Combine both rows: first row = values_line, second row = line
                row1 = values_line.split()
                row2 = line.split()
                tokens = row1 + row2
                vals: dict[str, float | None] = {}
                for aa, tok in zip(_AA_ORDER, tokens):
                    vals[aa] = None if tok == "NA" else float(tok)
                aaindex[current_acc] = vals
                current_acc = None
                values_line = None

    return aaindex


def load_selected_accessions(path: str | Path) -> list[str]:
    """Load the pruned list of accession IDs from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        accs = json.load(fh)
    if not isinstance(accs, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(accs).__name__}")
    return [str(a) for a in accs]


# ── Feature computation ───────────────────────────────────────────────────────

def compute_aaindex_mean(
    sequence: str,
    accessions: list[str],
    aaindex: dict[str, dict[str, float | None]],
) -> list[float]:
    """Return the mean AAindex value per accession for *sequence*.

    Residues not in the canonical 20-letter alphabet (X, B, Z, …) are
    silently skipped.  If no valid residues exist, 0.0 is returned for
    that index.
    """
    result: list[float] = []
    for acc in accessions:
        table = aaindex.get(acc, {})
        total = 0.0
        count = 0
        for aa in sequence:
            v = table.get(aa)
            if v is not None:
                total += v
                count += 1
        result.append(total / count if count > 0 else 0.0)
    return result


def build_aaindex_matrix(
    sequences: Iterable[str],
    accessions: list[str],
    aaindex: dict[str, dict[str, float | None]],
    desc: str = "AAindex features",
) -> list[list[float]]:
    """Compute mean AAindex features for every sequence.

    Returns a list of rows, each row being a ``list[float]`` of length
    ``len(accessions)``.
    """
    seqs = list(sequences)
    return [
        compute_aaindex_mean(seq, accessions, aaindex)
        for seq in tqdm(seqs, desc=desc, unit="seq")
    ]


# ── Config helpers ────────────────────────────────────────────────────────────

def load_aaindex_config(
    config: dict,
) -> tuple[str, list[str], dict[str, dict[str, float | None]]]:
    """Read ``config["extra_features"]`` and return ``(mode, accessions, aaindex_data)``.

    If the ``"extra_features"`` key is absent or mode is ``"builtin"``, returns
    ``("builtin", [], {})`` so callers can unconditionally destructure.
    """
    cfg = config.get("extra_features", {})
    mode = cfg.get("mode", "builtin")

    if mode not in VALID_MODES:
        raise ValueError(
            f"config[\"aaindex\"][\"mode\"] must be one of {VALID_MODES}, got {mode!r}"
        )

    if mode == "builtin":
        return "builtin", [], {}

    aaindex1_path = cfg.get("aaindex1_path")
    accessions_path = cfg.get("accessions_path")

    if not aaindex1_path:
        raise ValueError('config["extra_features"]["aaindex1_path"] is required when mode != "builtin"')
    if not accessions_path:
        raise ValueError('config["extra_features"]["accessions_path"] is required when mode != "builtin"')

    # Resolve relative to the config's base directory if possible
    from .configuration import resolve_pipeline_path
    aaindex1_resolved = resolve_pipeline_path(config, aaindex1_path)
    accessions_resolved = resolve_pipeline_path(config, accessions_path)

    accessions = load_selected_accessions(accessions_resolved)
    aaindex_data = load_aaindex1(aaindex1_resolved)

    # Keep only accessions that have no NA values
    valid = {acc for acc, table in aaindex_data.items() if None not in table.values()}
    accessions = [a for a in accessions if a in valid]

    return mode, accessions, aaindex_data


# ── Merging ───────────────────────────────────────────────────────────────────

def merge_features(
    base_features: list[list[float]],
    extra_features: list[list[float]],
    mode: str,
) -> list[list[float]]:
    """Combine built-in and AAindex features according to *mode*.

    Parameters
    ----------
    base_features:
        The original built-in feature matrix (n × 540).
    extra_features:
        The AAindex mean feature matrix (n × k).
    mode:
        ``"builtin"`` → return base_features unchanged.
        ``"combine"``  → return base + extra concatenated.
        ``"extra_only"`` → return extra_features only.
    """
    if mode == "builtin":
        return base_features
    if mode == "combine":
        return [b + e for b, e in zip(base_features, extra_features)]
    if mode == "extra_only":
        return extra_features
    raise ValueError(f"Unknown mode: {mode!r}")


def merge_single_feature_vector(
    base_row: list[float],
    extra_row: list[float],
    mode: str,
) -> list[float]:
    """Single-vector version of :func:`merge_features` for window-level inference."""
    if mode == "builtin":
        return base_row
    if mode == "combine":
        return base_row + extra_row
    if mode == "extra_only":
        return extra_row
    raise ValueError(f"Unknown mode: {mode!r}")
