"""Generic feature loader and dispatcher for the multi-model pipeline.

Each feature set is described by a JSON file with the following schema:

  {
    "type":        "builtin" | "lookup_table",
    "level":       "per_sequence" | "per_residue",
    "format":      "numeric" | "categorical",
    "aggregation": ["mean"] | ["std"] | ["mean", "std"] | "frequency",
    "description": "...",   // optional
    ... type-specific fields ...
  }

Type-specific required fields
------------------------------
builtin
    No extra fields.  Produces the original 540-dim vector:
    length(1) + AA freq(20) + dipeptide(400) + group freq(17) +
    intra-class transitions(101) + IUPred(1).

lookup_table  (single scale)
    table           : {AA: value}       for format="numeric"
                      {AA: class_label} for format="categorical"
    aggregation     : list of ["mean","std"] for numeric (default: ["mean"])
                      "frequency" for categorical (one fraction per class)
    → Produces len(aggregation) features (numeric) or len(classes) features (categorical).

lookup_table  (multi-scale, e.g. AAindex)
    tables          : {name: {AA: value}, ...}  — ordered dict of named scales
    aggregation     : list of ["mean","std"] for numeric (default: ["mean"])
    → Produces len(tables) * len(aggregation) features in tables insertion order.

Aggregation rules
-----------------
per_residue + numeric    : aggregation list — ["mean"] → 1 float per entry,
                           ["mean","std"] → 2 floats per entry
per_residue + categorical: aggregation="frequency" → fraction per class label
per_sequence + numeric   : no aggregation (values are already scalar)

Usage
-----
    specs = load_and_prepare_feature_specs(config, cli_paths)
    # training:
    features = compute_sequence_feature_matrix(sequences, specs, iupred_scores)
    # benchmark (live IUPred):
    row = compute_window_feature_vector_live(window, specs, iupred_script, ...)
    # benchmark_adaptive (pre-cached IUPred):
    #   handle builtin inline, then append:
    extra = compute_extra_for_window(window, specs)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from .configuration import resolve_pipeline_path

# ── JSON loading & validation ─────────────────────────────────────────────────

def load_feature_spec(path: str | Path) -> dict:
    """Load a feature spec JSON file.

    Minimal schema: must be a JSON object with either ``"tables"`` (multi-scale
    lookup) or ``"table"`` (single-scale lookup) key, plus an optional
    ``"aggregation"`` list.  No other metadata fields are required.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        spec = json.load(fh)
    if not isinstance(spec, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(spec).__name__}")
    if "tables" not in spec and "table" not in spec:
        raise ValueError(f"{path}: must contain 'tables' (multi-scale) or 'table' (single-scale) key")
    spec["_source_path"] = str(path)
    return spec


# ── Spec preparation (IO happens once here) ───────────────────────────────────

def prepare_feature_specs(specs: list[dict], config: dict) -> list[dict]:
    """Enrich specs with loaded data.

    Adds private ``_``-prefixed keys to each spec with pre-loaded data so
    that per-sequence/per-window computation does no repeated IO.

    Parameters
    ----------
    specs:
        Output of :func:`load_feature_spec` for each feature file.
    config:
        Pipeline config dict (used for path resolution).
    """
    prepared: list[dict] = []
    for spec in specs:
        s = dict(spec)

        if not spec.get("_builtin"):
            # Multi-scale lookup (e.g. 304 AAindex tables).
            # Drop any scale that has None values.
            valid: list[tuple[str, dict[str, float]]] = [
                (name, tbl)
                for name, tbl in spec["tables"].items()
                if None not in tbl.values()
            ]
            s["_tables"] = valid
        elif "table" in spec:
            table = spec["table"]
            s["_table"] = {aa: v for aa, v in table.items()}
            # categorical: classes inferred from unique values
            if all(isinstance(v, str) for v in table.values()):
                s["_classes"] = sorted(set(table.values()))

        prepared.append(s)
    return prepared


def load_and_prepare_feature_specs(
    config: dict,
    cli_paths: list[str] | None = None,
) -> list[dict]:
    """Resolve feature file paths, load, validate, and prepare all specs.

    Priority: CLI paths > ``config["feature_files"]`` > default (builtin only).

    Each entry in the paths list is either:
      - the special keyword ``"builtin"``  → inject the built-in 540-dim spec
      - a file path to a feature JSON file → loaded from disk

    Examples
    --------
    ``--features builtin``                          → builtin only (same as default)
    ``--features data/aaindex_features.json``       → aaindex only
    ``--features builtin data/aaindex_features.json`` → builtin + aaindex
    """
    paths = cli_paths or config.get("feature_files", ["builtin"])
    specs: list[dict] = []
    for p in paths:
        if str(p).strip() == "builtin":
            specs.append({"_builtin": True})
        else:
            specs.append(load_feature_spec(resolve_pipeline_path(config, p)))
    return prepare_feature_specs(specs, config)


def resolve_feature_paths_for_model(
    model_path: "str | Path | None",
    cli_feature_paths: "list[str] | None",
) -> "list[str] | None":
    """Resolve which feature paths to use for a benchmark run.

    Priority: ``cli_feature_paths`` > model's ``metadata.json`` > ``None``
    (callers then fall back to ``config["feature_files"]``).
    """
    if cli_feature_paths:
        return cli_feature_paths
    if model_path is None:
        return None
    meta = Path(model_path).parent / "metadata.json"
    if meta.exists():
        with meta.open(encoding="utf-8") as fh:
            data = json.load(fh)
        ff = data.get("feature_files")
        if ff:
            return ff
    return None


# ── Per-residue aggregation helpers ──────────────────────────────────────────

def _agg_numeric(values: list[float], aggregation: list[str]) -> list[float]:
    """Compute requested aggregations of a numeric value list."""
    if not values:
        return [0.0] * len(aggregation)
    n = len(values)
    mean = sum(values) / n
    result: list[float] = []
    for agg in aggregation:
        if agg == "mean":
            result.append(mean)
        elif agg == "std":
            result.append(
                (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
                if n >= 2 else 0.0
            )
    return result


def _compute_extra_row(sequence: str, spec: dict) -> list[float]:
    """Compute extra feature values for *one* sequence from a non-builtin spec."""
    row: list[float] = []

    if "_tables" in spec:
        # Multi-scale numeric: one aggregation per named table.
        agg: list[str] = spec.get("aggregation", ["mean"])
        for _name, table in spec["_tables"]:
            vals = [
                float(table[aa])
                for aa in sequence
                if aa in table and table[aa] is not None
            ]
            row.extend(_agg_numeric(vals, agg))
    elif "_table" in spec:
        table = spec["_table"]
        if "_classes" in spec:
            # Categorical single-scale.
            classes: list[str] = spec["_classes"]
            counts = Counter(table.get(aa) for aa in sequence if aa in table)
            total = sum(counts.values()) or 1
            row.extend(counts.get(cls, 0) / total for cls in classes)
        else:
            # Numeric single-scale.
            agg = spec.get("aggregation", ["mean"])
            vals = [float(table[aa]) for aa in sequence if aa in table]
            row.extend(_agg_numeric(vals, agg))

    return row


# ── Sequence-level computation ────────────────────────────────────────────────

def compute_sequence_feature_matrix(
    sequences: list[str],
    feature_specs: list[dict],
    iupred_scores: list[float],
    desc: str = "Extracting features",
) -> list[list[float]]:
    """Build combined feature matrix for a list of sequences.

    Parameters
    ----------
    sequences:
        List of amino acid strings.
    feature_specs:
        Output of :func:`prepare_feature_specs` (data already loaded).
    iupred_scores:
        Pre-computed per-sequence average IUPred scores (used by builtin).
    desc:
        tqdm progress bar description.

    Returns
    -------
    list of float rows, one per sequence.  Length of each row depends on
    which feature specs are active.
    """
    from .feature_pipeline import extract_sequence_features

    has_builtin = any(s.get("_builtin") for s in feature_specs)
    extra_specs = [s for s in feature_specs if not s.get("_builtin")]

    if has_builtin:
        builtin_rows = extract_sequence_features(sequences, iupred_scores, desc=desc)
    else:
        builtin_rows = [[] for _ in sequences]

    if not extra_specs:
        return builtin_rows

    combined: list[list[float]] = []
    for i, seq in enumerate(tqdm(sequences, desc=f"{desc} [extra]", unit="seq", leave=False)):
        row = list(builtin_rows[i])
        for spec in extra_specs:
            row.extend(_compute_extra_row(seq, spec))
        combined.append(row)
    return combined


# ── Window-level computation ──────────────────────────────────────────────────

def compute_window_feature_vector_live(
    window: str,
    feature_specs: list[dict],
    iupred_script,
    input_type: str = "long",
    use_v2: bool = False,
) -> list[float]:
    """Build feature vector for a single window, calling IUPred live.

    Used by :mod:`benchmark_pipeline` and :mod:`benchmark_pipeline_new`.
    """
    if use_v2:
        from .feature_pipeline import extract_window_features_v2
        builtin_row: list[float] = extract_window_features_v2(
            window, iupred_script, input_type=input_type
        )
    else:
        from .feature_pipeline import extract_window_features
        builtin_row = extract_window_features(window, iupred_script, input_type=input_type)

    has_builtin = any(s.get("_builtin") for s in feature_specs)
    row: list[float] = list(builtin_row) if has_builtin else []

    for spec in feature_specs:
        if not spec.get("_builtin"):
            row.extend(_compute_extra_row(window, spec))

    return row


def compute_extra_for_window(window: str, feature_specs: list[dict]) -> list[float]:
    """Return only the non-builtin extra features for a single window.

    Used by :mod:`benchmark_pipeline_adaptive` where the builtin features are
    computed separately via the cached IUPred path.
    """
    row: list[float] = []
    for spec in feature_specs:
        if not spec.get("_builtin"):
            row.extend(_compute_extra_row(window, spec))
    return row
