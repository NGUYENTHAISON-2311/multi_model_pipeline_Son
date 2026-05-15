"""Feature extraction utilities matching Chloe's model inputs."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .defaults import AMINO_ACIDS, CLASSIFICATIONS, GROUP_LABELS

# Try to import the compiled Cython extension for accelerated inner loops.
# Falls back silently to the pure-Python versions if not yet compiled.
# Build with:  python setup_cython.py build_ext --inplace
try:
    from ._feature_fast import (
        aa_frequencies      as _fast_aa_freq,
        dipeptide_transitions as _fast_dipeptide,
        group_transitions   as _fast_group_trans,
    )
    _CYTHON = True
except ImportError:
    _CYTHON = False


def load_json_records(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_sequence_strings(path: str | Path, field_name: str = "sequence") -> list[str]:
    records = load_json_records(path)
    return [record[field_name] for record in records]


def load_sequences_from_file(path: str | Path, field_name: str = "sequence") -> list[str]:
    """Load sequences from a JSON or FASTA file, detected by file suffix."""
    suffix = Path(path).suffix.lower()
    if suffix in (".fasta", ".fa", ".fna", ".faa"):
        return [seq for _, seq in read_fasta_records(path)]
    return load_sequence_strings(path, field_name)


def read_fasta_records(path: str | Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    current_sequence: list[str] = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_sequence)))
                current_id = line[1:]
                current_sequence = []
                continue
            current_sequence.append(line)

    if current_id is not None:
        records.append((current_id, "".join(current_sequence)))

    return records


def compute_amino_acid_frequencies(sequence: str, amino_acids: Iterable[str] = AMINO_ACIDS) -> list[float]:
    if _CYTHON:
        return _fast_aa_freq(sequence, list(amino_acids) if not isinstance(amino_acids, list) else amino_acids)
    counts = Counter(sequence)
    length = len(sequence)
    if length == 0:
        return [0.0 for _ in amino_acids]
    return [counts.get(amino_acid, 0) / length for amino_acid in amino_acids]


def encode_sequence(sequence: str, classification: dict[str, str]) -> str:
    return "".join(classification.get(amino_acid, "X") for amino_acid in sequence)


def compute_group_frequencies(encoded_sequence: str, groups: Iterable[str]) -> list[float]:
    counts = Counter(encoded_sequence)
    length = len(encoded_sequence)
    if length == 0:
        return [0.0 for _ in groups]
    return [counts.get(group, 0) / length for group in groups]


def compute_cross_classification_transitions(
    sequence: str,
    encoded_sequences: tuple[str, str],
    group_pairs: Iterable[tuple[str, str]],
) -> list[float]:
    counts = Counter()
    length = len(sequence)
    if length < 2:
        return [0.0 for _ in group_pairs]
    for index in range(length - 1):
        from_group = encoded_sequences[0][index]
        to_group = encoded_sequences[1][index + 1]
        counts[(from_group, to_group)] += 1
    return [counts.get(pair, 0) / (length - 1) for pair in group_pairs]


def compute_tripeptide_transitions(
    sequence: str,
    encoded_sequences: tuple[str, str, str],
    group_triplets: Iterable[tuple[str, str, str]],
) -> list[float]:
    counts = Counter()
    length = len(sequence)
    if length < 3:
        return [0.0 for _ in group_triplets]
    for index in range(length - 2):
        triplet = (
            encoded_sequences[0][index],
            encoded_sequences[1][index + 1],
            encoded_sequences[2][index + 2],
        )
        if triplet in group_triplets:
            counts[triplet] += 1
    return [counts.get(triplet, 0) / (length - 2) for triplet in group_triplets]


def compute_average_iupred_scores_from_fasta(
    fasta_path: str | Path,
    iupred_script: str | Path,
    input_type: str = "long",
) -> list[float]:
    fasta_records = read_fasta_records(fasta_path)
    result = subprocess.run(
        [sys.executable, str(iupred_script), str(fasta_path), input_type],
        capture_output=True,
        text=True,
        check=True,
    )
    disorder_scores: list[float] = []
    for line in result.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) == 3:
            disorder_scores.append(float(parts[2]))

    average_scores: list[float] = []
    current_index = 0
    for _, sequence in fasta_records:
        next_index = current_index + len(sequence)
        sequence_scores = disorder_scores[current_index:next_index]
        average = sum(sequence_scores) / len(sequence_scores) if sequence_scores else 0.0
        average_scores.append(average)
        current_index = next_index

    return average_scores


def compute_average_iupred_scores_from_sequences(
    sequences: list[str],
    iupred_script: str | Path,
    input_type: str = "long",
) -> list[float]:
    """Write sequences to a temporary FASTA, run IUPred, and return per-sequence
    average disorder scores.  This keeps the sequence list and IUPred input in
    perfect sync regardless of whether the caller loaded from JSON or FASTA."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp:
        for index, sequence in enumerate(sequences):
            tmp.write(f">seq_{index}\n{sequence}\n")
        tmp_path = Path(tmp.name)
    try:
        return compute_average_iupred_scores_from_fasta(tmp_path, iupred_script, input_type)
    finally:
        tmp_path.unlink(missing_ok=True)


def _load_iupred_library(iupred_script: str | Path):
    module_directory = str(Path(iupred_script).resolve().parent)
    if module_directory not in sys.path:
        sys.path.append(module_directory)
    return importlib.import_module("iupred3_lib")


def compute_average_iupred_for_window(
    sequence_window: str,
    iupred_script: str | Path,
    input_type: str = "long",
) -> float:
    iupred3_lib = _load_iupred_library(iupred_script)
    scores, _ = iupred3_lib.iupred(sequence_window, input_type, smoothing=False)
    return sum(scores) / len(scores) if scores else 0.0


def extract_sequence_features(
    sequences: list[str],
    average_iupred_scores: list[float],
    desc: str = "Extracting features",
) -> list[list[float]]:
    """Build 540-feature vectors: length + AA freq + dipeptide AA→AA + group freq
    + intra-class group transitions + IUPred."""
    if len(sequences) != len(average_iupred_scores):
        raise ValueError("Sequence count and IUPred score count do not match.")

    features: list[list[float]] = []
    seq_iter = tqdm(zip(sequences, average_iupred_scores), total=len(sequences), desc=desc, unit="seq")
    for sequence, iupred_score in seq_iter:
        row: list[float] = []

        # 1) Sequence length
        row.append(float(len(sequence)))

        # 2) AA frequencies (20)
        row.extend(compute_amino_acid_frequencies(sequence, AMINO_ACIDS))

        # 3) Dipeptide AA→AA transitions (400)
        row.extend(compute_dipeptide_aa_transitions(sequence, AMINO_ACIDS))

        # 4) Group frequencies (17)
        encoded_sequences = [encode_sequence(sequence, cls) for cls in CLASSIFICATIONS]
        for encoded_sequence, groups in zip(encoded_sequences, GROUP_LABELS):
            row.extend(compute_group_frequencies(encoded_sequence, groups))

        # 5) Intra-class group transitions (101 = 4² + 6² + 7²)
        for cls_index in range(len(CLASSIFICATIONS)):
            enc = encoded_sequences[cls_index]
            if _CYTHON:
                row.extend(_fast_group_trans(enc, GROUP_LABELS[cls_index]))
            else:
                group_pairs = [
                    (g_from, g_to)
                    for g_from in GROUP_LABELS[cls_index]
                    for g_to in GROUP_LABELS[cls_index]
                ]
                row.extend(
                    compute_cross_classification_transitions(
                        sequence,
                        (enc, enc),
                        group_pairs,
                    )
                )

        # 6) IUPred
        row.append(iupred_score)
        features.append(row)

    return features


def extract_window_features(
    sequence_window: str,
    iupred_script: str | Path,
    input_type: str = "long",
) -> list[float]:
    """Build a 540-feature vector for a single window: length + AA freq +
    dipeptide AA→AA + group freq + intra-class group transitions + IUPred."""
    row: list[float] = []

    # 1) Sequence length
    row.append(float(len(sequence_window)))

    # 2) AA frequencies (20)
    row.extend(compute_amino_acid_frequencies(sequence_window, AMINO_ACIDS))

    # 3) Dipeptide AA→AA transitions (400)
    row.extend(compute_dipeptide_aa_transitions(sequence_window, AMINO_ACIDS))

    # 4) Group frequencies (17)
    encoded_sequences = [encode_sequence(sequence_window, cls) for cls in CLASSIFICATIONS]
    for encoded, groups in zip(encoded_sequences, GROUP_LABELS):
        row.extend(compute_group_frequencies(encoded, groups))

    # 5) Intra-class group transitions (101 = 4² + 6² + 7²)
    for cls_index in range(len(CLASSIFICATIONS)):
        group_pairs = [
            (g_from, g_to)
            for g_from in GROUP_LABELS[cls_index]
            for g_to in GROUP_LABELS[cls_index]
        ]
        row.extend(
            compute_cross_classification_transitions(
                sequence_window,
                (encoded_sequences[cls_index], encoded_sequences[cls_index]),
                group_pairs,
            )
        )

    # 6) IUPred
    row.append(compute_average_iupred_for_window(sequence_window, iupred_script, input_type=input_type))
    return row


def compute_dipeptide_aa_transitions(
    sequence: str,
    amino_acids: list[str] = AMINO_ACIDS,
) -> list[float]:
    """Return row-normalised AA→AA transition frequencies as a flat list (20×20 = 400)."""
    if _CYTHON:
        return _fast_dipeptide(sequence, amino_acids)
    idx = {aa: i for i, aa in enumerate(amino_acids)}
    n = len(amino_acids)
    mat = [[0.0] * n for _ in range(n)]
    for a, b in zip(sequence[:-1], sequence[1:]):
        if a in idx and b in idx:
            mat[idx[a]][idx[b]] += 1
    result: list[float] = []
    for row in mat:
        row_sum = sum(row) or 1.0
        result.extend(v / row_sum for v in row)
    return result


def extract_window_features_v2(
    sequence_window: str,
    iupred_script: str | Path,
    input_type: str = "long",
) -> list[float]:
    """Build a 539-feature vector for the v2 model.

    Feature groups:
      1. AA frequencies                  — 20
      2. Group frequencies (3 cls)       — 17  (4 + 6 + 7)
      3. Intra-class group transitions   — 101 (4² + 6² + 7²)
      4. Dipeptide AA→AA transitions     — 400 (20×20)
      5. IUPred score                    —   1
    """
    row: list[float] = []

    # 1) AA frequencies
    row.extend(compute_amino_acid_frequencies(sequence_window, AMINO_ACIDS))

    # 2) Group frequencies
    encoded_sequences = [encode_sequence(sequence_window, cls) for cls in CLASSIFICATIONS]
    for encoded, groups in zip(encoded_sequences, GROUP_LABELS):
        row.extend(compute_group_frequencies(encoded, groups))

    # 3) Intra-class group transitions only (same classification index)
    for cls_index in range(len(CLASSIFICATIONS)):
        group_pairs = [
            (g_from, g_to)
            for g_from in GROUP_LABELS[cls_index]
            for g_to in GROUP_LABELS[cls_index]
        ]
        row.extend(
            compute_cross_classification_transitions(
                sequence_window,
                (encoded_sequences[cls_index], encoded_sequences[cls_index]),
                group_pairs,
            )
        )

    # 4) Dipeptide AA→AA transitions
    row.extend(compute_dipeptide_aa_transitions(sequence_window, AMINO_ACIDS))

    # 5) IUPred
    row.append(compute_average_iupred_for_window(sequence_window, iupred_script, input_type=input_type))
    return row