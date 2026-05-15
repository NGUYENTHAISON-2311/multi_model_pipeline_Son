"""Shared constants from Chloe's original prediction scripts."""

AMINO_ACIDS = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]

CLASSIFICATIONS = [
    {
        "R": "A", "K": "A", "E": "A", "D": "A", "Q": "A", "N": "A",
        "G": "B", "A": "B", "S": "B", "T": "B", "P": "B", "H": "B", "Y": "B",
        "C": "C", "V": "C", "L": "C", "I": "C", "M": "C", "F": "C", "W": "C",
        "X": "X",
    },
    {
        "H": "A", "N": "A", "T": "A", "Q": "A", "C": "A", "S": "A",
        "K": "B", "R": "B", "E": "B", "D": "B",
        "I": "C", "L": "C", "M": "C", "V": "C", "W": "C", "Y": "C", "F": "C", "A": "C",
        "G": "G", "P": "P", "X": "X",
    },
    {
        "H": "A", "T": "A", "C": "A", "S": "A",
        "K": "B", "R": "B", "E": "B", "D": "B",
        "I": "C", "L": "C", "M": "C", "V": "C", "W": "C", "Y": "C", "F": "C", "A": "C",
        "Q": "D", "N": "D", "G": "G", "P": "P", "X": "X",
    },
]

GROUP_LABELS = [
    ["A", "B", "C", "X"],
    ["A", "B", "C", "G", "P", "X"],
    ["A", "B", "C", "D", "G", "P", "X"],
]