"""Pre-selected negative sequences for deterministic padding in the padded benchmark.

Each dictionary maps a single amino-acid letter to the **longest** disordered
sequence from ``AbTools_train_test_dataset/disordered_regions_clustered.json``
that starts (RIGHT_PADDING_SEQS) or ends (LEFT_PADDING_SEQS) with that residue.

*  ``RIGHT_PADDING_SEQS[X]`` — appended **after** a target sequence whose last
   residue is X.  The padding sequence starts with X so the junction reads
   …X | X… (matching terminal residue).
*  ``LEFT_PADDING_SEQS[X]``  — prepended **before** a target sequence whose
   first residue is X.  The padding sequence ends with X so the junction
   reads …X | X… (matching terminal residue).

When the required pad length exceeds the stored sequence length, multiple
copies are concatenated (sharing junction residues) by :func:`build_padding`.
"""

from __future__ import annotations

# -- Right padding: sequence STARTS with amino acid key -----------------------
#    Used to pad the right end of a target sequence whose last residue = key.
RIGHT_PADDING_SEQS: dict[str, str] = {
    "A": "ADEETIRAKMVELRATAREQIISEI",
    "R": "RVGDKEKASETEENGSDSFMHSMD",
    "N": "NLSSDSSLSSPSALNSPGIEGLSRR",
    "D": "DMRPEIWIAQELRRIGDEFNAYYAR",
    "C": "CFQWQRNMRKVRGPPVSCIKRD",
    "Q": "QQSKVAPSSAASRPVLSSRSDQSQK",
    "E": "ECTLQENPFFSQPGAPILQCMGCCF",
    "G": "GAEDAQDDLVPSIQDDGCESGACKI",
    "H": "HPESSQLFAKLLQKMTDLRQIVTEH",
    "I": "INFRPEDRIKRGLMMLKRAKGVWI",
    "L": "LKNPLRSVDIETKEEMKAGKERTDI",
    "K": "KADMNTFPNFTFEDPKFEVVEKPQS",
    "M": "MLKAAAKRPELSGKNTISNNSDMAE",
    "F": "FLGALLKIGAKLLPSVVGLFKKKQQ",
    "P": "PAENDKPHDVEINKIISTTASKTET",
    "S": "SNPLIRKAMGMDTEGGGKDEKMSGL",
    "T": "TQVDSSSTDQTEPNPGESDTSEDSE",
    "W": "WTPKLDVNTSVDEFFQGCFL",
    "Y": "YEEKNKEHKRPTGPPAKKAISELP",
    "V": "VLLGTLAASTPGCDTSNQAKAQRPD",
}

# -- Left padding: sequence ENDS with amino acid key -------------------------
#    Used to pad the left end of a target sequence whose first residue = key.
LEFT_PADDING_SEQS: dict[str, str] = {
    "A": "MAAEDRQPADIVEGATAGDVEEEVA",
    "R": "DMRPEIWIAQELRRIGDEFNAYYAR",
    "N": "DNAISGGSNEGSTDTTSTHTTNTQN",
    "D": "AEEELETPTPTQRGEAESRGDGLVD",
    "C": "MAYSVQKSRLAKVAGVSLVLLLAAC",
    "Q": "SSSRRSPGEEVLRMPGDENQQQESQ",
    "E": "MLKAAAKRPELSGKNTISNNSDMAE",
    "G": "LKTEAESYEGLLAPSLIPKNWPDQG",
    "H": "HPESSQLFAKLLQKMTDLRQIVTEH",
    "I": "ADEETIRAKMVELRATAREQIISEI",
    "L": "SNPLIRKAMGMDTEGGGKDEKMSGL",
    "K": "QQSKVAPSSAASRPVLSSRSDQSQK",
    "M": "VGGPGHKARVLAEAMSQVTNSATIM",
    "F": "ECTLQENPFFSQPGAPILQCMGCCF",
    "P": "MGGKWSKSSVIGWPAVRERMRRAEP",
    "S": "MPPAQKTVKKAAPKDAKATKVVKVS",
    "T": "PAENDKPHDVEINKIISTTASKTET",
    "W": "ARIFSPHEPILEGSRSYTQAGVQW",
    "Y": "ETGTAEKMPSTSRPTAPSSEKGGNY",
    "V": "SSLSALSLDEPFIQKDVELRIMPPV",
}


def build_padding(junction_residue: str, pad_length: int, side: str) -> str:
    """Build a padding string of exactly *pad_length* residues.

    Parameters
    ----------
    junction_residue : str
        The amino acid at the junction point (first residue of the target
        sequence for left padding, last residue for right padding).
    pad_length : int
        Number of residues needed.
    side : ``"left"`` or ``"right"``
        Which padding pool to use.

    If the stored sequence is shorter than *pad_length*, it is repeated
    (concatenated at matching endpoints) until the required length is reached,
    then trimmed.

    Returns the appropriate *pad_length* tail (for left) or head (for right)
    of the constructed string.
    """
    pool = LEFT_PADDING_SEQS if side == "left" else RIGHT_PADDING_SEQS
    base = pool.get(junction_residue.upper())
    if base is None:
        # Unknown residue (e.g. X, U, …) → fall back to repeating the residue
        return junction_residue * pad_length

    if len(base) >= pad_length:
        # Enough in one copy
        if side == "left":
            return base[-pad_length:]      # take the LAST pad_length chars
        else:
            return base[:pad_length]       # take the FIRST pad_length chars

    # Need to concatenate copies.  Each copy shares the junction residue
    # with the previous one, so effective new chars per concat = len(base) - 1.
    # Example: "ABCD" concat "DEFG" → "ABCDEFG" (D shared).
    built = base
    while len(built) < pad_length:
        # The last char of current 'built' should match the first char of the
        # next copy (for right) or vice versa.  The stored sequences already
        # guarantee the first/last char equals junction_residue.
        # Right: built ends with some char; next copy starts with junction_residue.
        # Left:  built starts with some char; next copy ends with junction_residue.
        if side == "right":
            # built = "...X", base = "X...", overlap on X → drop first char of base
            built += base[1:]
        else:
            # built = "X...", base = "...X", overlap on X → drop last char of base
            built = base[:-1] + built

    if side == "left":
        return built[-pad_length:]
    else:
        return built[:pad_length]
