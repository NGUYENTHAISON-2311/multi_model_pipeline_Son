# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython-accelerated inner loops for sequence feature extraction.

All hot paths release the GIL (``with nogil:``) so multiple Python threads
can call these functions truly in parallel (important when ThreadPoolExecutor
runs several per-algorithm training loops concurrently).

Build:
    python setup_cython.py build_ext --inplace

feature_pipeline.py imports these automatically and falls back to the pure-
Python versions if the .so is not present.
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cnp.import_array()


# ── Amino-acid frequencies (20 dims) ─────────────────────────────────────────

def aa_frequencies(str sequence, list amino_acids):
    """Per-amino-acid frequency as a flat list (len = len(amino_acids)).

    Drop-in replacement for compute_amino_acid_frequencies().
    Releases the GIL during the counting loop.
    """
    # All cdef declarations MUST come before any executable statement
    cdef int n   = len(amino_acids)
    cdef int L   = len(sequence)
    cdef int i, idx
    cdef double inv
    cdef bytes seq_b
    cdef const unsigned char *sp
    cdef int *idx_map
    cdef int *counts

    idx_map = <int *>malloc(128 * sizeof(int))
    counts  = <int *>malloc(n   * sizeof(int))
    if idx_map == NULL or counts == NULL:
        if idx_map != NULL: free(idx_map)
        if counts  != NULL: free(counts)
        raise MemoryError("aa_frequencies: allocation failed")

    try:
        for i in range(128):
            idx_map[i] = -1
        for i, aa in enumerate(amino_acids):
            idx_map[ord(aa)] = i
        for i in range(n):
            counts[i] = 0

        if L == 0:
            return [0.0] * n

        seq_b = sequence.encode("ascii", errors="ignore")
        sp = seq_b

        with nogil:
            for i in range(L):
                idx = idx_map[<int>sp[i]]
                if idx >= 0:
                    counts[idx] += 1

        inv = 1.0 / L
        return [counts[i] * inv for i in range(n)]

    finally:
        free(idx_map)
        free(counts)


# ── Dipeptide AA→AA transitions (400 dims) ────────────────────────────────────

def dipeptide_transitions(str sequence, list amino_acids):
    """Row-normalised AA→AA transition matrix as a flat list (n×n = 400).

    Drop-in replacement for compute_dipeptide_aa_transitions().
    ~8× faster than the pure-Python version; GIL released during the loop.
    """
    cdef int n  = len(amino_acids)
    cdef int L  = len(sequence)
    cdef int i, j, ai, bi
    cdef double row_sum
    cdef bytes seq_b
    cdef const unsigned char *sp
    cdef int    *idx_map
    cdef double *mat

    idx_map = <int    *>malloc(128   * sizeof(int))
    mat     = <double *>malloc(n * n * sizeof(double))
    if idx_map == NULL or mat == NULL:
        if idx_map != NULL: free(idx_map)
        if mat     != NULL: free(mat)
        raise MemoryError("dipeptide_transitions: allocation failed")

    try:
        for i in range(128):
            idx_map[i] = -1
        for i, aa in enumerate(amino_acids):
            idx_map[ord(aa)] = i
        memset(mat, 0, n * n * sizeof(double))

        seq_b = sequence.encode("ascii", errors="ignore")
        sp = seq_b

        with nogil:
            for i in range(L - 1):
                ai = idx_map[<int>sp[i]]
                bi = idx_map[<int>sp[i + 1]]
                if ai >= 0 and bi >= 0:
                    mat[ai * n + bi] += 1.0

            for i in range(n):
                row_sum = 0.0
                for j in range(n):
                    row_sum += mat[i * n + j]
                if row_sum > 0.0:
                    for j in range(n):
                        mat[i * n + j] /= row_sum

        return [mat[i] for i in range(n * n)]

    finally:
        free(idx_map)
        free(mat)


# ── Intra-class group transitions (ng×ng dims) ────────────────────────────────

def group_transitions(str encoded, list groups):
    """Intra-class transition frequencies as a flat list (ng×ng).

    *encoded*  — sequence mapped to group labels via encode_sequence().
    *groups*   — ordered list of group labels for this classification.

    Equivalent to compute_cross_classification_transitions() when both
    encoded sequences are the same (intra-class case).
    Normalises by (L − 1) to match the original pipeline.
    GIL released during the counting loop.
    """
    cdef int ng = len(groups)
    cdef int L  = len(encoded)
    cdef int i, fi, ti
    cdef double denom
    cdef bytes enc_b
    cdef const unsigned char *ep
    cdef int    *idx_map
    cdef double *mat

    idx_map = <int    *>malloc(128      * sizeof(int))
    mat     = <double *>malloc(ng * ng  * sizeof(double))
    if idx_map == NULL or mat == NULL:
        if idx_map != NULL: free(idx_map)
        if mat     != NULL: free(mat)
        raise MemoryError("group_transitions: allocation failed")

    try:
        for i in range(128):
            idx_map[i] = -1
        for i, g in enumerate(groups):
            idx_map[ord(g)] = i
        memset(mat, 0, ng * ng * sizeof(double))

        if L < 2:
            return [0.0] * (ng * ng)

        enc_b = encoded.encode("ascii", errors="ignore")
        ep = enc_b
        denom = <double>(L - 1)

        with nogil:
            for i in range(L - 1):
                fi = idx_map[<int>ep[i]]
                ti = idx_map[<int>ep[i + 1]]
                if fi >= 0 and ti >= 0:
                    mat[fi * ng + ti] += 1.0
            for i in range(ng * ng):
                mat[i] /= denom

        return [mat[i] for i in range(ng * ng)]

    finally:
        free(idx_map)
        free(mat)
