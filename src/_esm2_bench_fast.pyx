# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython-accelerated inner loops for the ESM2 sliding-window benchmark.

Two core operations:

pool_windows_mean / pool_windows_max
    Given the (L_padded, H) per-residue ESM2 embedding matrix, produce the
    (N_windows, H) pooled feature matrix for every window in one C pass.
    pool_windows_mean uses a sliding running-sum (O(N·H)) instead of
    re-summing each window from scratch (O(N·H·W)).

accumulate_scores
    Given the (N_windows,) model scores and the window geometry, accumulate
    per-residue mean scores into a (seq_len,) array.

Both functions release the GIL (``with nogil:``) so they are safe to call
from multiple Python threads concurrently.

Build:
    python setup_cython.py build_ext --inplace
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

cnp.import_array()


# ── Sliding-window mean pooling (O(N·H) via running sum) ─────────────────────

def pool_windows_mean(
    cnp.ndarray[cnp.float32_t, ndim=2] residue_embs,
    int window_size,
):
    """Mean-pool a sliding window over per-residue ESM2 embeddings.

    Uses a running-sum so each output row costs O(H) instead of O(H·W).

    Parameters
    ----------
    residue_embs : (L, H) float32
        Per-residue hidden states (includes padding if any).
    window_size : int

    Returns
    -------
    (N_windows, H) float32   where N_windows = L - window_size + 1
    """
    cdef int L = residue_embs.shape[0]
    cdef int H = residue_embs.shape[1]
    cdef int N = L - window_size + 1
    cdef int i, j
    cdef float inv
    cdef float *run_sum
    cdef float[:, :] emb = residue_embs
    cdef cnp.ndarray[cnp.float32_t, ndim=2] out
    cdef float[:, :] o

    if N <= 0:
        return np.zeros((0, H), dtype=np.float32)

    out = np.empty((N, H), dtype=np.float32)
    o   = out
    inv = 1.0 / window_size

    run_sum = <float *>malloc(H * sizeof(float))
    if run_sum == NULL:
        raise MemoryError("pool_windows_mean: allocation failed")

    try:
        with nogil:
            # Initialise running sum for first window
            for j in range(H):
                run_sum[j] = 0.0
                for i in range(window_size):
                    run_sum[j] += emb[i, j]
                o[0, j] = run_sum[j] * inv

            # Slide: add incoming residue, subtract outgoing residue
            for i in range(1, N):
                for j in range(H):
                    run_sum[j] += emb[i + window_size - 1, j] - emb[i - 1, j]
                    o[i, j] = run_sum[j] * inv
    finally:
        free(run_sum)

    return out


# ── Sliding-window max pooling (O(N·H·W)) ────────────────────────────────────

def pool_windows_max(
    cnp.ndarray[cnp.float32_t, ndim=2] residue_embs,
    int window_size,
):
    """Max-pool a sliding window over per-residue ESM2 embeddings.

    Parameters / Returns: same as :func:`pool_windows_mean`.
    """
    cdef int L = residue_embs.shape[0]
    cdef int H = residue_embs.shape[1]
    cdef int N = L - window_size + 1
    cdef int i, j, k
    cdef float acc, val
    cdef float[:, :] emb = residue_embs
    cdef cnp.ndarray[cnp.float32_t, ndim=2] out
    cdef float[:, :] o

    if N <= 0:
        return np.zeros((0, H), dtype=np.float32)

    out = np.empty((N, H), dtype=np.float32)
    o   = out

    with nogil:
        for i in range(N):
            for j in range(H):
                acc = emb[i, j]
                for k in range(1, window_size):
                    val = emb[i + k, j]
                    if val > acc:
                        acc = val
                o[i, j] = acc

    return out


# ── Per-residue score accumulation ───────────────────────────────────────────

def accumulate_scores(
    cnp.ndarray[cnp.float64_t, ndim=1] window_scores,
    int window_size,
    int seq_len,
    int left_pad,
):
    """Map per-window scores back to per-residue mean scores.

    Window *i* (0-indexed) covers padded positions [i, i+window_size-1],
    which correspond to original residue positions
    [i - left_pad, i - left_pad + window_size - 1] (clamped to [0, seq_len-1]).

    For the standard padding scheme (left_pad = window_size - 1) every
    residue is covered by exactly *window_size* windows.

    Parameters
    ----------
    window_scores : (N_windows,) float64
    window_size   : int
    seq_len       : int   original sequence length
    left_pad      : int   number of padding residues prepended (= window_size - 1
                          for standard full-coverage padding)

    Returns
    -------
    (seq_len,) float64   per-residue mean score (0.5 where no coverage)
    """
    cdef int N = window_scores.shape[0]
    cdef int i, r, orig_start, orig_end
    cdef double[:] ws = window_scores
    cdef cnp.ndarray[cnp.float64_t, ndim=1] score_sum = np.zeros(seq_len, dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t,   ndim=1] count     = np.zeros(seq_len, dtype=np.int32)
    cdef double[:] ss  = score_sum
    cdef int[:]    cnt = count
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result
    cdef double[:] res

    with nogil:
        for i in range(N):
            orig_start = i - left_pad
            if orig_start < 0:
                orig_start = 0
            orig_end = i - left_pad + window_size - 1
            if orig_end >= seq_len:
                orig_end = seq_len - 1
            for r in range(orig_start, orig_end + 1):
                ss[r]  += ws[i]
                cnt[r] += 1

    result = np.empty(seq_len, dtype=np.float64)
    res = result
    with nogil:
        for r in range(seq_len):
            if cnt[r] > 0:
                res[r] = ss[r] / cnt[r]
            else:
                res[r] = 0.5   # neutral fallback — no window covered this position

    return result
