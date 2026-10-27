"""Shared helpers of the benchmark-B readers: binary symmetric snapshots on a fixed vertex set, file hashes."""
import hashlib

import numpy as np
import scipy.sparse as sp


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def to_csr(n, i, j):
    """Undirected simple graph on n vertices from edge endpoints (self-loops and duplicates dropped)."""
    i, j = np.asarray(i, np.int64), np.asarray(j, np.int64)
    keep = i != j
    a, b = np.minimum(i[keep], j[keep]), np.maximum(i[keep], j[keep])
    if a.size:
        key = np.unique(a * n + b)
        a, b = key // n, key % n
    A = sp.csr_matrix((np.ones(2 * a.size, np.int8), (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    A.sum_duplicates()
    A.sort_indices()
    return A


def check(A):
    assert A.shape[0] == A.shape[1]
    assert (A != A.T).nnz == 0, "not symmetric"
    assert A.diagonal().sum() == 0, "self-loop"
    assert A.data.size == 0 or A.data.max() == 1, "multi-edge"


def pack(T, snap, a, b, w):
    """One row per undirected pair (a != b) and snapshot -> both orientations sorted by (snapshot, row, col).

    Returns offsets (T + 1), rows, cols (int32) and weights (float32) of the stacked CSR blocks.
    """
    assert np.all(a != b)
    s2, r, c, w2 = np.r_[snap, snap], np.r_[a, b], np.r_[b, a], np.r_[w, w]
    o = np.lexsort((c, r, s2))
    s2, r, c, w2 = s2[o], r[o], c[o], w2[o]
    offsets = np.zeros(T + 1, np.int64)
    np.cumsum(np.bincount(s2, minlength=T), out=offsets[1:])
    return offsets, r.astype(np.int32), c.astype(np.int32), w2.astype(np.float32)


def from_mats(mats):
    """CSR list -> (offsets, rows, cols) of the stacked blocks."""
    offsets = np.r_[0, np.cumsum([A.nnz for A in mats])].astype(np.int64)
    rows = [np.repeat(np.arange(A.shape[0], dtype=np.int32), np.diff(A.indptr)) for A in mats]
    cols = [A.indices.astype(np.int32) for A in mats]
    cat = lambda x: np.concatenate(x) if offsets[-1] else np.zeros(0, np.int32)
    return offsets, cat(rows), cat(cols)
