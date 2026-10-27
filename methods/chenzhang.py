"""Chen-Zhang graph-based two-sample statistic (Chen & Zhang, Ann. Statist. 2015) at the fixed split of the window
rule (identity-aware, online, no training).

The N = w + w_ref snapshots before t are the points; their distance is the Hamming distance of the edge sets,
|E_i| + |E_j| - 2 |E_i & E_j|. G is the minimum spanning tree of these distances and R the number of its edges that
join a test snapshot (the last w) to a reference snapshot. Under the permutation null (n1 = w test labels placed at
random), with p1 = 2 n1 n2 / (N (N - 1)), p2 = 4 n1 (n1 - 1) n2 (n2 - 1) / (N (N - 1) (N - 2) (N - 3)) and
|G_i| the degree of point i in G,

    E R = p1 |G|,   Var R = p2 |G| + (p1 / 2 - p2) sum_i |G_i|^2 + (p2 - p1^2) |G|^2,

and D_t = -(R - E R) / sd(R): a change makes the two windows far apart, so few MST edges cross.

    D = scores(keys, seg)      keys[t] = sorted unique int64 pair ids of snapshot t
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree

from methods.interface import seg_ranges

W, W_REF = 2, 10


def hamming_matrix(window) -> np.ndarray:
    """Pairwise Hamming distances between the edge sets of the snapshots in `window` (a list of key arrays)."""
    L = len(window)
    k = np.concatenate(window) if L else np.empty(0, np.int64)
    _, inv = np.unique(k, return_inverse=True)
    row = np.repeat(np.arange(L), [len(x) for x in window])
    M = sp.csr_matrix((np.ones(len(k)), (row, inv)), shape=(L, inv.max() + 1 if len(k) else 1))
    G = (M @ M.T).toarray()
    s = np.diag(G)
    return s[:, None] + s[None, :] - 2 * G


def null_moments(deg, n1: int, n2: int) -> tuple[float, float]:
    """E R and Var R of the between-sample edge count of a graph with degrees `deg` under random labels."""
    N = n1 + n2
    m = deg.sum() / 2
    p1 = 2 * n1 * n2 / (N * (N - 1))
    p2 = 4 * n1 * (n1 - 1) * n2 * (n2 - 1) / (N * (N - 1) * (N - 2) * (N - 3))
    return p1 * m, p2 * m + (p1 / 2 - p2) * float((deg ** 2).sum()) + (p2 - p1 ** 2) * m ** 2


def mst_edges(Dm) -> np.ndarray:
    """Edges (i, j) of the MST; adding 1 to every distance keeps the tree and makes zero distances edges."""
    T = minimum_spanning_tree(np.triu(Dm + 1.0, 1)).tocoo()
    return np.c_[T.row, T.col]


def stat_at(keys, t: int, w: int = W, w_ref: int = W_REF) -> float:
    E = mst_edges(hamming_matrix([keys[i] for i in range(t - w - w_ref, t)]))
    test = np.arange(w + w_ref) >= w_ref
    R = int(np.sum(test[E[:, 0]] != test[E[:, 1]]))
    mu, var = null_moments(np.bincount(E.ravel(), minlength=w + w_ref).astype(float), w, w_ref)
    return float(-(R - mu) / np.sqrt(var)) if var > 0 else 0.0


def scores(keys, seg, w: int = W, w_ref: int = W_REF) -> np.ndarray:
    D = np.full(len(keys), np.nan)
    for s0, s1 in seg_ranges(seg):
        for t in range(s0 + w + w_ref, s1):
            D[t] = stat_at(keys, t, w, w_ref)
    return D
