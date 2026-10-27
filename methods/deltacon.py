"""DeltaCon between the window-mean adjacencies (Koutra, Vogelstein & Faloutsos, SDM 2013; Koutra et al., TKDD 2016),
identity-aware, online, no training.

The test graph has weights a_e = x_e / w and the reference graph b_e = y_e / w_ref (window means of pair presence).
For each graph, FaBP affinities S = [I + eps^2 D - eps A]^{-1} with eps = 1 / (1 + max_i d_i); the grouped version
uses g random vertex groups (indicator matrix E, n x g) and compares S E:

    D_t = sqrt( sum_{i, k} ( sqrt((S_test E)_ik) - sqrt((S_ref E)_ik) )^2 )      (root-Euclidean distance; DeltaCon's
                                                                                   similarity is 1 / (1 + D_t))
Vertices without an edge in either window have S_ii = 1 in both graphs and add 0, so only the active vertices enter.

    D = scores(keys, seg, n)    keys[t] = sorted unique int64 pair ids u * n + v (u < v) of snapshot t
"""
from __future__ import annotations

import functools

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from methods.interface import persistence, seg_ranges

W, W_REF, G, SEED = 2, 10, 10, 0
DENSE_MAX = 3000


@functools.lru_cache(maxsize=4)
def groups(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(G, size=n).astype(np.int8)


def affinity(i, j, val, na: int, grp) -> np.ndarray:
    """S E (na x g) for the weighted graph with edges (i, j, val) on na vertices."""
    A = sp.coo_matrix((np.r_[val, val], (np.r_[i, j], np.r_[j, i])), shape=(na, na)).tocsr()
    d = np.asarray(A.sum(1)).ravel()
    eps = 1.0 / (1.0 + (d.max() if na else 0.0))
    M = (sp.diags(1.0 + eps ** 2 * d) - eps * A).tocsr()
    E = np.zeros((na, G))
    E[np.arange(na), grp] = 1.0
    if na <= DENSE_MAX:
        return np.linalg.solve(M.toarray(), E)
    pre = sp.diags(1.0 / M.diagonal())
    return np.column_stack([cg(M, E[:, k], rtol=1e-10, M=pre, maxiter=10000)[0] for k in range(G)])


def stat_at(keys, t: int, n: int, w: int = W, w_ref: int = W_REF) -> float:
    u, x, y = persistence(keys, t, w, w_ref)
    if len(u) == 0:
        return 0.0
    a, b = np.divmod(u, n)
    act, inv = np.unique(np.r_[a, b], return_inverse=True)
    i, j = inv[:len(u)], inv[len(u):]
    grp = groups(n)[act]
    st, sr = (np.sqrt(np.clip(affinity(i[v > 0], j[v > 0], v[v > 0], len(act), grp), 0, None))
              for v in (x / w, y / w_ref))
    return float(np.sqrt(((st - sr) ** 2).sum()))


def scores(keys, seg, n: int, w: int = W, w_ref: int = W_REF) -> np.ndarray:
    D = np.full(len(keys), np.nan)
    for s0, s1 in seg_ranges(seg):
        for t in range(s0 + w + w_ref, s1):
            D[t] = stat_at(keys, t, n, w, w_ref)
    return D
