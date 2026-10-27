"""NetSimile (Berlingerio, Koutra, Eliassi-Rad, Faloutsos, arXiv 2012): a permutation-invariant graph signature.

Six of its seven features per active vertex i: degree d_i, clustering c_i, mean neighbour degree, mean neighbour
clustering, edges inside the egonet (d_i + t_i, t_i = triangles at i) and edges leaving the egonet (sum_{j in N(i)}
d_j - d_i - 2 t_i); each aggregated by median, mean, SD, skewness and kurtosis (30 numbers). The seventh, the number of
vertices adjacent to the egonet, needs the two-step neighbourhood, which graphs with hubs of degree ~1e5 (the
stablecoin network) do not allow; it is left out. Triangles are counted on the degree-ordered orientation.
Change score at t: Canberra distance between the mean signatures of the test window (the w snapshots before t) and
of the reference window (the w_ref snapshots before it), the window rule of the other rows.

    F = signature(A)            A: sparse adjacency (any weights; binarized, no self-loops) -> (30,)
    D = scores(F, seg)          F: (T, 30) signatures -> (T,) scores, nan before both windows fit
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.stats import kurtosis, skew

from methods.interface import seg_ranges

W, W_REF = 2, 10


def signature(A) -> np.ndarray:
    A = sp.csr_matrix(A, dtype=np.float64)
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    d = np.asarray(A.sum(1)).ravel()
    act = np.flatnonzero(d)
    if len(act) < 2:
        return np.zeros(30)
    A = A[act][:, act].tocsr()
    d = d[act]
    rank = np.empty(len(d), np.int64)
    rank[np.lexsort((np.arange(len(d)), d))] = np.arange(len(d))
    C = A.tocoo()
    up = rank[C.row] < rank[C.col]
    U = sp.csr_matrix((np.ones(up.sum()), (C.row[up], C.col[up])), shape=A.shape)
    W1 = (U @ U).multiply(U)                                             # lowest -> middle -> highest
    W2 = (U.T @ U).multiply(U)
    t = np.asarray(W1.sum(1)).ravel() + np.asarray(W1.sum(0)).ravel() + np.asarray(W2.sum(1)).ravel()
    c = np.where(d > 1, 2 * t / np.maximum(d * (d - 1), 1), 0.0)
    nd = (A @ d) / d
    nc = (A @ c) / d
    ego_in = d + t
    ego_out = A @ d - d - 2 * t
    X = np.c_[d, c, nd, nc, ego_in, ego_out]
    with np.errstate(invalid="ignore", divide="ignore"):
        agg = [np.median(X, 0), X.mean(0), X.std(0), np.nan_to_num(skew(X, 0)), np.nan_to_num(kurtosis(X, 0))]
    return np.concatenate(agg)


def canberra(a, b) -> float:
    s = np.abs(a) + np.abs(b)
    return float(np.sum(np.where(s > 0, np.abs(a - b) / np.where(s > 0, s, 1.0), 0.0)))


def scores(F, seg, w: int = W, w_ref: int = W_REF) -> np.ndarray:
    F = np.asarray(F, float)
    D = np.full(len(F), np.nan)
    for s0, s1 in seg_ranges(np.asarray(seg)):
        for t in range(s0 + w + w_ref, s1):
            D[t] = canberra(F[t - w:t].mean(0), F[t - w - w_ref:t - w].mean(0))
    return D
