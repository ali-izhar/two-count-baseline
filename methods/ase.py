"""ASE two-window score: random dot product graph estimates of the two windows compared (identity-aware, online, no
training; the estimate of Sussman et al. 2012, the change statistic in the spirit of Padilla, Yu & Priebe 2022).

The test graph has weights a_e = x_e / w and the reference graph b_e = y_e / w_ref (window means of pair presence, as
in methods/hamming.py). For each, the adjacency spectral embedding of dimension d keeps the d eigenpairs of largest
|lambda|, so P_hat = U diag(lambda) U^T (a generalized RDPG: negative eigenvalues allowed). The score is

    D_t = || P_hat_test - P_hat_ref ||_F,
    ||.||_F^2 = sum lambda_T^2 + sum lambda_R^2 - 2 sum_ij lambda_Ti lambda_Rj (u_Ti . u_Rj)^2,

on the vertices active in either window (an inactive vertex has a zero row in both). Unlike Hamming, D_t sees only
the low-rank (block) part of the change; an edge-level change without block structure is mostly filtered out.

    D = scores(keys, seg, n)    keys[t] = sorted unique int64 pair ids u * n + v (u < v) of snapshot t
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from methods.interface import persistence, seg_ranges

W, W_REF, D = 2, 10, 4
DENSE_MAX = 400


def embed(i, j, val, na: int, d: int = D):
    """(lambda, U): the d eigenpairs of largest |lambda| of the symmetric weighted adjacency on na vertices."""
    A = sp.coo_matrix((np.r_[val, val], (np.r_[i, j], np.r_[j, i])), shape=(na, na)).tocsr()
    if na <= max(DENSE_MAX, d + 2):
        lam, U = np.linalg.eigh(A.toarray())
    else:
        v0 = np.random.RandomState(0).rand(na)
        lam, U = eigsh(A, k=d, which="LM", v0=v0, tol=1e-6)
    k = np.argsort(-np.abs(lam))[:d]
    return lam[k], U[:, k]


def stat_at(keys, t: int, n: int, w: int = W, w_ref: int = W_REF, d: int = D) -> float:
    u, x, y = persistence(keys, t, w, w_ref)
    if len(u) == 0:
        return 0.0
    a, b = np.divmod(u, n)
    act, inv = np.unique(np.r_[a, b], return_inverse=True)
    i, j = inv[:len(u)], inv[len(u):]
    (lt, Ut), (lr, Ur) = (embed(i[v > 0], j[v > 0], v[v > 0], len(act), d) for v in (x / w, y / w_ref))
    f2 = (lt ** 2).sum() + (lr ** 2).sum() - 2 * (lt[:, None] * lr[None, :] * (Ut.T @ Ur) ** 2).sum()
    return float(np.sqrt(max(f2, 0.0)))


def scores(keys, seg, n: int, w: int = W, w_ref: int = W_REF, d: int = D) -> np.ndarray:
    S = np.full(len(keys), np.nan)
    for s0, s1 in seg_ranges(seg):
        for t in range(s0 + w + w_ref, s1):
            S[t] = stat_at(keys, t, n, w, w_ref, d)
    return S
