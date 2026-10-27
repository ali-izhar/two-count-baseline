"""NBS: the CUSUM inner product of Network Binary Segmentation (Wang, Yu & Rinaldo, Ann. Statist. 2021) at the fixed
split of the window rule (identity-aware, online, no training).

NBS needs two independent samples; here copy A holds the snapshots t - 1, t - 3, ... and copy B the snapshots t - 2,
t - 4, ... of the last w + w_ref steps. For copy c with nL^c reference and nR^c test snapshots,

    Y^c_e = sqrt(nL nR / (nL + nR)) (mean of pair e over the test part - mean over the reference part),
    D_t = sum_e Y^A_e Y^B_e,

the statistic a^{s,e}_t of NBS (their CUSUM matrices of A and B, Frobenius inner product) with s = t - w - w_ref,
split t - w, e = t. Under no change E D_t = 0 for independent snapshots.

    D = scores(keys, seg)      keys[t] = sorted unique int64 pair ids of snapshot t
"""
from __future__ import annotations

import numpy as np

from methods.interface import seg_ranges

W, W_REF = 2, 10


def _cusum(keys, idx_ref, idx_test):
    """(pair ids, Y) of one copy."""
    kr = np.concatenate([keys[i] for i in idx_ref]) if idx_ref else np.empty(0, np.int64)
    kt = np.concatenate([keys[i] for i in idx_test]) if idx_test else np.empty(0, np.int64)
    u, inv = np.unique(np.r_[kt, kr], return_inverse=True)
    nl, nr = len(idx_ref), len(idx_test)
    y = np.bincount(inv[:len(kt)], minlength=len(u)) / nr - np.bincount(inv[len(kt):], minlength=len(u)) / nl
    return u, np.sqrt(nl * nr / (nl + nr)) * y


def stat_at(keys, t: int, w: int = W, w_ref: int = W_REF) -> float:
    ref, test = range(t - w - w_ref, t - w), range(t - w, t)
    ua, ya = _cusum(keys, [i for i in ref if (t - i) % 2 == 1], [i for i in test if (t - i) % 2 == 1])
    ub, yb = _cusum(keys, [i for i in ref if (t - i) % 2 == 0], [i for i in test if (t - i) % 2 == 0])
    _, ia, ib = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
    return float(ya[ia] @ yb[ib])


def scores(keys, seg, w: int = W, w_ref: int = W_REF) -> np.ndarray:
    D = np.full(len(keys), np.nan)
    for s0, s1 in seg_ranges(seg):
        for t in range(s0 + w + w_ref, s1):
            D[t] = stat_at(keys, t, w, w_ref)
    return D
