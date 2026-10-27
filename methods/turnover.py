"""Vertex turnover: which vertices are active, compared across two windows (identity-aware, activity only).

a_t is the 0/1 vector of the active vertices of snapshot t (degree > 0 without self-loops) on a fixed set of n
vertices. With x_i and y_i the number of snapshots in which vertex i is active in the test window (the w snapshots
before t) and in the reference window (the w_ref snapshots before it),

    D_t = sum_i | x_i / w - y_i / w_ref | / n,

the L1 distance of the two window means of a, as a share of n (scale-free: no setting to tune, causal as it stands).

    D = scores(keys, seg, n)    keys[t] = sorted unique int64 pair ids u * n + v (u < v) of snapshot t
"""
from __future__ import annotations

import numpy as np

from methods.interface import persistence, seg_ranges

W, W_REF = 2, 5                                     # the windows of the count detector


def scores(keys, seg, n: int, w: int = W, w_ref: int = W_REF) -> np.ndarray:
    verts = [np.unique(np.r_[np.divmod(np.asarray(k, np.int64), n)]) for k in keys]
    D = np.full(len(keys), np.nan)
    for s0, s1 in seg_ranges(seg):
        for t in range(s0 + w + w_ref, s1):
            _, x, y = persistence(verts, t, w, w_ref)
            D[t] = float(np.abs(x / w - y / w_ref).sum()) / n
    return D
