"""Edge-set distances between the window-mean adjacencies (identity-aware, online, no training).

Pair e is present x_e times in the test window (the w snapshots before t) and y_e times in the reference window (the
w_ref snapshots before it); a_e = x_e / w and b_e = y_e / w_ref are the two window means.

    hamming  D_t = sum_e |a_e - b_e|                      (Hamming distance of the edge sets when w = w_ref = 1;
                                                           the two-sample CUSUM at the fixed split, up to a constant)
    jaccard  D_t = 1 - sum_e min(a_e, b_e) / sum_e max(a_e, b_e)      (weighted Jaccard distance)

    D = scores(keys, seg)      keys[t] = sorted unique int64 pair ids of snapshot t (u * n + v, u < v)
"""
from __future__ import annotations

import numpy as np

from methods.interface import persistence, seg_ranges

W, W_REF = 2, 10


def stat_at(keys, t: int, w: int = W, w_ref: int = W_REF) -> tuple[float, float]:
    """(hamming, jaccard) at step t from keys[t - w - w_ref:t]."""
    _, x, y = persistence(keys, t, w, w_ref)
    a, b = x / w, y / w_ref
    hi = np.maximum(a, b).sum()
    return float(np.abs(a - b).sum()), float(1.0 - np.minimum(a, b).sum() / hi) if hi > 0 else 0.0


def scores(keys, seg, w: int = W, w_ref: int = W_REF) -> tuple[np.ndarray, np.ndarray]:
    """(hamming, jaccard) series; nan before w + w_ref snapshots of each segment."""
    H, J = np.full(len(keys), np.nan), np.full(len(keys), np.nan)
    for s0, s1 in seg_ranges(seg):
        for t in range(s0 + w + w_ref, s1):
            H[t], J[t] = stat_at(keys, t, w, w_ref)
    return H, J
