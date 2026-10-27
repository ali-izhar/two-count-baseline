"""The conditional score test of a common activity shift (the paper's Rasch model, Sec. 3 and App. A), as detectors.

Units are vertices (present = active) or vertex pairs (present = edge). In the N = w + w_ref snapshots of the two
windows, unit i is present x_i times in the test window and y_i times in the reference window, c_i = x_i + y_i. The
conditional score for beta = 0 and its conditional variance with independent units are

    U = sum_i (x_i - w c_i / N),    V = sum_i c_i (N - c_i) w w_ref / (N^2 (N - 1)),    z = U / sqrt(V).

rows: |z| of the vertices (score_v), |z| of the pairs (score_e), and the covariance-weighted joint test of the two
numerators with snapshots exchangeable (the permutation covariance of the window means of n_+ and |E|):
T^2 = d^T Sigma^{-1} d, d = (mean n_+ test - ref, mean |E| test - ref), Sigma = S (1 / w + 1 / w_ref), S the sample
covariance of (n_+, |E|) over the N snapshots (pseudo-inverse if singular).

    Z = scores(keys, seg, n)    keys[t] = sorted unique int64 pair ids u * n + v (u < v) of snapshot t
"""
from __future__ import annotations

import numpy as np

from methods.interface import persistence, seg_ranges

W, W_REF = 2, 5                                     # the windows of the count detector


def _z(units, t: int, w: int, w_ref: int) -> float:
    N = w + w_ref
    _, x, y = persistence(units, t, w, w_ref)
    c = x + y
    U = float((x - w * c / N).sum())
    V = float((c * (N - c)).sum()) * w * w_ref / (N * N * (N - 1))
    return U / np.sqrt(V) if V > 0 else 0.0


def scores(keys, seg, n: int, w: int = W, w_ref: int = W_REF) -> dict:
    T = len(keys)
    out = {k: np.full(T, np.nan) for k in ("score_v", "score_e", "score_joint")}
    verts = [np.unique(np.r_[np.divmod(np.asarray(k, np.int64), n)]) for k in keys]
    nv = np.array([len(v) for v in verts], float)
    ne = np.array([len(k) for k in keys], float)
    N = w + w_ref
    for s0, s1 in seg_ranges(seg):
        for t in range(s0 + N, s1):
            out["score_v"][t] = abs(_z(verts, t, w, w_ref))
            out["score_e"][t] = abs(_z(keys, t, w, w_ref))
            X = np.c_[nv[t - N:t], ne[t - N:t]]
            d = X[N - w:].mean(0) - X[:N - w].mean(0)
            S = np.cov(X.T) * (1 / w + 1 / w_ref)
            out["score_joint"][t] = float(d @ np.linalg.pinv(S) @ d)
    return out
