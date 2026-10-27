"""The count detector: the two-window distance between per-snapshot count vectors (no training, one setting).

X (T, d) holds per-snapshot values, e.g. the number of active vertices and the number of edges, each divided by its
median absolute first difference (repro/common.mad_scale). The test window is the w snapshots before t, the
reference window the w_ref snapshots before it, and

    pw : D_t = mean L1 distance over all (test, reference) pairs
    cen: D_t = L1 distance between the two window means
    gam: D_t = sqrt(sum_j (mean_test - mean_ref)_j^2 / j^2)    (coordinate j weighted 1 / j^2: edges 1/4 for d = 2)

The same window rule also scores the wiring statistics of repro/ (transitivity, assortativity).

    D = scores(X, w=2, w_ref=5, mode="gam")
    alarms(D, threshold_value(D, "pct", 0.8), q=6, start=w + w_ref + 5)
"""

import numpy as np


def window_idx(t: int, w: int, w_ref: int) -> tuple[np.ndarray, np.ndarray]:
    """Test window: the w snapshots before t. Reference window: the w_ref snapshots before the test window."""
    return np.arange(t - w, t), np.arange(t - w - w_ref, t - w)


def scores(X: np.ndarray, w: int, w_ref: int, mode: str) -> np.ndarray:
    """D_t on per-snapshot vectors X (T, d); nan before both windows fit."""
    T, d = X.shape
    D = np.full(T, np.nan)
    inv_j2 = 1.0 / np.arange(1, d + 1) ** 2
    for t in range(w + w_ref, T):
        ti, ri = window_idx(t, w, w_ref)
        if mode == "pw":
            D[t] = np.abs(X[ti][:, None, :] - X[ri][None, :, :]).sum(-1).mean()
        else:
            diff = X[ti].mean(0) - X[ri].mean(0)
            D[t] = np.abs(diff).sum() if mode == "cen" else np.sqrt((diff ** 2 * inv_j2).sum())
    return D


def threshold_value(D: np.ndarray, kind: str, value: float) -> float:
    """'abs': value itself (runs online). 'pct': quantile `value` of the positive scores of the whole series."""
    if kind == "abs":
        return value
    pos = D[np.isfinite(D) & (D > 0)]
    return float(np.quantile(pos, value)) if pos.size else np.inf


def alarms(D: np.ndarray, theta: float, q: int, start: int) -> list[int]:
    """Steps t >= start with D_t >= theta and at least q steps since the previous alarm (the cooldown)."""
    with np.errstate(invalid="ignore"):
        cand = np.nonzero(D >= theta)[0]
    out, last = [], -10 ** 9
    for t in cand[cand >= start]:
        if t - last >= q:
            out.append(int(t))
            last = t
    return out
