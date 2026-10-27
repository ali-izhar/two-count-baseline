"""Common output type, alarm rule, segments and pair windows of the detectors: detect(bundle, params) -> Detections."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Detections:
    """Output of a detection method."""
    change_points: list          # Detected CP indices
    scores: np.ndarray           # Detection statistic at each timestep
    threshold_used: float        # Absolute threshold that was applied


def apply_cooldown(scores: np.ndarray, threshold: float, cooldown: int,
                   start: int = 0) -> list:
    """Extract change points from scores with cooldown suppression.

    A change point is declared at time t if scores[t] >= threshold
    and at least `cooldown` steps have passed since the last detection.
    """
    change_points = []
    last_det = -cooldown
    for t in range(start, len(scores)):
        if scores[t] >= threshold and (t - last_det) >= cooldown:
            change_points.append(t)
            last_det = t
    return change_points


def seg_ranges(seg) -> list[tuple[int, int]]:
    """[(s0, s1)]: the contiguous segments (independent recordings) of a segment-id array."""
    seg = np.asarray(seg)
    cut = np.r_[0, np.nonzero(np.diff(seg))[0] + 1, len(seg)]
    return list(zip(cut[:-1].tolist(), cut[1:].tolist()))


def persistence(keys, t: int, w: int, w_ref: int):
    """(units, x, y) at step t: the units (pair or vertex ids) seen in the two windows and their presence counts in
    the test window keys[t - w:t] and the reference window keys[t - w - w_ref:t - w]."""
    kt = np.concatenate(keys[t - w:t])
    kr = np.concatenate(keys[t - w - w_ref:t - w])
    u, inv = np.unique(np.r_[kt, kr], return_inverse=True)
    return u, np.bincount(inv[:len(kt)], minlength=len(u)), np.bincount(inv[len(kt):], minlength=len(u))
