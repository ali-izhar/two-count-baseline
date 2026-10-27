"""CUSUM: Cumulative Sum Control Chart.

Reference:
    Page, E. S. (1954). "Continuous Inspection Schemes".
    Biometrika, 41(1/2), 100-115.

Algorithm:
    1. Estimate mean and std from features during startup period ONLY.
    2. Standardize: z_t = (x_t - mean) / std.
    3. Accumulate:
         S_pos = max(0, S_pos + z_t - kappa)
         S_neg = max(0, S_neg - z_t - kappa)
    4. score_t = max over all 8 dimensions of max(S_pos, S_neg).
    5. If score_t >= threshold and cooldown satisfied: detect + RESET.

Key: normalization uses ONLY the startup period (no data leakage).
     After each detection, S_pos and S_neg are reset to zero.
"""

import numpy as np

from data.registry import DataBundle
from methods.interface import Detections, apply_cooldown

# ---------------------------------------------------------------------------
# Core CUSUM logic (operates on feature array)
# ---------------------------------------------------------------------------

def _cusum_scores(features: np.ndarray, threshold: float, kappa: float,
                  startup_period: int, cooldown: int) -> np.ndarray:
    """Compute CUSUM scores on a (T, D) feature array.

    Accumulators are reset after each detection (threshold + cooldown).
    Because resets affect future scores, threshold and cooldown are needed
    during score computation.

    Args:
        features: (T, D) raw feature array.
        threshold: Absolute detection threshold.
        kappa: Slack parameter (allowance for drift).
        startup_period: Number of initial observations for estimating mean/std.
        cooldown: Minimum steps between detections.

    Returns:
        scores: (T,) array of CUSUM scores.
    """
    T, D = features.shape
    scores = np.zeros(T)

    if T <= startup_period:
        return scores

    # Estimate baseline mean and std from startup period ONLY
    baseline = features[:startup_period]
    mu = baseline.mean(axis=0)          # (D,)
    sigma = baseline.std(axis=0)        # (D,)
    sigma = np.where(sigma < 1e-10, 1.0, sigma)  # Avoid division by zero

    # Initialize CUSUM accumulators
    S_pos = np.zeros(D)
    S_neg = np.zeros(D)

    last_det = -cooldown

    for t in range(startup_period, T):
        # Standardize
        z = (features[t] - mu) / sigma

        # Update accumulators
        S_pos = np.maximum(0.0, S_pos + z - kappa)
        S_neg = np.maximum(0.0, S_neg - z - kappa)

        # Score: max over all dimensions of max(S_pos, S_neg)
        scores[t] = max(np.max(S_pos), np.max(S_neg))

        # Reset accumulators after detection (standard CUSUM behavior)
        if scores[t] >= threshold and (t - last_det) >= cooldown:
            last_det = t
            S_pos = np.zeros(D)
            S_neg = np.zeros(D)

    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(bundle: DataBundle, params: dict) -> Detections:
    """Detect change points using precomputed features.

    Parameters (in params dict):
        threshold      : float — Absolute detection threshold.
        kappa          : float — Slack parameter.
        startup_period : int   — Observations for baseline estimation.
        cooldown       : int   — Minimum steps between detections.
    """
    threshold = float(params["threshold"])
    kappa = float(params["kappa"])
    startup_period = int(params["startup_period"])
    cooldown = int(params["cooldown"])

    scores = _cusum_scores(
        bundle.features_raw, threshold, kappa, startup_period, cooldown
    )
    change_points = apply_cooldown(scores, threshold, cooldown, start=startup_period)

    return Detections(
        change_points=change_points,
        scores=scores,
        threshold_used=threshold,
    )
