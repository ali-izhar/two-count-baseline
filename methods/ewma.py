"""EWMA: Exponentially Weighted Moving Average Control Chart.

Reference:
    Roberts, S. W. (1959). "Control Chart Tests Based on Geometric
    Moving Averages". Technometrics, 1(3), 239-250.

Algorithm:
    1. Estimate mean and std from features during startup period ONLY.
    2. Standardize: z_t = (x_t - mean) / std.
    3. Update EWMA: E_t = lambda * z_t + (1 - lambda) * E_{t-1}.
    4. score_t = max(|E_t|) over all 8 dimensions.
    5. If score_t >= threshold and cooldown satisfied: detect + RESET E = 0.

Key: normalization uses ONLY the startup period (no data leakage).
     After each detection, EWMA is reset to zero.
"""

import numpy as np

from data.registry import DataBundle
from methods.interface import Detections, apply_cooldown

# ---------------------------------------------------------------------------
# Core EWMA logic (operates on feature array)
# ---------------------------------------------------------------------------

def _ewma_scores(features: np.ndarray, threshold: float, lambda_param: float,
                 startup_period: int, cooldown: int) -> np.ndarray:
    """Compute EWMA scores on a (T, D) feature array.

    EWMA is reset after each detection (threshold + cooldown). Because resets
    affect future scores, threshold and cooldown are needed during score
    computation.

    Args:
        features: (T, D) raw feature array.
        threshold: Absolute detection threshold.
        lambda_param: Smoothing parameter in (0, 1]. Higher = more responsive.
        startup_period: Number of initial observations for estimating mean/std.
        cooldown: Minimum steps between detections.

    Returns:
        scores: (T,) array of EWMA scores.
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

    # Initialize EWMA
    ewma = np.zeros(D)

    last_det = -cooldown

    for t in range(startup_period, T):
        # Standardize
        z = (features[t] - mu) / sigma

        # Update EWMA
        ewma = lambda_param * z + (1.0 - lambda_param) * ewma

        # Score: max of absolute EWMA values across all dimensions
        scores[t] = np.max(np.abs(ewma))

        # Reset EWMA after detection
        if scores[t] >= threshold and (t - last_det) >= cooldown:
            last_det = t
            ewma = np.zeros(D)

    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(bundle: DataBundle, params: dict) -> Detections:
    """Detect change points using precomputed features.

    Parameters (in params dict):
        threshold      : float — Absolute detection threshold.
        lambda_param   : float — EWMA smoothing parameter.
        startup_period : int   — Observations for baseline estimation.
        cooldown       : int   — Minimum steps between detections.
    """
    threshold = params["threshold"]
    lambda_param = params["lambda_param"]
    startup_period = params["startup_period"]
    cooldown = params["cooldown"]

    scores = _ewma_scores(
        bundle.features_raw, threshold, lambda_param, startup_period, cooldown
    )
    change_points = apply_cooldown(scores, threshold, cooldown, start=startup_period)

    return Detections(
        change_points=change_points,
        scores=scores,
        threshold_used=threshold,
    )
