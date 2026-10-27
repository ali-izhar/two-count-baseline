"""MMD: Maximum Mean Discrepancy Two-Sample Test.

Reference:
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Scholkopf, B., & Smola, A.
    (2012). "A Kernel Two-Sample Test". JMLR, 13, 723-773.

Algorithm:
    For each t >= 2*window_size:
    1. ref = features[t-2w:t-w], test = features[t-w:t]
    2. Bandwidth: sigma^2 = median of all pairwise distances (median heuristic).
    3. Compute RBF kernel matrices K_xx, K_yy, K_xy.
    4. Unbiased MMD^2 estimator (Gretton Lemma 6):
         MMD^2 = (sum(K_xx) - trace(K_xx)) / (m*(m-1))
               + (sum(K_yy) - trace(K_yy)) / (n*(n-1))
               - 2 * sum(K_xy) / (m*n)
    5. score_t = max(0, MMD^2)
    6. If score_t >= threshold and cooldown satisfied: detect.
"""

import numpy as np
from scipy.spatial.distance import cdist

from data.registry import DataBundle
from methods.interface import Detections, apply_cooldown

# ---------------------------------------------------------------------------
# Core MMD logic
# ---------------------------------------------------------------------------

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma_sq: float) -> np.ndarray:
    """Compute RBF (Gaussian) kernel matrix.

    K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

    Args:
        X: (m, d) array.
        Y: (n, d) array.
        sigma_sq: Bandwidth parameter (sigma squared).

    Returns:
        (m, n) kernel matrix.
    """
    sq_dists = cdist(X, Y, metric='sqeuclidean')
    return np.exp(-sq_dists / (2.0 * sigma_sq))


def _median_bandwidth(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute bandwidth via median heuristic.

    sigma^2 = median of all pairwise squared distances between
    the combined set of X and Y points.

    Args:
        X: (m, d) array.
        Y: (n, d) array.

    Returns:
        sigma^2 (bandwidth squared). Returns 1.0 if median is zero.
    """
    combined = np.vstack([X, Y])
    sq_dists = cdist(combined, combined, metric='sqeuclidean')
    # Extract upper triangle (exclude diagonal)
    upper = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    if len(upper) == 0:
        return 1.0
    med = np.median(upper)
    return med if med > 1e-10 else 1.0


def _unbiased_mmd2(K_xx: np.ndarray, K_yy: np.ndarray,
                   K_xy: np.ndarray) -> float:
    """Compute unbiased MMD^2 estimator (Gretton et al., Lemma 6).

    MMD^2_u = (1/(m(m-1))) * (sum(K_xx) - trace(K_xx))
            + (1/(n(n-1))) * (sum(K_yy) - trace(K_yy))
            - (2/(m*n))    * sum(K_xy)

    Args:
        K_xx: (m, m) kernel matrix within reference sample.
        K_yy: (n, n) kernel matrix within test sample.
        K_xy: (m, n) kernel matrix between reference and test.

    Returns:
        Unbiased estimate of MMD^2.
    """
    m = K_xx.shape[0]
    n = K_yy.shape[0]

    if m < 2 or n < 2:
        return 0.0

    term_xx = (K_xx.sum() - np.trace(K_xx)) / (m * (m - 1))
    term_yy = (K_yy.sum() - np.trace(K_yy)) / (n * (n - 1))
    term_xy = K_xy.sum() / (m * n)

    return term_xx + term_yy - 2.0 * term_xy


def _mmd_scores(features: np.ndarray, window_size: int) -> np.ndarray:
    """Compute MMD^2 scores on sliding windows of features.

    Args:
        features: (T, D) raw feature array.
        window_size: Window size w. At time t, compares [t-2w:t-w] vs [t-w:t].

    Returns:
        scores: (T,) array of MMD^2 scores.
    """
    T = features.shape[0]
    w = window_size
    scores = np.zeros(T)

    for t in range(2 * w, T):
        ref = features[t - 2 * w:t - w]   # (w, D)
        test = features[t - w:t]           # (w, D)

        # Bandwidth via median heuristic
        sigma_sq = _median_bandwidth(ref, test)

        # Kernel matrices
        K_xx = _rbf_kernel(ref, ref, sigma_sq)
        K_yy = _rbf_kernel(test, test, sigma_sq)
        K_xy = _rbf_kernel(ref, test, sigma_sq)

        # Unbiased MMD^2
        mmd2 = _unbiased_mmd2(K_xx, K_yy, K_xy)
        scores[t] = max(0.0, mmd2)

    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(bundle: DataBundle, params: dict) -> Detections:
    """Detect change points using precomputed features.

    Parameters (in params dict):
        threshold   : float — Absolute threshold on MMD^2.
        window_size : int   — Window size for two-sample test.
        cooldown    : int   — Minimum steps between detections.
    """
    threshold = params["threshold"]
    window_size = params["window_size"]
    cooldown = params["cooldown"]

    scores = _mmd_scores(bundle.features_raw, window_size)
    change_points = apply_cooldown(scores, threshold, cooldown, start=2 * window_size)

    return Detections(
        change_points=change_points,
        scores=scores,
        threshold_used=threshold,
    )
