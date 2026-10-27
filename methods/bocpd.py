"""BOCPD: Bayesian Online Change Point Detection.

Reference:
    Adams, R. P. & MacKay, D. J. C. (2007). "Bayesian Online Changepoint
    Detection". arXiv:0710.3742.

Algorithm:
    Maintains a run-length distribution P(r_t | x_{1:t}) via message passing.
    At each step t:
    1. Compute predictive probability P(x_t | r_t) using Student-t posterior
       predictive from Normal-Inverse-Gamma conjugate prior.
    2. Compute growth probability (run length increases by 1).
    3. Compute changepoint probability (run length resets to 0).
    4. Normalize to get posterior over run lengths.
    5. score_t = max over dimensions of max posterior run-length probability.
    6. If score_t >= threshold and cooldown satisfied: detect.

Normal-Inverse-Gamma conjugate prior (mu, kappa, alpha, beta):
    Online update upon observing x:
        kappa_new = kappa + 1
        mu_new    = (kappa * mu + x) / kappa_new
        alpha_new = alpha + 0.5
        beta_new  = beta + kappa * (x - mu)^2 / (2 * kappa_new)

    Posterior predictive: Student-t with
        df    = 2 * alpha
        loc   = mu
        scale = sqrt(beta * (kappa + 1) / (alpha * kappa))
"""

import numpy as np
from scipy import stats

from data.registry import DataBundle
from methods.interface import Detections, apply_cooldown

# ---------------------------------------------------------------------------
# NIG prior hyperparameters (vague prior)
# ---------------------------------------------------------------------------
_MU0 = 0.0
_KAPPA0 = 0.01
_ALPHA0 = 1.0
_BETA0 = 0.01


# ---------------------------------------------------------------------------
# Core BOCPD logic
# ---------------------------------------------------------------------------

def _bocpd_scores(features: np.ndarray, hazard_rate: float) -> np.ndarray:
    """Run BOCPD on a (T, D) feature array and return scores.

    Processes each dimension independently with a separate NIG prior, then
    combines via the joint predictive probability (sum of log-likelihoods).

    Args:
        features: (T, D) raw feature array.
        hazard_rate: Constant hazard rate H = P(changepoint | run_length).

    Returns:
        scores: (T,) array of detection scores.
    """
    T, D = features.shape
    scores = np.zeros(T)

    # Constant hazard function
    H = hazard_rate

    # Run-length distribution: R[r] = P(run_length = r | data)
    # Start with P(r=0) = 1
    R = np.array([1.0])

    # NIG hyperparameters for each run length and dimension
    # Shape: (n_run_lengths, D) for each parameter
    mu = np.full((1, D), _MU0)
    kappa = np.full((1, D), _KAPPA0)
    alpha = np.full((1, D), _ALPHA0)
    beta = np.full((1, D), _BETA0)

    for t in range(T):
        x = features[t]  # (D,)

        # --- Predictive probability P(x_t | r) for each run length ---
        # Student-t parameters from NIG posterior
        nu = 2.0 * alpha                             # (n_rl, D)
        loc = mu                                      # (n_rl, D)
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa + 1e-10))  # (n_rl, D)

        # Joint log predictive: sum over dimensions
        log_probs = stats.t.logpdf(x, df=nu, loc=loc, scale=scale)  # (n_rl, D)
        log_joint = log_probs.sum(axis=1)  # (n_rl,)
        pred_probs = np.exp(log_joint)
        pred_probs = np.clip(pred_probs, 1e-300, None)

        # --- Update run-length distribution ---
        n_rl = len(R)
        R_new = np.zeros(n_rl + 1)

        # Changepoint probability: run length resets to 0
        R_new[0] = np.sum(R * pred_probs * H)

        # Growth probability: run length increases by 1
        R_new[1:] = R * pred_probs * (1.0 - H)

        # Normalize
        total = R_new.sum()
        if total > 1e-300:
            R_new /= total
        else:
            R_new = np.zeros(len(R_new))
            R_new[0] = 1.0

        R = R_new

        # --- Update NIG parameters ---
        n_old = len(mu)
        kappa_new = kappa + 1.0
        mu_new = (kappa * mu + x) / kappa_new
        alpha_new = alpha + 0.5
        beta_new = beta + kappa * (x - mu) ** 2 / (2.0 * kappa_new)

        # Extend: row 0 = prior (changepoint), rows 1: = growth
        new_mu = np.zeros((n_old + 1, D))
        new_kappa = np.zeros((n_old + 1, D))
        new_alpha = np.zeros((n_old + 1, D))
        new_beta = np.zeros((n_old + 1, D))

        new_mu[0] = _MU0
        new_kappa[0] = _KAPPA0
        new_alpha[0] = _ALPHA0
        new_beta[0] = _BETA0

        new_mu[1:] = mu_new
        new_kappa[1:] = kappa_new
        new_alpha[1:] = alpha_new
        new_beta[1:] = beta_new

        mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta

        # --- Detection score ---
        # Score = probability mass on short run lengths (r < 5)
        # This indicates a recent changepoint was detected by the model
        rl_thresh = min(5, len(R))
        scores[t] = R[:rl_thresh].sum()

    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(bundle: DataBundle, params: dict) -> Detections:
    """Detect change points using precomputed features.

    Parameters (in params dict):
        threshold   : float — Absolute threshold on short-run-length probability.
        hazard_rate : float — Constant hazard rate (1/expected_run_length).
        cooldown    : int   — Minimum steps between detections.
    """
    threshold = params["threshold"]
    hazard_rate = params["hazard_rate"]
    cooldown = params["cooldown"]

    scores = _bocpd_scores(bundle.features_raw, hazard_rate)
    change_points = apply_cooldown(scores, threshold, cooldown, start=15)

    return Detections(
        change_points=change_points,
        scores=scores,
        threshold_used=threshold,
    )
