"""LAD: Laplacian change point detection.

Reference:
    Huang, Hitti, Rabusseau and Rabbany (2020). "Laplacian Change Point Detection for
    Dynamic Graphs". KDD 2020.

Algorithm:
    1. For each graph, compute top eigenvalues of UNNORMALIZED Laplacian L = D - A.
    2. L2-normalize eigenvalue vector -> signature.
    3. Build two context matrices: the previous short_window and the previous long_window signatures.
    4. Compute principal vector of each via SVD (first right singular vector).
    5. Z_t = max over the two contexts of 1 - |cos(signature_t, principal_vector)|
    6. Z*_t = Z_t - Z_{t-1}  (first difference)
    7. Detect when Z*_t >= threshold.

Note: LAD uses the UNNORMALIZED Laplacian (L = D - A); its top eigenvalues are precomputed in the data bundle
      (data/registry.py, compute_lad_eigenvalues_sequence).
"""

import numpy as np

from data.registry import DataBundle
from methods.interface import Detections, apply_cooldown


def _svd_anomaly_scores(signatures: list, short_window: int,
                        long_window: int, num_eigen: int) -> np.ndarray:
    """Compute SVD-based anomaly scores with two-window max and first-difference.

    Following the reference LAD code (Anomaly_Detection.py):
    - For each window size w in {short_window, long_window}:
      Context = signatures[t-w : t]  (NO gap between context and test point)
      Z_t^w = 1 - |cos(signature_t, principal_vector_of_context)|
    - Z_t = max(Z_t^short, Z_t^long)
    - Z*_t = Z_t - Z_{t-1}

    Args:
        signatures: List of signature arrays (or None for skipped graphs).
        short_window: Short context window size.
        long_window: Long context window size.
        num_eigen: Dimension of each signature.

    Returns:
        (T,) array of Z* scores.
    """
    T = len(signatures)
    windows = [short_window, long_window]
    initial_window = max(windows)

    Z = np.zeros(T)

    for t in range(initial_window, T):
        sig_t = signatures[t]
        if sig_t is None:
            continue

        norm_sig = np.linalg.norm(sig_t)
        scores_for_t = []

        for w in windows:
            if w > t:
                continue

            # Collect valid signatures in [t-w, t) — NO gap
            ctx_sigs = []
            for i in range(t - w, t):
                if signatures[i] is not None:
                    ctx_sigs.append(signatures[i])

            if len(ctx_sigs) < 2:
                continue

            context = np.array(ctx_sigs)

            _, S, Vt = np.linalg.svd(context, full_matrices=False)
            if len(S) == 0 or S[0] < 1e-12:
                continue
            principal = Vt[0]

            norm_pv = np.linalg.norm(principal)
            if norm_sig > 1e-12 and norm_pv > 1e-12:
                cos_sim = abs(np.dot(sig_t, principal) / (norm_sig * norm_pv))
            else:
                cos_sim = 0.0

            scores_for_t.append(1.0 - cos_sim)

        Z[t] = max(scores_for_t) if scores_for_t else 0.0

    # First difference: Z*_t = Z_t - Z_{t-1}
    Z_star = np.zeros(T)
    for t in range(1, T):
        Z_star[t] = Z[t] - Z[t - 1]

    return Z_star


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(bundle: DataBundle, params: dict) -> Detections:
    """Detect change points using precomputed LAD eigenvalues from bundle.

    Parameters (in params dict):
        threshold    : float — Absolute threshold on Z* score.
        num_eigen    : int   — Number of top eigenvalues to use (subset of precomputed).
        short_window : int   — Short context window size.
        long_window  : int   — Long context window size.
        cooldown     : int   — Minimum steps between detections.
    """
    threshold = params["threshold"]
    num_eigen = params["num_eigen"]
    short_window = params["short_window"]
    long_window = params["long_window"]
    cooldown = params["cooldown"]

    T = bundle.n_graphs

    # Step 1: Extract eigenvalue signatures from precomputed data
    # bundle.lad_eigenvalues is (T, max_eigen). Take first num_eigen and L2-normalize.
    signatures = []
    for i in range(T):
        eigs = bundle.lad_eigenvalues[i, :num_eigen].copy()
        norm = np.linalg.norm(eigs)
        if norm > 1e-12:
            eigs /= norm
            signatures.append(eigs)
        else:
            signatures.append(None)

    # Step 2-3: SVD anomaly scores
    scores = _svd_anomaly_scores(signatures, short_window, long_window, num_eigen)

    # Step 4: threshold with cooldown
    change_points = apply_cooldown(scores, threshold, cooldown,
                                   start=long_window + short_window)

    return Detections(
        change_points=change_points,
        scores=scores,
        threshold_used=threshold,
    )
