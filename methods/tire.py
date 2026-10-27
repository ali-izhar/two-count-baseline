"""TIRE: Time-Invariant REpresentation Change Point Detection.

Reference:
    De Ryck, T., De Vos, M., & Bertrand, A. (2021). "Change Point Detection
    in Time Series Data Using Autoencoders with a Time-Invariant
    Representation". IEEE Transactions on Signal Processing, 69, 3513-3524.

Algorithm (offline, unsupervised):
    This is fundamentally different from the online methods in this project.
    TIRE trains autoencoders on the ENTIRE test sequence -- it sees all data
    before making any predictions. This is both its strength (no need for a
    separate clean training window) and its limitation (cannot run in an
    online/streaming setting).

    Phase 1 -- Windowing:
        TD windows: y_t = X[t-N+1 : t+1].flatten()  shape (N*d,)
        FD windows: DFT per channel, modulus of first nfft//2 bins  shape (M*d,)
        Scale all windows to [-1, 1] per-feature (min-max over the full sequence).

    Phase 2 -- Parallel Autoencoders with Time-Invariance Loss:
        Two autoencoders (TD and FD) with architecture:
            Encoder: input_dim -> h_dim (tanh)
            Decoder: h_dim -> input_dim (tanh)
        Latent code h_t is split: first s_dim dimensions are "shared"
        (time-invariant), rest are "unshared".

        Loss for K+1 consecutive windows starting at t:
            L = sum_{k=0}^{K} ||y_{t+k} - yhat_{t+k}||_2
              + (lambda/K) * sum_{k=0}^{K-1} ||s_{t+k+1} - s_{t+k}||_2

        First term: reconstruction (L2 norm per sample, NOT MSE).
        Second term: time-invariance penalty on shared features.

    Phase 3 -- Feature Extraction & Post-processing:
        1. Encode all windows, extract shared features s_t
        2. Combine TD and FD with quantile-based weighting:
             alpha = 95th percentile of FD dissimilarities
             beta  = 95th percentile of TD dissimilarities
             combined = concat(alpha * s_TD, beta * s_FD)
        3. Smooth features with triangular matched filter of width 2N-1
        4. Dissimilarity: D_t = ||s_smooth_t - s_smooth_{t+N}||_2
        5. Smooth dissimilarity with same matched filter

    Phase 4 -- Detection:
        Threshold the smoothed dissimilarity. Apply cooldown.
"""

import numpy as np

from data.registry import DataBundle
from methods.interface import Detections, apply_cooldown

# ===========================================================================
# Windowing: Time-Domain (TD) and Frequency-Domain (FD)
# ===========================================================================

def _td_windows(
    X: np.ndarray,  # (T, d) raw feature array
    N: int,         # window size
) -> np.ndarray:
    """Build time-domain windows from multivariate time series.

    Mathematical definition:
        y_t = X[t-N+1 : t+1].flatten()  for t = N-1, ..., T-1

    Window index i corresponds to original time t = i + N - 1.

    Args:
        X: (T, d) raw feature array.
        N: Window size.

    Returns:
        windows: (T_eff, N*d) array where T_eff = T - N + 1.
    """
    T, d = X.shape
    T_eff = T - N + 1
    if T_eff <= 0:
        return np.empty((0, N * d))
    windows = np.zeros((T_eff, N * d), dtype=np.float64)
    for i in range(T_eff):
        windows[i] = X[i:i + N].flatten()
    return windows


def _fd_windows(
    X: np.ndarray,  # (T, d) raw feature array
    N: int,         # window size (same as TD)
    nfft: int,      # number of DFT points
) -> np.ndarray:
    """Build frequency-domain windows from multivariate time series.

    Mathematical definition:
        For each window of length N, compute DFT per channel (zero-padded
        to nfft points), take the modulus, and keep the first nfft//2
        frequency bins.

    Args:
        X: (T, d) raw feature array.
        N: Window size (same as TD).
        nfft: Number of DFT points.

    Returns:
        windows: (T_eff, M*d) array where M = nfft//2, T_eff = T - N + 1.
    """
    T, d = X.shape
    T_eff = T - N + 1
    M = nfft // 2
    if T_eff <= 0 or M <= 0:
        return np.empty((0, max(M, 1) * d))
    windows = np.zeros((T_eff, M * d), dtype=np.float64)
    for i in range(T_eff):
        segment = X[i:i + N]                            # (N, d)
        fft_vals = np.fft.fft(segment, n=nfft, axis=0)  # (nfft, d)
        mag = np.abs(fft_vals[:M])                       # (M, d)
        windows[i] = mag.flatten()
    return windows


def _scale_to_unit(
    windows: np.ndarray,  # (T_eff, D) raw windows
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max scale each feature to [-1, 1] over the full sequence.

    This is the standard TIRE scaling -- offline, uses the entire sequence.
    Matches the [-1, 1] range expected by tanh output activations.

    Args:
        windows: (T_eff, D) array of raw windows.
        eps: Small constant to prevent division by zero.

    Returns:
        (scaled_windows, feat_min, feat_max).
    """
    if windows.shape[0] == 0:
        return windows, np.zeros(windows.shape[1]), np.ones(windows.shape[1])
    feat_min = windows.min(axis=0)
    feat_max = windows.max(axis=0)
    denom = feat_max - feat_min
    denom = np.where(denom < eps, 1.0, denom)
    scaled = 2.0 * (windows - feat_min) / denom - 1.0
    return scaled, feat_min, feat_max


# ===========================================================================
# Triangular matched filter (smoothing)
# ===========================================================================

def _matched_filter(x: np.ndarray, N: int) -> np.ndarray:
    """Apply a triangular matched filter of width 2N-1.

    Mathematical definition:
        kernel = [1, 2, ..., N, N-1, ..., 1] / N^2

    The kernel sums to 1 (unit area), so it acts as a weighted moving average.

    Args:
        x: Input array. 2D (T, d) applies per-column; 1D (T,) directly.
        N: Window size parameter (kernel width = 2N-1).

    Returns:
        Smoothed array of same shape as input.
    """
    kernel = np.concatenate([
        np.arange(1, N + 1, dtype=np.float64),
        np.arange(N - 1, 0, -1, dtype=np.float64),
    ]) / (N ** 2)

    def _convolve_same(signal, kern):
        """Convolve with 'same' output length, safe when len(signal) < len(kern).

        np.convolve mode='same' returns max(len(signal), len(kern)) elements.
        We use mode='full' and center-crop to len(signal) for correctness.
        """
        full = np.convolve(signal, kern, mode='full')
        n_full = len(full)
        n_out = len(signal)
        start = (n_full - n_out) // 2
        return full[start:start + n_out]

    if x.ndim == 1:
        return _convolve_same(x, kernel)
    elif x.ndim == 2:
        out = np.zeros_like(x)
        for j in range(x.shape[1]):
            out[:, j] = _convolve_same(x[:, j], kernel)
        return out
    else:
        raise ValueError(f"_matched_filter expects 1D or 2D, got {x.ndim}D")


# ===========================================================================
# Autoencoder training (NumPy, manual backpropagation)
# ===========================================================================

def _tanh(x):
    return np.tanh(x)

def _tanh_deriv(x):
    t = np.tanh(x)
    return 1.0 - t * t

def _train_autoencoder(
    windows: np.ndarray,
    h_dim: int,
    s_dim: int,
    lam: float,
    K: int,
    epochs: int,
    lr: float,
    seed: int,
) -> np.ndarray:
    """Train the TIRE autoencoder D -> h_dim (tanh) -> D (tanh) by full-batch gradient descent; return shared codes.

    The loss rewards reconstruction and penalizes change of the first s_dim latent units across K+1 consecutive
    windows, so those units learn what stays constant within a regime.
    """
    T_eff, D = windows.shape

    if T_eff < K + 1:
        return np.zeros((T_eff, s_dim))

    rng = np.random.RandomState(seed)

    scale_enc = np.sqrt(2.0 / (D + h_dim))
    scale_dec = np.sqrt(2.0 / (h_dim + D))
    W_enc = rng.randn(D, h_dim) * scale_enc
    b_enc = np.zeros(h_dim)
    W_dec = rng.randn(h_dim, D) * scale_dec
    b_dec = np.zeros(D)

    n_groups = T_eff - K

    for _ in range(epochs):
        dW_enc = np.zeros_like(W_enc)
        db_enc = np.zeros_like(b_enc)
        dW_dec = np.zeros_like(W_dec)
        db_dec = np.zeros_like(b_dec)

        for g in range(n_groups):
            Y = windows[g:g + K + 1]  # (K+1, D)

            # Forward
            z_pre = Y @ W_enc + b_enc
            h = _tanh(z_pre)
            r_pre = h @ W_dec + b_dec
            x_hat = _tanh(r_pre)

            # Reconstruction gradient: d/d(x_hat) ||Y - x_hat||_2
            recon_diff = Y - x_hat
            recon_norms = np.linalg.norm(
                recon_diff, axis=1, keepdims=True,
            )
            safe_norms = np.where(recon_norms < 1e-12, 1.0, recon_norms)
            d_xhat = -recon_diff / safe_norms

            # Time-invariance gradient
            shared = h[:, :s_dim]
            d_h_ti = np.zeros_like(h)
            if K > 0 and s_dim > 0:
                for k in range(K):
                    diff_k = shared[k + 1] - shared[k]
                    norm_k = np.linalg.norm(diff_k)
                    if norm_k > 1e-12:
                        grad_k = diff_k / norm_k
                    else:
                        grad_k = np.zeros(s_dim)
                    d_h_ti[k + 1, :s_dim] += (lam / K) * grad_k
                    d_h_ti[k, :s_dim] -= (lam / K) * grad_k

            # Backprop through decoder
            d_r_pre = d_xhat * _tanh_deriv(r_pre)
            dW_dec += h.T @ d_r_pre
            db_dec += d_r_pre.sum(axis=0)
            d_h_from_dec = d_r_pre @ W_dec.T

            # Backprop through encoder
            d_h_total = d_h_from_dec + d_h_ti
            d_z_pre = d_h_total * _tanh_deriv(z_pre)
            dW_enc += Y.T @ d_z_pre
            db_enc += d_z_pre.sum(axis=0)

        dW_enc /= n_groups
        db_enc /= n_groups
        dW_dec /= n_groups
        db_dec /= n_groups

        W_enc -= lr * dW_enc
        b_enc -= lr * db_enc
        W_dec -= lr * dW_dec
        b_dec -= lr * db_dec

    # Extract shared features
    z_pre_all = windows @ W_enc + b_enc
    h_all = _tanh(z_pre_all)
    return h_all[:, :s_dim]


# ===========================================================================
# TIRE pipeline: windowing -> AE training -> dissimilarity scoring
# ===========================================================================

def _tire_scores(
    features: np.ndarray,   # (T, d) raw features
    window_size: int,       # N
    parallel_k: int,        # K
    td_latent_dim: int,     # h^TD total latent dim
    td_shared_dim: int,     # s^TD shared dim
    fd_latent_dim: int,     # h^FD total latent dim
    fd_shared_dim: int,     # s^FD shared dim
    lambda_td: float,       # time-invariance weight for TD
    lambda_fd: float,       # time-invariance weight for FD
    domain: str,            # "td", "fd", or "both"
    nfft: int,              # DFT points
    epochs: int,
    lr: float,
    seed: int,
) -> np.ndarray:
    """Full TIRE pipeline: from raw features to dissimilarity scores.

    Phases:
        1. Build TD and/or FD windows, scale to [-1, 1].
        2. Train autoencoders with time-invariance loss.
        3. Extract shared features, combine with quantile weighting,
           smooth, compute dissimilarity, smooth again.
        4. Map scores back to original time indices.

    Args:
        features: (T, d) un-normalized feature array.
        window_size: N, the sliding window size.
        parallel_k: K, number of parallel consecutive windows.
        td_latent_dim: Total latent dim for TD autoencoder.
        td_shared_dim: Shared dims in TD latent space.
        fd_latent_dim: Total latent dim for FD autoencoder.
        fd_shared_dim: Shared dims in FD latent space.
        lambda_td: Lambda for TD time-invariance penalty.
        lambda_fd: Lambda for FD time-invariance penalty.
        domain: "td", "fd", or "both".
        nfft: Number of DFT points for FD windows.
        epochs: Training epochs per autoencoder.
        lr: Learning rate.
        seed: Random seed.

    Returns:
        scores: (T,) dissimilarity scores. Zeros where windows are
                not available (first N-1 and last N entries).
    """
    T, _ = features.shape
    N = window_size

    # --- Phase 1: Windowing ---
    td_win = _td_windows(features, N)
    td_win_scaled, _, _ = _scale_to_unit(td_win)

    fd_win = _fd_windows(features, N, nfft)
    fd_win_scaled, _, _ = _scale_to_unit(fd_win)

    T_eff = td_win.shape[0]  # T - N + 1

    if T_eff < 2:
        return np.zeros(T)

    # --- Phase 2: Train autoencoders, extract shared features ---
    shared_td = None
    shared_fd = None

    if domain in ("td", "both"):
        shared_td = _train_autoencoder(
            td_win_scaled, td_latent_dim, td_shared_dim,
            lambda_td, parallel_k, epochs, lr, seed,
        )

    if domain in ("fd", "both"):
        shared_fd = _train_autoencoder(
            fd_win_scaled, fd_latent_dim, fd_shared_dim,
            lambda_fd, parallel_k, epochs, lr, seed + 1,
        )

    # --- Phase 3a: Combine with quantile-based weighting ---
    if domain == "both" and shared_td is not None and shared_fd is not None:
        if T_eff >= 2:
            td_diss = np.linalg.norm(
                shared_td[1:] - shared_td[:-1], axis=1,
            )
            fd_diss = np.linalg.norm(
                shared_fd[1:] - shared_fd[:-1], axis=1,
            )
            alpha = np.percentile(fd_diss, 95) if len(fd_diss) > 0 else 1.0
            beta = np.percentile(td_diss, 95) if len(td_diss) > 0 else 1.0
            # NOTE: Degenerate guard -- avoid zeroing out a domain.
            alpha = alpha if alpha > 1e-12 else 1.0
            beta = beta if beta > 1e-12 else 1.0
            combined = np.concatenate([
                alpha * shared_td,
                beta * shared_fd,
            ], axis=1)
        else:
            combined = np.concatenate([shared_td, shared_fd], axis=1)
    elif shared_td is not None:
        combined = shared_td
    elif shared_fd is not None:
        combined = shared_fd
    else:
        return np.zeros(T)

    # --- Phase 3b: Smooth features ---
    combined_smooth = _matched_filter(combined, N)

    # --- Phase 3c: Dissimilarity D_t = ||s_smooth_t - s_smooth_{t+N}||_2 ---
    n_valid = T_eff - N
    if n_valid <= 0:
        return np.zeros(T)

    diss_raw = np.zeros(T_eff)
    for t in range(n_valid):
        diss_raw[t] = np.linalg.norm(
            combined_smooth[t] - combined_smooth[t + N],
        )

    # --- Phase 3d: Smooth dissimilarity ---
    diss_smooth = _matched_filter(diss_raw, N)

    # --- Phase 4: Map to original time indices ---
    # Window index i -> original time t = i + N - 1
    scores = np.zeros(T)
    for i in range(T_eff):
        t_orig = i + N - 1
        if t_orig < T:
            scores[t_orig] = diss_smooth[i]

    return scores


# ===========================================================================
# Public API
# ===========================================================================

def detect(bundle: DataBundle, params: dict) -> Detections:
    """Detect change points using TIRE on precomputed features.

    This is an OFFLINE method: it trains on the entire sequence.

    Parameters (in params dict):
        threshold     : float -- Detection threshold on smoothed dissimilarity.
        window_size   : int   -- N, sliding window size (default 5).
        parallel_k    : int   -- K, parallel consecutive windows (default 2).
        td_latent_dim : int   -- Total TD latent dim (default 2).
        td_shared_dim : int   -- TD shared/time-invariant dim (default 1).
        fd_latent_dim : int   -- Total FD latent dim (default 1).
        fd_shared_dim : int   -- FD shared dim (default 1).
        lambda_td     : float -- TD time-invariance penalty (default 1.0).
        lambda_fd     : float -- FD time-invariance penalty (default 1.0).
        domain        : str   -- "td", "fd", or "both" (default "both").
        nfft          : int   -- DFT points for FD windows (default 30).
        epochs        : int   -- Training epochs (default 200).
        lr            : float -- Learning rate (default 1e-3).
        cooldown      : int   -- Min steps between detections.
        seed          : int   -- Random seed (default 42).
    """
    threshold = float(params["threshold"])
    cooldown = int(params["cooldown"])
    window_size = int(params.get("window_size", 5))
    parallel_k = int(params.get("parallel_k", 2))
    td_latent_dim = int(params.get("td_latent_dim", 2))
    td_shared_dim = int(params.get("td_shared_dim", 1))
    fd_latent_dim = int(params.get("fd_latent_dim", 1))
    fd_shared_dim = int(params.get("fd_shared_dim", 1))
    lambda_td = float(params.get("lambda_td", 1.0))
    lambda_fd = float(params.get("lambda_fd", 1.0))
    domain = str(params.get("domain", "both"))
    nfft = int(params.get("nfft", 30))
    epochs = int(params.get("epochs", 200))
    lr = float(params.get("lr", 1e-3))
    seed = int(params.get("seed", 42))

    scores = _tire_scores(
        bundle.features_raw,
        window_size, parallel_k,
        td_latent_dim, td_shared_dim,
        fd_latent_dim, fd_shared_dim,
        lambda_td, lambda_fd,
        domain, nfft, epochs, lr, seed,
    )
    start = 2 * window_size
    change_points = apply_cooldown(scores, threshold, cooldown, start=start)

    return Detections(
        change_points=change_points,
        scores=scores,
        threshold_used=threshold,
    )
