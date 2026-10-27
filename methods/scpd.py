"""SCPD (Huang, Danovitch, Rabusseau, Rabbany, PAKDD 2023), faithful port of the authors' code
(github.com/shenyangHuang/SCPD, subroutines/dos.py get_dos + DOS/* and subroutines/Anomaly_Detection.py).

Signature: 50-bin density of states of L_sym rescaled by (L - I) (range [0, 2]), from Nm = 20 Chebyshev moments,
Nz = 100 Rademacher probes np.sign(np.random.randn(n, Nz)), Jackson damping, bins from the Chebyshev CDF on
linspace(-1+1e-8, 1-1e-8, 51) with negative bin counts clipped to 0 (plot_cheb.plot_chebhist).
Score: L2-normalised signatures; for each window w in (ws, wl), Z = 1 - |cos(s_t, u1(S[t-w:t]))| with u1 the
principal left singular vector; difference score Z_t - Z_{t-1}; max over windows; first max(w)+1 steps set to 0;
the top round(T * percent) steps are the anomalies.

Fixes needed to run the shipped code (not changes of method): Anomaly_Detection.py uses LA.norm and
metrics.compute_accuracy without importing them; Hits@n follows LAD's compute_CPD.compute_accuracy
(exact index match, correct / len(real_events)).
"""
import numpy as np
import scipy.sparse as sp

NZ, NM, NPTS = 100, 20, 50


def lsym(W):
    """networkx normalized_laplacian_matrix on a self-loop-free weighted graph; degree-0 rows give L_ii = 0 (nx)."""
    W = sp.csr_matrix(W, dtype=float)
    d = np.asarray(W.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        s = 1.0 / np.sqrt(d)
    s[~np.isfinite(s)] = 0.0
    S = sp.diags(s)
    return sp.csr_matrix(S @ (sp.diags(d) - W) @ S)


def jackson(c):
    N = c.shape[0]
    n = np.arange(N)
    tau = np.pi / (N + 1)
    g = ((N - n + 1) * np.cos(tau * n) + np.sin(tau * n) / np.tan(tau)) / (N + 1)
    return g * c


def moments(H, nZ=NZ, N=NM, rng=None):
    """moment_comp.moments_cheb_dos: tr(T_k(H)) estimates, k < N, probes sign(randn) from the legacy global RNG."""
    n = H.shape[0]
    Z = np.sign((np.random.randn if rng is None else rng.randn)(n, nZ))
    c = np.zeros(N)
    TVp, TVk = Z, H @ Z
    c[0] = np.sum(Z * TVp) / nZ
    c[1] = np.sum(Z * TVk) / nZ
    for i in range(2, N):
        TV = 2 * (H @ TVk) - TVp
        TVp, TVk = TVk, TV
        c[i] = np.sum(Z * TVk) / nZ
    return c


def chebhist(c, npts=NPTS, matlab=False):
    """Python plot_chebhist (grid +-(1-1e-8), negative bins clipped); matlab=True: NetworkDOS plot_chebhist.m
    as called by ADOS/compute_dos.m (grid linspace(-1, 1, 51), no clipping), used for the flight embedding."""
    xx = np.linspace(-1, 1, npts + 1) if matlab else np.linspace(-1 + 1e-8, 1 - 1e-8, npts + 1)
    txx = np.arccos(xx)
    yy = c[0] * (txx - np.pi) / 2
    for k in range(1, len(c)):
        yy = yy + c[k] * np.sin(k * txx) / k
    yy = yy * (-2 / np.pi)
    h = yy[1:] - yy[:-1]
    if not matlab:
        h[h < 0] = 0
    return h


def dos(W, rng=None, nZ=NZ, N=NM, npts=NPTS, matlab=False):
    """get_dos(L, Nmoment=20, nZ=100, npts=50) on the graph with adjacency W (rescale range [0, 2] -> H = L - I)."""
    L = lsym(W)
    H = (L - sp.eye(L.shape[0], format="csr")).tocsr()
    return chebhist(jackson(moments(H, nZ, N, rng)), npts, matlab)


def score(spec, windows, percent_ranked, difference=True):
    """Anomaly_Detection.change_detection_many_windows (principal=True); returns (z_overall, anomalies)."""
    S = np.asarray(spec, float).reshape(len(spec), -1)
    nr = np.linalg.norm(S, axis=1, keepdims=True)
    nr[nr == 0] = 1
    S = S / nr                                            # sklearn normalize(norm='l2')
    T, iw = len(S), max(windows)
    Z = [[0.0] * iw for _ in windows]
    for i in range(iw, T):
        for j, w in enumerate(windows):
            if w > i:
                Z[j].append(0.0)
                continue
            u = np.linalg.svd(S[i - w:i].T, full_matrices=False)[0][:, 0]
            cur = S[i]
            Z[j].append(1 - abs(np.dot(cur, u) / np.linalg.norm(cur) / np.linalg.norm(u)))
    if difference:
        Z = [[z[0]] + [z[k] - z[k - 1] for k in range(1, len(z))] for z in Z]
    zo = np.array([0.0 if k < iw else max(z[k] for z in Z) for k in range(T)])
    zo[:iw + 1] = 0
    k = round(T * percent_ranked)
    out = np.sort(zo.argsort()[-k:][::-1])
    return zo, out


def hits(anomalies, real_events):
    return sum(int(a in set(real_events)) for a in anomalies) / len(real_events)
