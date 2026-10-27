"""Step 2: per-snapshot counts and the KPM moments m_0..m_50 (five probe seeds) of the normalized-Laplacian spectrum.

The counts (active vertices and edges without self-loops) are the input of the count detector; the moments are the
density of states of SCPD and LADdos in step 3. Probe seed of snapshot t: base + t, base in {0, 1000, 2000, 3000,
4000} (keys s0..s4). Snapshots without edges (m_j = T_j(0)) or with one vertex and a self-loop (m_j = (-1)^j) are
computed by the same recurrence, which is exact there, and asserted against these closed forms.
Output: repro/data/moments_<ds>.npz.

Usage: 02_moments.py [dataset ...]
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from common import CHECK_SEED, DATA, K_MOM, NAMES, R_PROBES, SEEDS, Timer, adjacency, load_graphs, peak_rss_mb
from scipy.sparse import diags, eye


def laplacian(A):
    deg = np.asarray(A.sum(axis=1)).ravel()
    d = diags(1.0 / np.sqrt(np.where(deg == 0, 1, deg)), format="csr")
    return eye(A.shape[0], format="csr") - d @ A @ d


def counts(A) -> dict:
    """Vertices, and active vertices and edges without self-loops, of snapshot A."""
    B = (A - diags(A.diagonal())).tocsr()
    B.eliminate_zeros()
    return {"n": A.shape[0], "active_noloop": int((np.asarray(B.sum(1)).ravel() > 0).sum()), "edges_noloop": B.nnz / 2}


def run(name: str) -> str:
    from utils.spectral import chebyshev_moments_sparse
    t0 = Timer()
    g = load_graphs(name)
    T = len(g["n"])
    bases = {**SEEDS, **CHECK_SEED}
    mom = {k: np.zeros((T, K_MOM + 1)) for k in bases}
    cols = ("n", "active_noloop", "edges_noloop")
    q = {c: np.zeros(T) for c in cols}
    degenerate, notes = [], []
    Tj0 = np.cos(np.arange(K_MOM + 1) * np.pi / 2).round()
    t_kpm = 0.0                                      # wall time of the KPM calls (all seeds)
    for i in range(T):
        n = int(g["n"][i])
        if n == 0:                                   # no vertex: no measure; use delta_0 (does not occur here)
            for k in bases:
                mom[k][i] = Tj0
            degenerate.append(i)
            notes.append(f"t={i}: n=0, set to T_j(0)")
            continue
        A = adjacency(g, i)
        e = counts(A)
        for c in cols:
            q[c][i] = e[c]
        L = laplacian(A)
        tk = time.perf_counter()
        for k, base in bases.items():
            mom[k][i] = chebyshev_moments_sparse(L, K_MOM, R_PROBES, seed=base + i)
        t_kpm += time.perf_counter() - tk
        if n <= 1 or A.nnz == 0:                     # degenerate: check the KPM result against the closed form
            if A.nnz == 0:
                ref = Tj0
            else:                                    # single vertex with a self-loop: S = [1], m_j = (-1)^j
                ref = np.cos(np.arange(K_MOM + 1) * np.pi)
            err = max(np.abs(mom[k][i] - ref).max() for k in bases)
            assert err < 1e-12, (name, i, err)
            degenerate.append(i)
            notes.append(f"t={i}: n={n}, nnz={A.nnz}, m_1..m_4 = {np.round(mom['s0'][i, 1:5], 6).tolist()}")
    zero_cached = np.nonzero(np.abs(g["moments_cached"]).sum(1) == 0)[0]
    ok = np.setdiff1d(np.arange(T), zero_cached)
    err42 = float(np.abs(mom["s42"][ok] - g["moments_cached"][ok]).max())
    np.savez(DATA / f"moments_{name}.npz", **mom, **q, true_cps=g["true_cps"],
             degenerate=np.array(degenerate, np.int64), weights_max=float(g["data"].max()) if g["data"].size else 0,
             kpm_seconds_per_seed=t_kpm / len(bases))
    lines = [f"{name}: T={T}; degenerate snapshots {degenerate} {notes}",
             (f"  cached bundle rows that are all zero: {zero_cached.tolist()}; "
              f"max |s42 - cached| on the other rows {err42:.1e}"),
             (f"  KPM time for one probe seed (K = {K_MOM}, R = {R_PROBES}, all {T} snapshots): "
              f"{t_kpm / len(bases):.1f} s"),
             (f"  adjacency weights max {g['data'].max() if g['data'].size else 0}; "
              f"{t0():.1f} s, peak RSS {peak_rss_mb():.0f} MB")]
    return "\n".join(lines)


if __name__ == "__main__":
    names = sys.argv[1:] or list(NAMES)
    t = Timer()
    with ProcessPoolExecutor(max_workers=min(3, len(names))) as ex:
        for msg in ex.map(run, names):
            print(msg, flush=True)
    print(f"step 2 total {t():.1f} s")
