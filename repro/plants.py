"""Planted changes on real backgrounds: plants, per-snapshot statistics, diagnostic rows and metrics (steps 14, 16).

Plants act on every snapshot independently, rng = default_rng([seed, MODES.index(mode), round(1000 rho)]):
  rand      degree-preserving double-edge swaps, round(rho |E| / 2) accepted swaps (mixes; destroys triangles)
  assort    the same swaps, accepted only if they raise sum d_u d_v (degree pairing at fixed degrees)
  triclose  the same swaps, accepted only if the triangle count rises
  thin      delete a share rho of the edges uniformly (activity / density control)
  deact     delete every edge of a share rho of the active vertices (activity control)
  swap      round(rho n_+ / 2) disjoint pairs of active vertices exchange labels (isomorphic: a blind class)
A trial splices planted snapshots into the original series on blocks [t0, t0 + 15), t0 = s0 + 20 + o + 40 i in each
segment (offset o). A trial is (seed, offset, t0): pos = max score over [t0, t0 + 5) in the spliced series, neg = the
same window in the original series. Paired AUC = mean over trials of 1[pos > neg] + 1[pos = neg] / 2, 95% interval
by a block bootstrap whose unit is one period (segment, i): every offset and seed of that period stays together. The
seeds randomize only the plant; the background is fixed. `auc` (Mann-Whitney over all pos and all neg windows) is the
unpaired window-max AUC.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
from common import bundle, src_sha
from evaluate import SEED, online_scale, window_scores
from scipy.sparse.linalg import eigsh
from scipy.stats import rankdata

from data.registry import compute_lad_eigenvalues, extract_features_raw, start_vector
from methods import bocpd, cusum, ewma, lad, mmd
from methods.interface import seg_ranges
from methods.scpd import jackson

MODES = ("none", "rand", "assort", "thin", "deact", "triclose", "swap")
STRUCT = ("rand", "assort", "triclose")
ACTIV = ("thin", "deact")
START, PERIOD, LENGTH, H = 20, 40, 15, 5
NB_PAIRED = 2000                                           # block-bootstrap draws of the paired AUC
TIE_MAX = 0.5                                              # tie fraction at or above which a row is not evaluable


# ------------------------------------------------------------------------------------------------ plants
def edges_of(A):
    U = sp.triu(A, k=1).tocoo()
    return np.c_[U.row, U.col].astype(np.int64), A.diagonal()


def build(n, E, loops):
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    A = A + A.T
    if loops.any():
        A = A + sp.diags(loops)
    A.data[:] = np.minimum(A.data, 1.0)
    return sp.csr_matrix(A)


def swap_edges(E, rho, rng, accept=None):
    """Double-edge swaps (a, b), (c, d) -> (a, d), (c, b) until round(rho |E| / 2) are accepted (50x tries).

    accept: None (rand), ("assort", deg) or ("tri", None); the triangle change is exact and sequential.
    """
    m = len(E)
    target = round(rho * m / 2)
    if m < 2 or target == 0:
        return E, 0
    E = E.copy()
    key = lambda a, b: (a, b) if a < b else (b, a)
    S = set(map(tuple, E.tolist()))
    nb = {}
    if accept and accept[0] == "tri":
        for a, b in E.tolist():
            nb.setdefault(a, set()).add(b)
            nb.setdefault(b, set()).add(a)
    done = tries = 0
    while done < target and tries < 50 * target:
        tries += 1
        i, j = rng.integers(m, size=2)
        if i == j:
            continue
        a, b = E[i]
        c, d = E[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        e1, e2 = key(a, d), key(c, b)
        if e1 in S or e2 in S:
            continue
        if accept and accept[0] == "assort":
            deg = accept[1]
            if deg[a] * deg[d] + deg[c] * deg[b] <= deg[a] * deg[b] + deg[c] * deg[d]:
                continue
        if accept and accept[0] == "tri":           # remove (a,b), (c,d); add (a,d), (c,b); count triangles on the way
            nb[a].discard(b); nb[b].discard(a); dt = -len(nb[a] & nb[b])
            nb[c].discard(d); nb[d].discard(c); dt -= len(nb[c] & nb[d])
            dt += len(nb[a] & nb[d]); nb[a].add(d); nb[d].add(a)
            dt += len(nb[c] & nb[b]); nb[c].add(b); nb[b].add(c)
            if dt <= 0:
                nb[a].discard(d); nb[d].discard(a); nb[c].discard(b); nb[b].discard(c)
                nb[a].add(b); nb[b].add(a); nb[c].add(d); nb[d].add(c)
                continue
        S.discard(key(a, b)); S.discard(key(c, d)); S.add(e1); S.add(e2)
        E[i] = e1
        E[j] = e2
        done += 1
    return E, done


def plant(A, mode, rho, rng):
    """(planted csr snapshot, accepted swaps or nan)."""
    if mode == "none" or rho == 0:
        return sp.csr_matrix(A), np.nan
    n = A.shape[0]
    if mode == "swap":
        d = np.asarray(A.sum(1)).ravel()
        act = np.nonzero(d > 0)[0]
        k = round(rho * len(act) / 2)
        perm = np.arange(n)
        if k > 0:
            v = rng.choice(act, 2 * k, replace=False)
            perm[v[:k]], perm[v[k:]] = v[k:], v[:k]
        return sp.csr_matrix(A[perm][:, perm]), float(k)
    E, loops = edges_of(A)
    deg = np.asarray(A.sum(1)).ravel()
    done = np.nan
    if mode == "rand":
        E, _ = swap_edges(E, rho, rng)
    elif mode == "assort":
        E, _ = swap_edges(E, rho, rng, ("assort", deg))
    elif mode == "triclose":
        E, done = swap_edges(E, rho, rng, ("tri", None))
        done = float(done)
    elif mode == "thin":
        E = E[rng.random(len(E)) >= rho]
    elif mode == "deact":
        act = np.nonzero(deg > 0)[0]
        off = np.zeros(n, bool)
        off[rng.choice(act, round(rho * len(act)), replace=False)] = True
        E = E[~(off[E[:, 0]] | off[E[:, 1]])]
        loops = np.where(off, 0, loops)
    else:
        raise ValueError(mode)
    return sp.csr_matrix(build(n, E, loops)), done


def validity(A, P):
    """max |d_planted - d| (no self-loops), change of |E|, change of the sorted degrees, share of moved edges."""
    Al = A - sp.diags(A.diagonal())
    Pl = P - sp.diags(P.diagonal())
    d0 = np.asarray(Al.sum(1)).ravel()
    d1 = np.asarray(Pl.sum(1)).ravel()
    U0 = sp.triu(Al, 1).tocoo()
    U1 = sp.triu(Pl, 1).tocoo()
    s0 = set(zip(U0.row.tolist(), U0.col.tolist()))
    s1 = set(zip(U1.row.tolist(), U1.col.tolist()))
    return (float(np.abs(d1 - d0).max()) if len(d0) else 0.0, float(len(s1) - len(s0)),
            float(np.abs(np.sort(d1) - np.sort(d0)).max()) if len(d0) else 0.0,
            float(len(s0 - s1) / len(s0)) if s0 else 0.0)


# ------------------------------------------------------------------------------------------------ statistics
def structure_stats(A):
    """Triangles, transitivity, lambda_max(A) and Newman r of the active part (self-loops removed)."""
    Al = A - sp.diags(A.diagonal())
    Al.eliminate_zeros()
    d = np.asarray(Al.sum(1)).ravel()
    act = np.nonzero(d > 0)[0]
    B = Al[act][:, act]
    tri = float((B @ B).multiply(B).sum()) / 6 if B.nnz else 0.0
    wedges = float((d * (d - 1)).sum()) / 2
    trans = 3 * tri / wedges if wedges > 0 else 0.0
    if B.shape[0] > 2 and B.nnz:
        lam = (float(eigsh(B, k=1, which="LA", v0=start_vector(B.shape[0]), return_eigenvectors=False)[0])
               if B.shape[0] > 50
               else float(np.linalg.eigvalsh(B.toarray())[-1]))
    else:
        lam = 0.0
    Uc = sp.triu(B, 1).tocoo()
    da, db = d[act][Uc.row], d[act][Uc.col]
    x, y = np.r_[da, db], np.r_[db, da]
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 else 0.0
    return {"tri": tri, "trans": trans, "lam": lam, "assort": r}


def lad_nl(A):
    """20 largest eigenvalues of L_sym of the active part (dense), unit-normalised."""
    A = sp.csr_matrix(A, dtype=float)
    d = np.asarray(A.sum(1)).ravel()
    act = np.nonzero(d > 0)[0]
    if len(act) < 3:
        return np.zeros(20)
    assert len(act) < 5000, "dense LAD-NL only"
    isq = sp.diags(1 / np.sqrt(d[act]))
    L = sp.eye(len(act)) - isq @ A[act][:, act] @ isq
    ev = np.linalg.eigvalsh(L.toarray())[::-1][:20]
    z = np.zeros(20)
    z[:len(ev)] = ev
    nr = np.linalg.norm(z)
    return z / nr if nr > 1e-12 else z


# DOS row: 50 bins of L_sym on [0, 2] from 20 Jackson-damped KPM moments, one fixed block of 100 Rademacher probes.
# Not SCPD (methods/scpd.py): isolated vertices sit at lambda = 1, no clipping of negative bins,
# and the score has no first difference, so it isolates the signature from SCPD's scoring.
NZ, NMOM, NBIN = 100, 20, 50


def _binmat():
    jack = jackson(np.ones(NMOM))
    th = np.arccos(np.clip(np.linspace(0.0, 2.0, NBIN + 1) - 1.0, -1, 1))
    j = np.arange(1, NMOM)
    return np.c_[(th[:-1] - th[1:]) / np.pi,
                 2 * (np.sin(np.outer(th[:-1], j)) - np.sin(np.outer(th[1:], j))) / (np.pi * j)] * jack


BINMAT = _binmat()


def probes(n):
    return 2.0 * np.random.default_rng(0).integers(0, 2, size=(n, NZ)).astype(np.float64) - 1.0


def dos(A, Z):
    deg = np.asarray(A.sum(axis=1)).ravel()
    dg = sp.diags(1.0 / np.sqrt(np.where(deg == 0, 1, deg)), format="csr")
    n = A.shape[0]
    X = sp.csr_matrix(sp.eye(n, format="csr") - dg @ A @ dg - sp.eye(n, format="csr"))
    T0, T1 = Z, X @ Z
    mu = np.empty(NMOM)
    mu[0] = float((Z * T0).sum()) / (n * NZ)
    mu[1] = float((Z * T1).sum()) / (n * NZ)
    for j in range(2, NMOM):
        T0, T1 = T1, 2 * (X @ T1) - T0
        mu[j] = float((Z * T1).sum()) / (n * NZ)
    return BINMAT @ mu


def snapshot_stats(A, Z):
    """Every per-snapshot quantity a row or a table needs."""
    A = sp.csr_matrix(A, dtype=np.float64)
    d = np.asarray(A.sum(1)).ravel()
    nl = int((A.diagonal() != 0).sum())
    sumd2, nplus = float((d ** 2).sum()), int((d > 0).sum())
    out = {"n": A.shape[0], "nplus": nplus, "vol": float(d.sum()), "sumd2": sumd2,
           "E": (A.nnz - nl) / 2 + nl}
    out.update(structure_stats(A))
    Al = A - sp.diags(A.diagonal())
    Al.eliminate_zeros()
    out["nact_nl"] = float((np.asarray(Al.sum(1)).ravel() > 0).sum())
    out["E_nl"] = float(Al.nnz / 2)
    out["dhet"] = sumd2 / nplus if nplus > 0 else 0.0
    G = nx.from_scipy_sparse_array(A)
    out["raw8"] = extract_features_raw(G)
    out["ladu"] = compute_lad_eigenvalues(G, 20)
    out["lad_nl"] = lad_nl(A)
    out["dos"] = dos(A, Z[:A.shape[0]])
    return out


def series_stats(mats, mode, rho, seed):
    """Plant every snapshot in order; statistics of the planted series plus plant validity."""
    rng = np.random.default_rng([seed, MODES.index(mode), round(rho * 1000)])
    Z = probes(max(A.shape[0] for A in mats))
    rows, val, ndone = [], [], []
    for A in mats:
        P, done = plant(A, mode, rho, rng)
        val.append(validity(A, P))
        ndone.append(done)
        rows.append(snapshot_stats(P, Z))
    z = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    val = np.array(val)
    z.update(maxdd=val[:, 0], dE=val[:, 1], ddsorted=val[:, 2], moved=val[:, 3], ndone=np.array(ndone))
    return z


STATS_SHA = src_sha(edges_of, build, swap_edges, plant, validity, structure_stats, lad_nl, _binmat,
                    probes, dos, snapshot_stats, series_stats)


# fixed settings of the benchmark (one per method, never tuned on plants)
SET = {"cusum": {"threshold": 45.0, "kappa": 1.5, "startup_period": 10, "cooldown": 7},
       "ewma": {"threshold": 30.0, "lambda_param": 0.05, "startup_period": 5, "cooldown": 3},
       "bocpd": {"threshold": 0.1, "hazard_rate": 0.1, "cooldown": 5},
       "mmd": {"threshold": 0.0005, "window_size": 3, "cooldown": 5},
       "lad": {"threshold": 0.0005, "num_eigen": 5, "short_window": 7, "long_window": 10, "cooldown": 5},
       "lad_nl": {"threshold": 0.0, "num_eigen": 20, "short_window": 2, "long_window": 10, "cooldown": 1},
       "tire": {"window_size": 5, "parallel_k": 1, "td_latent_dim": 3, "td_shared_dim": 2, "lambda_td": 1.0,
                "lambda_fd": 1.0, "domain": "both", "epochs": 100, "threshold": 0.0, "cooldown": 1, "seed": 42}}
BASELINES = {"cusum": cusum, "ewma": ewma, "bocpd": bocpd, "mmd": mmd}
BURN = {"cusum": 10, "ewma": 5, "bocpd": 15, "mmd": 6, "lad": 17, "lad_nl": 17, "dos": 10}


# ------------------------------------------------------------------------------------------------ rows (steps 14, 16)
def feats(z):
    return {"cnt": np.c_[z["nact_nl"], z["E_nl"]],
            "dhet": z["dhet"], "tritr": np.c_[z["tri"], z["trans"]], "trans": z["trans"], "assort": z["assort"],
            "ladu": z["ladu"], "lad_nl": z["lad_nl"], "dos": z["dos"], "raw8": z["raw8"]}


def per_segment(seg, f, start):
    out = np.full(len(seg), np.nan)
    for s0, s1 in seg_ranges(seg):
        d = np.asarray(f(s0, s1), float)
        d[:start] = np.nan
        out[s0:s1] = d
    return out


def score_all(F, seg, U):
    """Score series of every row on one (spliced or original) feature set."""
    z = lambda X: online_scale(X, seg)[0]
    D = {"constant": 1.0 + 1e-9 * U,                                   # fixed random tie-break: AUC 0.5 exactly
         "counts": window_scores(z(F["cnt"]), seg, w=2, w_ref=5, mode="gam"),
         "dhet": window_scores(z(F["dhet"]), seg),
         "transitivity": window_scores(z(F["tritr"]), seg),
         "assortativity": window_scores(z(F["assort"]), seg)}
    D["fair_pair"] = np.fmax(window_scores(z(F["trans"]), seg), D["assortativity"])
    for r, key in (("lad", "ladu"), ("lad_nl", "lad_nl")):
        D[r] = per_segment(seg, lambda s0, s1, k=key, r=r: lad.detect(bundle(eig=F[k][s0:s1]), SET[r]).scores,
                           BURN[r])
    D["dos"] = per_segment(seg, lambda s0, s1: dos_scores(F["dos"][s0:s1]), BURN["dos"])
    for b, mod in BASELINES.items():
        D[b] = per_segment(seg, lambda s0, s1, b=b, mod=mod: mod.detect(bundle(X=F["raw8"][s0:s1]), SET[b]).scores,
                           BURN[b])
    return D


# ------------------------------------------------------------------------------------------------ trials, metrics
def block_onsets(seg, o):
    out = []
    for s0, s1 in seg_ranges(np.asarray(seg)):
        t = s0 + START + o
        while t + LENGTH <= s1:
            out.append(t)
            t += PERIOD
    return np.array(out, int)


def splice(base, planted, t0s):
    out = np.array(base, copy=True)
    for t0 in t0s:
        out[t0:t0 + LENGTH] = planted[t0:t0 + LENGTH]
    return out


def auc(pos, neg):
    """Mann-Whitney AUC, ties count 1/2."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(np.r_[pos, neg])
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def win_max(D, t0s):
    return np.array([np.nanmax(D[t0:t0 + H]) for t0 in t0s])


def block_ids(seg, t0s):
    """Bootstrap unit of each onset: segment index * 10^6 + period index i (t0 = s0 + START + o + PERIOD i)."""
    starts = np.array([a for a, _ in seg_ranges(np.asarray(seg))])
    t0s = np.asarray(t0s, np.int64)
    k = np.searchsorted(starts, t0s, side="right") - 1
    return k * 10 ** 6 + (t0s - starts[k] - START) // PERIOD


def paired_scores(pos, neg) -> np.ndarray:
    """Per trial 1[pos > neg] + 1[pos = neg] / 2 (nan where a window has no finite score)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    out = (pos > neg) + 0.5 * (pos == neg)
    return np.where(np.isfinite(pos) & np.isfinite(neg), out, np.nan)


def block_draws(x, blk, B: int = NB_PAIRED, seed: int = SEED) -> np.ndarray:
    """(B,) block-bootstrap means of x; the same (blk, B, seed) gives the same draws for every row (paired)."""
    ok = np.isfinite(x)
    u, inv = np.unique(np.asarray(blk)[ok], return_inverse=True)
    S, n = np.bincount(inv, x[ok], len(u)), np.bincount(inv, minlength=len(u)).astype(float)
    idx = np.random.default_rng(seed).integers(len(u), size=(B, len(u)))
    return S[idx].sum(1) / n[idx].sum(1)


def paired_summary(pos, neg, blk) -> dict:
    """Paired AUC with its block-bootstrap interval, mean increment in null-SD units, tie and zero fractions."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    x = paired_scores(pos, neg)
    ok = np.isfinite(x)
    if not ok.any():
        return {"auc_paired": np.nan}
    d = block_draws(x, blk)
    sd = float(np.std(neg[ok]))
    tie = float(np.mean(pos[ok] == neg[ok]))
    return {"auc_paired": float(np.mean(x[ok])), "auc_paired_ci95": [float(np.quantile(d, 0.025)),
                                                                  float(np.quantile(d, 0.975))],
            "increment_null_sd": float(np.mean(pos[ok] - neg[ok]) / sd) if sd > 0 else np.nan,
            "tie_fraction": tie, "zero_fraction": float(np.mean(np.r_[pos[ok], neg[ok]] == 0)),
            "evaluable": tie < TIE_MAX, "n_trials": int(ok.sum()), "n_blocks": len(np.unique(blk[ok]))}


def paired_contrast(pos_a, neg_a, pos_b, neg_b, blk) -> dict:
    """Paired AUC of row a minus row b on the same trials and the same block resample."""
    xa, xb = paired_scores(pos_a, neg_a), paired_scores(pos_b, neg_b)
    ok = np.isfinite(xa) & np.isfinite(xb)
    blk = np.asarray(blk)[ok]
    d = block_draws(xa[ok], blk) - block_draws(xb[ok], blk)
    return {"diff": float(np.mean(xa[ok]) - np.mean(xb[ok])),
            "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}


class Metrics:
    """Window maxima of one row over the trials: planted (pos) and original (neg) at the same onsets, and the
    bootstrap block of each trial; result() gives the unpaired and the paired window-max AUC."""

    def __init__(self, Du, seg):
        self.Du, self.seg = Du, seg
        self.pos, self.neg, self.blk = [], [], []

    def add(self, Dp, t0s):
        self.pos.append(win_max(Dp, t0s))
        self.neg.append(win_max(self.Du, t0s))
        self.blk.append(block_ids(self.seg, t0s))

    def trials(self) -> dict:
        return {"pos": np.concatenate(self.pos), "neg": np.concatenate(self.neg), "blk": np.concatenate(self.blk)}

    def result(self):
        return trial_result(**self.trials())


def trial_result(pos, neg, blk) -> dict:
    return {"auc": auc(pos, neg), "n_pos": len(pos), "n_neg": len(neg), **paired_summary(pos, neg, blk)}


def effect(zp, zb, seg):
    """Median |planted - original| from s0 + 20 over the median |first difference| of the original series."""
    ok = np.zeros(len(seg), bool)
    for s0, s1 in seg_ranges(seg):
        ok[s0 + START:s1] = True
    q = {"E": ("E", None), "nplus": ("nplus", None),
         "trans": ("trans", None), "assort": ("assort", None), "lam": ("lam", None), "dhet": ("dhet", None)}
    out = {}
    for k, (f, j) in q.items():
        p, b = (zp[f], zb[f]) if j is None else (zp[f][:, j], zb[f][:, j])
        nat = np.median(np.abs(np.concatenate([np.diff(b[s0:s1]) for s0, s1 in seg_ranges(seg)])))
        out[k] = float(np.median(np.abs(p - b)[ok]) / max(nat, 1e-12))
    return out


def dos_scores(sig, ws=5, wl=10):
    """max over w in (5, 10) of 1 - |cos(sigma_t, first right singular vector of sigma_{t-w..t-1})|, no difference."""
    T = len(sig)
    S = [v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-12 else None for v in sig]
    Z = np.full(T, np.nan)
    for t in range(wl, T):
        if S[t] is None:
            continue
        zs = []
        for w in (ws, wl):
            ctx = [S[i] for i in range(t - w, t) if S[i] is not None]
            if len(ctx) < 2:
                continue
            _, sv, Vt = np.linalg.svd(np.array(ctx), full_matrices=False)
            if sv[0] < 1e-12:
                continue
            zs.append(1.0 - abs(float(S[t] @ Vt[0])))
        Z[t] = max(zs) if zs else 0.0
    return Z
