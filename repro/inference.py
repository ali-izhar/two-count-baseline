"""Inference on per-event scores of the benchmark: paired bootstraps at three levels, exact sign-flip tests, Holm, rank
distributions, coverage and power.

Every bootstrap here is paired: one resample of datasets / clusters / events serves every row, so a contrast of two
rows uses the same draws. Seeds and draw counts are evaluate.SEED and evaluate.NBOOT unless stated.

API
  pooled_draws(E, level, weight="equal", B, seed, clusters=None) -> (B, R) pooled means
        E: {dataset: (n_events, R) array}, R rows in a fixed order.
        level "event"      events resampled within each dataset (conditional on the datasets: the paper's interval)
              "two_stage"  datasets resampled, then events within each drawn dataset (each copy its own draw)
              "three_stage" datasets resampled, then the clusters of each drawn dataset (clusters: {dataset: label
                           per event}, e.g. subjects), then events within each drawn cluster; the dataset mean is the
                           mean over its drawn events; a dataset without labels resamples its events (two-stage)
              "dataset"    datasets resampled, the per-dataset means fixed
        weight "equal" (mean of the per-dataset means) or "event" (mean over all drawn events)
  pooled_point(E, weight) -> (R,)               the point estimate of the same pool
  pval(d) -> two-sided bootstrap p of a contrast from its draws: min(1, 2 min(P(d <= 0), P(d >= 0)))
  holm(p) -> Holm-adjusted p-values (same order)
  ranks(draws) -> (B, R) ranks (1 = highest) of each row in each draw
  signflip(x, alpha=0.05) -> dict(mean, p, ci95, exact, bounded)
        exact sign-flip test of mean 0 (all 2^n patterns if n <= 12, else the identity and 5,000 random ones)
        and the interval of shifts it does not reject (grid step 0.002 in [-1, 1]); with n <= 5 nothing is
        rejected at 5%
  coverage(pool, ns, reps, B, seed) -> {n: {"percentile": c, "signflip": c}}
        coverage at mean 0 of the percentile bootstrap and of the sign-flip interval, n events drawn from the
        centred values `pool`
  ep_variant(x, mask, seg, events, tol, all_events=None, guard=0, within=False) -> {e: EP_e}
                                                evaluate.ep with the null windows that start in [e, e + tol + 1 +
                                                guard) after a labelled event also removed (guard band), and
                                                (within) each event ranked among the null windows of its own segment
                                                (or of its own `group`, e.g. the case of a multi-file subject)
  t_interval(x) -> dict(mean, ci95, p)          Student t_{n-1} interval and two-sided p of the mean of x (one value
                                                per dataset: the small-sample dataset-level interval)
  upper_bounds(d, point, x, alpha=0.05) -> dict  one-sided upper bounds of R contrasts, simultaneous over the R:
                                                max-t and Bonferroni from the bootstrap draws d (B, R); Bonferroni
                                                t_{K-1} and Bonferroni exact sign-flip from the per-dataset values
                                                x (K, R)
  signflip_upper(x, alpha) -> (bound, bounded)  the largest shift not rejected by the one-sided exact sign-flip test
                                                of mean = shift against mean < shift (grid step 0.001 in [-1, 1])
  best_set(draws, point, alpha=0.05) -> dict    rows not excluded from being best (bootstrap form of Hsu's multiple
                                                comparisons with the best): row r is excluded if some j has
                                                (pt_j - pt_r) - c se_jr > 0, se_jr the bootstrap SD of the contrast,
                                                c the (1 - alpha) quantile over draws of the largest studentized
                                                centred contrast over all ordered pairs
  coverage_pooled(D, reps, B, seed) -> dict     super-population coverage of three intervals of an equal-weight
                                                pooled mean: datasets drawn with replacement from D ({dataset:
                                                per-event values}, dataset effects centred to mean 0, each keeps its
                                                event count and deviations), events resampled; covers 0?
  mde80(se) -> (1.959964 + 0.841621) se: the lead with 80% power
  pct(null, x) -> mid-rank percentile of each x among null
"""
from __future__ import annotations

import itertools

import numpy as np
from evaluate import NBOOT, SEED, _wmax, windows
from scipy.stats import t as student

Z95, Z80 = 1.959963984540054, 0.8416212335729143


# ------------------------------------------------------------------------------------------------ pooled bootstrap
def _event_draws(X, B: int, rng) -> np.ndarray:
    """(B, R) means of n events drawn with replacement from the rows of X (n, R)."""
    n = len(X)
    return X[rng.integers(n, size=(B, n))].mean(1)


def pooled_point(E: dict, weight: str = "equal") -> np.ndarray:
    names = sorted(E)
    if weight == "equal":
        return np.mean([E[d].mean(0) for d in names], axis=0)
    return np.concatenate([E[d] for d in names]).mean(0)


def _cluster_draws(X, cl, B: int, rng) -> np.ndarray:
    """(B, R) means of the events of nc clusters drawn with replacement, each drawn cluster's events resampled."""
    mem = [np.flatnonzero(cl == c) for c in np.unique(cl)]
    pick = rng.integers(len(mem), size=(B, len(mem)))
    S, N = np.zeros((B, X.shape[1])), np.zeros(B)
    for i in range(len(mem)):
        for c, m in enumerate(mem):
            b = np.flatnonzero(pick[:, i] == c)
            if b.size:
                S[b] += X[m[rng.integers(len(m), size=(b.size, len(m)))]].sum(1)
                N[b] += len(m)
    return S / N[:, None]


def pooled_draws(E: dict, level: str, weight: str = "equal", B: int = NBOOT, seed: int = SEED,
                 clusters: dict | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    names = sorted(E)
    D = len(names)
    n = np.array([len(E[d]) for d in names], float)
    if level == "event":
        M = np.stack([_event_draws(E[d], B, rng) for d in names], 1)              # (B, D, R)
        w = np.ones(D) if weight == "equal" else n
        return np.einsum("bdr,d->br", M, w) / w.sum()
    pick = rng.integers(D, size=(B, D))                                           # dataset drawn into each slot
    if level == "dataset":
        mu = np.stack([E[d].mean(0) for d in names])                              # (D, R)
        if weight == "equal":
            return mu[pick].mean(1)
        return (mu[pick] * n[pick][:, :, None]).sum(1) / n[pick].sum(1)[:, None]
    if level not in ("two_stage", "three_stage"):
        raise ValueError(level)
    cl = clusters if level == "three_stage" else {}
    R = next(iter(E.values())).shape[1]
    acc, wsum = np.zeros((B, R)), np.zeros(B)
    for k in range(D):                                                            # slot k: its own event draws
        for j, d in enumerate(names):
            b = np.flatnonzero(pick[:, k] == j)
            if b.size:
                w = 1.0 if weight == "equal" else n[j]
                acc[b] += w * (_cluster_draws(E[d], np.asarray(cl[d]), b.size, rng) if d in cl
                               else _event_draws(E[d], b.size, rng))
                wsum[b] += w
    return acc / wsum[:, None]


# ------------------------------------------------------------------------------------------------ EP variants
def ep_variant(x, mask, seg, events, tol: int, all_events=None, guard: int = 0, within: bool = False,
               group=None) -> dict:
    L = tol + 1
    evs = events if all_events is None else all_events
    ev, null = windows(mask, seg, events, evs, L)
    if guard:
        bad = np.zeros(len(mask), bool)
        for e in evs:
            bad[max(0, e):min(len(mask), e + L + guard)] = True
        null = null[~bad[null]]
    seg = np.asarray(seg)
    nm_all, s = _wmax(x, null, L), _wmax(x, ev, L)
    out = {}
    for e, v in zip(ev, s):
        g = seg if group is None else np.asarray(group)
        nm = np.sort(nm_all[g[null] == g[e]] if within else nm_all)
        if len(nm):
            lo, hi = np.searchsorted(nm, v, "left"), np.searchsorted(nm, v, "right")
            out[e] = float((lo + 0.5 * (hi - lo)) / len(nm))
    return out


# ------------------------------------------------------------------------------------------------ tests
def pval(d) -> float:
    d = np.asarray(d, float)
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def holm(p) -> list[float]:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[k]))
        adj[k] = run
    return adj.tolist()


def ranks(draws) -> np.ndarray:
    """Rank 1 = highest; ties share the lowest rank of their group (min rank)."""
    d = np.asarray(draws)
    return (d[:, None, :] > d[:, :, None]).sum(2) + 1


def _patterns(n: int, rng):
    if n <= 12:
        return np.array(list(itertools.product((1.0, -1.0), repeat=n))), True
    return np.vstack([np.ones((1, n)), rng.choice((1.0, -1.0), size=(5000, n))]), False


def signflip(x, alpha: float = 0.05, seed: int = SEED) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    S, exact = _patterns(n, np.random.default_rng(seed))
    a, c = S @ x, S.sum(1)                                     # pattern statistic at shift d: a - d c
    grid = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.002), 3)
    obs = np.abs(x.sum() - grid * n)                           # identity pattern
    stat = np.abs(a[:, None] - grid[None, :] * c[:, None])     # (P, G)
    p_grid = (stat >= obs[None, :] - 1e-12).mean(0)
    keep = grid[p_grid > alpha]
    p0 = float(p_grid[np.argmin(np.abs(grid))])
    bounded = bool(keep.size) and keep.min() > -1.0 and keep.max() < 1.0
    return {"mean": float(x.mean()), "n": n, "p": p0, "ci95": [float(keep.min()), float(keep.max())] if keep.size
            else None, "exact": exact, "bounded": bounded}


def coverage(pool, ns, reps: int = 4000, B: int = 2000, seed: int = SEED) -> dict:
    pool = np.asarray(pool, float)
    pool = pool - pool.mean()
    rng = np.random.default_rng(seed)
    out = {}
    for n in ns:
        hit_p = hit_s = 0
        for _ in range(reps):
            x = rng.choice(pool, n)
            bm = x[rng.integers(n, size=(B, n))].mean(1)
            lo, hi = np.quantile(bm, [0.025, 0.975])
            hit_p += lo <= 0 <= hi
            ci = signflip(x)["ci95"] if n <= 12 else None
            hit_s += ci is None or ci[0] <= 0 <= ci[1]
        out[int(n)] = {"percentile": hit_p / reps, "signflip": hit_s / reps if n <= 12 else None}
    return out


def t_interval(x) -> dict:
    x = np.asarray(x, float)
    n, m = len(x), float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    q = float(student.ppf(0.975, n - 1)) if n > 1 else np.nan
    p = float(2 * student.sf(abs(m) / se, n - 1)) if se > 0 else (1.0 if m == 0 else 0.0)
    return {"mean": m, "n": n, "ci95": [m - q * se, m + q * se], "p": p}


def signflip_upper(x, alpha: float) -> tuple[float, bool]:
    x = np.asarray(x, float)
    S, _ = _patterns(len(x), np.random.default_rng(SEED))
    grid = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.001), 3)
    a, c = S @ x, S.sum(1)
    obs = x.sum() - grid * len(x)                              # identity pattern
    p = ((a[:, None] - grid[None, :] * c[:, None]) <= obs[None, :] + 1e-12).mean(0)
    keep = grid[p > alpha]
    return (float(keep.max()), bool(keep.max() < 1.0)) if keep.size else (-1.0, True)


def upper_bounds(d, point, x, alpha: float = 0.05) -> dict:
    d, pt, x = np.asarray(d, float), np.asarray(point, float), np.asarray(x, float)
    R, K = d.shape[1], len(x)
    se = d.std(0)
    c = float(np.quantile(np.max((d - pt) / np.where(se > 0, se, 1.0), 1), 1 - alpha))
    q = float(student.ppf(1 - alpha / R, K - 1))
    mu, s = x.mean(0), x.std(0, ddof=1)
    sf = [signflip_upper(x[:, r], alpha / R) for r in range(R)]
    return {"max_t_c": c, "t_quantile": q, "max_t": (pt + c * se).tolist(),
            "bonferroni": np.quantile(d, 1 - alpha / R, axis=0).tolist(),
            "t_bonferroni": (mu + q * s / np.sqrt(K)).tolist(),
            "signflip_bonferroni": [b for b, _ in sf], "signflip_bounded": [ok for _, ok in sf]}


def best_set(draws, point, alpha: float = 0.05) -> dict:
    d, pt = np.asarray(draws, float), np.asarray(point, float)
    C = d[:, None, :] - d[:, :, None]                     # C[b, r, j] = draw_j - draw_r
    P = pt[None, :] - pt[:, None]                          # P[r, j] = pt_j - pt_r
    se = C.std(0)
    off = ~np.eye(len(pt), dtype=bool) & (se > 0)
    tmax = np.max(np.where(off[None], np.abs(C - P[None]) / np.where(off, se, 1.0)[None], 0.0).reshape(len(d), -1), 1)
    c = float(np.quantile(tmax, 1 - alpha))
    excl = np.any(off & (P - c * se > 0), axis=1)
    return {"c": c, "not_excluded": [int(i) for i in np.flatnonzero(~excl)], "excluded": [int(i) for i in
                                                                                            np.flatnonzero(excl)]}


def coverage_pooled(D: dict, reps: int = 2000, B: int = 1000, seed: int = SEED) -> dict:
    names = sorted(D)
    mu = np.array([np.mean(D[d]) for d in names])
    eff = mu - mu.mean()                                   # population mean of the dataset effects: 0
    dev = [np.asarray(D[d], float) - m for d, m in zip(names, mu)]
    rng = np.random.default_rng(seed)
    K = len(names)
    hit = {"two_stage_percentile": 0, "t_datasets": 0, "event_level_pop": 0, "event_level_cond": 0}
    for r in range(reps):
        pick = rng.integers(K, size=K)
        E = {f"{k:02d}": (eff[j] + dev[j][rng.integers(len(dev[j]), size=len(dev[j]))])[:, None]
             for k, j in enumerate(pick)}
        ts = pooled_draws(E, "two_stage", "equal", B, seed + 1 + r)[:, 0]
        evl = pooled_draws(E, "event", "equal", B, seed + 1 + r)[:, 0]
        lo, hi = np.quantile(ts, [0.025, 0.975])
        hit["two_stage_percentile"] += lo <= 0 <= hi
        ti = t_interval([E[k].mean() for k in E])["ci95"]
        hit["t_datasets"] += ti[0] <= 0 <= ti[1]
        lo, hi = np.quantile(evl, [0.025, 0.975])
        hit["event_level_pop"] += lo <= 0 <= hi
        cond = float(np.mean(eff[pick]))
        hit["event_level_cond"] += lo <= cond <= hi
    return {k: v / reps for k, v in hit.items()} | {"reps": reps, "B": B, "n_datasets": K,
                                                     "n_events": [len(D[d]) for d in names]}


# ------------------------------------------------------------------------------------------------ power
def mde80(se: float) -> float:
    return float((Z95 + Z80) * se)


# ------------------------------------------------------------------------------------------------ percentiles
def pct(null, x) -> np.ndarray:
    """Mid-rank percentile of each x among the values null."""
    s = np.sort(null)
    lo, hi = np.searchsorted(s, x, "left"), np.searchsorted(s, x, "right")
    return (lo + 0.5 * (hi - lo)) / len(s)
