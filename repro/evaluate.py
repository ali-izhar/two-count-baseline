"""Shared evaluation of score series against labelled events (benchmark B).

A score D_t uses snapshots before t. An event e is hit inside its window [e, e + tol]. A series can have several
segments (independent recordings); every rule restarts at a segment start.

API
  SEED, NBOOT                                   20261023, 10000 (bootstrap seed and draws)
  seg_ranges(seg) -> [(s0, s1)]                 contiguous segments of a segment-id array
  window_scores(X, seg, w=2, w_ref=10, mode="cen", start=None) -> D
                                                methods.counts.scores per segment, nan before s0 + start
                                                (default start = w + w_ref + 5)
  online_scale(X, seg, n0=20, floor=1e-12) -> (Z, scales)
                                                each feature / median |first difference| of the first n0 steps of
                                                its segment (uses only the past of the segment)
  mad_scale(P, seg) -> Z                        common.mad_scale: each feature / median |first difference| of the
                                                whole series (differences across segment boundaries left out)
  scored_mask(seg, x, burn) -> bool[T]          steps after the burn-in of every segment where x is finite
  common_mask(scores, burn, rows, seg) -> bool[T]
                                                steps after the largest burn-in of `rows`, finite for every row
  windows(mask, seg, events, all_events, L) -> (evaluable events, null window starts)
  ep(x, mask, seg, events, tol, all_events=None) -> ({e: EP_e}, n_null)
                                                event percentile: s_e = max x on [e, e + tol]; EP_e =
                                                P(null max < s_e) + P(null max = s_e) / 2 over the null windows
                                                (length tol + 1, inside the mask, touching no labelled event)
  shifted_events(mask, seg, events, rng) -> list
                                                one cyclic offset per segment inside its scored part
  shift_null(scores, rows, mask, seg, events, tol, rng, nshift) -> {row: mean EP under each shift}
                                                the same shifted event set serves every row
  shift_p(null, obs) -> float                   one-sided p = (1 + #{null >= obs}) / (1 + nshift)
  boot_draws(x, clusters=None, B=NBOOT, seed=SEED) -> (B,) bootstrap means of one dataset (one stage, or two
                                                stages with clusters); np.mean(draws <= 0) is the one-sided p
  boot_pooled(diffs, clusters=None, B=NBOOT, seed=SEED) -> dict(mean, lo, hi, per_ds, B)
                                                equal weight per dataset; the draws of boot_draws per dataset
  paired(ep_a, ep_b) -> (events, a - b)         per-event differences on the events both rows score
  delay(x, mask, events, H, alpha, seed=0) -> {e: delay}
                                                first step in [e, e + H] above the (1 - alpha) null quantile, else
                                                H + 1; null steps lie outside every [e, e + H]
  auc(x, seg, events, tol, burn) -> float       Mann-Whitney AUC, positives = scored steps in [e, e + tol]
  individual_split(T, cps, rng, val_len=60, test_frac=0.3) -> (train, val, test)
                                                s-GNN individual-level split of one subject
  cp_targets(labels, tol) -> 0/1 targets        1 within +-tol of a change of the regime label
  adjusted_f1(score, target, threshold) -> F1   point-adjusted F1 of score > threshold
  best_threshold(score, target) -> threshold    best adjusted F1 over 100 thresholds (ties: nearest the median)
"""
from __future__ import annotations

import numpy as np
from common import mad_scale  # noqa: F401  (re-exported; common also puts the repository root on sys.path)
from scipy.stats import rankdata

from methods.counts import scores as _window
from methods.interface import seg_ranges

SEED, NBOOT = 20261023, 10000


# ------------------------------------------------------------------------------------------------ window rule
def window_scores(X, seg, w: int = 2, w_ref: int = 10, mode: str = "cen", start: int | None = None) -> np.ndarray:
    X = np.asarray(X, float)
    X = X[:, None] if X.ndim == 1 else X
    start = w + w_ref + 5 if start is None else start
    D = np.full(len(seg), np.nan)
    for s0, s1 in seg_ranges(seg):
        d = _window(X[s0:s1], w, w_ref, mode)
        d[:start] = np.nan
        D[s0:s1] = d
    return D


def online_scale(X, seg, n0: int = 20, floor: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, float)
    X2 = X[:, None] if X.ndim == 1 else X
    out = np.empty_like(X2)
    scales = []
    for s0, s1 in seg_ranges(seg):
        s = np.maximum(np.median(np.abs(np.diff(X2[s0:min(s0 + n0, s1)], axis=0)), axis=0), floor)
        out[s0:s1] = X2[s0:s1] / s
        scales.append(s)
    return (out if X.ndim > 1 else out[:, 0]), np.array(scales)


# ------------------------------------------------------------------------------------------------ masks, EP
def scored_mask(seg, x, burn: int) -> np.ndarray:
    ok = np.zeros(len(seg), bool)
    for s0, s1 in seg_ranges(seg):
        ok[s0 + burn:s1] = True
    return ok & np.isfinite(x)


def common_mask(scores: dict, burn: dict, rows, seg) -> np.ndarray:
    ok = np.ones(len(seg), bool)
    b = max(burn[m] for m in rows)
    for s0, s1 in seg_ranges(seg):
        ok[s0:min(s0 + b, s1)] = False
    for m in rows:
        ok &= np.isfinite(scores[m])
    return ok


def windows(mask, seg, events, all_events, L: int) -> tuple[list[int], np.ndarray]:
    seg = np.asarray(seg)
    T = len(mask)
    good = np.zeros(T, bool)                       # good[u]: [u, u + L) inside the mask and one segment
    cs = np.r_[0, np.cumsum(mask)]
    u = np.arange(T - L + 1)
    good[u] = (cs[u + L] - cs[u] == L) & (seg[u] == seg[u + L - 1])
    busy = np.zeros(T, bool)
    for e in all_events:                           # every labelled event blocks the null, not only the evaluated
        busy[max(0, e - L + 1):min(T, e + L)] = True
    ev_ok = [e for e in events if e + L <= T and good[e]]
    return ev_ok, np.nonzero(good & ~busy)[0]


def _wmax(x, starts, L: int) -> np.ndarray:
    starts = np.asarray(starts, int)
    if starts.size == 0:
        return np.array([])
    xp = np.r_[x, np.full(L, -np.inf)]
    return np.max(np.stack([xp[starts + j] for j in range(L)]), axis=0)


def ep(x, mask, seg, events, tol: int, all_events=None) -> tuple[dict, int]:
    L = tol + 1
    ev, null = windows(mask, seg, events, events if all_events is None else all_events, L)
    nm = np.sort(_wmax(x, null, L))
    s = _wmax(x, ev, L)
    lo = np.searchsorted(nm, s, "left")
    hi = np.searchsorted(nm, s, "right")
    return dict(zip(ev, ((lo + 0.5 * (hi - lo)) / len(nm)).tolist())), len(null)


# ------------------------------------------------------------------------------------------------ chance
def shifted_events(mask, seg, events, rng) -> list[int]:
    out = []
    for s0, s1 in seg_ranges(seg):
        idx = np.nonzero(mask[s0:s1])[0] + s0
        if idx.size == 0:
            continue
        a, n = idx[0], idx[-1] - idx[0] + 1
        off = rng.integers(n)
        out += [int(a + (e - a + off) % n) for e in events if s0 <= e < s1]
    return out


def shift_null(scores: dict, rows, mask, seg, events, tol: int, rng, nshift: int) -> dict:
    null = {m: np.empty(nshift) for m in rows}
    for i in range(nshift):
        sev = shifted_events(mask, seg, events, rng)
        for m in rows:
            v, _ = ep(scores[m], mask, seg, sev, tol, all_events=events)
            null[m][i] = np.mean(list(v.values())) if v else np.nan
    return null


def shift_p(null, obs: float) -> float:
    null = np.asarray(null)
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


# ------------------------------------------------------------------------------------------------ bootstrap
def _draws(x, c, B: int, rng) -> np.ndarray:
    x = np.asarray(x, float)
    if c is None:
        return x[rng.integers(len(x), size=(B, len(x)))].mean(1)
    c = np.asarray(c)
    groups = [x[c == u] for u in np.unique(c)]
    G = len(groups)
    out = np.zeros(B)
    for b in range(B):
        pick = rng.integers(G, size=G)
        out[b] = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in (groups[i] for i in pick)]).mean()
    return out


def boot_draws(x, clusters=None, B: int = NBOOT, seed: int = SEED) -> np.ndarray:
    return _draws(x, clusters, B, np.random.default_rng(seed))


def boot_pooled(diffs: dict, clusters: dict | None = None, B: int = NBOOT, seed: int = SEED) -> dict:
    """diffs: {ds: per-event values}; clusters: {ds: cluster id per event} for the two-stage bootstrap.

    One generator serves the datasets in sorted order, so a result depends only on (diffs, clusters, B, seed).
    """
    rng = np.random.default_rng(seed)
    clusters = clusters or {}
    names = sorted(diffs)
    bs = np.stack([_draws(diffs[ds], clusters.get(ds), B, rng) for ds in names], axis=1)
    pooled = bs.mean(1)
    per = {ds: (float(np.mean(diffs[ds])), float(np.quantile(bs[:, j], 0.025)), float(np.quantile(bs[:, j], 0.975)))
           for j, ds in enumerate(names)}
    return {"mean": float(np.mean([np.mean(diffs[ds]) for ds in names])), "lo": float(np.quantile(pooled, 0.025)),
                "hi": float(np.quantile(pooled, 0.975)), "per_ds": per, "B": B}


def paired(ep_a: dict, ep_b: dict) -> tuple[list[int], np.ndarray]:
    ev = sorted(set(ep_a) & set(ep_b))
    return ev, np.array([ep_a[e] - ep_b[e] for e in ev], float)


# ------------------------------------------------------------------------------------------------ other metrics
def delay(x, mask, events, H: int, alpha: float, seed: int = 0) -> dict:
    T = len(mask)
    x = np.asarray(x, float)
    x = x + np.random.default_rng(seed).random(T) * 1e-9 * (np.nanstd(x[mask]) + 1e-12)   # random tie-break
    busy = np.zeros(T, bool)
    for e in events:
        busy[e:min(T, e + H + 1)] = True
    theta = np.quantile(x[mask & ~busy], 1 - alpha)
    out = {}
    for e in events:
        if e + H >= T or not mask[e:e + H + 1].all():
            continue
        hit = np.nonzero(x[e:e + H + 1] > theta)[0]
        out[e] = int(hit[0]) if hit.size else H + 1
    return out


def auc(x, seg, events, tol: int, burn: int) -> float:
    x = np.asarray(x, float)
    T = len(x)
    scored = scored_mask(seg, x, burn)
    pos = np.zeros(T, bool)
    for e in events:
        pos[e:min(e + tol + 1, T)] = True
    y, v = pos[scored], x[scored]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(v)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------------------------------------ s-GNN protocol (PM)
def individual_split(T: int, cps, rng, val_len: int = 60, test_frac: float = 0.3):
    """s-GNN individual-level split -> (train, val, test) index arrays; val is centred on a random change point."""
    c = int(rng.choice(cps))
    v0 = max(0, min(T - val_len, c - val_len // 2))
    val = np.arange(v0, v0 + val_len)
    tl = int(np.round(test_frac * T))
    while True:
        t0 = int(rng.integers(0, T - tl + 1))
        test = np.arange(t0, t0 + tl)
        if not np.intersect1d(test, val).size:
            break
    return np.setdiff1d(np.arange(T), np.concatenate([val, test])), val, test


def cp_targets(labels, tol: int) -> np.ndarray:
    """Regime labels -> 0/1 targets: 1 within +-tol of a label change (DyNNet's adjusted change-point labels)."""
    labels = np.asarray(labels)
    cps = np.r_[0, (np.abs(np.diff(labels)) > 0).astype(int)]
    for i in range(1, tol + 1):
        cps = (cps + np.r_[np.zeros(i), cps[:-i]] + np.r_[cps[i:], np.zeros(i)] > 0)
    return cps.astype(int)


def _point_adjust(pred, target) -> np.ndarray:
    """Xu et al. (2018): one hit inside a target run marks the whole run (as DyNNet: the back-fill stops at index 1)."""
    pred = np.array(pred, copy=True)
    state = 0
    for i in range(len(pred)):
        if target[i] == 1 and pred[i] == 1 and state != 1:
            state = 1
            for j in range(i, 0, -1):
                if target[j] != 1:
                    break
                pred[j] = 1
        elif target[i] != 1:
            state = 0
        if state == 1:
            pred[i] = 1
    return pred


def adjusted_f1(score, target, threshold: float) -> float:
    """F1 of the point-adjusted predictions score > threshold (0 when there is no positive at all)."""
    target = np.asarray(target)
    pred = _point_adjust(1 * (np.asarray(score) > threshold), target)
    tp = int(np.sum((pred == 1) & (target == 1)))
    den = 2 * tp + int(np.sum((pred == 1) & (target != 1))) + int(np.sum((pred != 1) & (target == 1)))
    return 2 * tp / den if den else 0.0


def best_threshold(score, target) -> float:
    """DyNNet find_best_threshold: best adjusted F1 over 100 thresholds; among ties the one nearest their median."""
    th = np.linspace(np.min(score), np.max(score), 100)
    f = np.array([adjusted_f1(score, target, t) for t in th])
    best = th[f == f[np.argmax(f)]]
    return float(best[np.argmin(np.abs(best - np.median(best)))]) if len(best) > 1 else float(th[np.argmax(f)])

