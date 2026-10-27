"""SpotLight sketch (Eswaran, Faloutsos, Guha & Mishra, KDD 2018) with a nearest-neighbour outlier score in place of
the random cut forest (identity-aware, streaming, no training).

Sketch: K random query subgraphs; query k holds each vertex as a source with probability p and as a destination with
probability q (a fixed hash of the vertex id, so the sketch needs no n x K table); v_t[k] is the number of edges of
snapshot t between the sources and the destinations of query k (an undirected edge counts in both orientations).
A dense subgraph that appears or disappears moves the queries that cover it. Score at step t: the outlyingness of
v_{t-1} among the H sketches before it, each coordinate scaled by its median absolute deviation over those H,

    D_t = mean distance from v_{t-1} to its k nearest neighbours among v_{t-H-1} .. v_{t-2}.

SpotLight's own score is the collusive displacement of a robust random cut forest; the k-NN distance is an
outlier score of the same kind with no randomness and no training (a stated simplification).

    D = scores(keys, seg, n)    keys[t] = sorted unique int64 pair ids u * n + v (u < v) of snapshot t
"""
from __future__ import annotations

import numpy as np

from methods.interface import seg_ranges

K, P_SRC, Q_DST, H, KNN, SEED = 50, 0.2, 0.2, 10, 3, 0
_PRIME = 2_147_483_647


def _members(v, a, b, p: float) -> np.ndarray:
    """(len(v), K) bool: vertex v in the set of query k, by the hash ((a_k v + b_k) mod P) / P < p."""
    return ((np.multiply.outer(v % _PRIME, a) + b) % _PRIME) < p * _PRIME


def sketch(keys, n: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a_s, b_s, a_d, b_d = (rng.integers(1, _PRIME, size=K, dtype=np.int64) for _ in range(4))
    V = np.zeros((len(keys), K))
    for t, k in enumerate(keys):
        if len(k) == 0:
            continue
        u, v = np.divmod(np.asarray(k, np.int64), n)
        su, sv = _members(u, a_s, b_s, P_SRC), _members(v, a_s, b_s, P_SRC)
        du, dv = _members(u, a_d, b_d, Q_DST), _members(v, a_d, b_d, Q_DST)
        V[t] = (su & dv).sum(0) + (sv & du).sum(0)
    return V


def score_at(V, t: int) -> float:
    """D_t from the sketches V[t - H - 1 .. t - 1]."""
    hist, x = V[t - H - 1:t - 1], V[t - 1]
    sc = np.median(np.abs(hist - np.median(hist, 0)), 0)
    sc = np.where(sc > 0, sc, 1.0)
    return float(np.sort(np.sqrt((((hist - x) / sc) ** 2).sum(1)))[:KNN].mean())


def scores(keys, seg, n: int) -> np.ndarray:
    V = sketch(keys, n)
    S = np.full(len(keys), np.nan)
    for s0, s1 in seg_ranges(seg):
        for t in range(s0 + H + 2, s1):
            S[t] = score_at(V, t)
    return S
