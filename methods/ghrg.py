"""GHRG change-point test of Peel & Clauset (AAAI 2015), and their Gaussian baselines on scalar summaries.

Per window of w graphs: the authors' fitHRG / consensusHRG (C++, data/raw/peel_code/cpdetectr, compiled once) give
one consensus dendrogram fitted jointly to the window. Each internal node r has n_r leaf pairs across its children
(the authors count ordered pairs, n_r = sum_k s_k (S - s_k)) and e_{r,g} edges of graph g whose ends meet at r.
With a Beta(1 + E, 1 + n_r m - E) posterior from E edges over m graphs, the score of graph g is the Beta-binomial
term L = lnB(e + a, n_r - e + b) - lnB(a, b) summed over r. The test statistic is the largest split gain
S_c = L(G[:c] | prior of G[:c]) + L(G[c:] | prior of G[c:]) - L(G | prior of G) over c = 1 .. w-1; its null comes
from 1,000 windows sampled from the fitted model (parametric bootstrap). level = share of null draws <= S; the
authors call a change when the level of the best split exceeds 0.95.

This is a vectorised port of changepointDetection.py (cpdetectr 1.3): the same likelihood, priors and sampler, with
the node counts as sufficient statistics instead of one dendropy mrca call per pair. Kept from the authors' code:
sampled graphs list each edge in both orientations (so sampled counts are 2 x Binomial(n_r / 2, a / (a + b))) while
observed graphs list it once; leaves absent from the consensus tree hang from the root.
"""
from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm

NSAMPLES, ALPHA = 1000, 0.95


# ------------------------------------------------------------------------------------------------ dendrogram
def fit_consensus(graphs: list[np.ndarray], n: int, bin_dir: Path) -> str:
    """Consensus tree text of the authors' fitHRG -t joint + consensusHRG on edge lists (k, 2) of one window."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "names.lut").write_text("virtual\treal\n" + "".join(f"{i}\t{i}\n" for i in range(n)))
        files = []
        for k, E in enumerate(graphs):
            files.append(f"g{k:02d}.pairs")
            (d / files[-1]).write_text("".join(f"{a}\t{b}\n" for a, b in E))
        args = ["-n", "names.lut", "-m", str(len(files)), *files]
        last = files[-1][:-6]                            # both programs exit with status 1 on success
        subprocess.run([str(bin_dir.resolve() / "fitHRG_GPL_Bayes" / "fitHRG"), "-t", "joint", *args], check=False, cwd=d,
                       stdout=subprocess.DEVNULL)
        subprocess.run([str(bin_dir.resolve() / "consensusHRG_GPL_Bayes" / "consensusHRG"), *args, "-f",
                        f"{last}_joint_best-dendro.hrg"], check=False, cwd=d, stdout=subprocess.DEVNULL)
        return (d / f"{last}_joint_s1-consensus.tree").read_text()


def parse_tree(text: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Consensus tree -> (leaf_parent (n,), node_parent (R,)) with internal node 0 the root (the file's last row)."""
    rows = [r for r in text.splitlines() if r.strip()]
    R = len(rows)
    order = {R - 1: 0}                                   # file row -> internal id, root first
    node_parent = np.full(R, -1)
    leaf_parent = np.zeros(n, int)                       # absent leaves hang from the root
    stack = [R - 1]
    while stack:
        i = stack.pop()
        tok = rows[i].split("\t")[3].split()[2:]
        for val, typ in zip(tok[0::2], tok[1::2]):
            if typ.strip("()") == "D":
                order[int(val)] = len(order)
                node_parent[order[int(val)]] = order[i]
                stack.append(int(val))
            else:
                leaf_parent[int(val)] = order[i]
    return leaf_parent, node_parent


def lca_and_pairs(leaf_parent, node_parent) -> tuple[np.ndarray, np.ndarray]:
    """LCA internal node of every leaf pair (n, n), and n_r = sum_k s_k (S - s_k) per internal node."""
    n, R = len(leaf_parent), len(node_parent)
    anc = [[] for _ in range(n)]
    for v in range(n):
        r = leaf_parent[v]
        while r >= 0:
            anc[v].append(r)
            r = node_parent[r]
    sets = [{r: k for k, r in enumerate(a)} for a in anc]
    lca = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            r = next(r for r in anc[j] if r in sets[i])
            lca[i, j] = lca[j, i] = r
    size = np.zeros(R, int)                              # leaves below each internal node
    for v in range(n):
        for r in anc[v]:
            size[r] += 1
    child_sizes = [[] for _ in range(R)]
    for v in range(n):
        child_sizes[leaf_parent[v]].append(1)
    for r in range(1, R):
        child_sizes[node_parent[r]].append(size[r])
    ni = np.array([sum(s * (sum(cs) - s) for s in cs) for cs in child_sizes], float)
    return lca, ni


def node_counts(graphs, lca, R: int) -> np.ndarray:
    """(m, R) edges per internal node (each undirected edge once, as in the observed .pairs files)."""
    out = np.zeros((len(graphs), R))
    for g, E in enumerate(graphs):
        if len(E):
            out[g] = np.bincount(lca[E[:, 0], E[:, 1]], minlength=R)
    return out


# ------------------------------------------------------------------------------------------------ test
def _lik(e, a, b, ni):
    """Beta-binomial term of the authors' calculateLikelihood (ng = 1), summed over nodes (last axis)."""
    return (gammaln(e + a) + gammaln(ni - e + b) - gammaln(ni + a + b)
            + gammaln(a + b) - gammaln(a) - gammaln(b)).sum(-1)


def split_gains(e, ni) -> np.ndarray:
    """S_c for c = 1 .. m-1 of node counts e (..., m, R)."""
    m = e.shape[-2]

    def part(x):                                         # likelihood of graphs x under their own posterior
        E = x.sum(-2, keepdims=True)
        return _lik(x, 1 + E, 1 + ni * x.shape[-2] - E, ni).sum(-1)
    full = part(e)
    return np.stack([part(e[..., :c, :]) + part(e[..., c:, :]) - full for c in range(1, m)], -1)


def null_draws(e_obs, ni, rng, nsamples: int = NSAMPLES) -> np.ndarray:
    """Largest split gain of `nsamples` windows sampled from the model fitted to the observed window."""
    m = e_obs.shape[0]
    E = e_obs.sum(0)
    mu = (1 + E) / (2 + ni * m)                          # a / (a + b) of the authors' null model
    half = np.round(ni / 2).astype(np.int64)             # unordered pairs at node r
    e = 2.0 * rng.binomial(half, mu, size=(nsamples, m, len(ni)))
    return split_gains(e, ni).max(-1)


def levels(gains, null) -> np.ndarray:
    """The authors' 'p-value' of each split: share of null draws <= the gain (change when > ALPHA)."""
    null = np.sort(null)
    return np.searchsorted(null, gains, "right") / len(null)


def window_test(e_obs, ni, rng, nsamples: int = NSAMPLES) -> dict:
    g = split_gains(e_obs, ni)
    lv = levels(g, null_draws(e_obs, ni, rng, nsamples))
    c = int(np.argmax(g))
    return {"gains": g, "levels": lv, "best": c + 1, "level": float(lv[c]), "stat": float(g[c]),
            "change": bool(lv[c] > ALPHA)}


# ------------------------------------------------------------------------------------------------ baselines
def gaussian_window(x, rng, nsamples: int = NSAMPLES) -> dict:
    """createNullBaselineModel: Gaussian GLR split gains of a scalar window, bootstrap from N(mean, sd)."""
    x = np.asarray(x, float)
    m = len(x)
    mu, sd = x.mean(), x.std()

    def gains(s, sdv):
        sdv = np.asarray(sdv)[..., None]
        full = norm.logpdf(s, s.mean(-1, keepdims=True), sdv).sum(-1)
        return np.stack([norm.logpdf(s[..., :c], s[..., :c].mean(-1, keepdims=True), sdv).sum(-1)
                         + norm.logpdf(s[..., c:], s[..., c:].mean(-1, keepdims=True), sdv).sum(-1) - full
                         for c in range(1, m)], -1)
    with np.errstate(all="ignore"):
        S = rng.normal(mu, sd, size=(nsamples, m)) if sd > 0 else np.full((nsamples, m), mu)
        null = np.nan_to_num(gains(S, S.std(-1)), nan=-np.inf).max(-1)
        full = norm.logpdf(x, mu, sd).sum()
        g = np.array([norm.logpdf(x[:c], x[:c].mean(), sd).sum() + norm.logpdf(x[c:], x[c:].mean(), sd).sum() - full
                      for c in range(1, m)])
    g = np.nan_to_num(g, nan=-np.inf)
    lv = levels(g, null)
    c = int(np.argmax(g))
    return {"gains": g, "levels": lv, "best": c + 1, "level": float(lv[c]), "stat": float(g[c]),
            "change": bool(lv[c] > ALPHA)}


# ------------------------------------------------------------------------------------------------ protocols
def score_series(res: dict, T: int, w: int, key: str = "level") -> np.ndarray:
    """D_t from the window of snapshots t-w .. t-1 (res[s] = result of the window starting at s); nan before w."""
    D = np.full(T, np.nan)
    for s, r in res.items():
        if s + w < T:
            D[s + w] = r[key]
    return D


def sequential(res: dict, T: int, w: int) -> list[int]:
    """detectAnomaliesInSequence: slide by one; after a change, restart at it. Returns the first snapshot of each
    detected new regime."""
    out, t = [], 0
    while t < T - w:
        r = res[t]
        if r["change"]:
            c = int(np.argmax(r["levels"]))
            out.append(t + c + 1)
            t += c
        t += 1
    return out


def window_key(graphs, n: int, w: int, bin_dir: Path) -> str:
    h = hashlib.sha256(f"{n} {w}".encode())
    for E in graphs:
        h.update(np.ascontiguousarray(E, np.int64).tobytes() + b"|")
    for f in ("fitHRG_GPL_Bayes/fitHRG.cc", "consensusHRG_GPL_Bayes/consensusHRG.cc"):
        h.update((bin_dir / f).read_bytes())
    return h.hexdigest()[:24]
