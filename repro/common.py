"""Shared code of repro/: paths, data loading, F1, the count / constant grid and the selection rules.

F1: an alarm d matches an unmatched event e if e <= d <= e + tol (tolerance 5 MIT, 2 AskUbuntu, 4 Enron).
Selection on the seed-mean F1: best = argmax per dataset; fixed = argmax of the mean over the three datasets;
transfer = argmax of the mean over two datasets, applied to the third. Ties go to the first setting in grid order.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import hashlib
import inspect
import itertools
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent            # repro/
REPO = ROOT.parent                                # repository root (library code: utils/, data/, config.py)
DATA = ROOT / "data"                              # graphs_<ds>.npz, moments_<ds>.npz (generated)
RESULTS = ROOT / "results"                        # small JSON / CSV results
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

NAMES = ("mit_reality", "askubuntu", "enron")
LABEL = {"mit_reality": "MIT", "askubuntu": "AskUbuntu", "enron": "Enron"}
TOL = {"mit_reality": 5, "askubuntu": 2, "enron": 4}
SEEDS = {"s0": 0, "s1": 1000, "s2": 2000, "s3": 3000, "s4": 4000}    # probe base seeds (seed of t: base + t)
CHECK_SEED = {"s42": 42}                                              # base seed of the cached bundle moments
K_MOM, R_PROBES = 50, 30

# ---- grid of the counts and the constant score: 4*4*3*5*21 = 5,040 settings (w, w_ref, mode, q, kind, value) ----
W = (2, 3, 5, 7)
WR = (2, 5, 7, 10)
MODES = ("pw", "cen", "gam")
QS = (3, 5, 6, 7, 10)
PCTS = (0.2, 0.4, 0.6, 0.7, 0.8, 0.84, 0.85, 0.9, 0.95)
ABS_COUNT = (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16)

# ---- grid of SCPD and LADdos (on the KPM moments of step 2): 5*3*3*4*5*46 = 41,400 settings ----
NM = (10, 15, 20, 25, 30)
NB = (20, 30, 50)
SW = (3, 5, 7)
LW = (10, 15, 20, 25)
CD = (3, 5, 7, 10, 15)
LOGT = tuple(10 ** (-7 + i / 4) for i in range(25))
ABS_SCPD = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3)
THR_SCPD = [("abs", t) for t in LOGT] + [("pct", p) for p in PCTS] + [("abs", t) for t in ABS_SCPD]


# ------------------------------------------------------------------------------------------------ utilities
class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def __call__(self) -> float:
        return time.perf_counter() - self.t0


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def write_json(path: Path, obj) -> None:
    def conv(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=1, default=conv))


def sha(*parts) -> str:
    """First 12 hex digits of the sha256 of the parts (a Path: its bytes; bytes as they are; else str)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.read_bytes() if isinstance(p, Path) else p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()[:12]


def src_sha(*objs) -> str:
    """Cache key of the source code of functions (a changed function recomputes its cached results)."""
    return sha(*[inspect.getsource(o) for o in objs])


# ------------------------------------------------------------------------------------------------ data
def load_graphs(name: str) -> dict:
    return dict(np.load(DATA / f"graphs_{name}.npz"))


def adjacency(g: dict, t: int) -> csr_matrix:
    """Exact adjacency of snapshot t (as nx.adjacency_matrix: self-loop entry 1) from graphs_<ds>.npz."""
    ipo = np.r_[0, np.cumsum(g["indptr_len"])]
    nzo = np.r_[0, np.cumsum(g["nnz"])]
    n = int(g["n"][t])
    return csr_matrix((g["data"][nzo[t]:nzo[t + 1]], g["indices"][nzo[t]:nzo[t + 1]],
                       g["indptr"][ipo[t]:ipo[t + 1]]), shape=(n, n))


def load(name: str) -> dict:
    """moments_<ds>.npz -> {'cps', 'mom': {seed: (T, K+1)}, 'f': per-snapshot exact quantities}."""
    z = dict(np.load(DATA / f"moments_{name}.npz"))
    mom = {s: z[s].copy() for s in SEEDS}
    return {"cps": [int(c) for c in z["true_cps"]], "mom": mom, "f": z}


def mad_scale(P, seg=None) -> np.ndarray:
    """Each feature / median |first difference| of the whole series (differences across segment boundaries left
    out; zero scale -> 0)."""
    P = np.asarray(P, float)
    dif = np.diff(P, axis=0) if seg is None else np.diff(P, axis=0)[np.diff(np.asarray(seg)) == 0]
    s = np.median(np.abs(dif), axis=0)
    return np.where(s > 0, P / np.where(s > 0, s, 1.0), 0.0)


def count_vectors(f: dict) -> np.ndarray:
    """(T, 2): active vertices and edges without self-loops, each / median |first difference|."""
    return mad_scale(np.c_[f["active_noloop"], f["edges_noloop"]])


def bundle(X=None, eig=None, cps=(), tol: int = 0, n: int = 0, name: str = ""):
    """A DataBundle without graphs for the methods/ baselines: raw features X (T, 8) and LAD eigenvalues eig."""
    from data.registry import DataBundle
    T = len(X if X is not None else eig)
    return DataBundle(name=name, graphs=[], true_cps=list(cps), tolerance=tol, symmetric=False, n_graphs=T,
                      max_nodes=n, features_raw=np.zeros((T, 8)) if X is None else np.asarray(X),
                      moments=np.zeros((T, 1)), lad_eigenvalues=np.zeros((T, 20)) if eig is None else np.asarray(eig))


# ------------------------------------------------------------------------------------------------ scores
# The count detector's score, threshold and alarm rule are the library code (methods/counts.py).
from methods.counts import alarms
from methods.counts import scores as score_series
from methods.counts import threshold_value as theta_of


def constant_alarms(T: int, q: int, start: int) -> list[int]:
    """A constant score exceeds any threshold at or below it: alarms only through the cooldown."""
    return list(range(start, T, q))


def f1_score(det, cps, tol: int) -> tuple[float, float, float]:
    det = sorted(det)
    if not det:
        return 0.0, 0.0, 0.0
    used, tp = set(), 0
    for e in sorted(cps):
        for d in det:
            if d not in used and e <= d <= e + tol:
                used.add(d)
                tp += 1
                break
    p, r = tp / len(det), tp / len(cps)
    return (0.0 if tp == 0 else 2 * p * r / (p + r)), p, r


# ------------------------------------------------------------------------------------------------ grids
def thresholds() -> list[tuple[str, float]]:
    return [("pct", p) for p in PCTS] + [("abs", a) for a in ABS_COUNT]


def grid() -> list[tuple]:
    """Settings (w, w_ref, mode, q, kind, value) in the order of run_grid."""
    return [(w, wr, md, q, kind, v) for w, wr, md in itertools.product(W, WR, MODES) for q in QS
            for kind, v in thresholds()]


def run_grid(X: np.ndarray, cps, tol: int, det: str = "counts") -> np.ndarray:
    """F1 over grid() of the count detector on X (T, d), or of the constant score (det "constant")."""
    T = len(X)
    res = []
    for w, wr, md in itertools.product(W, WR, MODES):
        start = w + wr + 5
        if det == "constant":
            for q in QS:
                res.extend([f1_score(constant_alarms(T, q, start), cps, tol)[0]] * len(thresholds()))
            continue
        D = score_series(X, w, wr, md)
        for q in QS:
            for kind, v in thresholds():
                res.append(f1_score(alarms(D, theta_of(D, kind, v), q, start), cps, tol)[0])
    return np.array(res)


# ------------------------------------------------------------------------------------------------ selection
def summarize(F: dict) -> dict:
    """F: name -> (n_seeds, n_settings) F1. Selection on the seed mean (first setting among ties)."""
    M = {n: F[n].mean(0) for n in NAMES}
    out = {"best": {}}
    for n in NAMES:
        i = int(np.argmax(M[n]))
        out["best"][n] = {"f1": M[n][i], "sd_over_seeds": F[n][:, i].std(), "index": i,
                          "n_ties": int(np.isclose(M[n], M[n][i]).sum())}
    out["mean_best"] = float(np.mean([out["best"][n]["f1"] for n in NAMES]))
    allm = np.mean([M[n] for n in NAMES], 0)
    i = int(np.argmax(allm))
    out["fixed"] = {"f1": allm[i], "index": i, "n_ties": int(np.isclose(allm, allm[i]).sum()),
                    "per_dataset": {n: M[n][i] for n in NAMES}}
    tr = {}
    for n in NAMES:
        m = np.mean([M[o] for o in NAMES if o != n], 0)
        i = int(np.argmax(m))
        ties = np.nonzero(np.isclose(m, m[i]))[0]
        tr[n] = {"f1": M[n][i], "index": i, "n_ties": len(ties),
                 "tie_min": M[n][ties].min(), "tie_mean": M[n][ties].mean(), "tie_max": M[n][ties].max()}
    out["transfer"] = {"f1": float(np.mean([tr[n]["f1"] for n in NAMES])), "per_dataset": tr,
                       "tie_mean": float(np.mean([tr[n]["tie_mean"] for n in NAMES]))}
    return out
