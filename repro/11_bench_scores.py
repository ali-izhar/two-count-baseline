"""Step 11: the score series of every benchmark-B row on every dataset, one fixed setting per row.

Rows
  counts, constant              fixed Table 1 settings (results/main_table.json; methods/counts.py)
  cusum, bocpd, mmd, lad, tire  methods/*.py at their fixed Table 1 settings (results/baselines.json, tire.json)
  scpd      methods/scpd.py, windows 5 / 10, probes RandomState(t) on the active vertices of snapshot t
Every rule restarts at a segment start. Inputs: repro/data/bench/<ID>.npz (step 10, binary, symmetric, no self-loops,
fixed vertex set); MIT, ENRON and AskUbuntu use the Table 1 view (steps 1-2: graphs, counts, bundles).
Outputs: repro/data/bench/scores_<ID>.npz (s_<row> series), results/bench_settings.json.

Per snapshot the features come from the active part (vertices of degree > 0) with exact corrections for the isolated
vertices: they add zeros to the adjacency and Laplacian spectra, one component each, and (-1)^k to every SCPD probe
moment (z_i^2 T_k(-1), z_i = +-1). `--check` compares them with data.registry on full matrices.

Usage: 11_bench_scores.py [ID ...] [--check]
"""
from __future__ import annotations

import functools
import json
import sys
from multiprocessing import Pool

import evaluate as ev
import numpy as np
import scipy.sparse as sp
from common import DATA, ROOT, Timer, count_vectors, load, peak_rss_mb, sha, src_sha
from rows import (
    BENCH,
    REPO_VIEW,
    SCPD_WIN,
    B,
    baseline,
    lam_max_abs,
    lap_top,
    matrices,
    meta,
    scpd_signature,
    settings,
    stored,
)
from scipy.sparse.csgraph import connected_components

from methods import scpd

ROWS = ["counts", "constant", "cusum", "bocpd", "mmd", "lad", "tire", "scpd"]
CACHE = BENCH / "cache"


@functools.cache
def input_key(ID: str) -> str:
    if ID in REPO_VIEW:
        return sha(*[(DATA / f"{s}_{REPO_VIEW[ID]}.npz").read_bytes() for s in ("graphs", "moments")])
    return sha((BENCH / f"{ID}.npz").read_bytes())


# ------------------------------------------------------------------------------------------------ per snapshot
def snapshot(A, t: int, repo_view: bool = False) -> dict:
    """Features of one snapshot. Self-loops are dropped (counts, SCPD and the baselines use the loop-free graph)."""
    A = sp.csr_matrix(A, dtype=np.float64)
    n = A.shape[0]
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    deg = np.diff(A.indptr)
    act = np.flatnonzero(deg)
    Aa = A[act][:, act].tocsr()
    E = Aa.nnz / 2
    out = {"n": n, "nplus": len(act), "E": E, "dos": scpd_signature(Aa, n, np.random.RandomState(t))}
    if repo_view:
        return out
    d = deg.astype(float)
    lam = lam_max_abs(Aa)
    ncomp = (connected_components(Aa, directed=False)[0] if len(act) else 0) + (n - len(act))
    out.update(lad5=lap_top(Aa),
               raw8=np.array([n, E, d.mean(), d.std(), 2 * E / (n * (n - 1)) if n > 1 else 0.0, lam, lam, ncomp]))
    return out


FEAT_SHA = sha(src_sha(snapshot, lam_max_abs, lap_top, scpd_signature, matrices, stored),
               (ROOT.parent / "methods" / "scpd.py").read_bytes(), (ROOT.parent / "data" / "registry.py").read_bytes())


def feats(ID: str) -> dict:
    p = CACHE / f"feats_{ID}_{sha(input_key(ID), FEAT_SHA)}.npz"
    if p.exists():
        return dict(np.load(p))
    t0 = Timer()
    rows = [snapshot(A, t, ID in REPO_VIEW) for t, A in enumerate(matrices(ID))]
    z = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    z["seconds"] = t0()
    np.savez(p, **z)
    print(f"{ID}: features {z['seconds']:.0f} s, peak RSS {peak_rss_mb():.0f} MB", flush=True)
    return z


# ------------------------------------------------------------------------------------------------ rows
def window_row(X, seg, s):
    """The window rule of the counts with setting s; scores kept from t = w + w_ref (the burn-in drops the rest)."""
    w, wr, md = s[:3]
    return ev.window_scores(X, seg, w, wr, md, start=w + wr)


def row_scores(ID: str, row: str, F: dict, mt: dict, st: dict, inputs: dict) -> np.ndarray:
    seg = mt["seg"]
    if row == "counts":
        return window_row(inputs["C"], seg, st["counts"])
    if row == "constant":
        return np.where(np.isfinite(window_row(inputs["C"], seg, st["constant"])), 1.0, np.nan)
    if row == "scpd":
        D = np.full(len(seg), np.nan)
        for s0, s1 in ev.seg_ranges(seg):
            if s1 - s0 > max(SCPD_WIN) + 1:
                D[s0:s1] = scpd.score(F["dos"][s0:s1], SCPD_WIN, 0.05)[0]
        return D
    X8, lad = inputs["X8"], inputs["lad"]
    return baseline(row, ID, X8, lad, seg, inputs["n"], st)


ROW_SHA = sha(src_sha(row_scores, baseline, window_row), (ROOT / "rows.py").read_bytes(),
              (ROOT / "common.py").read_bytes(), (ROOT / "evaluate.py").read_bytes())


def method_sha(row: str) -> str:
    f = row if row in ("cusum", "bocpd", "mmd", "lad", "tire", "scpd") else "counts"
    return sha((ROOT.parent / "methods" / f"{f}.py").read_bytes())


def inputs_of(ID: str, F: dict, rows) -> dict:
    """Counts and baseline features: from the snapshots, or from steps 1-2 for the Table 1 view."""
    if ID not in REPO_VIEW:
        seg = meta(ID)["seg"]
        return {"C": ev.mad_scale(np.c_[F["nplus"], F["E"]], seg), "X8": F["raw8"], "lad": F["lad5"],
                "n": int(F["n"][0])}
    d = load(REPO_VIEW[ID])
    out = {"C": count_vectors(d["f"])}
    if any(r in ("cusum", "bocpd", "mmd", "lad", "tire") for r in rows):
        from data.registry import DataRegistry
        b = DataRegistry().load(REPO_VIEW[ID])           # the Table 1 bundle (the Enron pickle needs ~5 GB)
        out.update(X8=np.asarray(b.features_raw), lad=np.asarray(b.lad_eigenvalues), n=b.max_nodes)
    return out


def scores(ID: str) -> str:
    st = settings()
    mt = meta(ID)
    F = feats(ID)
    key = {r: sha(input_key(ID), FEAT_SHA, ROW_SHA, method_sha(r), json.dumps(st.get(r))) for r in ROWS}
    paths = {r: CACHE / f"row_{ID}_{r}_{key[r]}.npy" for r in ROWS}
    todo = [r for r in ROWS if not paths[r].exists()]
    secs = {}
    if todo:
        inputs = inputs_of(ID, F, todo)
        for r in todo:
            t0 = Timer()
            np.save(paths[r], row_scores(ID, r, F, mt, st, inputs))
            secs[r] = round(t0(), 1)
    np.savez(BENCH / f"scores_{ID}.npz", seg=mt["seg"], **{f"s_{r}": np.load(paths[r]) for r in ROWS})
    return f"{ID}: rows {secs or 'cached'}, peak RSS {peak_rss_mb():.0f} MB"


# ------------------------------------------------------------------------------------------------ check
def check() -> None:
    """Active-part features equal data.registry / full-matrix SCPD moments (first 40 snapshots of PS, UCI, HT09)."""
    import networkx as nx

    from data.registry import compute_lad_eigenvalues, extract_features_raw
    for ID in ("PS", "UCI", "HT09"):
        worst = {"raw8": 0.0, "lad5": 0.0, "dos": 0.0}
        for t, A in enumerate(matrices(ID)):
            if t >= 40:
                break
            s = snapshot(A, t)
            G = nx.from_scipy_sparse_array(A)
            r8 = extract_features_raw(G)
            worst["raw8"] = max(worst["raw8"], float(np.max(np.abs(r8 - s["raw8"]) / np.maximum(1, np.abs(r8)))))
            worst["lad5"] = max(worst["lad5"], float(np.max(np.abs(compute_lad_eigenvalues(G, 20)[:5] - s["lad5"]))))
            n, act = A.shape[0], np.flatnonzero(np.diff(A.indptr))
            Z = np.sign(np.random.RandomState(7 + t).randn(n, scpd.NZ))

            class Fixed:                                   # feeds one probe matrix to scpd.moments
                def __init__(self, Z):
                    self.Z = Z

                def randn(self, *a):
                    return self.Z
            cf = scpd.moments((scpd.lsym(A) - sp.eye(n, format="csr")).tocsr(), rng=Fixed(Z))
            Aa = A[act][:, act]
            ca = scpd.moments((scpd.lsym(Aa) - sp.eye(len(act), format="csr")).tocsr(), rng=Fixed(Z[act])) \
                + (n - len(act)) * (-1.0) ** np.arange(scpd.NM)
            worst["dos"] = max(worst["dos"], float(np.abs(cf - ca).max() / max(1.0, np.abs(cf).max())))
        print(ID, "max relative difference to the full-matrix route:", worst)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--check" in sys.argv:
        check()
        return
    t0 = Timer()
    settings()
    ids = args or B
    big = [i for i in ("SCC", "EEG", "PM", "MAG", "ENRON") if i in ids]
    with Pool(4, maxtasksperchild=1) as pool:                # largest first; one dataset per process
        for line in pool.imap_unordered(scores, big + [i for i in ids if i not in big]):
            print(line, flush=True)
    print(f"step 11: {t0():.0f} s")


if __name__ == "__main__":
    main()
