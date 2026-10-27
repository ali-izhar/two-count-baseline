"""Shared code of the benchmark steps (11, 12, 14, 16, 17, 19, 22, 23): datasets, fixed settings, stored snapshots,
per-snapshot features of the baselines and the baseline rows.

Benchmark B: B (dataset IDs), REPO_VIEW (MIT, ENRON, AskUbuntu use the Table 1 view of steps 1-2), settings() (one
fixed setting per row), meta(ID) (T, segments, events, tolerance), stored(ID) / matrices(ID) (the snapshots),
PRIMARY (the pooled rows; their burn-in and finiteness define the common mask).
Features: lam_max_abs, lap_top (LAD's input), scpd_signature (methods/scpd.py from the active part).
Rows: baseline(m, ...) runs a methods/ baseline per segment.
"""
from __future__ import annotations

import json

import numpy as np
import scipy.sparse as sp
from common import DATA, RESULTS, TOL, adjacency, bundle, load, load_graphs, sha, write_json
from scipy.sparse.linalg import eigsh

from data.registry import start_vector
from methods import scpd
from methods.interface import seg_ranges

B = ["UCI", "USLegis", "canVote", "MAG", "FL", "SCC", "PM", "MIT", "ENRON", "PS", "HT09", "EEG", "AskUbuntu"]
REPO_VIEW = {"MIT": "mit_reality", "ENRON": "enron", "AskUbuntu": "askubuntu"}
BENCH = DATA / "bench"
SCPD_WIN = (5, 10)
PRIMARY = ["counts", "constant", "cusum", "bocpd", "mmd", "lad", "scpd", "tire"]


# ------------------------------------------------------------------------------------------------ settings
def settings() -> dict:
    """The fixed settings of Table 1 (F1 on MIT, Enron, AskUbuntu: steps 3, 8, 9) and SCPD's windows (its paper's)."""
    p = RESULTS / "bench_settings.json"
    src = [json.loads((RESULTS / f"{f}.json").read_text()) for f in ("main_table", "baselines", "tire")]
    st = {"counts": src[0]["counts"]["fixed"]["setting"], "constant": src[0]["constant"]["fixed"]["setting"],
          **{m: src[1][m]["fixed"]["setting"] for m in ("cusum", "bocpd", "mmd", "lad")},
          "tire": src[2]["fixed"]["setting"],
          "scpd": {"windows": list(SCPD_WIN), "percent": 0.05, "Nz": scpd.NZ, "Nm": scpd.NM, "bins": scpd.NPTS}}
    key = sha(json.dumps(st, sort_keys=True))
    if p.exists() and json.loads(p.read_text()).get("key") == key:
        return json.loads(p.read_text())
    st = {"key": key, **st, "burn": burn_in(st)}
    write_json(p, st)
    return st


def window_burn(s) -> int:
    """First alarm step of a window-rule setting (w, w_ref, mode, q, kind, value): w + w_ref + 5."""
    return int(s[0]) + int(s[1]) + 5


def burn_in(st: dict) -> dict:
    """First scored step of each row inside a segment (the rows' own burn-in)."""
    return {"counts": window_burn(st["counts"]), "constant": window_burn(st["constant"]),
            "cusum": int(st["cusum"]["startup_period"]), "bocpd": 15, "mmd": 2 * int(st["mmd"]["window_size"]),
            "lad": int(st["lad"]["long_window"]) + int(st["lad"]["short_window"]),
            "tire": 2 * int(st["tire"][0][0]), "scpd": max(SCPD_WIN) + 1}


# ------------------------------------------------------------------------------------------------ data
def meta(ID: str) -> dict:
    if ID in REPO_VIEW:
        d = load(REPO_VIEW[ID])
        T = len(d["f"]["n"])
        return {"T": T, "seg": np.zeros(T, int), "events": d["cps"], "tol": TOL[REPO_VIEW[ID]], "sec": {}}
    with np.load(BENCH / f"{ID}.npz", allow_pickle=False) as z:
        sec = {k: [int(e) for e in z[k]] for k in ("events_secondary", "events_major") if k in z.files}
        return {"T": int(z["T"]), "n": int(z["n"]), "seg": z["segment"].astype(int),
                "events": [int(e) for e in z["events"]], "tol": int(z["tol"]), "sec": sec}


def stored(ID: str):
    """The snapshots of repro/data/bench/<ID>.npz (step 10: binary, symmetric, no self-loops, fixed vertex set)."""
    with np.load(BENCH / f"{ID}.npz", allow_pickle=False) as z:
        n, T, off, ix = int(z["n"]), int(z["T"]), z["offsets"], z["indices"]
        rows, ip = (z["rows"], None) if "rows" in z.files else (None, z["indptr"])
    for t in range(T):
        a, b = off[t], off[t + 1]
        if rows is None:
            yield sp.csr_matrix((np.ones(b - a), ix[a:b], ip[t]), shape=(n, n))
        else:
            yield sp.csr_matrix((np.ones(b - a), (rows[a:b], ix[a:b])), shape=(n, n))


def matrices(ID: str):
    """The snapshots of benchmark B: stored, or the Table 1 view (common.adjacency, self-loops as stored). In the Table 1
    view, MIT and ENRON keep one vertex set; each AskUbuntu snapshot holds only that month's active users, numbered
    afresh (step 16 maps them to the raw user ids for the identity-aware rows)."""
    if ID not in REPO_VIEW:
        yield from stored(ID)
        return
    g = load_graphs(REPO_VIEW[ID])
    for t in range(len(g["n"])):
        yield adjacency(g, t)


# ------------------------------------------------------------------------------------------------ per snapshot
def lam_max_abs(Aa) -> float:
    if Aa.nnz == 0:
        return 0.0
    if Aa.shape[0] < 1000:
        return float(np.max(np.abs(np.linalg.eigvalsh(Aa.toarray()))))
    return float(np.max(np.abs(eigsh(Aa, k=1, which="LM", v0=start_vector(Aa.shape[0]), return_eigenvectors=False))))


def lap_top(Aa, k: int = 5) -> np.ndarray:
    """Top k eigenvalues of D - A (LAD's input; isolated vertices add zeros only)."""
    na = Aa.shape[0]
    out = np.zeros(k)
    if na <= 1 or Aa.nnz == 0:
        return out
    L = (sp.diags(np.asarray(Aa.sum(1)).ravel()) - Aa).tocsr()
    ev_ = (np.linalg.eigvalsh(L.toarray()) if na < 1000
           else eigsh(L, k=min(k, na - 2), which="LM", v0=start_vector(na), return_eigenvectors=False))
    ev_ = np.sort(ev_)[::-1][:k]
    out[:len(ev_)] = ev_
    return out


def scpd_signature(Aa, n: int, rng) -> np.ndarray:
    """SCPD's DOS of the n-vertex snapshot from its active part (each isolated vertex adds (-1)^k to moment k)."""
    na = Aa.shape[0]
    c = scpd.moments((scpd.lsym(Aa) - sp.eye(na, format="csr")).tocsr(), rng=rng) if na else np.zeros(scpd.NM)
    return scpd.chebhist(scpd.jackson(c + (n - na) * (-1.0) ** np.arange(scpd.NM)))


# ------------------------------------------------------------------------------------------------ rows
def tire_params(st) -> dict:
    (ws, pk, tdl, tds, lam, dom, ep), _, _ = st
    return {"window_size": ws, "parallel_k": pk, "td_latent_dim": tdl, "td_shared_dim": tds, "lambda_td": lam,
            "lambda_fd": lam, "domain": dom, "epochs": ep, "threshold": 0.0, "cooldown": 1, "seed": 42}


def baseline(m: str, ID: str, X8, lad, seg, n, st) -> np.ndarray:
    """A repo baseline per segment; nan where a segment is too short for it (e.g. TIRE windows)."""
    detect = __import__(f"methods.{m}", fromlist=["detect"]).detect
    s = np.full(len(seg), np.nan)
    for s0, s1 in seg_ranges(seg):
        try:
            s[s0:s1] = detect(bundle(X8[s0:s1], lad[s0:s1], n=n, name=ID),
                              tire_params(st[m]) if m == "tire" else st[m]).scores
        except (ValueError, IndexError, np.linalg.LinAlgError) as e:
            print(f"{ID} {m}: segment {s0}-{s1} not scored ({repr(e)[:80]})", flush=True)
    return s
