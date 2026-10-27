"""Step 16: identity-aware detectors on benchmark B, on the planted changes of step 14 and on the source papers'
synthetic block-model changes (spec, checks and reading rules: prereg/IDENTITY.md).

Rows (methods/, one setting each, window rule w = 2, w_ref = 10 unless stated)
  hamming    L1 distance of the window-mean adjacencies (methods/hamming.py)
  jaccard    weighted Jaccard distance of the window means (methods/hamming.py)
  deltacon   DeltaCon, FaBP affinities, g = 10 groups (methods/deltacon.py)
  chenzhang  Chen-Zhang MST two-sample statistic on Hamming distances (methods/chenzhang.py)
  nbs        NBS CUSUM inner product of two interleaved copies (methods/nbs.py)
  turnover   vertex turnover, activity only (methods/turnover.py, w = 2, w_ref = 5; benchmark only)
Parts
  bench   every dataset of step 12, its mask, EP, bootstrap and shifts -> results/identity.json
  plants  the 14 cells of step 14 (independent plants regenerated exactly) and a persistent variant (drawn at the
          onset, carried through the block); identity rows on both, the step-14 rows (except TIRE) on the persistent
          variant at rho 0.5 -> results/identity_plants.json
  synth   LAD / SCPD SBM generator and s-GNN merge / swap generators, with matched-density versions; every benchmark
          row plus the identity rows -> results/identity_synth.json; LAD on its generator with 5 or 499 eigenvalues
          and windows (7, 10) or (5, 10) ("lad_check", spec v4 V9)
  new     two more identity-aware rows, ASE (methods/ase.py) and SpotLight (methods/spotlight.py), on the benchmark and
          the synthetic series (own cache keys) -> results/identity_new.json, with the paired EP difference of every
          row minus the counts on the synthetic series; plants: step 14 stage X
  plantsnew  ASE and SpotLight on the persistent plants (and the independent ones regenerated on PS, HT09, MIT as a
          check of stage X), ASE's test-window energy at t0 + 2 (spec v6 X) -> results/identity_plants_new.json
Caches (repro/data/identity/, keys hash inputs and code)
  keys_<ID>_<h>.npz              off (T + 1,), keys: per-snapshot sorted pair ids u * n + v (u < v), n
  bench_<ID>_<row>_<h>.npy       (T,) score series of one row on one benchmark dataset (nan before its burn-in)
  scores_<ID>.npz                s_<row>: the same series of every identity row (input of steps 19 and 23)
  plbase_<bg>_<h>.npz            s_<row> (T,): every identity row and every step-14 row on the original background
  pl_<variant>_<bg>_<mode>_<rho>_s<seed>_<h>.npz
                                 t0, offset (n_onsets,); s_<row> (n_onsets, 5): the score at t0 .. t0 + 4 with the
                                 block planted (identity rows; step-14 rows too in the persistent rho-0.5 cells);
                                 validity arrays (moved share, max |delta degree|, consecutive edge Jaccard)
  syn_<scenario>_s<seed>_<h>.npz events, s_<row> (T,): every row's score series on one synthetic series
  np_<variant>_<bg>_<mode>_<rho>_s<seed>_<h>.npz
                                 t0, blk; ase, spotlight (n_onsets,): window maxima with the block planted (variant
                                 none: the original); ase_t2, energy_t2: ASE's score and test energy at t0 + 2
Usage: 16_identity.py [checks] [bench] [plants] [synth] [new] [plantsnew] [--workers 4] [--summary]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import functools
import itertools
import json
import time
from multiprocessing import get_context

import evaluate as ev
import numpy as np
import plants as P
import scipy.sparse as sp
from common import DATA, REPO, RESULTS, ROOT, sha, src_sha, write_json
from rows import (
    BENCH,
    PRIMARY,
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

from methods import ase, chenzhang, deltacon, hamming, nbs, scpd, spotlight, turnover
from methods.interface import persistence

CACHE = DATA / "identity"
CACHE.mkdir(parents=True, exist_ok=True)
ID_ROWS = ("hamming", "jaccard", "deltacon", "chenzhang", "nbs")
BENCH_ROWS = ID_ROWS + ("turnover",)
ID_BURN = {"hamming": 12, "jaccard": 12, "deltacon": 12, "chenzhang": 12, "nbs": 12, "turnover": 7}
LOOK = 12                                          # longest window of an identity row (w + w_ref)
CODE = sha(*(REPO / "methods" / f"{m}.py" for m in ("hamming", "deltacon", "chenzhang", "nbs", "turnover",
                                                    "interface")))
MASK_ROWS = PRIMARY                                # step 12: rows whose burn-in and finiteness define the mask
NSHIFT = 1000
PL_ROWS = ("constant", "counts", "dhet", "transitivity", "assortativity", "fair_pair", "lad", "lad_nl", "dos",
           "cusum", "ewma", "bocpd", "mmd")


def keys_of(A) -> np.ndarray:
    U = sp.triu(A, 1).tocoo()
    return np.sort(U.row.astype(np.int64) * A.shape[0] + U.col.astype(np.int64))


def identity_at(win, n: int) -> dict:
    """Every identity row at the step right after `win` (a list of >= LOOK snapshot key arrays)."""
    L = len(win)
    h, j = hamming.stat_at(win, L)
    return {"hamming": h, "jaccard": j, "deltacon": deltacon.stat_at(win, L, n), "chenzhang": chenzhang.stat_at(win, L),
            "nbs": nbs.stat_at(win, L)}


def identity_series(keys, seg, n: int, rows=ID_ROWS) -> dict:
    out = {}
    if "hamming" in rows or "jaccard" in rows:
        out["hamming"], out["jaccard"] = hamming.scores(keys, seg)
    for r, f in (("deltacon", lambda: deltacon.scores(keys, seg, n)), ("chenzhang", lambda: chenzhang.scores(keys, seg)),
                 ("nbs", lambda: nbs.scores(keys, seg)), ("turnover", lambda: turnover.scores(keys, seg, n))):
        if r in rows:
            out[r] = f()
    return {r: out[r] for r in rows}


def pool_map(fn, jobs, workers):
    with get_context("fork").Pool(workers, maxtasksperchild=8) as pool:
        yield from pool.imap_unordered(fn, jobs)


def save_npz(p, **z):
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, **z)
    os.replace(tmp, p)


# ================================================================================================ A. benchmark
@functools.cache
def input_key(ID: str) -> str:
    if ID in REPO_VIEW:
        return sha(*[(DATA / f"{s}_{REPO_VIEW[ID]}.npz").read_bytes() for s in ("graphs", "moments")])
    return sha((BENCH / f"{ID}.npz").read_bytes())


def pickle_keys(ID: str):
    """Pair ids of the Table 1 view with the vertex labels of data/cache/<ds>.pkl: the matrices of step 1 number the
    vertices of each snapshot afresh (AskUbuntu), so identity needs the labels."""
    import pickle

    import data.registry  # noqa: F401  (registers the pickle class aliases)
    with open(REPO / "data" / "cache" / f"{REPO_VIEW[ID]}.pkl", "rb") as f:
        gs = pickle.load(f).graphs
    lab = np.array(sorted(set().union(*[set(G.nodes()) for G in gs])))
    n = len(lab)
    ks = []
    for G in gs:
        e = np.array([(u, v) for u, v in G.edges() if u != v]).reshape(-1, 2)
        e = np.searchsorted(lab, e)
        ks.append(np.unique(np.minimum(e[:, 0], e[:, 1]).astype(np.int64) * n + np.maximum(e[:, 0], e[:, 1])))
    return ks, n


def bench_keys(ID: str):
    p = CACHE / f"keys_{ID}_{sha(input_key(ID), src_sha(keys_of, matrices, stored, pickle_keys))}.npz"
    if not p.exists():
        ks, n = [], 0
        if ID in REPO_VIEW:
            ks, n = pickle_keys(ID)
            ne = [int(sp.triu(A, 1).nnz) for A in matrices(ID)]
            assert ne == [len(k) for k in ks], f"{ID}: pickle edges differ from the Table 1 matrices"
        for A in () if ID in REPO_VIEW else matrices(ID):
            n = A.shape[0]
            ks.append(keys_of(A))
        save_npz(p, off=np.r_[0, np.cumsum([len(k) for k in ks])], keys=np.concatenate(ks), n=n)
    z = np.load(p)
    off, k = z["off"], z["keys"]
    return [k[off[t]:off[t + 1]] for t in range(len(off) - 1)], int(z["n"])


def bench_path(ID, row):
    return CACHE / f"bench_{ID}_{row}_{sha(input_key(ID), CODE, src_sha(identity_series, keys_of))}.npy"


def bench_job(job):
    ID, row = job
    p = bench_path(ID, row)
    if not p.exists():
        t0 = time.time()
        keys, n = bench_keys(ID)
        np.save(p, identity_series(keys, meta(ID)["seg"], n, (row,))[row])
        print(f"bench {ID} {row}: {time.time() - t0:.0f} s", flush=True)
    return job


def pooled(E: dict, dss, rows) -> dict:
    out = {"ep": {}, "minus_counts": {}}
    for m in rows:
        b = ev.boot_pooled({d: E[d][m] - 0.5 for d in dss})
        out["ep"][m] = {"mean": b["mean"] + 0.5, "lo": b["lo"] + 0.5, "hi": b["hi"] + 0.5,
                        "per_ds": {d: [v + 0.5 for v in b["per_ds"][d]] for d in dss}}
        if m != "counts":
            out["minus_counts"][m] = ev.boot_pooled({d: E[d][m] - E[d]["counts"] for d in dss})
    return out


def run_bench(workers: int) -> None:
    jobs = [(ID, r) for ID in ("SCC", "MAG", "PM", "EEG", "ENRON") + tuple(x for x in B if x not in
            ("SCC", "MAG", "PM", "EEG", "ENRON")) for r in BENCH_ROWS]
    for ID in B:
        bench_keys(ID)
    for _ in pool_map(bench_job, [j for j in jobs if not bench_path(*j).exists()], workers):
        pass
    st = settings()
    burn = {**st["burn"], **ID_BURN}
    bj = json.loads((RESULTS / "bench.json").read_text())
    rows = ["counts", *BENCH_ROWS]
    per, E, nulls, c1 = {}, {}, {}, {}
    for ID in B:
        mt = meta(ID)
        seg, evs, tol = mt["seg"], mt["events"], mt["tol"]
        z = np.load(BENCH / f"scores_{ID}.npz")
        S = {k[2:]: z[k] for k in z.files if k.startswith("s_")}
        S.update({r: np.load(bench_path(ID, r)) for r in BENCH_ROWS})
        np.savez(CACHE / f"scores_{ID}.npz", **{f"s_{r}": S[r] for r in BENCH_ROWS})  # read by steps 19 and 23
        mask = ev.common_mask(S, burn, MASK_ROWS, seg)
        ev_ok, null = ev.windows(mask, seg, evs, evs, tol + 1)
        ref = bj["datasets"][ID]
        c1[ID] = {"n_mask": [int(mask.sum()), ref["n_mask"]], "finite": bool(all(np.isfinite(S[r][mask]).all()
                                                                                  for r in BENCH_ROWS))}
        r = {"T": mt["T"], "n_mask": int(mask.sum()), "n_events_evaluable": len(ev_ok), "n_null_windows": len(null),
             "tol": tol}
        per[ID] = r
        if not ev_ok or not len(null):
            continue
        E[ID] = {m: np.array([ev.ep(S[m], mask, seg, evs, tol)[0][e] for e in ev_ok]) for m in rows}
        c1[ID]["counts_ep"] = [float(E[ID]["counts"].mean()), ref["m2"]["ep_mean"]["counts"]]
        nulls[ID] = ev.shift_null(S, rows, mask, seg, evs, tol, np.random.default_rng([ev.SEED, B.index(ID)]), NSHIFT)
        r.update(events=[int(e) for e in ev_ok], ep_mean={m: float(E[ID][m].mean()) for m in rows},
                 ep_per_event={m: E[ID][m].round(4).tolist() for m in rows},
                 shift_p={m: ev.shift_p(nulls[ID][m], E[ID][m].mean()) for m in rows})
        print(ID, {m: round(float(E[ID][m].mean()), 3) for m in rows}, flush=True)
    dss = sorted(E)
    big = [d for d in dss if per[d]["n_events_evaluable"] >= 10]
    out = {"rows": rows, "datasets": per, "pooled_datasets": dss, "pooled_datasets_10plus": big,
           "pooled": pooled(E, dss, rows), "pooled_10plus": pooled(E, big, rows),
           "shift_p_pooled": {m: ev.shift_p(np.mean([nulls[d][m] for d in dss], 0),
                                            np.mean([E[d][m].mean() for d in dss])) for m in rows},
           "settings": {"window": [hamming.W, hamming.W_REF], "turnover": [turnover.W, turnover.W_REF],
                        "deltacon_groups": deltacon.G,
                        "burn": ID_BURN, "common_burn": max(burn[m] for m in MASK_ROWS)},
           "checks": {"C1": {"per_ds": c1, "status": "OK" if all(
               v["n_mask"][0] == v["n_mask"][1] and v["finite"] and
               ("counts_ep" not in v or abs(v["counts_ep"][0] - v["counts_ep"][1]) < 1e-12) for v in c1.values())
               else "FAIL"}}}
    write_json(RESULTS / "identity.json", out)
    print("wrote results/identity.json")


# ================================================================================================ B. plants
BGS = ("PS", "HT09", "PM", "MIT")
HELD = ("PS", "HT09", "PM")
SEEDS, OFFSETS = tuple(range(10)), tuple(range(40))
CELLS = [(m, r) for m in P.STRUCT for r in (0.1, 0.2, 0.5)] + [(m, r) for m in P.ACTIV for r in (0.1, 0.3)] + \
        [("swap", 0.5)]
FULL = [(m, 0.5) for m in (*P.STRUCT, "swap")]        # persistent cells that also get the step-14 rows


def pl_set() -> dict:
    return json.loads((RESULTS / "plants.json").read_text())["meta"]["settings"]


@functools.cache
def bg(ID: str):
    """(original snapshots, keys, segments) of a background."""
    mats = list(stored(ID))
    with np.load(BENCH / f"{ID}.npz", allow_pickle=False) as z:
        seg = z["segment"].astype(int)
    return mats, [keys_of(A) for A in mats], seg


def onsets(seg):
    """Every block onset of every offset: (t0, offset) arrays."""
    t0s = [(t0, o) for o in OFFSETS for t0 in P.block_onsets(seg, o)]
    return np.array([t for t, _ in t0s], int), np.array([o for _, o in t0s], int)


# ------------------------------------------------------------------------------------------------ persistent plants
def pattern_swaps(Es, n: int, mode: str, rho: float, rng):
    """Double-edge swaps on the union of the block's edge sets, only between edges with the same presence pattern
    over the block: every vertex keeps its degree in every snapshot and a moved edge carries the presence pattern of
    the edge it replaces (persistence as in the original). A new pair must be absent at every time of the pattern
    (it may be present at other times: then its presence joins two patterns). round(rho |union| / 2) accepted swaps,
    50x tries; acceptance as plants.swap_edges, summed over the snapshots of the pattern (assort: sum d_u d_v rises;
    triclose: the triangle count rises). Returns the planted edge arrays, one per snapshot."""
    L = len(Es)
    ids = [e[:, 0] * n + e[:, 1] for e in Es]
    U, inv = np.unique(np.concatenate(ids), return_inverse=True)
    m = len(U)
    pres = np.zeros((m, L), bool)
    pres[inv, np.repeat(np.arange(L), [len(x) for x in ids])] = True
    _, cls = np.unique(pres, axis=0, return_inverse=True)
    cls = cls.ravel()
    members = [np.nonzero(cls == c)[0] for c in range(cls.max() + 1)] if m else []
    EU = np.c_[U // n, U % n]
    bits = (pres * (1 << np.arange(L))).sum(1).tolist()
    mask = dict(zip(U.tolist(), bits))              # pair id -> presence bits of the current planted state
    target = round(rho * m / 2)
    deg = [np.bincount(e.ravel(), minlength=n) for e in Es] if mode == "assort" else None
    nb = None
    if mode == "triclose":
        nb = [{} for _ in range(L)]
        for t, e in enumerate(Es):
            for a, b in e.tolist():
                nb[t].setdefault(a, set()).add(b)
                nb[t].setdefault(b, set()).add(a)
    done = tries = 0
    while done < target and tries < 50 * target:
        tries += 1
        i = int(rng.integers(m))
        grp = members[cls[i]]
        j = int(grp[rng.integers(len(grp))])
        if i == j:
            continue
        a, b = EU[i]
        c, d = EU[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        e1, e2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        k1, k2, bi = int(e1[0] * n + e1[1]), int(e2[0] * n + e2[1]), bits[i]
        if mask.get(k1, 0) & bi or mask.get(k2, 0) & bi:
            continue
        times = np.nonzero(pres[i])[0]
        if mode == "assort" and sum(int(g[a] * g[d] + g[c] * g[b] - g[a] * g[b] - g[c] * g[d])
                                    for g in (deg[t] for t in times)) <= 0:
            continue
        if mode == "triclose":                      # as plants.swap_edges, in every snapshot of the pattern
            dt = 0
            for t in times:
                q = nb[t]
                q[a].discard(b); q[b].discard(a); dt -= len(q[a] & q[b])
                q[c].discard(d); q[d].discard(c); dt -= len(q[c] & q[d])
                dt += len(q[a] & q.setdefault(d, set())); q[a].add(d); q[d].add(a)
                dt += len(q[c] & q.setdefault(b, set())); q[c].add(b); q[b].add(c)
            if dt <= 0:
                for t in times:
                    q = nb[t]
                    q[a].discard(d); q[d].discard(a); q[c].discard(b); q[b].discard(c)
                    q[a].add(b); q[b].add(a); q[c].add(d); q[d].add(c)
                continue
        for k_old in (int(a * n + b) if a < b else int(b * n + a), int(c * n + d) if c < d else int(d * n + c)):
            mask[k_old] ^= bi
        mask[k1] = mask.get(k1, 0) | bi
        mask[k2] = mask.get(k2, 0) | bi
        EU[i], EU[j] = e1, e2
        done += 1
    return [EU[pres[:, t]] for t in range(L)]


def persistent_block(mats, t0: int, L: int, mode: str, rho: float, seed: int):
    """Planted snapshots t0 .. t0 + L - 1 of the persistent variant (prereg/IDENTITY.md, B)."""
    rng = np.random.default_rng([seed, P.MODES.index(mode), round(1000 * rho), t0])
    out = []
    A0 = mats[t0]
    n = A0.shape[0]
    if mode == "swap":
        act = np.nonzero(np.asarray(A0.sum(1)).ravel() > 0)[0]
        k = round(rho * len(act) / 2)
        perm = np.arange(n)
        if k > 0:
            v = rng.choice(act, 2 * k, replace=False)
            perm[v[:k]], perm[v[k:]] = v[k:], v[:k]
        return [sp.csr_matrix(mats[t][perm][:, perm]) for t in range(t0, t0 + L)]
    if mode == "deact":
        act = np.nonzero(np.asarray(A0.sum(1)).ravel() > 0)[0]
        off = np.zeros(n, bool)
        off[rng.choice(act, round(rho * len(act)), replace=False)] = True
    if mode in P.STRUCT:
        Es, loops = zip(*(P.edges_of(mats[t]) for t in range(t0, t0 + L)))
        return [sp.csr_matrix(P.build(n, E, lp)) for E, lp in zip(pattern_swaps(list(Es), n, mode, rho, rng), loops)]
    u = {}
    for t in range(t0, t0 + L):
        E, loops = P.edges_of(mats[t])
        if mode == "thin":
            ids = E[:, 0] * n + E[:, 1]
            for e in np.sort(ids):
                if e not in u:
                    u[e] = rng.random()
            E = E[np.array([u[e] for e in ids]) >= rho] if len(E) else E
        elif mode == "deact":
            E = E[~(off[E[:, 0]] | off[E[:, 1]])]
            loops = np.where(off, 0, loops)
        out.append(sp.csr_matrix(P.build(n, E, loops)))
    return out


def edge_jaccard(a, b) -> float:
    i = len(np.intersect1d(a, b, assume_unique=True))
    u = len(a) + len(b) - i
    return i / u if u else 1.0


# ------------------------------------------------------------------------------------------------ step-14 rows
def stats_of(mats, Z) -> dict:
    rows = [P.snapshot_stats(A, Z) for A in mats]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


PL_CODE = sha(CODE, P.STATS_SHA, src_sha(keys_of, identity_at, identity_series, edge_jaccard, P.feats,
                                         P.per_segment, P.score_all, stats_of, onsets, P.block_onsets, P.plant,
                                         P.swap_edges),
              *(REPO / "methods" / f"{m}.py" for m in ("counts", "lad", "cusum", "ewma", "bocpd", "mmd", "scpd")),
              REPO / "data" / "registry.py", ROOT / "evaluate.py")


def pl_key(ID):
    return sha(PL_CODE, (BENCH / f"{ID}.npz").read_bytes(), json.dumps(pl_set()))


def base_path(ID):
    return CACHE / f"plbase_{ID}_{pl_key(ID)}.npz"


def base_job(ID):
    p = base_path(ID)
    if not p.exists():
        t0 = time.time()
        mats, keys, seg = bg(ID)
        S = identity_series(keys, seg, mats[0].shape[0])
        zb = stats_of(mats, P.probes(max(A.shape[0] for A in mats)))
        D = P.score_all(P.feats(zb), seg, np.random.default_rng(0).random(len(seg)))
        save_npz(p, **{f"s_{r}": v for r, v in S.items()}, **{f"s_{r}": v for r, v in D.items()},
                 **{f"z_{k}": v for k, v in zb.items()})
        print(f"base {ID}: {time.time() - t0:.0f} s", flush=True)
    return ID


def cell_path(variant, ID, mode, rho, seed):
    plant = () if variant == "indep" else (persistent_block, pattern_swaps) if mode in P.STRUCT else (persistent_block,)
    return CACHE / f"pl_{variant}_{ID}_{mode}_{rho:g}_s{seed}_{sha(pl_key(ID), src_sha(cell_job, *plant))}.npz"


def cell_job(job):
    """Scores at t0 .. t0 + 4 of every onset with the block planted, for one (variant, background, cell, seed)."""
    variant, ID, mode, rho, seed = job
    p = cell_path(*job)
    if p.exists():
        return job
    t_start = time.time()
    mats, keys, seg = bg(ID)
    n = mats[0].shape[0]
    T0, OF = onsets(seg)
    S = {r: np.full((len(T0), P.H), np.nan) for r in ID_ROWS}
    val = {"moved": [], "maxdd": [], "jac_orig": [], "jac_plant": [], "ndone": []}
    if variant == "indep":                          # plants.series_stats' generator: every snapshot in order
        rng = np.random.default_rng([seed, P.MODES.index(mode), round(rho * 1000)])
        planted = []
        for A in mats:
            Pm, done = P.plant(A, mode, rho, rng)
            planted.append(Pm)
            val["ndone"].append(done)
        pk = [keys_of(A) for A in planted]
        val["E_nl"] = [len(k) for k in pk]
        va = np.array([P.validity(A, Pm) for A, Pm in zip(mats, planted)])
        val["moved_all"], dd = va[:, 3], va[:, 0]
        blocks = {t0: pk[t0:t0 + P.H] for t0 in set(T0.tolist())}
    else:
        blk = {t0: persistent_block(mats, t0, P.LENGTH, mode, rho, seed) for t0 in sorted(set(T0.tolist()))}
        blocks = {t0: [keys_of(A) for A in b[:P.H]] for t0, b in blk.items()}
    for t0, b in blocks.items():
        orig, pl = keys[t0:t0 + P.H], b
        val["jac_orig"].append(np.mean([edge_jaccard(orig[i], orig[i + 1]) for i in range(P.H - 1)]))
        val["jac_plant"].append(np.mean([edge_jaccard(pl[i], pl[i + 1]) for i in range(P.H - 1)]))
        if variant == "persist":
            v = [P.validity(mats[t0 + i], blk[t0][i]) for i in range(P.H)]
            val["moved"].append(np.mean([x[3] for x in v]))
            val["maxdd"].append(max(x[0] for x in v))
        else:
            val["moved"].append(float(np.mean(val["moved_all"][t0:t0 + P.H])))
            val["maxdd"].append(float(np.max(dd[t0:t0 + P.H])))
    with np.load(base_path(ID)) as z:
        zb0 = {f"s_{r}": z[f"s_{r}"] for r in ID_ROWS}
    for k, t0 in enumerate(T0):                     # h = 0: no planted snapshot in the window yet
        for h, d in enumerate([{r: zb0[f"s_{r}"][t0] for r in ID_ROWS}] +
                              [identity_at(keys[t0 + h - LOOK:t0] + blocks[t0][:h], n) for h in range(1, P.H)]):
            for r in ID_ROWS:
                S[r][k, h] = d[r]
    out = {f"s_{r}": v for r, v in S.items()}
    if variant == "persist" and (mode, rho) in FULL:
        zb = {k[2:]: v for k, v in np.load(base_path(ID)).items() if k.startswith("z_")}
        Z = P.probes(n)
        st = {t0: stats_of(b, Z) for t0, b in blk.items()}
        U = np.random.default_rng(0).random(len(seg))
        R = {r: np.full((len(T0), P.H), np.nan) for r in PL_ROWS}
        for o in OFFSETS:
            ks = np.nonzero(OF == o)[0]
            if not len(ks):
                continue
            z = {k: np.array(v, copy=True) for k, v in zb.items()}
            for t0 in T0[ks]:
                for name, arr in z.items():
                    arr[t0:t0 + P.LENGTH] = st[t0][name]
            D = P.score_all(P.feats(z), seg, U)
            for k in ks:
                for r in PL_ROWS:
                    R[r][k] = D[r][T0[k]:T0[k] + P.H]
        out.update({f"s_{r}": v for r, v in R.items()})
    save_npz(p, t0=T0, offset=OF, **out, **{f"v_{k}": np.array(v, float) for k, v in val.items()},
             seconds=time.time() - t_start)
    print(f"cell {job}: {time.time() - t_start:.0f} s", flush=True)
    return job


def cell_metrics(ID, variant, mode, rho, rows) -> dict:
    seg = bg(ID)[2]
    zb = np.load(base_path(ID))
    met = {r: P.Metrics(zb[f"s_{r}"], seg) for r in rows}
    val = {}
    for s in SEEDS:
        z = dict(np.load(cell_path(variant, ID, mode, rho, s)))
        for o in OFFSETS:
            ks = np.nonzero(z["offset"] == o)[0]
            if not len(ks):
                continue
            t0s = z["t0"][ks]
            for r in rows:
                Dp = np.full(len(seg), np.nan)
                for k, t0 in zip(ks, t0s):
                    Dp[t0:t0 + P.H] = z[f"s_{r}"][k]
                met[r].add(Dp, t0s)
        for k in ("jac_orig", "jac_plant", "moved", "maxdd"):
            if len(z[f"v_{k}"]):
                val.setdefault(k, []).append(z[f"v_{k}"])
    keep = ("auc", "n_pos", "auc_paired", "auc_paired_ci95", "increment_null_sd", "tie_fraction", "zero_fraction",
            "evaluable")
    out = {"rows": {r: {k: v for k, v in met[r].result().items() if k in keep} for r in rows}}
    out["validity"] = {k: float(np.median(np.concatenate(v))) if k != "maxdd" else float(np.max(np.concatenate(v)))
                       for k, v in val.items()}
    return out


def check_c2(ID) -> dict:
    """Regenerated independent plants vs step 14's cached series statistics (every cached series found)."""
    out = {"checked": 0, "mismatch": 0}
    for mode, rho in CELLS:
        for s in SEEDS:
            z = np.load(cell_path("indep", ID, mode, rho, s))
            cands = list((DATA / "plants").glob(f"stats_{ID}_{mode}_{rho:g}_s{s}_*.npz"))
            ok = False
            for c in cands:
                y = np.load(c)
                ok |= bool(np.array_equal(y["E_nl"], z["v_E_nl"]) and np.allclose(y["moved"], z["v_moved_all"]) and
                           np.array_equal(np.nan_to_num(y["ndone"], nan=-1), np.nan_to_num(z["v_ndone"], nan=-1)))
            out["checked"] += bool(cands)
            out["mismatch"] += bool(cands) and not ok
    return out


def run_plants(workers: int, bgs) -> None:
    for _ in pool_map(base_job, [ID for ID in bgs if not base_path(ID).exists()], workers):
        pass
    jobs = [("persist", ID, m, r, s) for ID in bgs for m, r in FULL for s in SEEDS]
    jobs += [(v, ID, m, r, s) for v in ("indep", "persist") for ID in bgs for m, r in CELLS for s in SEEDS
             if not (v == "persist" and (m, r) in FULL)]
    jobs.sort(key=lambda j: (j[1] != "PM", (j[2], j[3]) not in FULL))
    for _ in pool_map(cell_job, [j for j in jobs if not cell_path(*j).exists()], workers):
        pass
    out = {"meta": {"backgrounds": list(bgs), "seeds": list(SEEDS), "offsets": [OFFSETS[0], OFFSETS[-1]],
                    "H": P.H, "block": P.LENGTH, "id_rows": list(ID_ROWS), "pl_rows": list(PL_ROWS),
                    "full_persistent_cells": [list(c) for c in FULL]}, "cells": {}}
    ref = json.loads((RESULTS / "plants.json").read_text())["backgrounds"]
    for ID in bgs:
        for m, r in CELLS:
            k = f"{ID}/{m}/{r:g}"
            c = {"indep": cell_metrics(ID, "indep", m, r, ID_ROWS),
                 "persist": cell_metrics(ID, "persist", m, r, ID_ROWS + (PL_ROWS if (m, r) in FULL else ()))}
            rc = next(x for x in ref[ID]["e1"] + ref[ID]["e2"] + [ref[ID]["e5"]] if x["mode"] == m and x["rho"] == r)
            c["indep_step14"] = {q: rc["rows"][q]["auc"] for q in PL_ROWS}
            out["cells"][k] = c
            print(k, {q: round(c["indep"]["rows"][q]["auc"], 3) for q in ID_ROWS}, "|",
                  {q: round(c["persist"]["rows"][q]["auc"], 3) for q in ID_ROWS}, flush=True)
    out["checks"] = {"C2": {ID: check_c2(ID) for ID in bgs}}
    sw = [out["cells"][f"{ID}/swap/0.5"]["persist"]["rows"][q]["auc"] for ID in bgs for q in PL_ROWS]
    out["checks"]["C5_persistent_swap_invariant_rows"] = {
        "max_abs_dev": float(max(abs(x - 0.5) for x in sw)), "status": "OK" if max(abs(x - 0.5) for x in sw) <= 0.03
        else "FAIL"}
    out["reading"] = plant_reading(out)
    write_json(RESULTS / "identity_plants.json", out)
    print("wrote results/identity_plants.json")


def plant_reading(out) -> dict:
    """Rules R2 and R4 of prereg/IDENTITY.md on PS, HT09, PM."""
    C = out["cells"]
    res = {}
    for q in ID_ROWS + PL_ROWS:
        for m in P.STRUCT:
            ind = [C[f"{ID}/{m}/0.5"]["indep"]["rows"][q]["auc"] if q in ID_ROWS
                   else C[f"{ID}/{m}/0.5"]["indep_step14"][q] for ID in HELD if f"{ID}/{m}/0.5" in C]
            per = [C[f"{ID}/{m}/0.5"]["persist"]["rows"][q]["auc"] for ID in HELD if f"{ID}/{m}/0.5" in C]
            res.setdefault(q, {})[m] = {"indep": ind, "persist": per, "n_seen_indep": sum(x >= 0.65 for x in ind),
                                        "n_seen_persist": sum(x >= 0.65 for x in per),
                                        "persistence_only": sum(x >= 0.65 for x in ind) >= 2 and
                                        not any(x >= 0.65 for x in per)}
    sees = {q: [m for m in P.STRUCT if res[q][m]["n_seen_persist"] >= 2] for q in ID_ROWS}
    # the persistent plant moves fewer edges: compare it with the independent cell of the nearest moved share
    matched = {}
    for k, c in C.items():
        ID, m, r = k.split("/")
        if m not in P.STRUCT or r != "0.5":
            continue
        mv = c["persist"]["validity"]["moved"]
        near = min((f"{ID}/{m}/{x:g}" for x in (0.1, 0.2, 0.5)),
                   key=lambda j: abs(C[j]["indep"]["validity"]["moved"] - mv))
        matched[k] = {"persist_moved": mv, "indep_cell": near, "indep_moved": C[near]["indep"]["validity"]["moved"],
                      "auc_persist": {q: c["persist"]["rows"][q]["auc"] for q in ID_ROWS},
                      "auc_indep": {q: C[near]["indep"]["rows"][q]["auc"] for q in ID_ROWS}}
    return {"per_row": res, "identity_rows_seeing_persistent_plants": {q: v for q, v in sees.items() if v},
            "moved_matched": matched}


# ================================================================================================ C. synthetic
SCEN = {"lad_sbm": 10, "lad_sbm_matched": 10, "sgnn_merge": 20, "sgnn_merge_matched": 20, "sgnn_swap": 20}
SYN_TOL, SYN_BURN = 2, 17
SYN_ROWS = ("counts", "constant", "cusum", "bocpd", "mmd", "lad", "scpd", "tire") + ID_ROWS


def sbm(sizes, Pm, rng, perm=None) -> np.ndarray:
    """Pair ids of one SBM draw (block matrix Pm), optionally relabelled by perm."""
    lab = np.repeat(np.arange(len(sizes)), sizes)
    n = len(lab)
    iu, ju = np.triu_indices(n, 1)
    keep = rng.random(len(iu)) < Pm[lab[iu], lab[ju]]
    a, b = iu[keep], ju[keep]
    if perm is not None:
        a, b = perm[a], perm[b]
        a, b = np.minimum(a, b), np.maximum(a, b)
    return np.sort(a.astype(np.int64) * n + b)


def blockmat(K, p_in, p_out):
    M = np.full((K, K), p_out)
    np.fill_diagonal(M, p_in)
    return M


def matched(N, K, c, ratio=6.0):
    """(p_in, p_out) of K equal blocks with expected degree c and p_in = ratio p_out."""
    s = N // K
    po = c / (ratio * (s - 1) + N - s)
    return ratio * po, po


def synth_series(scen: str, seed: int):
    """(keys per snapshot, n, events, expected degree per snapshot)."""
    rng = np.random.default_rng([20261024, list(SCEN).index(scen), seed])
    if scen.startswith("lad_sbm"):
        N, inter, intra, inc = 1000, 0.005, 0.030, 0.010
        cps = [15, 30, 60, 75, 90, 105, 135]
        Ks = [2, 4, 10]
        idx, is_event = 1, True
        c4 = (N // 4 - 1) * intra + (N - N // 4) * inter

        def probs(K):
            return blockmat(K, *(matched(N, K, c4) if scen == "lad_sbm_matched" else (intra, inter)))
        out, cdeg, evs = [], [], []
        K = Ks[idx]
        snaps = [(K, probs(K))]
        for t in range(150):
            if t in cps and is_event:
                is_event = False
                if scen == "lad_sbm":
                    Pm = probs(K).copy()
                    Pm[Pm < intra] = inter + inc
                    snaps.append((K, Pm))
                    evs.append(t + 1)
                    continue
            elif t in cps:
                is_event = True
                idx = 0 if idx + 1 > len(Ks) - 1 else idx + 1
                K = Ks[idx]
                evs.append(t + 1)
            snaps.append((K, probs(K)))
        for K, Pm in snaps:
            s = N // K
            out.append(sbm([s] * K, Pm, rng))
            cdeg.append((s - 1) * Pm[0, 0] + (N - s) * Pm[0, 1])
        return out, N, evs, np.array(cdeg)
    T, cp = 200, int(rng.integers(50, 150))
    if scen.startswith("sgnn_merge"):
        p, q = 0.5, 0.2
        c1 = 99 * p + 300 * q
        p2 = (c1 - 200 * q) / 199 if scen == "sgnn_merge_matched" else p
        pre, post = ([100] * 4, blockmat(4, p, q)), ([200] * 2, blockmat(2, p2, q))
        out = [sbm(*(pre if t < cp else post), rng) for t in range(T)]
        return out, 400, [cp], np.array([c1 if t < cp else 199 * p2 + 200 * q for t in range(T)])
    # swap: the s-GNN swap generator's intent (its sequence version draws no swap with the default sizes): 40 pairs
    # of vertices from blocks {0, 1} and {2, 3} exchange labels at the change, as in its pair version
    sizes, p, q = [100] * 4, 0.25, 0.2
    c1, c2 = rng.permutation(200), 200 + rng.permutation(200)
    perm = np.arange(400)
    perm[c1[:40]], perm[c2[:40]] = c2[:40], c1[:40]
    out = [sbm(sizes, blockmat(4, p, q), rng, None if t < cp else perm) for t in range(T)]
    return out, 400, [cp], np.full(T, 99 * p + 300 * q)


def snapshot(A, t: int) -> dict:
    """Step 11's per-snapshot features (check C8 compares with its cache on PS)."""
    A = sp.csr_matrix(A, dtype=np.float64)
    n = A.shape[0]
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    deg = np.diff(A.indptr)
    act = np.flatnonzero(deg)
    Aa = A[act][:, act].tocsr()
    E = Aa.nnz / 2
    d = deg.astype(float)
    lam = lam_max_abs(Aa)
    ncomp = (connected_components(Aa, directed=False)[0] if len(act) else 0) + (n - len(act))
    return {"n": n, "nplus": len(act), "E": E, "dos": scpd_signature(Aa, n, np.random.RandomState(t)),
            "lad5": lap_top(Aa),
            "raw8": np.array([n, E, d.mean(), d.std(), 2 * E / (n * (n - 1)) if n > 1 else 0.0, lam, lam, ncomp])}


def bench_rows(F, seg, st, rows, name="") -> dict:
    """Step 11's rows (row_scores) from the features of `snapshot`."""
    C = ev.mad_scale(np.c_[F["nplus"], F["E"]], seg)
    wr = lambda s: ev.window_scores(C, seg, s[0], s[1], s[2], start=s[0] + s[1])
    out = {}
    for r in rows:
        if r == "counts":
            out[r] = wr(st["counts"])
        elif r == "constant":
            out[r] = np.where(np.isfinite(wr(st["constant"])), 1.0, np.nan)
        elif r == "scpd":
            out[r] = np.full(len(seg), np.nan)
            for s0, s1 in ev.seg_ranges(seg):
                if s1 - s0 > max(SCPD_WIN) + 1:
                    out[r][s0:s1] = scpd.score(F["dos"][s0:s1], SCPD_WIN, 0.05)[0]
        else:
            out[r] = baseline(r, name, F["raw8"], F["lad5"], seg, int(F["n"][0]), st)
    return out


def to_csr(k, n):
    a, b = np.divmod(k, n)
    return sp.csr_matrix((np.ones(2 * len(k)), (np.r_[a, b], np.r_[b, a])), shape=(n, n))


SYN_CODE = sha(CODE, src_sha(sbm, blockmat, matched, synth_series, snapshot, bench_rows, to_csr, identity_series),
               src_sha(lam_max_abs, lap_top, scpd_signature, baseline),
               *(REPO / "methods" / f"{m}.py" for m in ("counts", "lad", "cusum", "bocpd", "mmd", "scpd", "tire")),
               ROOT / "evaluate.py", ROOT / "rows.py")


def syn_path(scen, seed):
    return CACHE / f"syn_{scen}_s{seed}_{sha(SYN_CODE, json.dumps({k: v for k, v in settings().items() if k != 'key'}))}.npz"


def syn_job(job):
    scen, seed = job
    p = syn_path(scen, seed)
    if not p.exists():
        t0 = time.time()
        keys, n, evs, cdeg = synth_series(scen, seed)
        seg = np.zeros(len(keys), int)
        rows = [snapshot(to_csr(k, n), t) for t, k in enumerate(keys)]
        F = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        S = bench_rows(F, seg, settings(), SYN_ROWS[:8], name=f"{scen}_{seed}")
        S.update(identity_series(keys, seg, n))
        save_npz(p, events=np.array(evs), cdeg=cdeg, mean_degree=2 * F["E"] / n,
                 **{f"s_{r}": v for r, v in S.items()}, seconds=time.time() - t0)
        print(f"synth {scen} {seed}: {time.time() - t0:.0f} s", flush=True)
    return job


def run_synth(workers: int) -> None:
    jobs = [(s, k) for s, n in SCEN.items() for k in range(n)]
    for _ in pool_map(syn_job, [j for j in jobs if not syn_path(*j).exists()], workers):
        pass
    st = settings()
    burn = {**st["burn"], **ID_BURN}
    out = {"meta": {"tol": SYN_TOL, "burn": SYN_BURN, "rows": list(SYN_ROWS), "seeds": SCEN}, "scenarios": {}}
    for scen, ns in SCEN.items():
        E = {r: [] for r in SYN_ROWS}
        cl, c7 = [], []
        for k in range(ns):
            z = np.load(syn_path(scen, k))
            S = {r: z[f"s_{r}"] for r in SYN_ROWS}
            T = len(S["counts"])
            seg = np.zeros(T, int)
            evs = [int(e) for e in z["events"]]
            mask = ev.common_mask(S, {r: max(burn.get(r, 0), SYN_BURN) for r in SYN_ROWS}, SYN_ROWS, seg)
            ev_ok, _ = ev.windows(mask, seg, evs, evs, SYN_TOL + 1)
            for r in SYN_ROWS:
                v, _ = ev.ep(S[r], mask, seg, evs, SYN_TOL)
                E[r] += [v[e] for e in ev_ok]
            cl += [k] * len(ev_ok)
            md, cd = z["mean_degree"], z["cdeg"]
            for e in evs:                           # realised mean degree, 10 snapshots before / after each change
                c7.append([float(md[max(0, e - 10):e].mean()), float(md[e + 1:e + 11].mean() if e + 1 < T else np.nan),
                           float(cd[e - 1]), float(cd[min(e + 1, T - 1)])])
        res = {}
        for r in SYN_ROWS:
            x = np.array(E[r])
            d = ev.boot_draws(x, np.array(cl))
            res[r] = {"ep": float(x.mean()), "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975)),
                      "n_events": len(x), "sees": bool(np.quantile(d, 0.025) > 0.5)}
        c7 = np.array(c7)
        out["scenarios"][scen] = {"rows": res, "mean_degree_before_after": c7.round(3).tolist()}
        if scen.endswith("matched") or scen == "sgnn_swap":
            rel = np.nanmax(np.abs(c7[:, 1] / c7[:, 0] - 1))
            out["scenarios"][scen]["C7"] = {"max_rel_change": float(rel), "status": "OK" if rel <= 0.02 else "FAIL"}
        print(scen, {r: round(v["ep"], 3) for r, v in res.items()}, flush=True)
    out["lad_check"] = lad_check(workers, burn)
    write_json(RESULTS / "identity_synth.json", out)
    print("wrote results/identity_synth.json")


# LAD on its own generator (spec v4 V9): the number of eigenvalues (5 as on the benchmark, 499 as LAD's code for its
# synthetic data) and the windows ((7, 10) fixed, (5, 10) LAD's synthetic run), on the regenerated lad_sbm series.
LAD_K, LAD_WIN, LAD_SCEN = (5, 499), ((7, 10), (5, 10)), ("lad_sbm", "lad_sbm_matched")


def lad_eigs(keys, n: int, k: int) -> np.ndarray:
    """(T, k) top eigenvalues of D - A of every snapshot (dense, exact; isolated vertices add zeros)."""
    out = np.zeros((len(keys), k))
    for t, q in enumerate(keys):
        A = to_csr(q, n)
        L = (sp.diags(np.asarray(A.sum(1)).ravel()) - A).toarray()
        out[t] = np.sort(np.linalg.eigvalsh(L))[::-1][:k]
    return out


def lad_path(scen, seed):
    return CACHE / f"synlad_{scen}_s{seed}_{sha(src_sha(sbm, blockmat, matched, synth_series, to_csr, lad_eigs))}.npy"


def lad_job(job):
    p = lad_path(*job)
    if not p.exists():
        keys, n, _, _ = synth_series(*job)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, lad_eigs(keys, n, max(LAD_K)))
        os.replace(tmp, p)
    return job


def lad_check(workers: int, burn: dict) -> dict:
    """EP of LAD under each (k, windows) on the masks and events of the synth rows; check: k = 5, (7, 10) = s_lad."""
    from common import bundle

    from methods import lad
    jobs = [(s, k) for s in LAD_SCEN for k in range(SCEN[s])]
    for _ in pool_map(lad_job, [j for j in jobs if not lad_path(*j).exists()], workers):
        pass
    out = {"k": list(LAD_K), "windows": [list(w) for w in LAD_WIN], "check_max_abs_diff_fixed": 0.0}
    for scen in LAD_SCEN:
        E = {f"k{k}_w{a}-{b}": [] for k in LAD_K for a, b in LAD_WIN}
        cl = []
        for seed in range(SCEN[scen]):
            z = np.load(syn_path(scen, seed))
            S = {r: z[f"s_{r}"] for r in SYN_ROWS}
            seg = np.zeros(len(S["counts"]), int)
            evs = [int(e) for e in z["events"]]
            mask = ev.common_mask(S, {r: max(burn.get(r, 0), SYN_BURN) for r in SYN_ROWS}, SYN_ROWS, seg)
            ev_ok, _ = ev.windows(mask, seg, evs, evs, SYN_TOL + 1)
            eig = np.load(lad_path(scen, seed))
            for k in LAD_K:
                for a, b in LAD_WIN:
                    x = lad.detect(bundle(eig=eig[:, :k]), {"threshold": 0.0, "num_eigen": k, "short_window": a,
                                                            "long_window": b, "cooldown": 1}).scores
                    if (k, a, b) == (5, 7, 10):
                        out["check_max_abs_diff_fixed"] = max(out["check_max_abs_diff_fixed"],
                                                              float(np.max(np.abs(x - S["lad"]))))
                    v, _ = ev.ep(x, mask, seg, evs, SYN_TOL)
                    E[f"k{k}_w{a}-{b}"] += [v[e] for e in ev_ok]
            cl += [seed] * len(ev_ok)
        res = {}
        for name, x in E.items():
            x = np.array(x)
            d = ev.boot_draws(x, np.array(cl))
            res[name] = {"ep": float(x.mean()), "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975)),
                         "n_events": len(x)}
        out[scen] = res
        print("LAD check", scen, {k: round(v["ep"], 3) for k, v in res.items()}, flush=True)
    return out


# ================================================================================================ D. two more rows
# ASE (RDPG estimates of the two windows) and SpotLight (sketch + k-NN outlier score): the same benchmark and synthetic
# series, cached apart (their own code key) so that the rows above keep their caches. Results: identity_new.json;
# series: repro/data/identity/scores_new_<ID>.npz (read by steps 19 and 23).
NEW_ROWS = ("ase", "spotlight")
NEW_CODE = sha(*(REPO / "methods" / f"{m}.py" for m in ("ase", "spotlight", "interface")))


def new_series(keys, seg, n: int, row: str) -> np.ndarray:
    return ase.scores(keys, seg, n) if row == "ase" else spotlight.scores(keys, seg, n)


def new_bench_path(ID, row):
    return CACHE / f"newbench_{ID}_{row}_{sha(input_key(ID), NEW_CODE, src_sha(new_series))}.npy"


def new_bench_job(job):
    ID, row = job
    p = new_bench_path(ID, row)
    if not p.exists():
        t0 = time.time()
        keys, n = bench_keys(ID)
        np.save(p, new_series(keys, meta(ID)["seg"], n, row))
        print(f"new bench {ID} {row}: {time.time() - t0:.0f} s", flush=True)
    return job


def new_syn_path(scen, seed):
    return CACHE / f"newsyn_{scen}_s{seed}_{sha(NEW_CODE, src_sha(new_series, sbm, blockmat, matched, synth_series))}.npz"


def new_syn_job(job):
    scen, seed = job
    p = new_syn_path(scen, seed)
    if not p.exists():
        keys, n, evs, _ = synth_series(scen, seed)
        seg = np.zeros(len(keys), int)
        save_npz(p, events=np.array(evs), **{f"s_{r}": new_series(keys, seg, n, r) for r in NEW_ROWS})
    return job


def run_new(workers: int) -> None:
    jobs = [(ID, r) for ID in ("SCC", "MAG", "PM", "EEG", "ENRON") + tuple(x for x in B if x not in
            ("SCC", "MAG", "PM", "EEG", "ENRON")) for r in NEW_ROWS]
    for _ in pool_map(new_bench_job, [j for j in jobs if not new_bench_path(*j).exists()], workers):
        pass
    sj = [(s, k) for s, n in SCEN.items() for k in range(n)]
    for _ in pool_map(new_syn_job, [j for j in sj if not new_syn_path(*j).exists()], workers):
        pass
    st = settings()
    burn = {**st["burn"], **ID_BURN, "ase": 12, "spotlight": 12}
    rows = ["counts", "hamming", *NEW_ROWS]
    per, E = {}, {}
    for ID in B:
        mt = meta(ID)
        seg, evs, tol = mt["seg"], mt["events"], mt["tol"]
        z = np.load(BENCH / f"scores_{ID}.npz")
        S = {k[2:]: z[k] for k in z.files if k.startswith("s_")}
        S["hamming"] = np.load(bench_path(ID, "hamming"))
        S.update({r: np.load(new_bench_path(ID, r)) for r in NEW_ROWS})
        np.savez(CACHE / f"scores_new_{ID}.npz", **{f"s_{r}": S[r] for r in NEW_ROWS})       # read by steps 19, 23
        mask = ev.common_mask(S, burn, MASK_ROWS, seg)
        ev_ok, null = ev.windows(mask, seg, evs, evs, tol + 1)
        per[ID] = {"finite_on_mask": {r: bool(np.isfinite(S[r][mask]).all()) for r in NEW_ROWS},
                   "n_events_evaluable": len(ev_ok)}
        if not ev_ok or not len(null):
            continue
        E[ID] = {m: np.array([ev.ep(S[m], mask, seg, evs, tol)[0][e] for e in ev_ok]) for m in rows}
        per[ID]["ep_mean"] = {m: float(E[ID][m].mean()) for m in rows}
        per[ID]["ep_per_event"] = {m: E[ID][m].round(4).tolist() for m in rows}
    dss = sorted(E)
    big = [d for d in dss if per[d]["n_events_evaluable"] >= 10]
    out = {"rows": rows, "settings": {"ase": {"w": ase.W, "w_ref": ase.W_REF, "d": ase.D},
                                      "spotlight": {"K": spotlight.K, "p": spotlight.P_SRC, "q": spotlight.Q_DST,
                                                    "history": spotlight.H, "knn": spotlight.KNN}},
           "datasets": per, "pooled_datasets": dss, "pooled": pooled(E, dss, rows), "pooled_10plus": pooled(E, big, rows)}
    syn, allr = {}, SYN_ROWS + NEW_ROWS
    for scen, ns in SCEN.items():
        Es = {r: [] for r in allr}
        cl = []
        for k in range(ns):
            z0, z1 = np.load(syn_path(scen, k)), np.load(new_syn_path(scen, k))
            S = {r: z0[f"s_{r}"] for r in SYN_ROWS} | {r: z1[f"s_{r}"] for r in NEW_ROWS}
            seg = np.zeros(len(S["counts"]), int)
            evs = [int(e) for e in z0["events"]]
            mask = ev.common_mask(S, {r: max(burn.get(r, 0), SYN_BURN) for r in SYN_ROWS}, SYN_ROWS, seg)
            ev_ok, _ = ev.windows(mask, seg, evs, evs, SYN_TOL + 1)
            for r in allr:
                v, _ = ev.ep(S[r], mask, seg, evs, SYN_TOL)
                Es[r] += [v[e] for e in ev_ok]
            cl += [k] * len(ev_ok)
        res, cl = {}, np.array(cl)
        for r in NEW_ROWS:
            x = np.array(Es[r])
            d = ev.boot_draws(x, cl)
            res[r] = {"ep": float(x.mean()), "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975)),
                      "n_events": len(x), "n_series": ns}
        mc = {}                                     # paired: row minus counts on the same events, series then events
        for r in allr:
            if r != "counts":
                x = np.array(Es[r]) - np.array(Es["counts"])
                d = ev.boot_draws(x, cl)
                mc[r] = {"diff": float(x.mean()), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
                         "p": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))}
        res["minus_counts"] = mc
        syn[scen] = res
        print(scen, {r: round(v["ep"], 3) for r, v in res.items() if r != "minus_counts"}, flush=True)
    out["synth"] = syn
    write_json(RESULTS / "identity_new.json", out)
    print("wrote results/identity_new.json", {m: round(out["pooled"]["ep"][m]["mean"], 3) for m in rows})


# ------------------------------------------------------------------------------------------------ D2. their plants
# ASE and SpotLight on the persistent plants (spec v6 X), scored as step 14 stage X scores them on the independent
# plants; the independent plants are regenerated on PS, HT09 and MIT as a check (X0) and for the energy diagnostic:
# the low-rank energy of ASE's two-snapshot test mean at t0 + 2, planted against original.
NP_CELLS = [(m, 0.5) for m in (*P.STRUCT, "swap")] + [(m, 0.3) for m in P.ACTIV]
NP_INDEP = ("PS", "HT09", "MIT")


def ase_energy(win, t: int, n: int) -> float:
    """Sum of the squared top-d eigenvalues of ASE's test-window mean x / w at step t (its own estimate, ase.embed)."""
    u, x, _ = persistence(win, t, ase.W, ase.W_REF)
    v = x / ase.W
    if not (v > 0).any():
        return 0.0
    a, b = np.divmod(u, n)
    act, inv = np.unique(np.r_[a, b], return_inverse=True)
    i, j = inv[:len(u)], inv[len(u):]
    lt, _ = ase.embed(i[v > 0], j[v > 0], v[v > 0], len(act), ase.D)
    return float((lt ** 2).sum())


def np_path(variant, ID, mode, rho, seed):
    code = sha(NEW_CODE, src_sha(np_job, ase_energy, keys_of, persistent_block, pattern_swaps, P.plant, P.swap_edges,
                                 P.edges_of, P.build, P.block_onsets, onsets))
    return CACHE / f"np_{variant}_{ID}_{mode}_{rho:g}_s{seed}_{sha(code, (BENCH / f'{ID}.npz').read_bytes())}.npz"


def np_job(job):
    """ASE and SpotLight window maxima over [t0, t0 + H) of every onset with the block planted (variant indep or
    persist; none: the original series), and ASE's score and test energy at t0 + 2."""
    variant, ID, mode, rho, seed = job
    p = np_path(*job)
    if p.exists():
        return job
    t_start = time.time()
    mats, keys, seg = bg(ID)
    n = mats[0].shape[0]
    T0, _ = onsets(seg)
    if variant == "indep":                          # the generator of stage X: every snapshot in order
        rng = np.random.default_rng([seed, P.MODES.index(mode), round(rho * 1000)])
        pk = [keys_of(P.plant(A, mode, rho, rng)[0]) for A in mats]
        block = lambda t0: pk[t0:t0 + P.H]
    elif variant == "persist":
        block = lambda t0: [keys_of(A) for A in persistent_block(mats, t0, P.LENGTH, mode, rho, seed)[:P.H]]
    else:
        block = lambda t0: keys[t0:t0 + P.H]
    Vb = spotlight.sketch(keys, n)
    out = {k: [] for k in ("ase", "spotlight", "ase_t2", "energy_t2")}
    for t0 in T0:
        b = block(t0)
        win = keys[t0 - 12:t0] + b
        a = [ase.stat_at(win, 12 + h, n) for h in range(P.H)]
        Vs = np.r_[Vb[t0 - 12:t0], spotlight.sketch(b, n)]
        out["ase"].append(max(a))
        out["spotlight"].append(max(spotlight.score_at(Vs, 12 + h) for h in range(P.H)))
        out["ase_t2"].append(a[2])
        out["energy_t2"].append(ase_energy(win, 14, n))
    save_npz(p, t0=T0, blk=P.block_ids(seg, T0), **{k: np.array(v) for k, v in out.items()},
             seconds=time.time() - t_start)
    print(f"np {job}: {time.time() - t_start:.0f} s", flush=True)
    return job


def run_plants_new(workers: int, bgs) -> None:
    jobs = [("none", ID, "none", 0.0, 0) for ID in bgs]
    jobs += [("persist", ID, m, r, s) for ID in bgs for m, r in NP_CELLS for s in SEEDS]
    jobs += [("indep", ID, m, r, s) for ID in bgs if ID in NP_INDEP for m, r in NP_CELLS for s in SEEDS]
    jobs.sort(key=lambda j: j[1] != "PM")
    for _ in pool_map(np_job, [j for j in jobs if not np_path(*j).exists()], workers):
        pass
    ref = json.loads((RESULTS / "plants.json").read_text())["x_rows"]
    out = {"meta": {"backgrounds": list(bgs), "cells": [list(c) for c in NP_CELLS], "seeds": list(SEEDS),
                    "indep_regenerated": list(NP_INDEP), "rows": list(NEW_ROWS)}, "cells": {}, "check_X0": {}}
    dev = 0.0
    for ID in bgs:
        neg = np.load(np_path("none", ID, "none", 0.0, 0))
        for m, r in NP_CELLS:
            c = {}
            for v in ("indep", "persist"):
                if v == "indep" and ID not in NP_INDEP:
                    continue
                z = [np.load(np_path(v, ID, m, r, s)) for s in SEEDS]
                blk = np.concatenate([q["blk"] for q in z])
                res = {}
                for q in NEW_ROWS:
                    pos, ng = np.concatenate([x[q] for x in z]), np.tile(neg[q], len(SEEDS))
                    res[q] = {"auc": P.auc(pos, neg[q]), **P.paired_summary(pos, ng, blk)}
                e1, e0 = np.concatenate([x["energy_t2"] for x in z]), np.tile(neg["energy_t2"], len(SEEDS))
                a1, a0 = np.concatenate([x["ase_t2"] for x in z]), np.tile(neg["ase_t2"], len(SEEDS))
                ok = e0 > 0
                res["ase_energy_t2"] = {"median_ratio": float(np.median(e1[ok] / e0[ok])),
                                        "share_lower": float(np.mean(e1 < e0)), "share_higher": float(np.mean(e1 > e0))}
                res["ase_score_t2"] = {"share_lower": float(np.mean(a1 < a0)), "share_higher": float(np.mean(a1 > a0))}
                c[v] = res
                if v == "indep":
                    for q in NEW_ROWS:
                        x = ref[ID][f"{m}/{r:g}"][q]
                        dev = max(dev, abs(res[q]["auc_paired"] - x["auc_paired"]), abs(res[q]["auc"] - x["auc"]))
            out["cells"][f"{ID}/{m}/{r:g}"] = c
            print(f"{ID}/{m}/{r:g}", {v: {q: round(c[v][q]["auc_paired"], 3) for q in NEW_ROWS} for v in c}, flush=True)
    out["check_X0"] = {"max_abs_dev_vs_plants_json_x_rows": dev, "status": "OK" if dev < 1e-12 else "FAIL"}
    write_json(RESULTS / "identity_plants_new.json", out)
    print("wrote results/identity_plants_new.json", out["check_X0"])


def checks() -> dict:
    """C4 (Chen-Zhang null moments, exact enumeration), C6 (NBS null mean), C8 (step-11 replica on PS), C9 (turnover:
    0 on identical snapshots, invariant to one relabelling of every snapshot, in [0, 1])."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(50):
        Dm = rng.random((12, 12))
        Ed = chenzhang.mst_edges(Dm + Dm.T)
        deg = np.bincount(Ed.ravel(), minlength=12).astype(float)
        for n1 in (2, 5):
            mu, var = chenzhang.null_moments(deg, n1, 12 - n1)
            R = []
            for c in itertools.combinations(range(12), n1):
                lab = np.zeros(12, bool)
                lab[list(c)] = True
                R.append(np.sum(lab[Ed[:, 0]] != lab[Ed[:, 1]]))
            worst = max(worst, abs(np.mean(R) - mu), abs(np.var(R) - var))
    out = {"C4": {"max_abs_error_exact_enumeration": worst, "status": "OK" if worst < 1e-9 else "FAIL"}}
    keys = [np.flatnonzero(rng.random(4950) < 0.05).astype(np.int64) for _ in range(600)]
    d = nbs.scores(keys, np.zeros(600, int))[12:]
    se = d.std() / np.sqrt(len(d) / 12)                # overlapping windows: about one independent value per 12
    out["C6"] = {"mean": float(d.mean()), "se_blocks": float(se), "status": "OK" if abs(d.mean()) < 3 * se else "FAIL"}
    st = settings()
    mt = meta("PS")
    F = [snapshot(A, t) for t, A in enumerate(matrices("PS"))]
    F = {k: np.array([r[k] for r in F]) for k in F[0]}
    S = bench_rows(F, mt["seg"], st, ("counts", "cusum", "bocpd", "mmd", "lad", "scpd"), "PS")
    z = np.load(BENCH / "scores_PS.npz")
    diff = {r: float(np.nanmax(np.abs(S[r] - z[f"s_{r}"]))) if np.array_equal(np.isnan(S[r]), np.isnan(z[f"s_{r}"]))
            else float("inf") for r in S}
    out["C8"] = {"max_abs_diff": diff, "status": "OK" if max(diff.values()) < 1e-9 else "FAIL"}
    keys = [np.flatnonzero(rng.random(4950) < 0.05).astype(np.int64) for _ in range(30)]      # turnover (v3 spec)
    perm = rng.permutation(100)
    relab = [np.sort(np.minimum(perm[k // 100], perm[k % 100]) * 100 + np.maximum(perm[k // 100], perm[k % 100]))
             for k in keys]
    d, dr = turnover.scores(keys, np.zeros(30, int), 100), turnover.scores(relab, np.zeros(30, int), 100)
    same = turnover.scores([keys[0]] * 30, np.zeros(30, int), 100)
    ok = np.isfinite(d)
    out["C9_turnover"] = {"identical_snapshots_max": float(np.nanmax(same)), "relabel_max_diff": float(np.nanmax(
        np.abs(d - dr))), "range": [float(d[ok].min()), float(d[ok].max())]}
    out["C9_turnover"]["status"] = "OK" if (out["C9_turnover"]["identical_snapshots_max"] == 0 and
                                            out["C9_turnover"]["relabel_max_diff"] < 1e-12 and
                                            0 <= d[ok].min() and d[ok].max() <= 1) else "FAIL"
    return out


# ================================================================================================ summary
def f3(x):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def summary() -> None:
    p = RESULTS / "identity.json"
    if p.exists():
        d = json.loads(p.read_text())
        dss = d["pooled_datasets"]
        print("## A. Benchmark B: EP per dataset (mask of step 12)\n")
        print("| row | " + " | ".join(dss) + " | pooled [95% CI] | >= 10 events | row - counts [95% CI] |")
        for r in d["rows"]:
            e = d["pooled"]["ep"][r]
            e10 = d["pooled_10plus"]["ep"][r]
            c = d["pooled"]["minus_counts"].get(r)
            print(f"| {r} | " + " | ".join(f3(d['datasets'][x]['ep_mean'][r]) for x in dss) +
                  f" | {f3(e['mean'])} [{f3(e['lo'])}, {f3(e['hi'])}] | {f3(e10['mean'])} [{f3(e10['lo'])}, "
                  f"{f3(e10['hi'])}] | " + (f"{f3(c['mean'])} [{f3(c['lo'])}, {f3(c['hi'])}]" if c else "-") + " |")
        print(f"\nchecks: {json.dumps(d['checks']['C1']['status'])}")
    p = RESULTS / "identity_plants.json"
    if p.exists():
        d = json.loads(p.read_text())
        print("\n## B. Plants: AUC independent / persistent\n")
        rows = d["meta"]["id_rows"]
        print("| cell | " + " | ".join(rows) + " | counts (14) / persistent | moved share pers | jac orig/plant |")
        for k, c in d["cells"].items():
            pr = c["persist"]["rows"]
            print(f"| {k} | " + " | ".join(f"{f3(c['indep']['rows'][r]['auc'])} / {f3(pr[r]['auc'])}" for r in rows) +
                  f" | {f3(c['indep_step14']['counts'])} / {f3(pr.get('counts', {}).get('auc'))} | "
                  f"{f3(c['persist']['validity'].get('moved'))} | "
                  f"{f3(c['persist']['validity'].get('jac_orig'))}/{f3(c['indep']['validity'].get('jac_plant'))}/"
                  f"{f3(c['persist']['validity'].get('jac_plant'))} |")
        print("\n## B2. Plants: paired AUC [95% block bootstrap], independent / persistent\n")
        pa = lambda x: (f"{f3(x['auc_paired'])} [{f3(x['auc_paired_ci95'][0])}, {f3(x['auc_paired_ci95'][1])}]"
                        if x and "auc_paired_ci95" in x else "-")
        for k, c in d["cells"].items():
            print(f"| {k} | " + " | ".join(f"{r} {pa(c['indep']['rows'][r])} / {pa(c['persist']['rows'][r])}"
                                          for r in rows) + " |")
        print(f"\nchecks: {json.dumps(d['checks'])}")
        print(f"reading: {json.dumps(d['reading']['identity_rows_seeing_persistent_plants'])}")
    p = RESULTS / "identity_plants_new.json"
    if p.exists():
        d = json.loads(p.read_text())
        print("\n## B3. ASE and SpotLight: paired AUC [95% block bootstrap] independent (stage X) / persistent; ASE "
              "test energy at t0 + 2, planted / original: median ratio, share lower\n")
        x = json.loads((RESULTS / "plants.json").read_text())["x_rows"]
        for k, c in d["cells"].items():
            ID, m, r = k.split("/")
            en = lambda v, c=c: (f"{c[v]['ase_energy_t2']['median_ratio']:.3f}, "
                                 f"{c[v]['ase_energy_t2']['share_lower']:.3f}" if v in c else "-")
            print(f"| {k} | " + " | ".join(f"{q} {pa(x[ID][f'{m}/{r}'][q])} / {pa(c['persist'][q])}" for q in NEW_ROWS)
                  + f" | energy indep {en('indep')} / persist {en('persist')} |")
        print(f"\ncheck X0 (independent plants vs plants.json x_rows): {json.dumps(d['check_X0'])}")
    p = RESULTS / "identity_synth.json"
    if p.exists():
        d = json.loads(p.read_text())
        print("\n## C. Synthetic block-model changes: EP [95% CI] (tolerance 2)\n")
        sc = list(d["scenarios"])
        print("| row | " + " | ".join(sc) + " |")
        for r in d["meta"]["rows"]:
            print(f"| {r} | " + " | ".join(
                f"{f3(d['scenarios'][s]['rows'][r]['ep'])} [{f3(d['scenarios'][s]['rows'][r]['lo'])}, "
                f"{f3(d['scenarios'][s]['rows'][r]['hi'])}]" for s in sc) + " |")
        print("\nevents: " + ", ".join(f"{s} {d['scenarios'][s]['rows']['counts']['n_events']}" for s in sc))
        print("C7: " + json.dumps({s: d["scenarios"][s].get("C7") for s in sc}))
    p = RESULTS / "identity_new.json"
    if p.exists():
        d = json.loads(p.read_text())
        print("\n## D. ASE, SpotLight on the benchmark: pooled EP [95% CI], all / >= 10 events")
        for r in d["rows"]:
            e, e10 = d["pooled"]["ep"][r], d["pooled_10plus"]["ep"][r]
            print(f"| {r} | {f3(e['mean'])} [{f3(e['lo'])}, {f3(e['hi'])}] | {f3(e10['mean'])} [{f3(e10['lo'])}, "
                  f"{f3(e10['hi'])}] |")
        print("\n## D2. Synthetic series: EP of ASE, SpotLight; every row minus the counts, paired [95% CI, series then "
              "events]\n")
        for sc, res in d["synth"].items():
            print(f"* {sc}: " + "; ".join(f"{r} {f3(v['ep'])} [{f3(v['lo'])}, {f3(v['hi'])}]" for r, v in res.items()
                                          if r != "minus_counts"))
            print("  minus counts: " + "; ".join(f"{r} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                                                 for r, v in res["minus_counts"].items()))
    p = RESULTS / "identity_checks.json"
    if p.exists():
        print(f"\nchecks C4, C6, C8: {p.read_text()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*", default=["checks", "bench", "plants", "synth"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bg", nargs="+", default=list(BGS))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
        return
    t0 = time.time()
    if "checks" in a.parts:
        write_json(RESULTS / "identity_checks.json", checks())
    if "bench" in a.parts:
        run_bench(a.workers)
    if "plants" in a.parts:
        run_plants(a.workers, a.bg)
    if "synth" in a.parts:
        run_synth(a.workers)
    if "new" in a.parts:
        run_new(a.workers)
    if "plantsnew" in a.parts:
        run_plants_new(a.workers, a.bg)
    print(f"step 16: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
