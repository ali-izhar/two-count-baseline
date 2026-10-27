"""Step 14: planted wiring changes at fixed degrees on four real backgrounds (PS, HT09, PM, MIT) -> results/plants.json.

Does a detector see a change of wiring when every degree stays the same? Every snapshot of a background is planted
(plants.py); trials splice planted snapshots into the original series on 15-step blocks. Cells:
  E1  rand / assort / triclose at rho 0.1, 0.2, 0.5      seeds 0-9, block offsets 0-39 (wiring)
  E2  thin / deact at rho 0.1, 0.3                       same trials (activity controls)
  E5  swap at rho 0.5                                    same trials (isomorphic relabelling: every row is blind)
  TIRE rand / assort / triclose at rho 0.5               seeds 0-2, offsets 0 and 20 (TIRE on the eight features)
  X    rand / assort / triclose / swap at 0.5, thin / deact at 0.3, seeds 0-9, offsets 0-39: SCPD itself (not DOS)
       and the identity-aware ASE and SpotLight rows (methods/ase.py, methods/spotlight.py)
  P    SCPD's probe control: the original background with re-drawn probe vectors (spec v4 V8)
Effect table (spec v4 V6, V7): every cell and row with paired and unpaired AUC, increment, blocks; control cells.
Rows: methods (LAD, CUSUM, EWMA, BOCPD, MMD, TIRE from methods/), counts, constant, and the diagnostics transitivity,
assortativity (Newman r), FAIR PAIR = max of the two, degree heterogeneity, LAD-NL (LAD scoring on L_sym eigenvalues)
and the DOS row (plants.py; SCPD's signature, not SCPD). Online scale: each feature divided by its median |first
difference| over the first 20 snapshots of each segment. Metrics (plants.py): paired AUC with a block bootstrap over
periods, increment in null-SD units, tie and zero fractions (a row with ties in >= half of the trials is not
evaluable), null cells (median moved-edge share < 0.01), paired contrasts of every row with the counts; the
unpaired window-max AUC. Per-trial window maxima are cached per cell (trials_*.npz).

Usage: 14_plants.py [--workers 4] [--bg PS HT09 PM MIT] [--summary]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import json
import time
from multiprocessing import get_context

import numpy as np
import plants as P
import scipy.sparse as sp
from common import DATA, REPO, RESULTS, bundle, sha, src_sha, write_json
from evaluate import online_scale, seg_ranges, window_scores
from rows import SCPD_WIN, scpd_signature, stored

from methods import ase, scpd, spotlight, tire

BENCH_DIR = DATA / "bench"                     # <ID>.npz of the benchmark step (10_bench_data)
CACHE = DATA / "plants"                       # cached series statistics, cells and TIRE scores
CACHE.mkdir(parents=True, exist_ok=True)
OUT = RESULTS / "plants.json"
BGS = ("PS", "HT09", "PM", "MIT")
HELD = ("PS", "HT09", "PM")                    # the plan's rules use these three
SEEDS, OFFSETS = tuple(range(10)), tuple(range(40))
CELLS_E1 = [(m, r) for m in P.STRUCT for r in (0.1, 0.2, 0.5)]
CELLS_E2 = [(m, r) for m in P.ACTIV for r in (0.1, 0.3)]
CELLS_E5 = [("swap", 0.5)]
ROWS = ("constant", "counts", "dhet", "transitivity", "assortativity", "fair_pair", "lad", "lad_nl", "dos",
        "cusum", "ewma", "bocpd", "mmd")
TIRE_RHO, TIRE_SEEDS, TIRE_OFFSETS = 0.5, (0, 1, 2), (0, 20)


METHODS_SHA = sha(*(REPO / "methods" / f for f in ("counts.py", "lad.py", "cusum.py", "ewma.py", "bocpd.py", "mmd.py",
                                                    "interface.py", "scpd.py")), REPO / "data" / "registry.py",
                  REPO / "repro" / "common.py")
DATA_SHA = {}


def bench_path(ID):
    return BENCH_DIR / f"{ID}.npz"


def segments(ID):
    with np.load(bench_path(ID), allow_pickle=False) as z:
        return z["segment"].astype(int)


def stats_key(ID):
    if ID not in DATA_SHA:
        DATA_SHA[ID] = sha(bench_path(ID))
    return sha(P.STATS_SHA, src_sha(stored), DATA_SHA[ID], METHODS_SHA)


# ------------------------------------------------------------------------------------------------ stage 1: series
def stats_path(ID, mode, rho, seed):
    return CACHE / f"stats_{ID}_{mode}_{rho:g}_s{seed}_{stats_key(ID)}.npz"


def stats_job(job):
    p = stats_path(*job)
    if not p.exists():
        t0 = time.time()
        z = P.series_stats(list(stored(job[0])), *job[1:])
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **z, seconds=time.time() - t0)
        os.replace(tmp, p)
    return job


def series(ID, mode, rho, seed):
    return dict(np.load(stats_path(ID, mode, rho, seed)))


def series_jobs(ID):
    J = [(ID, "none", 0.0, 0)]
    J += [(ID, m, r, s) for m, r in CELLS_E1 + CELLS_E2 + CELLS_E5 for s in SEEDS]
    return J


# ------------------------------------------------------------------------------------------------ stage 2: cells
def trials_path(ID, mode, rho):
    """Per-trial window maxima of every row (pos_<row>, neg_<row>), the bootstrap block of each trial (blk) and the
    plant's validity, achieved share and effect sizes (info_json)."""
    return CACHE / f"trials_{ID}_{mode}_{rho:g}_{stats_key(ID)}_{CELL_SHA}.npz"


def trials_job(args):
    ID, mode, rho = args
    p = trials_path(ID, mode, rho)
    if p.exists():
        return args
    t_start = time.time()
    zb = series(ID, "none", 0.0, 0)
    seg = segments(ID)
    T = len(seg)
    U = np.random.default_rng(0).random(T)
    Fb = P.feats(zb)
    Du = P.score_all(Fb, seg, U)
    met = {r: P.Metrics(Du[r], seg) for r in ROWS}
    ok20 = np.zeros(T, bool)
    for s0, s1 in seg_ranges(seg):
        ok20[s0 + P.START:s1] = True
    eff, val, moved, ratio = [], [], [], []
    for sd in SEEDS:
        zp = series(ID, mode, rho, sd)
        Fp = P.feats(zp)
        eff.append(P.effect(zp, zb, seg))
        val.append([float(np.max(zp["maxdd"])), float(np.max(np.abs(zp["dE"]))), float(np.max(zp["ddsorted"]))])
        moved.append(zp["moved"][ok20])
        if mode in ("triclose", "swap"):
            tgt = np.round(rho * (zp["E_nl"] if mode == "triclose" else zp["nplus"]) / 2)
            ratio.append(np.where(tgt > 0, zp["ndone"] / np.maximum(tgt, 1), np.nan)[ok20])
        for o in OFFSETS:
            t0s = P.block_onsets(seg, o)
            if len(t0s) == 0:
                continue
            D = P.score_all({k: P.splice(Fb[k], Fp[k], t0s) for k in Fb}, seg, U)
            for r in ROWS:
                met[r].add(D[r], t0s)
    tr = {r: met[r].trials() for r in ROWS}
    info = {"effect": {k: float(np.mean([e[k] for e in eff])) for k in eff[0]},
            "validity": {"max_abs_delta_degree": max(v[0] for v in val), "max_abs_delta_E": max(v[1] for v in val),
                         "max_abs_delta_sorted_degree": max(v[2] for v in val)},
            "achieved": {"moved_edge_share_median": float(np.median(np.concatenate(moved)))},
            "seconds": time.time() - t_start}
    if ratio:
        info["achieved"]["accepted_over_target_median"] = float(np.nanmedian(np.concatenate(ratio)))
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, blk=tr[ROWS[0]]["blk"], info_json=json.dumps(info),
                        **{f"{k}_{r}": tr[r][k] for r in ROWS for k in ("pos", "neg")})
    os.replace(tmp, p)
    return args


CELL_SHA = sha(src_sha(P.feats, P.per_segment, bundle, P.score_all, P.block_onsets, P.splice, P.win_max, P.Metrics,
                       P.effect, P.dos_scores, P.block_ids, online_scale, window_scores, seg_ranges, trials_job),
               json.dumps(P.SET), json.dumps(P.BURN))


def cell(ID, mode, rho) -> dict:
    """Metrics of every row in one cell from its cached trials."""
    z = np.load(trials_path(ID, mode, rho))
    blk = z["blk"]
    info = json.loads(str(z["info_json"]))
    out = {"bg": ID, "mode": mode, "rho": rho, "family": "wiring" if mode in P.STRUCT else "activity",
           "rows": {r: P.trial_result(z[f"pos_{r}"], z[f"neg_{r}"], blk) for r in ROWS},
           "paired_minus_counts": {r: P.paired_contrast(z[f"pos_{r}"], z[f"neg_{r}"], z["pos_counts"], z["neg_counts"],
                                                        blk) for r in ROWS if r != "counts"}}
    return out | {k: info[k] for k in ("effect", "validity", "achieved")}


# ------------------------------------------------------------------------------------------------ stage 3: TIRE
TIRE_SHA = sha(REPO / "methods" / "tire.py", REPO / "data" / "registry.py", REPO / "repro" / "common.py")


def tire_path(X, seg):
    h = hashlib.sha256(np.ascontiguousarray(X, float).tobytes() + np.asarray(seg, np.int64).tobytes()
                       + json.dumps(P.SET["tire"]).encode() + TIRE_SHA.encode()).hexdigest()[:24]
    return CACHE / f"tire_{h}.npy"


def tire_job(args):
    """methods/tire.py trained on each segment separately (offline); cached by input, setting and code."""
    X, seg = args
    p = tire_path(X, seg)
    if not p.exists():
        out = np.zeros(len(seg))
        for s0, s1 in seg_ranges(seg):
            out[s0:s1] = tire.detect(bundle(X=X[s0:s1]), P.SET["tire"]).scores
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, out)
        os.replace(tmp, p)
    return str(p)


def tire_inputs(ID):
    """(key, X, seg) of every TIRE training: the original series and the spliced trials (eight raw features)."""
    zb = series(ID, "none", 0.0, 0)
    seg = segments(ID)
    out = [((None, None, None), zb["raw8"])]
    for m in P.STRUCT:
        for s in TIRE_SEEDS:
            zp = series(ID, m, TIRE_RHO, s)
            out += [((m, s, o), P.splice(zb["raw8"], zp["raw8"], P.block_onsets(seg, o))) for o in TIRE_OFFSETS]
    return [(k, X, seg) for k, X in out]


def tire_rows(ID):
    inp = tire_inputs(ID)
    seg = inp[0][2]
    SC = {k: np.load(tire_path(X, seg)) for k, X, _ in inp}
    out = {}
    for m in P.STRUCT:
        met = P.Metrics(SC[(None, None, None)], seg)
        for s in TIRE_SEEDS:
            for o in TIRE_OFFSETS:
                met.add(SC[(m, s, o)], P.block_onsets(seg, o))
        out[m] = {"tire": {k: v for k, v in met.result().items()
                           if k in ("auc", "n_pos", "auc_paired", "auc_paired_ci95", "tie_fraction", "evaluable",
                                    "increment_null_sd", "n_blocks", "n_trials")}}
    return out


# ------------------------------------------------------------------------------------------------ reading
NULL_MOVED = 0.01                                  # a wiring cell whose median moved-edge share is below: null


def reading(B):
    """Null cells, the rows whose paired-AUC interval lies above 0.5, the rows that are not evaluable (tie fraction
    >= 0.5), plant validity, and the relabelling control with round-off ties (maxima within 1e-9 relative count as
    tied)."""
    out = {"null_cells": [], "above_chance": {}, "not_evaluable": {}}
    for ID in B:
        for c in B[ID]["e1"] + B[ID]["e2"] + [B[ID]["e5"]]:
            key = f"{ID}/{c['mode']}/{c['rho']:g}"
            if c["mode"] in P.STRUCT and c["achieved"]["moved_edge_share_median"] < NULL_MOVED:
                out["null_cells"].append(key)
            R = c["rows"]
            out["above_chance"][key] = [r for r in ROWS if R[r]["auc_paired_ci95"][0] > 0.5]
            out["not_evaluable"][key] = [r for r in ROWS if not R[r]["evaluable"]]
    rnd = {}
    for ID in B:
        z = np.load(trials_path(ID, "swap", 0.5))
        rnd[ID] = {}
        for r in ROWS:
            p_, n_ = z[f"pos_{r}"], z[f"neg_{r}"]
            ok = np.isfinite(p_) & np.isfinite(n_)
            p_, n_ = p_[ok], n_[ok]
            rel = np.abs(p_ - n_) / np.maximum(np.maximum(np.abs(p_), np.abs(n_)), 1e-300)
            tie = rel <= 1e-9
            rnd[ID][r] = {"auc_paired_rtol1e-9": float(np.mean((p_ > n_) & ~tie) + 0.5 * np.mean(tie)),
                          "tie_fraction_rtol1e-9": float(tie.mean()), "max_rel_diff": float(rel.max())}
    out["swap_roundoff"] = rnd
    dd = [c["validity"][k] for ID in B for c in B[ID]["e1"] for k in ("max_abs_delta_degree", "max_abs_delta_E")]
    out["plant_degrees"] = {"status": "INVALID" if max(dd) > 0 else "OK", "max_abs_ddeg": max(dd), "n_values": len(dd)}
    return out


# ------------------------------------------------------------------------------------------------ stage 4: SCPD, ASE, SpotLight
# SCPD itself (methods/scpd.py, its signature with probes RandomState(t) as step 11, windows 5 / 10, first-difference
# score) and two identity-aware rows (methods/ase.py, methods/spotlight.py) on the plants of stage 1, regenerated with
# the same generator: rho 0.5 structural and relabelling, the activity controls at 0.3. The score at t needs the
# snapshots t - 12 .. t, so each trial is scored on a short slice of the spliced series (planted on [t0, t0 + 15)).
CELLS_X = [(m, 0.5) for m in P.STRUCT] + [("swap", 0.5)] + [(m, 0.3) for m in P.ACTIV]
X_ROWS = ("scpd", "ase", "spotlight")
X_SHA = sha(src_sha(scpd_signature, P.plant, P.swap_edges, P.edges_of, P.build, stored),
            *(REPO / "methods" / f"{m}.py" for m in ("scpd", "ase", "spotlight", "interface")))


def x_path(ID, mode, rho, seed):
    return CACHE / f"xrows_{ID}_{mode}_{rho:g}_s{seed}_{sha(X_SHA, DATA_SHA.get(ID) or sha(bench_path(ID)))}.npz"


def _keys(Q) -> np.ndarray:
    U = sp.triu(Q, 1).tocoo()
    return np.sort(U.row.astype(np.int64) * Q.shape[0] + U.col.astype(np.int64))


def _clean(A):
    Q = sp.csr_matrix(A, dtype=np.float64)
    Q.setdiag(0)
    Q.eliminate_zeros()
    Q.data[:] = 1.0
    return Q


def x_series(ID, mode, rho, seed):
    """(SCPD signatures, pair keys) of the planted series (generator and order of P.series_stats) or the original."""
    rng = np.random.default_rng([seed, P.MODES.index(mode), round(rho * 1000)])
    sig, keys = [], []
    for t, A in enumerate(stored(ID)):
        Q = _clean(A if mode == "none" else P.plant(A, mode, rho, rng)[0])
        act = np.flatnonzero(np.diff(Q.indptr))
        sig.append(scpd_signature(Q[act][:, act].tocsr(), Q.shape[0], np.random.RandomState(t)))
        keys.append(_keys(Q))
    return np.array(sig), keys, Q.shape[0]


def x_job(job):
    """Window maxima over [t0, t0 + H) of every X row in every trial (offset, onset) of one planted series; the
    original series (mode none) gives the negatives."""
    ID, mode, rho, seed = job
    p = x_path(*job)
    if p.exists():
        return job
    seg = segments(ID)
    sig, keys, n = x_series(ID, mode, rho, seed)
    base = sig, keys, n
    if mode != "none":
        base = x_series(ID, "none", 0.0, 0)
    V, Vb = spotlight.sketch(keys, n), spotlight.sketch(base[1], n)
    out = {r: [] for r in X_ROWS}
    blk = []
    for o in OFFSETS:
        t0s = P.block_onsets(seg, o)
        for t0 in t0s:
            sl = np.r_[base[0][t0 - 12:t0], sig[t0:t0 + P.H]]
            z = scpd.score(sl, SCPD_WIN, 0.05)[0]
            out["scpd"].append(float(z[12:12 + P.H].max()))
            win = base[1][t0 - 12:t0] + keys[t0:t0 + P.H]
            out["ase"].append(max(ase.stat_at(win, 12 + h, n) for h in range(P.H)))
            Vs = np.r_[Vb[t0 - 12:t0], V[t0:t0 + P.H]]
            out["spotlight"].append(max(spotlight.score_at(Vs, 12 + h) for h in range(P.H)))
        blk += P.block_ids(seg, t0s).tolist()
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, blk=np.array(blk), **{r: np.array(v) for r, v in out.items()})
    os.replace(tmp, p)
    return job


def x_cell(args) -> dict:
    ID, mode, rho = args
    neg = np.load(x_path(ID, "none", 0.0, 0))
    z = [np.load(x_path(ID, mode, rho, s)) for s in SEEDS]
    blk = np.concatenate([q["blk"] for q in z])
    res = {r: {"auc": P.auc(np.concatenate([q[r] for q in z]), neg[r]),
               **P.paired_summary(np.concatenate([q[r] for q in z]), np.tile(neg[r], len(SEEDS)), blk)} for r in X_ROWS}
    return {"bg": ID, "mode": mode, "rho": rho, "rows": res}


# ------------------------------------------------------------------------------------------------ stage 5: SCPD probes
# SCPD's DOS draws Rademacher probes per snapshot in vertex order (RandomState(t)); a relabelling therefore acts as a
# fresh draw of the probes. Control (spec v4 V8): realization k of the ORIGINAL background uses RandomState(10^6 k + t)
# (k = 0: the probes of stage X); for each negative realization a = 0..9 the positives splice realizations b = 10..19
# into [t0, t0 + H) with the context [t0 - 12, t0) of a, as stage X does for a plant.
PROBE_A, PROBE_B = tuple(range(10)), tuple(range(10, 20))


def probe_path(ID, k):
    return CACHE / f"probe_{ID}_k{k}_{sha(X_SHA, src_sha(probe_job), DATA_SHA.get(ID) or sha(bench_path(ID)))}.npy"


def probe_job(job):
    """SCPD signatures of the original background with probe realization k."""
    ID, k = job
    p = probe_path(ID, k)
    if not p.exists():
        sig = []
        for t, A in enumerate(stored(ID)):
            Q = _clean(A)
            act = np.flatnonzero(np.diff(Q.indptr))
            sig.append(scpd_signature(Q[act][:, act].tocsr(), Q.shape[0], np.random.RandomState(10 ** 6 * k + t)))
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, np.array(sig))
        os.replace(tmp, p)
    return job


def probe_cell(job) -> dict:
    """Paired AUC of SCPD on the re-probed original against negative realization a (10 positive realizations)."""
    ID, a = job
    seg = segments(ID)
    sa = np.load(probe_path(ID, a))
    za = np.full(len(seg), np.nan)
    for s0, s1 in seg_ranges(seg):
        za[s0:s1] = scpd.score(sa[s0:s1], SCPD_WIN, 0.05)[0]
    pos, neg, blk = [], [], []
    for b in PROBE_B:
        sb = np.load(probe_path(ID, b))
        for o in OFFSETS:
            t0s = P.block_onsets(seg, o)
            for t0 in t0s:
                z = scpd.score(np.r_[sa[t0 - 12:t0], sb[t0:t0 + P.H]], SCPD_WIN, 0.05)[0]
                pos.append(float(z[12:12 + P.H].max()))
                neg.append(float(za[t0:t0 + P.H].max()))
            blk += P.block_ids(seg, t0s).tolist()
    n1 = len(pos) // len(PROBE_B)
    return {"bg": ID, "a": a, "auc": P.auc(pos, neg[:n1]), "neg_max_sum": float(np.sum(neg[:n1])),
            **P.paired_summary(np.array(pos), np.array(neg), np.array(blk))}


def run_probe(bgs, workers) -> dict:
    J = [(ID, k) for ID in bgs for k in PROBE_A + PROBE_B if not probe_path(ID, k).exists()]
    J.sort(key=lambda j: j[0] != "PM")
    for k, j in pool_map(probe_job, J, workers):
        if k % 10 == 0:
            print(f"[probe signatures {k}/{len(J)}] {j}", flush=True)
    out = {}
    for _, c in pool_map(probe_cell, [(ID, a) for ID in bgs for a in PROBE_A], workers):
        out.setdefault(c.pop("bg"), {})[str(c.pop("a"))] = c
    for ID, R in out.items():
        R["check_a0_negatives_equal_stage_x"] = R["0"].pop("neg_max_sum") == float(np.sum(np.load(
            x_path(ID, "none", 0.0, 0))["scpd"]))
        for a in PROBE_A[1:]:
            R[str(a)].pop("neg_max_sum")
        v = np.array([R[str(a)]["auc_paired"] for a in PROBE_A])
        R["summary"] = {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
                        "n_ci_excluding_0.5": sum(not (R[str(a)]["auc_paired_ci95"][0] <= 0.5 <=
                                                       R[str(a)]["auc_paired_ci95"][1]) for a in PROBE_A),
                        "n_realizations": len(PROBE_A)}
    return out


# ------------------------------------------------------------------------------------------------ effect sizes
def effect_table(out) -> dict:
    """Every cell and row (spec v4 V6): paired AUC [CI], unpaired AUC, increment in null-SD units, blocks; and the
    control cells (V7): relabelling and null cells whose paired interval excludes 0.5, per row."""
    tab = {}
    for ID, b in out["backgrounds"].items():
        for c in b["e1"] + b["e2"] + [b["e5"]]:
            tab[f"{ID}/{c['mode']}/{c['rho']:g}"] = {r: c["rows"][r] for r in ROWS}
        for m, R in b["tire"].items():
            tab.setdefault(f"{ID}/{m}/{TIRE_RHO:g}", {})["tire"] = R["tire"]
        for k, R in out["x_rows"][ID].items():
            tab.setdefault(f"{ID}/{k}", {}).update(R)
    keep = ("auc_paired", "auc_paired_ci95", "auc", "increment_null_sd", "n_blocks", "n_trials", "evaluable")
    tab = {k: {r: {q: v[q] for q in keep if q in v} for r, v in R.items()} for k, R in tab.items()}
    nulls = out["reading"]["null_cells"]
    swaps = [f"{ID}/swap/0.5" for ID in out["backgrounds"]]
    rnd = out["reading"]["swap_roundoff"]           # relabelling: maxima equal to 1e-9 (relative) are ties
    for ID in out["backgrounds"]:
        for r, t in rnd[ID].items():
            if t["tie_fraction_rtol1e-9"] >= 0.5:
                tab[f"{ID}/swap/0.5"][r]["roundoff_tie"] = True
    excl = lambda v: not (v.get("roundoff_tie") or v["auc_paired_ci95"][0] <= 0.5 <= v["auc_paired_ci95"][1])
    rows = sorted({r for R in tab.values() for r in R})
    ctrl = {}
    for r in rows:
        ctrl[r] = {}
        for name, cells in (("relabelling", swaps), ("null", nulls)):
            got = [(c, tab[c][r]) for c in cells if r in tab.get(c, {}) and "auc_paired_ci95" in tab[c][r]]
            ctrl[r][name] = {"n_cells": len(got), "n_excluding_0.5": sum(excl(v) for _, v in got),
                             "cells_excluding": {c: round(v["auc_paired"], 3) for c, v in got if excl(v)}}
    weak = {k: [r for r, v in R.items() if "auc_paired_ci95" in v and v["auc_paired_ci95"][0] > 0.5
                and v.get("auc", 1.0) < 0.55 and abs(v.get("increment_null_sd", 1.0)) < 0.1] for k, R in tab.items()}
    return {"cells": tab, "controls": ctrl, "consistent_but_negligible": {k: v for k, v in weak.items() if v},
            "n_blocks": {ID: tab[f"{ID}/rand/0.5"]["counts"]["n_blocks"] for ID in out["backgrounds"]}}


def run_x(bgs, workers) -> dict:
    J = [(ID, "none", 0.0, 0) for ID in bgs] + [(ID, m, r, s) for ID in bgs for m, r in CELLS_X for s in SEEDS]
    J = [j for j in J if not x_path(*j).exists()]
    J.sort(key=lambda j: j[0] != "PM")
    for k, j in pool_map(x_job, J, workers):
        if k % 20 == 0:
            print(f"[x rows {k}/{len(J)}] {j}", flush=True)
    out = {}
    for _, c in pool_map(x_cell, [(ID, m, r) for ID in bgs for m, r in CELLS_X], workers):
        out.setdefault(c["bg"], {})[f"{c['mode']}/{c['rho']:g}"] = c["rows"]
    return out


# ------------------------------------------------------------------------------------------------ main
def pool_map(fn, jobs, workers):
    with get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        yield from enumerate(pool.imap_unordered(fn, jobs), 1)


def run(bgs, workers):
    t0 = time.time()
    J = [j for ID in bgs for j in series_jobs(ID) if not stats_path(*j).exists()]
    J.sort(key=lambda j: (j[0] != "PM", j[1] != "none"))
    for k, j in pool_map(stats_job, J, workers):
        print(f"[series {k}/{len(J)}] {j}", flush=True)
    t1 = time.time()
    C = [(ID, m, r) for ID in bgs for m, r in CELLS_E1 + CELLS_E2 + CELLS_E5]
    todo = sorted((c for c in C if not trials_path(*c).exists()), key=lambda c: c[0] != "PM")
    for k, c in pool_map(trials_job, todo, workers):
        print(f"[cell {k}/{len(todo)}] {c}", flush=True)
    cells = {c: cell(*c) for c in C}
    t2 = time.time()
    tj = [(X, seg) for ID in bgs for _, X, seg in tire_inputs(ID)]
    tj = [j for j in tj if not tire_path(*j).exists()]
    for k, _ in pool_map(tire_job, tj, workers):
        print(f"[tire {k}/{len(tj)}]", flush=True)
    t3 = time.time()
    B = {}
    for ID in bgs:
        B[ID] = {"e1": [cells[(ID, m, r)] for m, r in CELLS_E1], "e2": [cells[(ID, m, r)] for m, r in CELLS_E2],
                 "e5": cells[(ID, "swap", 0.5)], "tire": tire_rows(ID)}
    out = {"meta": {"backgrounds": list(bgs), "seeds": list(SEEDS), "offsets": [OFFSETS[0], OFFSETS[-1]], "H": P.H,
                    "block": P.LENGTH, "period": P.PERIOD, "start": P.START, "rows": list(ROWS), "settings": P.SET,
                    "tire": {"rho": TIRE_RHO, "seeds": list(TIRE_SEEDS), "offsets": list(TIRE_OFFSETS)},
                    "data_sha12": {ID: DATA_SHA[ID] for ID in bgs}, "stats_sha": P.STATS_SHA, "cell_sha": CELL_SHA,
                    "seconds": {"series": t1 - t0, "cells": t2 - t1, "tire": t3 - t2}},
           "backgrounds": B}
    out["reading"] = reading(B)
    out["x_rows"] = run_x(bgs, workers)
    out["effect_table"] = effect_table(out)
    out["scpd_probe"] = run_probe(bgs, workers)
    write_json(OUT, out)
    print(f"wrote {OUT} ({time.time() - t0:.0f} s)")


def summary():
    d = json.loads(OUT.read_text())
    B, f3 = d["backgrounds"], (lambda x: "-" if x is None else f"{x:.3f}")
    pa = lambda x: f"{x['auc_paired']:.3f} [{x['auc_paired_ci95'][0]:.3f}, {x['auc_paired_ci95'][1]:.3f}]"
    print("A. Wiring plants: paired AUC [95% block bootstrap] (10 seeds, offsets 0-39); unpaired window-max AUC")
    for ID, b in B.items():
        hdr = [f"{c['mode']} {c['rho']:g}" for c in b["e1"]]
        print(f"\n{ID} paired\n| row | " + " | ".join(hdr) + " |")
        for r in d["meta"]["rows"]:
            print(f"| {r} | " + " | ".join(pa(c["rows"][r]) for c in b["e1"]) + " |")
        print(f"\n{ID} unpaired\n| row | " + " | ".join(hdr) + " |")
        for r in d["meta"]["rows"]:
            print(f"| {r} | " + " | ".join(f3(c["rows"][r]["auc"]) for c in b["e1"]) + " |")
    print("\nA2. Mean paired increment (null SD units) | tie fraction | zero fraction, every cell")
    for ID, b in B.items():
        for c in b["e1"] + b["e2"] + [b["e5"]]:
            print(f"| {ID} {c['mode']} {c['rho']:g} | " + " | ".join(
                f"{r} {c['rows'][r]['increment_null_sd']:+.2f}/{c['rows'][r]['tie_fraction']:.2f}/"
                f"{c['rows'][r]['zero_fraction']:.2f}" for r in d["meta"]["rows"]) + " |")
    print("\nA3. Row minus counts, paired AUC difference [95% block bootstrap], rho 0.5")
    for ID, b in B.items():
        for c in b["e1"]:
            if c["rho"] == 0.5:
                print(f"| {ID} {c['mode']} | " + " | ".join(
                    f"{r} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                    for r, v in c["paired_minus_counts"].items()) + " |")
    print("\nValidity: max |Delta d| / max |Delta |E|| / median moved-edge share")
    for ID, b in B.items():
        print(f"| {ID} | " + " | ".join(f"{c['validity']['max_abs_delta_degree']:g}/"
                                        f"{c['validity']['max_abs_delta_E']:g}/"
                                        f"{c['achieved']['moved_edge_share_median']:.2f}" for c in b["e1"]) + " |")
    print("\nEffect at rho 0.5 (median |planted - original| / median |first difference|): trans assort lam dhet")
    for ID, b in B.items():
        for c in b["e1"]:
            if c["rho"] == 0.5:
                ks = ("trans", "assort", "lam", "dhet")
                print(f"| {ID} {c['mode']} | " + " | ".join(f"{c['effect'][k]:.2f}" for k in ks) + " |")
    print("\nB. TIRE (eight features), rho 0.5, seeds 0-2, offsets 0, 20: unpaired AUC (paired) | n_pos")
    for ID, b in B.items():
        for m, R in b["tire"].items():
            print(f"| {ID} | {m} | {f3(R['tire']['auc'])} (paired {pa(R['tire'])}) | {R['tire']['n_pos']} |")
    print("\nC. Activity controls: paired AUC")
    for ID, b in B.items():
        print(f"\n{ID}\n| row | " + " | ".join(f"{c['mode']} {c['rho']:g}" for c in b["e2"]) + " |")
        for r in d["meta"]["rows"]:
            print(f"| {r} | " + " | ".join(pa(c["rows"][r]) for c in b["e2"]) + " |")
    print("\nE. Swap (rho 0.5): paired AUC per row; with round-off ties (rtol 1e-9)")
    for r in d["meta"]["rows"]:
        print(f"| {r} | " + " | ".join(f"{f3(b['e5']['rows'][r]['auc_paired'])} / "
                                       f"{f3(d['reading']['swap_roundoff'][ID][r]['auc_paired_rtol1e-9'])}"
                                       for ID, b in B.items()) + " |")
    q = d["reading"]
    print(f"\nNull cells {q['null_cells']}; plant degrees {json.dumps(q['plant_degrees'])}")
    print("rows with paired-AUC CI above 0.5: " + json.dumps(q["above_chance"]))
    print("not evaluable (tie fraction >= 0.5): " + json.dumps(q["not_evaluable"]))
    print("\nX. SCPD, ASE, SpotLight: paired AUC")
    for ID, cs in d["x_rows"].items():
        print(f"| {ID} | " + " | ".join(f"{k} " + "; ".join(f"{r} {pa(v)}" for r, v in R.items())
                                        for k, R in cs.items()) + " |")
    et = d["effect_table"]
    print(f"\nV6. Effect sizes (paired AUC [CI] / unpaired AUC / increment in null SD / blocks); blocks {et['n_blocks']}")
    for k, R in et["cells"].items():
        print(f"| {k} | " + " | ".join(
            f"{r} {pa(v)} / {f3(v.get('auc'))} / {v.get('increment_null_sd', float('nan')):+.3f} / {v.get('n_blocks')}"
            for r, v in R.items() if "auc_paired_ci95" in v) + " |")
    print("consistent but negligible (CI above 0.5, unpaired AUC < 0.55, |increment| < 0.1 SD): "
          + json.dumps(et["consistent_but_negligible"]))
    print("\nV7. Control cells whose paired interval excludes 0.5 (relabelling of 4 | null cells): " + "; ".join(
        f"{r} {v['relabelling']['n_excluding_0.5']}/{v['relabelling']['n_cells']} {v['relabelling']['cells_excluding']}"
        f" | {v['null']['n_excluding_0.5']}/{v['null']['n_cells']} {v['null']['cells_excluding']}"
        for r, v in et["controls"].items()))
    print("\nV8. SCPD re-probed original (10 negative realizations x 10 positive): paired AUC per realization")
    for ID, R in d["scpd_probe"].items():
        print(f"| {ID} | " + " | ".join(f"{a} {pa(v)} ({f3(v['auc'])})" for a, v in R.items()
                                        if isinstance(v, dict) and a != "summary")
              + f" | {json.dumps(R['summary'])} | check {R['check_a0_negatives_equal_stage_x']} |")
    print(f"runtime (s): {json.dumps(d['meta']['seconds'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bg", nargs="+", default=list(BGS))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
    else:
        run(a.bg, a.workers)
