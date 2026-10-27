"""Step 22: settings and protocol sensitivity of benchmark B -> results/settings.json.

Does the ranking of Table 2 depend on the setting of each detector, the burn-in, the tolerance, the alignment of
events, the scale of the counts, the graph construction or, for the plants, the setting? Spec (written before the
first run): the module docstring sections A-I below, with the reading rules stated in results/settings.json "spec".

Common: score inputs of step 11 (cached features; Table 1 view for MIT, ENRON, AskUbuntu), EP of evaluate.py, the
ten pooled datasets of Table 2 and the four with >= 10 events, equal weight per dataset. Mask = the Table 2 common
mask (17-step burn-in per segment) unless a section says otherwise; a score a setting leaves undefined on the mask
takes the lowest value of its series (it cannot alarm there). Checks C1-C4 run first (results "checks").
  A  grids of the score-shaping parameters (seeded samples): EP at the authors' defaults, median / IQR over the grid,
     oracle (best pooled EP in hindsight), leave-one-dataset-out selection by EP; Table 2 (F1-fixed) setting
  B  per-detector burn-in (B1, every dataset) and shortest windows on the three short series (B2): Hits and EP
  C  tolerance tau = 0..10 and tau = each detector's window span; guard band after each event (17, own span)
  D  alignment: events moved to e + 1; scores of the detectors that see only G_{<t} shifted to include G_t
  E  counts: whole-series and online scale, weights on the squared edge difference, each count alone
  F  the source detectors' own constructions next to the counts on the same constructions
  G  plants: oracle AUC per detector and cell over the grids of A; tie diagnostics of MMD and CUSUM
  H  ties inside EP (event maxima tied with null maxima)
  I  s-GNN-I tuned on the validation interval of its protocol (needs the DGL environment of repro/README.md)
  J  identity rows (Hamming, Jaccard, NBS) under the rules of A on the (w, w_ref) grid of common.py, from step
     16's key caches; tolerance, own span and guard bands for all six identity rows (step 16's series)
  K  the conditional score test of Sec. 5 as detectors (methods/scoretest.py; step 16's key caches)
  L  more baselines: NetSimile (methods/netsimile.py), CUSUM without reset, BOCPD on the two counts
  N  grid sizes: the F1 grid of every row and the number of its settings that change the score (EP-relevant)
  M  LAD at its fixed windows with 6 and 499 Laplacian eigenvalues (its code's values on real and synthetic data;
     the benchmark row keeps 5); check M0 against the stored input; series repro/data/settings/lad_<ID>.npz
  R  the residual design of prereg/specs/RESIDUAL.md, calibration only (constant, counts, noise); detector
     residuals only if every calibration check passes
Output: results/settings.json; caches under repro/data/settings/ (also scores_<ID>.npz, section E, an input of
step 23: the causal count row and four count reference rows: the counts at w = 2, w_ref = 10; the causal edge count;
the counts with an expanding past scale; the edge count alone with the whole-series scale).

Usage: 22_settings.py [--sections ABCDEFGH] [--sgnn] [--workers 8] [--summary]
"""
from __future__ import annotations

import functools
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CACHE = HERE / "data" / "settings"
SGNN_PY = os.environ.get("SGNN_PYTHON", str(REPO / "data" / "raw" / "sgnn" / "venv" / "bin" / "python"))
LAD_FILES = {"UCI": "UCI_processed/OCnodeslinks_chars.txt", "USLegis": "USLegis_processed/LegisEdgelist.txt",
             "canVote": "canVote_processed/canVote_edgelist.txt"}


# ------------------------------------------------------------------------------------------------ workers (-I)
def own_counts_worker(ID: str, out: str) -> None:
    """Active vertices, edges and total weight per snapshot of LAD's own construction, parsed as the authors'
    loaders do (UCI: DiGraph per date, last weight kept; Senate: Graph per Congress; bills: DiGraph per year)."""
    import re
    f = REPO / "data" / "raw" / "lad" / "datasets" / LAD_FILES[ID]
    snaps, cur, E = [], None, {}
    for line in f.read_text().splitlines():
        if ID == "UCI":
            v = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
            if len(v) < 3:
                continue
            t, (u, x, w) = re.search(r"\d{4}-\d{2}-\d{2}", line).group(0), (int(v[-3]), int(v[-2]), int(v[-1]))
        else:
            p = line.split(",")
            if len(p) < 4:
                continue
            t, x, u, w = p[0], p[1], p[2], int(p[3])
        if cur is not None and t != cur:
            snaps.append(E)
            E = {}
        cur = t
        key = (u, x) if ID != "USLegis" else tuple(sorted((u, x)))
        E[key] = w
    snaps.append(E)
    rows = [[len({a for e in S for a in e}), len(S), float(sum(S.values()))] for S in snaps]
    np.save(out, np.array(rows, float))


def sgnn_worker(job: str, out: str) -> None:
    """DGL environment: train s-GNN-I with one configuration on the train part; Z on the validation and test parts."""
    import scipy.sparse as sp
    sys.path.insert(0, str(REPO))
    from methods.sgnn import SGNN
    j = np.load(job, allow_pickle=False)
    n, ip, ix, off = int(j["n"]), j["indptr"], j["indices"], j["offsets"]
    mats = [sp.csr_matrix((np.ones(off[t + 1] - off[t]), ix[off[t]:off[t + 1]], ip[t]), shape=(n, n))
            for t in range(len(ip))]
    cpf = np.zeros(len(mats), int)
    cpf[j["cps"]] = 1
    tr, va, te = j["train"], j["val"], j["test"]
    cfg = json.loads(str(j["cfg"]))
    t0 = time.time()
    m = SGNN(seed=int(j["rep"]), workdir=out + ".work", **cfg).fit([mats[i] for i in tr], cpf[tr],
                                                                 [mats[i] for i in va], cpf[va])
    Zv, _ = m.similarity([mats[i] for i in va])
    Zt, it = m.similarity([mats[i] for i in te])
    res = json.loads(next(Path(out + ".work").glob("models/*/results.json")).read_text())
    np.savez(out, Zv=np.asarray(Zv, float), Zt=np.asarray(Zt, float), it=np.asarray(it, int),
             val_pair_f1=float(res["f1"][1]), seconds=time.time() - t0)
    import shutil
    shutil.rmtree(out + ".work", ignore_errors=True)


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] in ("_own", "_sgnn"):
    (own_counts_worker if sys.argv[1] == "_own" else sgnn_worker)(sys.argv[2], sys.argv[3])
    sys.exit(0)

sys.path.insert(0, str(HERE))
import evaluate as ev
import plants as P
from common import (
    DATA,
    MODES,
    RESULTS,
    WR,
    Timer,
    W,
    load,
    mad_scale,
    sha,
    src_sha,
    write_json,
)
from rows import BENCH, PRIMARY, REPO_VIEW, B, baseline, matrices, meta, settings, tire_params

from methods import scpd
from methods.counts import scores as window_scores
from utils.param_grids import get_grid

OUT = RESULTS / "settings.json"
SEED = 20261024
ROWS = PRIMARY
DETS = ["counts", "cusum", "bocpd", "mmd", "lad", "scpd", "tire"]
ROW_OF = {"counts": "counts", "cusum": "cusum", "bocpd": "bocpd", "mmd": "mmd", "lad": "lad",
          "scpd": "scpd", "tire": "tire"}
SOURCE9 = ["UCI", "USLegis", "canVote", "MAG", "FL", "SCC", "PM", "MIT", "ENRON"]
SHORT = ["USLegis", "canVote", "FL"]
HITS_N = {"UCI": 10, "USLegis": 2, "canVote": 2, "FL": 2}
LAD_PAPER = {"USLegis": (1, 2), "UCI": (7, 14), "canVote": (2, 4)}      # authors' Anomaly_Detection.py
TIRE_TRAIN = [c for c in itertools.product([3, 5, 7], [1, 2], [1, 2, 3], [1, 2], [0.1, 1.0, 10.0], ["td", "both"],
                                           [100, 200]) if c[3] <= c[2]]  # the training grid of step 9
DEFAULT = {"counts": (2, 10, "cen"),
           "cusum": {"threshold": 5.0, "kappa": 0.5, "startup_period": 10, "cooldown": 1},
           "bocpd": {"hazard_rate": 1 / 250}, "mmd": None, "lad": (5, 15), "scpd": (5, 10),
           "tire": (20, 2, 1, 1, 1.0, "both", 200)}
DEFAULT_NOTE = {"counts": "no published default: windows (2, 10) of the identity-aware rows, centroid L1",
                "cusum": "textbook k = 0.5, h = 5 (Montgomery, Introduction to SQC, Sec. 9.1); startup 10, cooldown 1 "
                         "(our choice)",
                "bocpd": "hazard 1/250 (Adams & MacKay 2007, well-log example); the NIG prior of methods/bocpd.py",
                "mmd": "no published window: not reported",
                "lad": "LAD's code: windows 1/2 Senate, 7/14 UCI, 2/4 bills (its paper); 5/15 elsewhere "
                       "(change_detection_two_windows default); 5 eigenvalues stored (the paper uses 6)",
                "scpd": "SCPD's paper: windows 2/4 on the flights, 5/10 elsewhere",
                "tire": "TIRE's code defaults (De Ryck et al.): N = 20, K = 2, latent 1, shared 1, lambda 1, TD + FD, "
                        "nfft 30, 200 epochs"}


# ------------------------------------------------------------------------------------------------ inputs
def newest(pattern: str) -> Path:
    """The latest step-11 cache file of one dataset (`feats_<ID>_<12 hex>.npz`; IDs like ENRON_PC do not match)."""
    import re
    rx = re.compile(pattern.replace("*", "[0-9a-f]{12}").replace(".", r"\."))
    return max((p for p in (BENCH / "cache").iterdir() if rx.fullmatch(p.name)), key=os.path.getmtime)


@functools.cache
def inputs(ID: str) -> dict:
    """Per-snapshot inputs of every row: raw counts, eight raw features, LAD eigenvalues, SCPD DOS."""
    mt = meta(ID)
    F = dict(np.load(newest(f"feats_{ID}_*.npz")))
    out = {"seg": mt["seg"], "events": mt["events"], "tol": mt["tol"], "T": mt["T"], "sec": mt["sec"],
           "dos": F["dos"]}
    if ID not in REPO_VIEW:
        out.update(C=np.c_[F["nplus"], F["E"]].astype(float), X8=F["raw8"], lad=F["lad5"],
                   n=int(F["n"][0]))
        return out
    d = load(REPO_VIEW[ID])
    p = CACHE / f"repo_{ID}_{sha(*[(DATA / f'{s}_{REPO_VIEW[ID]}.npz').read_bytes() for s in ('graphs', 'moments')])}.npz"
    if not p.exists():
        from data.registry import DataRegistry
        b = DataRegistry().load(REPO_VIEW[ID])            # the Table 1 bundle (the Enron pickle needs ~5 GB)
        np.savez(p, X8=np.asarray(b.features_raw), lad=np.asarray(b.lad_eigenvalues), n=b.max_nodes)
    z = np.load(p)
    out.update(C=np.c_[d["f"]["active_noloop"], d["f"]["edges_noloop"]].astype(float),
               X8=z["X8"], lad=z["lad"], n=int(z["n"]))
    return out


@functools.cache
def table2(ID: str) -> dict:
    """The Table 2 score series (step 11) and the Table 2 common mask (step 12's rule)."""
    z = np.load(BENCH / f"scores_{ID}.npz")
    S = {k[2:]: z[k] for k in z.files if k.startswith("s_")}
    st = settings()
    mask = ev.common_mask(S, st["burn"], ROWS, inputs(ID)["seg"])
    return {"S": S, "mask": mask}


# ------------------------------------------------------------------------------------------------ scores
def vscores(X, seg, w: int, wr: int, mode: str, start: int | None = None, wts=None) -> np.ndarray:
    """methods.counts.scores per segment, vectorised; gam weights wts (default 1/j^2) on the squared differences."""
    X = np.asarray(X, float)
    X = X[:, None] if X.ndim == 1 else X
    d = X.shape[1]
    wv = 1.0 / np.arange(1, d + 1) ** 2 if wts is None else np.asarray(wts, float)
    start = w + wr if start is None else start
    D = np.full(len(seg), np.nan)
    for s0, s1 in ev.seg_ranges(seg):
        Y = X[s0:s1]
        T = len(Y)
        if T <= w + wr:
            continue
        t = np.arange(w + wr, T)
        if mode == "pw":
            pair = [np.abs(Y[t - w + a] - Y[t - w - wr + b]).sum(1) for a in range(w) for b in range(wr)]
            v = np.ascontiguousarray(np.stack(pair, 1)).mean(1)        # the summation order of methods.counts
        else:
            sw = lambda k, Y=Y: np.lib.stride_tricks.sliding_window_view(Y, k, axis=0).mean(-1)   # sw(k)[i] = Y[i:i+k] mean
            diff = sw(w)[t - w] - sw(wr)[t - w - wr]
            v = np.abs(diff).sum(1) if mode == "cen" else np.sqrt((diff ** 2 * wv).sum(1))
        out = np.full(T, np.nan)
        out[t] = v
        out[:start] = np.nan
        D[s0:s1] = out
    return D


def expanding_scores(C, seg, w: int, wr: int, wts) -> np.ndarray:
    """The gam count score with an expanding past scale: at step t each window difference is divided by the median
    |first difference| of its feature over the snapshots s0 .. t-1 of the segment (the snapshots the score sees)."""
    import bisect
    C = np.asarray(C, float)
    D = np.zeros(len(seg))
    for j, wj in enumerate(wts):
        diff = vscores(C[:, j], seg, w, wr, "cen")    # |mean of the test window - mean of the reference window|
        sc = np.full(len(seg), np.nan)
        for s0, s1 in ev.seg_ranges(seg):
            seen = []
            for t in range(s0 + 2, s1):               # differences of snapshots s0 .. t-1
                bisect.insort(seen, abs(C[t - 1, j] - C[t - 2, j]))
                k = len(seen)
                sc[t] = 0.5 * (seen[(k - 1) // 2] + seen[k // 2])
        D += wj * (diff / np.maximum(sc, 1e-12)) ** 2
    return np.sqrt(D)


def counts_X(I: dict, scale: str = "mad", cols=(0, 1)) -> np.ndarray:
    C = I["C"][:, list(cols)]
    return mad_scale(C, I["seg"]) if scale == "mad" else ev.online_scale(C, I["seg"])[0]


def lad_params(s) -> dict:
    return {"threshold": 0.0, "num_eigen": 5, "short_window": int(s[0]), "long_window": int(s[1]), "cooldown": 1}


def compute(det: str, s, ID: str) -> np.ndarray:
    """Score series of detector det with score-shaping setting s on dataset ID."""
    I = inputs(ID)
    seg = I["seg"]
    if det == "counts":
        w, wr, md = s[:3]
        return vscores(counts_X(I, s[3] if len(s) > 3 else "mad"), seg, w, wr, md)
    if det == "scpd":
        D = np.full(len(seg), np.nan)
        for s0, s1 in ev.seg_ranges(seg):
            if s1 - s0 > max(s) + 1:
                D[s0:s1] = scpd.score(I["dos"][s0:s1], tuple(s), 0.05)[0]
        return D
    st = {"lad": lambda: lad_params(s), "cusum": lambda: s,
          "mmd": lambda: {"threshold": 0.0, "window_size": s, "cooldown": 1},
          "bocpd": lambda: {"threshold": 0.5, "hazard_rate": s, "cooldown": 1}, "tire": lambda: [list(s), 0.0, 1]}
    return baseline(det, ID, I["X8"], I["lad"], seg, I["n"], {det: st[det]()})


def method_src(det: str) -> str:
    f = det
    return sha((REPO / "methods" / f"{f}.py").read_bytes(), src_sha(compute, vscores, counts_X, baseline, lad_params,
                                                                     tire_params))


@functools.cache
def input_key(ID: str) -> str:
    """The cached features of step 11 (their file name carries the hash of inputs and code) and, for the Table 1
    view, the moment and graph files of steps 1-2."""
    parts = [newest(f"feats_{ID}_*.npz").name]
    if ID in REPO_VIEW:
        parts += [(DATA / f"{s}_{REPO_VIEW[ID]}.npz").read_bytes() for s in ("graphs", "moments")]
    return sha(*parts)


def series(det: str, s, ID: str) -> np.ndarray:
    key = sha(det, json.dumps(s), ID, input_key(ID), method_src(det))
    p = CACHE / f"s_{det}_{ID}_{key}.npy"
    if not p.exists():
        x = compute(det, s, ID)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, x)
        os.replace(tmp, p)
    return np.load(p)


def _job(a):
    det, s, ID = a
    series(det, s, ID)
    return a


# ------------------------------------------------------------------------------------------------ grids
def grids() -> dict:
    """Score-shaping settings per detector (seeded samples for CUSUM and TIRE); the fixed and default ones first."""
    st = settings()
    rng = np.random.default_rng(SEED)
    cus = get_grid("cusum")
    G = {"counts": [(w, wr, md) for w, wr, md in itertools.product(W, WR, MODES)],
         "cusum": [cus[i] for i in sorted(rng.choice(len(cus), 200, replace=False))],
         "bocpd": sorted({p["hazard_rate"] for p in get_grid("bocpd")}),
         "mmd": sorted({p["window_size"] for p in get_grid("mmd")}),
         "lad": [(a, b) for a in (3, 5, 7) for b in (10, 15, 20, 25)],
         "scpd": [(a, b) for a in (3, 5, 7) for b in (10, 15, 20, 25)] + [(2, 4)],
         "tire": [TIRE_TRAIN[i] for i in sorted(rng.choice(len(TIRE_TRAIN), 40, replace=False)[:16])]}
    fixed = {"counts": tuple(st["counts"][:3]),
             "cusum": st["cusum"], "bocpd": st["bocpd"]["hazard_rate"], "mmd": st["mmd"]["window_size"],
             "lad": (st["lad"]["short_window"], st["lad"]["long_window"]), "scpd": (5, 10),
             "tire": tuple(st["tire"][0])}
    extra = {"cusum": [DEFAULT["cusum"]], "bocpd": [DEFAULT["bocpd"]["hazard_rate"]],
             "lad": [(5, 15)] + list(LAD_PAPER.values()), "tire": [DEFAULT["tire"]]}
    out = {}
    for d in DETS:
        lst = [fixed[d]] + extra.get(d, []) + G[d]
        seen, uniq = set(), []
        for s in lst:
            k = json.dumps(s, sort_keys=True)
            if k not in seen:
                seen.add(k)
                uniq.append(s)
        out[d] = {"fixed": fixed[d], "settings": uniq, "grid": G[d]}
    return out


def default_of(det: str, ID: str):
    if det == "lad":
        return LAD_PAPER.get(ID, DEFAULT["lad"])
    if det == "scpd":
        return (2, 4) if ID == "FL" else DEFAULT["scpd"]
    if det == "bocpd":
        return DEFAULT["bocpd"]["hazard_rate"]
    return DEFAULT[det]


# ------------------------------------------------------------------------------------------------ EP
def ep_local(x, mask, seg, events, tol: int, guard: int = 0, all_events=None) -> tuple[dict, int]:
    """evaluate.ep with null windows also kept out of [e, e + tol + guard] after every labelled event."""
    L = tol + 1
    evs = events if all_events is None else all_events
    ev_ok, null = ev.windows(mask, seg, events, evs, L)
    if guard:
        bad = np.zeros(len(mask), bool)
        for e in evs:
            bad[e:min(len(mask), e + L + guard)] = True
        null = null[~bad[null]]
    if not len(null) or not ev_ok:
        return {}, len(null)
    nm = np.sort(ev._wmax(x, null, L))
    s = ev._wmax(x, ev_ok, L)
    lo, hi = np.searchsorted(nm, s, "left"), np.searchsorted(nm, s, "right")
    return dict(zip(ev_ok, ((lo + 0.5 * (hi - lo)) / len(nm)).tolist())), len(null)


def fill(x, mask):
    x = np.asarray(x, float)
    ok = np.isfinite(x) & mask
    lo = np.min(x[ok]) if ok.any() else 0.0
    return np.where(np.isfinite(x), x, lo)


def ds_ep(x, ID, mask=None, tol=None, guard=0, events=None) -> np.ndarray:
    I = inputs(ID)
    mask = table2(ID)["mask"] if mask is None else mask
    evs = I["events"] if events is None else events
    v, _ = ep_local(fill(x, mask), mask, I["seg"], evs, I["tol"] if tol is None else tol, guard,
                    all_events=I["events"] if events is None else events)
    return np.array([v[e] for e in sorted(v)])


def pooled(E: dict, dss) -> float | None:
    vals = [E[d].mean() for d in dss if d in E and len(E[d])]
    return float(np.mean(vals)) if len(vals) == len(dss) else None


def ci(E: dict, dss) -> dict:
    b = ev.boot_pooled({d: E[d] - 0.5 for d in dss})
    return {"mean": b["mean"] + 0.5, "lo": b["lo"] + 0.5, "hi": b["hi"] + 0.5}


def pools() -> tuple[list, list]:
    R = json.loads((RESULTS / "bench.json").read_text())
    return R["pooled_datasets"], R["pooled_datasets_10plus"]


# ------------------------------------------------------------------------------------------------ checks
def checks() -> dict:
    """C1 recomputed rows = step 11; C2 local EP = step 12; C3 vectorised = methods.counts.scores."""
    st = settings()
    out = {"C1": {}, "C2": {}, "C3": {}}
    bench = json.loads((RESULTS / "bench.json").read_text())["datasets"]
    g = grids()
    for ID in B:
        S, I = table2(ID)["S"], inputs(ID)
        rec = {"counts": vscores(counts_X(I), I["seg"], st["counts"][0], st["counts"][1], st["counts"][2])}
        rec.update({d: series(d, g[d]["fixed"], ID) for d in ("cusum", "bocpd", "mmd", "lad", "scpd", "tire")})
        diffs = {}
        for r, x in rec.items():
            a, b = S[ROW_OF.get(r, r)], x
            both = np.isfinite(a) & np.isfinite(b)
            diffs[r] = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
            diffs[r + "_nan_mismatch_on_mask"] = int((np.isfinite(a) != np.isfinite(b))[table2(ID)["mask"]].sum())
        out["C1"][ID] = diffs
        m2 = bench[ID]["m2"]
        if "ep_per_event" in m2:
            out["C2"][ID] = max(float(np.max(np.abs(ds_ep(S[r], ID) - np.array(m2["ep_per_event"][r]))))
                                for r in m2["ep_per_event"] if r in S)
    for ID in ("UCI", "HT09"):
        I = inputs(ID)
        worst = 0.0
        for md in MODES:
            for s0, s1 in ev.seg_ranges(I["seg"]):
                a = window_scores(counts_X(I)[s0:s1], 3, 7, md)
                b = vscores(counts_X(I)[s0:s1], np.zeros(s1 - s0, int), 3, 7, md)
                worst = max(worst, float(np.nanmax(np.abs(a - b))))
        out["C3"][ID] = worst
    tol = {"C1": 1e-6, "C2": 5e-5, "C3": 0.0}           # bench.json keeps 4 decimals
    out["pass"] = {"C1": all(v <= tol["C1"] for d in out["C1"].values() for k, v in d.items() if "nan" not in k)
                   and all(v == 0 for d in out["C1"].values() for k, v in d.items() if "nan" in k),
                   "C2": all(v <= tol["C2"] for v in out["C2"].values()),
                   "C3": all(v <= tol["C3"] for v in out["C3"].values())}
    return out


# ------------------------------------------------------------------------------------------------ A: settings
def section_a(g, dss, big) -> dict:
    out = {"defaults_note": DEFAULT_NOTE, "grid_sizes": {d: len(g[d]["grid"]) for d in DETS}}
    for d in DETS:
        E = {}                                            # setting index -> {ds: per-event EP}
        for i, s in enumerate(g[d]["settings"]):
            E[i] = {ID: ds_ep(series(d, s, ID), ID) for ID in dss}
        P_all = {i: pooled(E[i], dss) for i in E}
        P_big = {i: pooled(E[i], big) for i in E}
        gi = [g[d]["settings"].index(s) for s in g[d]["grid"]]
        pg = np.array([P_all[i] for i in gi])
        lodo, sel = {}, {}
        for ID in dss:
            rest = [x for x in dss if x != ID]
            j = max(gi, key=lambda i: (pooled(E[i], rest), -i))
            sel[ID] = g[d]["settings"][j]
            lodo[ID] = E[j][ID]
        res = {"fixed": {"setting": g[d]["fixed"], **ci(E[0], dss), "pooled_10plus": P_big[0],
                         "per_ds": {ID: float(E[0][ID].mean()) for ID in dss}},
               "grid": {"n": len(gi), "median": float(np.median(pg)), "q25": float(np.quantile(pg, 0.25)),
                        "q75": float(np.quantile(pg, 0.75)), "min": float(pg.min()), "max": float(pg.max()),
                        "median_10plus": float(np.median([P_big[i] for i in gi]))},
               "oracle": {"setting": g[d]["settings"][max(gi, key=lambda i: P_all[i])], "pooled": float(pg.max()),
                          "pooled_10plus": float(max(P_big[i] for i in gi)),
                          "per_ds_oracle_mean": float(np.mean([max(E[i][ID].mean() for i in gi) for ID in dss]))},
               "lodo": {**ci(lodo, dss), "selected": {ID: sel[ID] for ID in dss},
                        "pooled_10plus": pooled(lodo, big), "per_ds": {ID: float(lodo[ID].mean()) for ID in dss}}}
        if DEFAULT[d] is not None:
            Ed = {ID: ds_ep(series(d, default_of(d, ID), ID), ID) for ID in dss}
            res["default"] = {**ci(Ed, dss), "pooled_10plus": pooled(Ed, big),
                              "per_ds": {ID: float(Ed[ID].mean()) for ID in dss}}
        out[d] = res
    rules = {"fixed": lambda r: r["fixed"]["mean"], "default": lambda r: r.get("default", {}).get("mean"),
             "grid_median": lambda r: r["grid"]["median"], "oracle": lambda r: r["oracle"]["pooled"],
             "lodo": lambda r: r["lodo"]["mean"]}
    out["ranking"] = {k: sorted(((d, f(out[d])) for d in DETS if f(out[d]) is not None), key=lambda x: -x[1])
                      for k, f in rules.items()}
    oth = [d for d in DETS if d != "counts"]
    out["counts_lead"] = {
        k: {"counts": f(out["counts"]), "best_other": max(((d, f(out[d])) for d in oth if f(out[d]) is not None),
                                                          key=lambda x: x[1])} for k, f in rules.items()}
    out["counts_vs_other_oracles"] = {"counts_fixed": out["counts"]["fixed"]["mean"],
                                      "counts_lodo": out["counts"]["lodo"]["mean"],
                                      "others_oracle": {d: out[d]["oracle"]["pooled"] for d in oth}}
    return out


# ------------------------------------------------------------------------------------------------ B: burn-in
BURN_MIN = {"counts": lambda s: s[0] + s[1], "cusum": lambda s: s["startup_period"],
            "bocpd": lambda s: 2, "mmd": lambda s: 2 * s, "lad": lambda s: s[1], "scpd": lambda s: max(s) + 1,
            "tire": lambda s: 2 * s[0]}


def short_setting(det: str, ID: str, g):
    fx = g[det]["fixed"]
    return {"counts": lambda: (2, 2, fx[2]),
            "cusum": lambda: {**fx, "startup_period": 5}, "bocpd": lambda: fx, "mmd": lambda: 3,
            "lad": lambda: LAD_PAPER.get(ID, (3, 10)), "scpd": lambda: (2, 4),
            "tire": lambda: (3,) + tuple(fx[1:])}[det]()


def hits(x, ok, events, n, tol=0) -> float | None:
    idx = np.flatnonzero(ok & np.isfinite(x))
    if not idx.size:
        return None
    top = idx[np.argsort(-x[idx], kind="stable")][:n]
    return sum(int(any(e <= t <= e + tol for t in top)) for e in events) / len(events)


def score_own(x, ID, burn, events=None) -> dict:
    I = inputs(ID)
    ok = ev.scored_mask(I["seg"], x, burn)
    evs = I["events"] if events is None else events
    v, nn = ep_local(np.where(np.isfinite(x), x, -np.inf), ok, I["seg"], evs, I["tol"])
    r = {"burn": int(burn), "n_scored": int(ok.sum()), "n_events_scored": len(v), "n_null": nn,
         "ep": float(np.mean(list(v.values()))) if v else None, "ep_per_event": {int(k): v[k] for k in v}}
    if ID in HITS_N:
        r[f"hits@{HITS_N[ID]}"] = hits(x, ok, evs, HITS_N[ID], I["tol"] if ID == "FL" else 0)
        if ID == "canVote" and events is None:
            r["hits@2_secondary"] = hits(x, ok, I["sec"]["events_secondary"], 2)
    return r


def section_b(g, dss) -> dict:
    st = settings()
    burn_t2 = {d: st["burn"][ROW_OF[d]] for d in DETS}
    out = {"B1": {}, "B2": {}, "B1_lad_paper": {}, "rule": "B1: Table 2 setting, own burn-in (rows.burn_in); B2: on the "
           "three short series the shortest windows (its paper's where stated) and the minimal burn-in"}
    for ID in B:
        out["B1"][ID] = {d: score_own(series(d, g[d]["fixed"], ID), ID, burn_t2[d]) for d in DETS}
        if ID in LAD_PAPER:
            s = LAD_PAPER[ID]
            out["B1_lad_paper"][ID] = score_own(series("lad", s, ID), ID, BURN_MIN["lad"](s))
    for ID in SHORT:
        out["B2"][ID] = {}
        for d in DETS:
            s = short_setting(d, ID, g)
            out["B2"][ID][d] = {"setting": s, **score_own(series(d, s, ID), ID, BURN_MIN[d](s))}
    scored = {}
    for rule, blk in (("common", None), ("B1", out["B1"]), ("B1+B2", None)):
        scored[rule] = {}
        for d in DETS:
            if rule == "common":
                n = sum(ID in dss for ID in SOURCE9)
            elif rule == "B1":
                n = sum(blk[ID][d]["n_events_scored"] > 0 for ID in SOURCE9)
            else:
                n = sum((out["B2"][ID][d] if ID in SHORT else out["B1"][ID][d])["n_events_scored"] > 0
                        for ID in SOURCE9)
            scored[rule][d] = n
    out["source_datasets_scored"] = scored
    full = {}
    for d in DETS:
        E = {ID: np.array(list((out["B2"][ID][d] if ID in SHORT else out["B1"][ID][d])["ep_per_event"].values()))
             for ID in B}
        ok = [ID for ID in B if len(E[ID])]
        full[d] = {"datasets": ok, "pooled": float(np.mean([E[ID].mean() for ID in ok])),
                   **({"ci": ci(E, ok)} if ok else {})}
    out["pooled_all_scored"] = full
    out["paired_all_scored"] = paired_all(out)
    E10 = {d: {ID: np.array(list(out["B1"][ID][d]["ep_per_event"].values())) for ID in dss} for d in DETS}
    out["pooled_B1_ten"] = {d: ci(E10[d], dss) for d in DETS if all(len(E10[d][ID]) for ID in dss)}
    return out


def paired_all(out) -> dict:
    """TIRE and the counts minus every detector over the 13 datasets, each with its own burn-in: per dataset the events
    both rows score; equal weight; event-level and two-stage (datasets, then events) paired intervals; rank of each
    detector in each two-stage draw (on the events every detector scores)."""
    import inference as inf
    ep = {d: {ID: (out["B2"][ID][d] if ID in SHORT else out["B1"][ID][d])["ep_per_event"] for ID in B} for d in DETS}
    res = {}
    for ref in ("counts", "tire"):
        res[ref] = {}
        for d in DETS:
            if d == ref:
                continue
            X = {}
            for ID in B:
                evs = sorted(set(ep[ref][ID]) & set(ep[d][ID]))
                if evs:
                    X[ID] = np.array([[ep[ref][ID][e] - ep[d][ID][e]] for e in evs])
            pt = float(inf.pooled_point(X)[0])
            q = {lv: inf.pooled_draws(X, lv)[:, 0] for lv in ("event", "two_stage")}
            res[ref][d] = {"diff": pt, "n_datasets": len(X), **{f"ci95_{lv}": [float(np.quantile(v, 0.025)),
                                                                                 float(np.quantile(v, 0.975))]
                                                                 for lv, v in q.items()},
                           "p_two_stage": inf.pval(q["two_stage"])}
    X = {}
    for ID in B:
        evs = sorted(set.intersection(*[set(ep[d][ID]) for d in DETS]))
        if evs:
            X[ID] = np.array([[ep[d][ID][e] for d in DETS] for e in evs])
    pt, dr = inf.pooled_point(X), inf.pooled_draws(X, "two_stage")
    rk = inf.ranks(dr)
    res["common_events"] = {"datasets": sorted(X), "ep": dict(zip(DETS, pt.tolist())),
                            "p_rank1": {d: float(np.mean(rk[:, j] == 1)) for j, d in enumerate(DETS)}}
    return res


# ------------------------------------------------------------------------------------------------ C: tolerance
SPAN = {"counts": 7, "mmd": 6, "lad": 10, "scpd": 10, "tire": 10,
        "cusum": None, "bocpd": None, "constant": None}


def section_c(dss, big) -> dict:
    rows = [r for r in ROWS if r != "constant"]
    out = {"tau": {}, "tau_own": {}, "guard17": {}, "guard_own": {}, "span": SPAN,
           "tau_per_dataset": {ID: inputs(ID)["tol"] for ID in B}}
    for tau in range(11):
        blk = {}
        for r in rows:
            E = {ID: ds_ep(table2(ID)["S"][r], ID, tol=tau) for ID in dss}
            ok = [ID for ID in dss if len(E[ID])]
            blk[r] = {"pooled": float(np.mean([E[ID].mean() for ID in ok])), "n_ds": len(ok),
                      "pooled_10plus": pooled(E, big)}
        out["tau"][tau] = blk
    for r in rows:
        tau = SPAN[r]
        E = {ID: ds_ep(table2(ID)["S"][r], ID, tol=tau) for ID in dss} if tau is not None else \
            {ID: ds_ep(table2(ID)["S"][r], ID) for ID in dss}
        ok = [ID for ID in dss if len(E[ID])]
        out["tau_own"][r] = {"tau": tau, "pooled": float(np.mean([E[ID].mean() for ID in ok])), "n_ds": len(ok)}
        for key, gd in (("guard17", 17), ("guard_own", SPAN[r] or 0)):
            E = {ID: ds_ep(table2(ID)["S"][r], ID, guard=gd) for ID in dss}
            ok = [ID for ID in dss if len(E[ID])]
            out[key][r] = {"guard": gd, "pooled": float(np.mean([E[ID].mean() for ID in ok])), "n_ds": len(ok),
                           "pooled_10plus": pooled(E, big), **ci(E, ok)}
    lead = {}
    for tau, blk in out["tau"].items():
        best = max((r for r in blk if r != "counts"), key=lambda r: blk[r]["pooled"])
        lead[tau] = {"counts": blk["counts"]["pooled"], "best_other": [best, blk[best]["pooled"]]}
    out["counts_first"] = lead
    return out


# ------------------------------------------------------------------------------------------------ D: alignment
PAST_ONLY = ("counts", "mmd")


def shift1(x, seg):
    y = np.full(len(x), np.nan)
    for s0, s1 in ev.seg_ranges(seg):
        y[s0:s1 - 1] = x[s0 + 1:s1]
    return y


def section_d(dss, g) -> dict:
    rows = [r for r in ROWS if r != "constant"]
    out = {"note": "score at t sees G_{t-1} and earlier for " + ", ".join(PAST_ONLY) + "; G_t and earlier for CUSUM, "
                   "BOCPD, LAD, SCPD; TIRE (offline) sees both sides", "per_ds": {}, "pooled": {}}
    for r in rows:
        E0, E1, E2 = {}, {}, {}
        for ID in dss:
            I, S = inputs(ID), table2(ID)["S"]
            T = I["T"]
            E0[ID] = ds_ep(S[r], ID)
            E1[ID] = ds_ep(S[r], ID, events=[e + 1 for e in I["events"] if e + 1 < T])
            E2[ID] = ds_ep(shift1(S[r], I["seg"]) if r in PAST_ONLY else S[r], ID)
        out["pooled"][r] = {"as_is": pooled(E0, dss), "events_e+1": pooled(E1, dss), "include_G_t": pooled(E2, dss)}
        out["per_ds"][r] = {"UCI": {"as_is": float(E0["UCI"].mean()), "events_e+1": float(E1["UCI"].mean()),
                                    "include_G_t": float(E2["UCI"].mean())}}
    short = {}
    for ID in SHORT + ["UCI"]:
        short[ID] = {}
        for d in DETS:
            s = short_setting(d, ID, g) if ID in SHORT else g[d]["fixed"]
            x = series(d, s, ID)
            row = ROW_OF[d]
            burn = BURN_MIN[d](s) if ID in SHORT else settings()["burn"][row]
            xs = shift1(x, inputs(ID)["seg"]) if row in PAST_ONLY else x
            short[ID][d] = {"as_is": score_own(x, ID, burn), "include_G_t": score_own(xs, ID, burn)}
            for k in short[ID][d]:
                short[ID][d][k].pop("ep_per_event")
    out["lad_sets"] = short
    return out


# ------------------------------------------------------------------------------------------------ E: counts
def section_e(dss, big) -> dict:
    st = settings()
    w, wr, md = st["counts"][:3]
    out = {"setting": [w, wr, md], "variants": {}}
    var = {"mad_w1/4 (Table 2)": ("mad", (0, 1), (1, .25)), "online_w1/4": ("online", (0, 1), (1, .25))}
    for sc in ("mad", "online"):
        for c in (1, .5, .25, 0):
            var[f"{sc}_w{c:g}"] = (sc, (0, 1), (1, c))
        var[f"{sc}_edges_only"] = (sc, (1,), (1,))
        var[f"{sc}_vertices_only"] = (sc, (0,), (1,))
    for ID in B:                                      # the causal count row and the count reference rows of step 23
        I = inputs(ID)
        np.savez(CACHE / f"scores_{ID}.npz",
                 s_counts_causal=vscores(counts_X(I, "online"), I["seg"], w, wr, md, wts=(1, .25)),
                 s_counts_w2_10=vscores(counts_X(I), I["seg"], 2, 10, md, wts=(1, .25)),
                 s_counts_causal_edges=vscores(counts_X(I, "online", (1,)), I["seg"], w, wr, md, wts=(1,)),
                 s_counts_causal_expanding=expanding_scores(I["C"], I["seg"], w, wr, (1, .25)),
                 s_counts_edges=vscores(counts_X(I, "mad", (1,)), I["seg"], w, wr, md, wts=(1,)))
    for k, (sc, cols, wts) in var.items():
        E = {ID: ds_ep(vscores(counts_X(inputs(ID), sc, cols), inputs(ID)["seg"], w, wr, md, wts=wts), ID)
             for ID in dss}
        out["variants"][k] = {**ci(E, dss), "pooled_10plus": pooled(E, big),
                              "per_ds": {ID: float(E[ID].mean()) for ID in dss}}
    grid = [(a, b, m, "online") for a, b, m in itertools.product(W, WR, MODES)]
    P_ = {}
    for s in grid:
        P_[s] = {ID: ds_ep(series("counts", s, ID), ID) for ID in dss}
    pg = np.array([pooled(P_[s], dss) for s in grid])
    lodo = {}
    for ID in dss:
        rest = [x for x in dss if x != ID]
        s = max(grid, key=lambda s: pooled(P_[s], rest))
        lodo[ID] = P_[s][ID]
    out["online_grid"] = {"n": len(grid), "median": float(np.median(pg)), "q25": float(np.quantile(pg, .25)),
                          "q75": float(np.quantile(pg, .75)), "oracle": float(pg.max()),
                          "oracle_setting": grid[int(np.argmax(pg))], "lodo": ci(lodo, dss)}
    return out


# ------------------------------------------------------------------------------------------------ F: own constructions
def own_counts(ID: str) -> np.ndarray:
    """(T, 3): active vertices, edges, total weight of the source paper's own construction."""
    if ID in LAD_FILES:
        f = REPO / "data" / "raw" / "lad" / "datasets" / LAD_FILES[ID]
        p = CACHE / f"own_{ID}_{sha(f, src_sha(own_counts_worker))}.npy"
        if not p.exists():
            subprocess.run([sys.executable, "-I", str(Path(__file__)), "_own", ID, str(p)], check=True)
        return np.load(p)
    z = np.load(BENCH / f"{ID}.npz", allow_pickle=False)
    off, w = z["offsets"], z["weights"]
    I = inputs(ID)
    tw = np.array([w[off[t]:off[t + 1]].astype(float).sum() / 2 for t in range(len(off) - 1)])
    return np.c_[I["C"], tw]


def section_f(dss) -> dict:
    st = settings()
    w, wr, md = st["counts"][:3]
    g0 = np.load(BENCH / "g0_rows.npz")
    own_burn = {"UCI": 14, "USLegis": 2, "canVote": 4, "MAG": 11, "FL": 5, "SCC": 11}
    out = {}
    for ID, row in (("UCI", "lad"), ("USLegis", "lad"), ("canVote", "lad"), ("MAG", "scpd"), ("FL", "scpd"),
                    ("SCC", "scpd")):
        I = inputs(ID)
        own = g0[f"{row}_own_{ID}"]
        C = own_counts(ID)
        if len(C) != I["T"]:
            out[ID] = {"error": f"own construction has {len(C)} snapshots, stored {I['T']}"}
            continue
        burn = max(own_burn[ID], w + wr)
        X = {f"{row}_own": own, f"{row}_shared": table2(ID)["S"][row],
             "counts_own": vscores(mad_scale(C[:, :2], I["seg"]), I["seg"], w, wr, md),
             "counts_own_weighted": vscores(mad_scale(C[:, [0, 2]], I["seg"]), I["seg"], w, wr, md),
             "counts_shared": table2(ID)["S"]["counts"]}
        r = {"burn": burn, "own_counts_first5": C[:5].tolist()}
        for k, x in X.items():
            o = score_own(x, ID, burn)
            o.pop("ep_per_event")
            if ID in dss:
                o["ep_common_mask"] = float(ds_ep(x, ID).mean())
            r[k] = o
        r["same_events"] = {"rows": "a = the source detector, b = the counts",
                            "own": score_pair(X[f"{row}_own"], X["counts_own"], ID, burn, burn),
                            "own_weighted": score_pair(X[f"{row}_own"], X["counts_own_weighted"], ID, burn, burn),
                            "shared": score_pair(X[f"{row}_shared"], X["counts_shared"], ID, burn, burn)}
        if ID in SHORT or ID == "UCI":                 # each row after its own burn-in (the counts' shortest windows)
            cs = (2, 2, md) if ID in SHORT else (w, wr, md)
            own_rows = {f"{row}_own": (own, own_burn[ID]),
                        "counts_own": (vscores(mad_scale(C[:, :2], I["seg"]), I["seg"], *cs), cs[0] + cs[1]),
                        "counts_own_weighted": (vscores(mad_scale(C[:, [0, 2]], I["seg"]), I["seg"], *cs),
                                                cs[0] + cs[1])}
            r["own_burn"] = {"counts_windows": list(cs)}
            for k, (x, bn) in own_rows.items():
                o = score_own(x, ID, bn)
                o.pop("ep_per_event")
                r["own_burn"][k] = o
            (xo, bo), (xc, bc) = own_rows[f"{row}_own"], own_rows["counts_own"]
            r["same_events"]["own_burn"] = score_pair(xo, xc, ID, bo, bc)
        out[ID] = r
    return out


def score_pair(xa, xb, ID, burn_a, burn_b) -> dict:
    """Two rows on the SAME events and null windows: the steps both score after their own burn-in (6e)."""
    I = inputs(ID)
    ok = ev.scored_mask(I["seg"], xa, burn_a) & ev.scored_mask(I["seg"], xb, burn_b)
    out = {"n_scored": int(ok.sum())}
    for k, x in (("a", xa), ("b", xb)):
        v, nn = ep_local(np.where(np.isfinite(x), x, -np.inf), ok, I["seg"], I["events"], I["tol"])
        out[f"ep_{k}"] = float(np.mean(list(v.values()))) if v else None
        out["events"], out["n_null"] = sorted(int(e) for e in v), nn
        if ID in HITS_N:
            out[f"hits@{HITS_N[ID]}_{k}"] = hits(x, ok, I["events"], HITS_N[ID], I["tol"] if ID == "FL" else 0)
    return out


# ------------------------------------------------------------------------------------------------ J: identity rows
ID_ROWS = ("hamming", "jaccard", "deltacon", "chenzhang", "nbs")
ID_GRID_ROWS = ("hamming", "jaccard", "nbs")
ID_FIXED = {"hamming": (2, 10), "jaccard": (2, 10), "deltacon": (2, 10), "chenzhang": (2, 10), "nbs": (2, 10)}


def id_keys(ID: str):
    """Step 16's per-snapshot pair ids (repro/data/identity/keys_<ID>_<h>.npz) -> (file name, keys, n)."""
    import re
    rx = re.compile(rf"keys_{ID}_[0-9a-f]{{12}}\.npz")
    p = max((q for q in (DATA / "identity").iterdir() if rx.fullmatch(q.name)), key=os.path.getmtime)
    z = np.load(p)
    off, k = z["off"], z["keys"]
    return p.name, [k[off[t]:off[t + 1]] for t in range(len(off) - 1)], int(z["n"])


def id_path(row: str, s, ID: str) -> Path:
    from methods import hamming, interface, nbs
    src = sha(*(Path(m.__file__).read_bytes() for m in (interface, hamming, nbs)))
    return CACHE / f"id_{row}_{ID}_{sha(id_keys(ID)[0], json.dumps(list(s)), src)}.npy"


def id_job(a) -> tuple:
    from methods import hamming, nbs
    det, s, ID = a
    rows = ("hamming", "jaccard") if det == "hamming" else (det,)
    if all(id_path(r, s, ID).exists() for r in rows):
        return a
    _, keys, _ = id_keys(ID)
    seg = inputs(ID)["seg"]
    w, wr = s
    out = dict(zip(("hamming", "jaccard"), hamming.scores(keys, seg, w, wr))) if det == "hamming" else \
        {det: nbs.scores(keys, seg, w, wr)}
    for r, x in out.items():
        np.save(id_path(r, s, ID), x)
    return a


def id_series16(row: str, ID: str) -> np.ndarray:
    return np.asarray(np.load(DATA / "identity" / f"scores_{ID}.npz")[f"s_{row}"], float)


def section_j(dss, big, workers: int) -> dict:
    """Identity rows: setting rules on the (w, w_ref) grid of common.py; tolerance and guard bands (step-16 series)."""
    from multiprocessing import get_context
    grid = [(a, b) for a in W for b in WR]
    jobs = [(d, s, ID) for d in ("hamming", "nbs") for s in [ID_FIXED[d]] + grid for ID in dss]
    with get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        for i, _ in enumerate(pool.imap_unordered(id_job, jobs, chunksize=1)):
            if i % 40 == 0:
                print(f"J series {i}/{len(jobs)}", flush=True)
    out = {"grid": grid, "fixed": ID_FIXED, "check_fixed_equals_step16": {}}
    for row in ID_GRID_ROWS:
        sets = [ID_FIXED[row]] + [s for s in grid if s != ID_FIXED[row]]
        E = {s: {ID: ds_ep(np.load(id_path(row, s, ID)), ID) for ID in dss} for s in sets}
        out["check_fixed_equals_step16"][row] = max(float(np.max(np.abs(
            E[sets[0]][ID] - ds_ep(id_series16(row, ID), ID)))) for ID in dss)
        pg = np.array([pooled(E[s], dss) for s in grid])
        lodo, sel = {}, {}
        for ID in dss:
            rest = [x for x in dss if x != ID]
            j = max(range(len(grid)), key=lambda i: (pooled(E[grid[i]], rest), -i))
            sel[ID], lodo[ID] = list(grid[j]), E[grid[j]][ID]
        out[row] = {"fixed": {"setting": list(sets[0]), **ci(E[sets[0]], dss), "pooled_10plus": pooled(E[sets[0]], big)},
                    "grid": {"n": len(grid), "median": float(np.median(pg)), "q25": float(np.quantile(pg, .25)),
                             "q75": float(np.quantile(pg, .75)),
                             "median_10plus": float(np.median([pooled(E[s], big) for s in grid]))},
                    "oracle": {"setting": list(grid[int(np.argmax(pg))]), "pooled": float(pg.max()),
                               "pooled_10plus": float(max(pooled(E[s], big) for s in grid))},
                    "lodo": {**ci(lodo, dss), "selected": sel, "pooled_10plus": pooled(lodo, big)}}
    out["tau"], out["tau_own"], out["guard17"], out["guard_own"] = {}, {}, {}, {}
    ok = lambda E: [ID for ID in dss if len(E[ID])]          # datasets that keep an evaluable event
    for row in ID_ROWS:
        S = {ID: id_series16(row, ID) for ID in dss}
        span = sum(ID_FIXED[row])
        out["tau"][row] = {}
        for tau in range(11):
            E = {ID: ds_ep(S[ID], ID, tol=tau) for ID in dss}
            out["tau"][row][tau] = {"pooled": pooled(E, ok(E)), "n_ds": len(ok(E))}
        E = {ID: ds_ep(S[ID], ID, tol=span) for ID in dss}
        out["tau_own"][row] = {"tau": span, "pooled": pooled(E, ok(E)), "n_ds": len(ok(E)),
                               "pooled_10plus": pooled(E, big)}
        for key, gd in (("guard17", 17), ("guard_own", span)):
            E = {ID: ds_ep(S[ID], ID, guard=gd) for ID in dss}
            out[key][row] = {"guard": gd, **ci(E, ok(E)), "n_ds": len(ok(E)), "pooled_10plus": pooled(E, big)}
    return out


# ------------------------------------------------------------------------------------------------ K: score tests
def sc_path(ID: str) -> Path:
    from methods import interface, scoretest
    src = sha(*(Path(m.__file__).read_bytes() for m in (interface, scoretest)))
    return CACHE / f"scoretest_{ID}_{sha(id_keys(ID)[0], src)}.npz"


def sc_job(ID: str) -> str:
    from methods import scoretest
    p = sc_path(ID)
    if not p.exists():
        _, keys, n = id_keys(ID)
        np.savez(p, **scoretest.scores(keys, inputs(ID)["seg"], n))
    return ID


def section_k(dss, big, workers: int) -> dict:
    """The conditional score test as detectors (methods/scoretest.py): vertex and pair numerators over
    their independent-unit conditional SD, and the covariance-weighted joint test; EP next to the count detector."""
    from multiprocessing import get_context
    with get_context("fork").Pool(min(workers, 4)) as pool:
        list(pool.imap_unordered(sc_job, dss))
    rows = ("score_v", "score_e", "score_joint")
    E = {r: {ID: ds_ep(np.load(sc_path(ID))[r], ID) for ID in dss} for r in rows}
    E["counts"] = {ID: ds_ep(table2(ID)["S"]["counts"], ID) for ID in dss}
    out = {r: {**ci(E[r], dss), "pooled_10plus": pooled(E[r], big), "per_ds": {ID: float(E[r][ID].mean())
                                                                               for ID in dss}} for r in E}
    out["counts_minus"] = {r: ev.boot_pooled({ID: E["counts"][ID] - E[r][ID] for ID in dss}) for r in rows}
    return out


# ------------------------------------------------------------------------------------------------ L: more baselines
def ns_path(ID: str) -> Path:
    from methods import netsimile
    return CACHE / f"netsimile_{ID}_{sha(input_key(ID), Path(netsimile.__file__).read_bytes())}.npy"


def ns_job(ID: str) -> str:
    from methods import netsimile
    p = ns_path(ID)
    if not p.exists():
        np.save(p, np.array([netsimile.signature(A) for A in matrices(ID)]))
    return ID


def section_l(dss, big, workers: int) -> dict:
    """Round 5 (R2): NetSimile (methods/netsimile.py, w = 2, w_ref = 10); CUSUM with its fixed setting but no
    threshold, hence no reset; BOCPD on the two MAD-scaled counts instead of the eight raw features; next to the
    counts on the Table 2 mask, with the paired two-stage interval of counts minus each row; the pair-level score test
    of section K is added for its two-stage contrast with the counts."""
    from multiprocessing import get_context

    import inference as inf

    from methods import netsimile
    with get_context("fork").Pool(min(workers, 4)) as pool:
        list(pool.imap_unordered(ns_job, dss))
    st = settings()
    S = {}
    for ID in dss:
        I = inputs(ID)
        S[ID] = {"netsimile": netsimile.scores(np.load(ns_path(ID)), I["seg"]),
                 "cusum_noreset": baseline("cusum", ID, I["X8"], I["lad"], I["seg"], I["n"],
                                           {"cusum": {**st["cusum"], "threshold": 1e300}}),
                 "bocpd_counts": baseline("bocpd", ID, counts_X(I), I["lad"], I["seg"], I["n"], {"bocpd": st["bocpd"]}),
                 "score_e": np.load(sc_path(ID))["score_e"],
                 **{r: table2(ID)["S"][r] for r in ("counts", "cusum", "bocpd")}}
    rows = list(S[dss[0]])
    E = {ID: np.stack([ds_ep(S[ID][r], ID) for r in rows], 1) for ID in dss}
    out = {"rows": rows}
    for name, ds in (("all", dss), ("10plus", big)):
        X = {ID: E[ID] for ID in ds}
        pt, dr = inf.pooled_point(X, "equal"), inf.pooled_draws(X, "two_stage", "equal")
        ic = rows.index("counts")
        q = lambda d: [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
        out[name] = {"ep": {r: float(pt[i]) for i, r in enumerate(rows)},
                     "ep_ci95": {r: q(dr[:, i]) for i, r in enumerate(rows)},
                     "counts_minus": {r: {"diff": float(pt[ic] - pt[i]), "ci95": q(dr[:, ic] - dr[:, i])}
                                      for i, r in enumerate(rows) if i != ic}}
    out["per_ds"] = {ID: {r: float(E[ID][:, i].mean()) for i, r in enumerate(rows)} for ID in dss}
    return out


# ------------------------------------------------------------------------------------------------ N: grid sizes
def section_n() -> dict:
    """Per row: the F1 grid it was selected from (steps 3, 8, 9) and how many of its settings change the score (and
    so the EP); thresholds and cooldowns change only the alarms, except for CUSUM and EWMA, whose accumulators reset
    after an alarm. Rows without a grid run with one setting."""
    from common import CD, LW, NB, NM, SW, THR_SCPD
    from common import grid as count_grid
    lad = get_grid("lad")
    one = {"windows of its paper": ["scpd"],
           "w = 2, w_ref = 10 (untuned)": ["hamming", "jaccard", "deltacon", "chenzhang", "nbs", "ase", "transitivity",
                                           "assortativity", "fair_pair"],
           "G_{t-1} against the 10 snapshots before it (untuned)": ["spotlight"],
           "w = 2, w_ref = 5 (the counts' windows)": ["turnover"]}
    out = {"counts": {"f1_grid": len(count_grid()), "score_settings": len(W) * len(WR) * len(MODES),
                      "score_parameters": "w, w_ref, mode (pw, cen, gam)"},
           "constant": {"f1_grid": len(count_grid()), "score_settings": 1, "score_parameters": "none"},
           "cusum": {"f1_grid": len(get_grid("cusum")), "score_settings": len(get_grid("cusum")),
                     "score_parameters": "threshold, kappa, startup, cooldown (reset after an alarm)"},
           "ewma": {"f1_grid": len(get_grid("ewma")), "score_settings": len(get_grid("ewma")),
                    "score_parameters": "every parameter (reset after an alarm); plants only"},
           "bocpd": {"f1_grid": len(get_grid("bocpd")),
                     "score_settings": len({p["hazard_rate"] for p in get_grid("bocpd")}),
                     "score_parameters": "hazard rate"},
           "mmd": {"f1_grid": len(get_grid("mmd")), "score_settings": len({p["window_size"] for p in get_grid("mmd")}),
                   "score_parameters": "window"},
           "lad": {"f1_grid": len(lad), "score_settings": len({(p["num_eigen"], p["short_window"], p["long_window"])
                                                               for p in lad}),
                   "score_parameters": "number of eigenvalues, short and long window"},
           "tire": {"f1_grid": len(TIRE_TRAIN) * 9 * 5, "score_settings": len(TIRE_TRAIN),
                    "score_parameters": "N, K, latent and shared dimension, lambda, domain, epochs"}}
    for note, rows in one.items():
        out.update({r: {"f1_grid": 0, "score_settings": 1, "score_parameters": note} for r in rows})
    out["scpd"]["f1_grid_table1"] = len(NM) * len(NB) * len(SW) * len(LW) * len(CD) * len(THR_SCPD)
    return out


# ------------------------------------------------------------------------------------------------ M: LAD eigenvalues
# LAD's code uses 6 top Laplacian eigenvalues on its real data and 499 on its synthetic SBM (data/raw/lad/README.md);
# the benchmark row keeps 5. Spec v5 M: LAD at its fixed windows with k = 6 and k = 499, reference rows of step 23.
LAD_K = 499
LAD_KS = (6, 499)


def lap_eigs(A, k: int = LAD_K) -> np.ndarray:
    """Top k eigenvalues of D - A of the active part of the loop-free binary snapshot (isolated vertices add zeros;
    zeros fill when the active part has fewer)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    from data.registry import start_vector
    A = sp.csr_matrix(A, dtype=np.float64)
    A = A.maximum(A.T).tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    A.data[:] = 1.0
    act = np.flatnonzero(np.diff(A.indptr))
    out = np.zeros(k)
    if len(act) < 2:
        return out
    Aa = A[act][:, act]
    L = (sp.diags(np.asarray(Aa.sum(1)).ravel()) - Aa).tocsr()
    v = (np.linalg.eigvalsh(L.toarray()) if len(act) < 2000
         else eigsh(L, k=k, which="LA", v0=start_vector(len(act)), tol=1e-6, return_eigenvectors=False))
    v = np.sort(v)[::-1][:k]
    out[:len(v)] = v
    return out


def lad_chunks(ID: str) -> list[tuple]:
    T = meta(ID)["T"]
    step = 12 if ID in ("SCC", "MAG", "ENRON", "AskUbuntu") else T
    return [(ID, a, min(a + step, T)) for a in range(0, T, step)]


def lad_path(ID: str, a: int, b: int) -> Path:
    return CACHE / f"lad{LAD_K}_{ID}_{a}-{b}_{sha(input_key(ID), src_sha(lap_eigs))}.npy"


def lad_job(job) -> str:
    ID, a, b = job
    p = lad_path(*job)
    if not p.exists():
        t0 = time.time()
        E = np.array([lap_eigs(A) for A in itertools.islice(matrices(ID), a, b)])
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, E)
        os.replace(tmp, p)
        return f"{ID} {a}-{b}: {time.time() - t0:.0f} s"
    return f"{ID} {a}-{b}: cached"


def section_m(dss, big, workers: int) -> dict:
    """LAD with k = 6 and k = 499 eigenvalues at its fixed windows; check M0: k = 5 from the new eigenvalues against
    LAD's stored input and the Table 2 LAD row. Writes repro/data/settings/lad_<ID>.npz (input of step 23)."""
    from multiprocessing import get_context
    jobs = [j for ID in B for j in lad_chunks(ID)]
    jobs.sort(key=lambda j: j[0] not in ("SCC", "MAG", "ENRON", "AskUbuntu"))
    with get_context("fork").Pool(min(workers, 6), maxtasksperchild=4) as pool:
        for line in pool.imap_unordered(lad_job, [j for j in jobs if not lad_path(*j).exists()]):
            print(line, flush=True)
    st = settings()
    out = {"k": list(LAD_KS), "windows": [st["lad"]["short_window"], st["lad"]["long_window"]], "M0": {}, "ep": {}}
    for ID in B:
        I = inputs(ID)
        E = np.concatenate([np.load(lad_path(*j)) for j in lad_chunks(ID)])
        run = {k: baseline("lad", ID, I["X8"], E[:, :k], I["seg"], I["n"], {"lad": {**st["lad"], "num_eigen": k}})
               for k in (5, *LAD_KS)}
        s5, S = run[5], {f"lad_k{k}": run[k] for k in LAD_KS}
        np.savez(CACHE / f"lad_{ID}.npz", **{f"s_{r}": x for r, x in S.items()})
        ref, both = table2(ID)["S"]["lad"], np.isfinite(table2(ID)["S"]["lad"]) & np.isfinite(s5)
        dif = np.abs(E[:, :5] - I["lad"][:, :5])
        out["M0"][ID] = {"eig5_max_abs": float(dif.max()), "eig5_max_rel": float((dif / np.maximum(1, np.abs(
            I["lad"][:, :5]))).max()), "lad_k5_series_max_abs": float(np.max(np.abs(ref[both] - s5[both])))}
        if ID in dss:
            out["M0"][ID]["ep_k5_recomputed"] = float(ds_ep(s5, ID).mean())
            out["M0"][ID]["ep_table2_lad"] = float(ds_ep(ref, ID).mean())
            out["ep"][ID] = {r: float(ds_ep(x, ID).mean()) for r, x in S.items()}
    out["pooled"] = {r: {"all": float(np.mean([out["ep"][d][r] for d in dss])),
                         "10plus": float(np.mean([out["ep"][d][r] for d in big]))} for r in out["ep"][dss[0]]}
    return out


# ------------------------------------------------------------------------------------------------ R: residual design
# Spec prereg/specs/RESIDUAL.md (fixed 2026-10-24), calibration only (spec v5 R): OLS of a score on an activity
# basis per dataset (EEG per case), residual ranked as a score. Detector residuals only if every check passes.
RES_WIN = ((2, 5), (2, 10))
RES_ROWS = ["tire", "cusum", "lad", "mmd", "bocpd", "scpd", "counts", "constant"]


def wdelta(x, seg, w: int, wr: int) -> np.ndarray:
    """Signed window change mean x[t-w, t) - mean x[t-w-wr, t-w) per segment; nan where the windows do not fit."""
    D = np.full(len(x), np.nan)
    for s0, s1 in ev.seg_ranges(seg):
        c = np.r_[0.0, np.cumsum(x[s0:s1])]
        t = np.arange(w + wr, s1 - s0)
        D[s0 + t] = (c[t] - c[t - w]) / w - (c[t - w] - c[t - w - wr]) / wr
    return D


def activity_basis(ID: str) -> np.ndarray:
    I = inputs(ID)
    Z = mad_scale(I["C"], I["seg"])
    cols = [np.ones(len(I["seg"]))]
    for w, wr in RES_WIN:
        a, b = (wdelta(Z[:, j], I["seg"], w, wr) for j in (0, 1))
        cols += [a, b, np.abs(a), np.abs(b), a ** 2, b ** 2, a * b]
    return np.stack(cols, 1)


def fit_groups(ID: str) -> np.ndarray:
    """Fit group of every step: the benchmark EEG per case (its segments are files of cases chb01, chb03, chb05)."""
    seg = inputs(ID)["seg"]
    if ID != "EEG":
        return np.zeros(len(seg), int)
    files = {s["segment"]: s["file"] for s in json.loads((BENCH / "EEG_meta.json").read_text())["segments"]}
    names = sorted({f.split("_")[0] for f in files.values()})
    return np.array([names.index(files[int(k)].split("_")[0]) for k in seg])


def residual(x, ID: str, rank: bool = False) -> np.ndarray:
    from scipy.stats import rankdata
    mask, X, g = table2(ID)["mask"], activity_basis(ID), fit_groups(ID)
    out = np.full(len(mask), np.nan)
    for u in np.unique(g[mask]):
        m = mask & (g == u)
        y = np.asarray(x, float)[m]
        y = rankdata(y) if rank else y
        beta = np.linalg.lstsq(X[m], y, rcond=None)[0]
        out[m] = y - X[m] @ beta
    return out


def section_r(dss, big) -> dict:
    out = {"spec": "prereg/specs/RESIDUAL.md; calibration only (spec v5 R)", "windows": [list(w) for w in RES_WIN],
           "basis_finite_on_mask": {ID: bool(np.isfinite(activity_basis(ID)[table2(ID)["mask"]]).all()) for ID in dss}}
    pool = lambda E: {"all": ci(E, dss), "10plus": ci({d: E[d] for d in big}, big)}
    const = {ID: np.where(table2(ID)["mask"], 1.0, np.nan) for ID in dss}
    R = {ID: residual(const[ID], ID) for ID in dss}
    out["C1_constant"] = {**pool({ID: ds_ep(R[ID], ID) for ID in dss}),
                          "max_abs_residual": float(max(np.nanmax(np.abs(R[ID])) for ID in dss))}
    out["C2_counts"] = pool({ID: ds_ep(residual(table2(ID)["S"]["counts"], ID), ID) for ID in dss})
    draws = []
    for i in range(20):
        rng = np.random.default_rng(ev.SEED + i)
        E = {ID: ds_ep(residual(rng.standard_normal(len(const[ID])), ID), ID) for ID in dss}
        draws.append({"all": float(np.mean([E[d].mean() for d in dss])), "10plus": float(np.mean([E[d].mean()
                                                                                                  for d in big]))})
    out["C3_noise"] = {"draws": draws, **{k: float(np.mean([d[k] for d in draws])) for k in ("all", "10plus")}}
    out["pass"] = {"C1": abs(out["C1_constant"]["all"]["mean"] - 0.5) <= 0.005,
                   "C2": abs(out["C2_counts"]["all"]["mean"] - 0.5) <= 0.05,
                   "C3": abs(out["C3_noise"]["all"] - 0.5) <= 0.02}
    out["pass"]["all"] = all(out["pass"].values())
    print("residual calibration", out["pass"], flush=True)
    if out["pass"]["all"]:                         # the spec's reading applies only to a calibrated test
        out["residual_ep"] = {r: pool({ID: ds_ep(residual(table2(ID)["S"][r], ID), ID) for ID in dss})
                              for r in RES_ROWS}
        out["residual_ep_rank"] = {r: pool({ID: ds_ep(residual(table2(ID)["S"][r], ID, True), ID) for ID in dss})
                                   for r in RES_ROWS}
    return out


# ------------------------------------------------------------------------------------------------ H: ties
def section_h(dss) -> dict:
    out = {}
    for ID in dss:
        I, S, mask = inputs(ID), table2(ID)["S"], table2(ID)["mask"]
        L = I["tol"] + 1
        evs, null = ev.windows(mask, I["seg"], I["events"], I["events"], L)
        out[ID] = {}
        for r in ROWS:
            x = S[r]
            nm, se = np.sort(ev._wmax(x, null, L)), ev._wmax(x, evs, L)
            lo, hi = np.searchsorted(nm, se, "left"), np.searchsorted(nm, se, "right")
            out[ID][r] = {"events_tied": float(np.mean(hi > lo)), "tie_mass": float(np.mean((hi - lo) / len(nm))),
                          "null_at_min": float(np.mean(nm == np.min(x[mask])))}
    return out


# ------------------------------------------------------------------------------------------------ G: plants
PL_OFFSETS = (0, 10, 20, 30)


def plant_series(ID, mode, rho, seed):
    return dict(np.load(max((DATA / "plants").glob(f"stats_{ID}_{mode}_{rho:g}_s{seed}_*.npz"),
                            key=os.path.getmtime)))


def plant_score(det, s, F, seg):
    """Score of one plant row on (spliced) per-snapshot statistics F (online scale as step 14)."""
    from common import bundle

    from methods import bocpd, cusum, lad, mmd
    out = np.full(len(seg), np.nan)
    for s0, s1 in ev.seg_ranges(seg):
        if det == "dos":
            d = scpd_nodiff(F["dos"][s0:s1], s)
            burn = max(s)
        elif det == "lad":
            d = lad.detect(bundle(eig=F["ladu"][s0:s1]), lad_params(s)).scores
            burn = s[0] + s[1]
        else:
            if det == "cusum":
                mod, prm, burn = cusum, s, s["startup_period"]
            elif det == "bocpd":
                mod, prm, burn = bocpd, {"threshold": .5, "hazard_rate": s, "cooldown": 1}, 15
            else:
                mod, prm, burn = mmd, {"threshold": 0.0, "window_size": s, "cooldown": 1}, 2 * s
            d = mod.detect(bundle(X=F["raw8"][s0:s1]), prm).scores
        d = np.asarray(d, float)
        d[:burn] = np.nan
        out[s0:s1] = d
    return out


def scpd_nodiff(sig, s):
    if tuple(s) == (5, 10):
        return P.dos_scores(sig)
    return P.dos_scores(sig, ws=s[0], wl=s[1])


def plant_grid(g):
    rng = np.random.default_rng(SEED + 1)
    cs = g["cusum"]["grid"]
    return {"cusum": [g["cusum"]["fixed"]] + [cs[i] for i in rng.choice(len(cs), 30,
                                                                                                       replace=False)],
            "bocpd": [0.1, 0.002, 0.01, 0.03, 0.3, 0.5], "mmd": g["mmd"]["grid"], "lad": g["lad"]["grid"],
            "dos": g["scpd"]["grid"]}


def plant_offsets(ID, det):
    """BOCPD costs ~20 s per scoring on PM (run-length posterior over 2,400-step segments): 20 trials there."""
    return (0, 20) if (ID, det) == ("PM", "bocpd") else PL_OFFSETS


def plant_cell(args):
    """AUC (window max, planted vs original at the same onsets) of every grid setting in one cell."""
    ID, mode, det, settings_, offsets = args
    key = sha(ID, mode, det, json.dumps(settings_), offsets, src_sha(plant_score, vscores, scpd_nodiff, lad_params),
              P.STATS_SHA, *[(REPO / "methods" / f"{m}.py").read_bytes() for m in ("cusum", "bocpd", "mmd", "lad")])
    p = CACHE / f"plant_{ID}_{mode}_{det}_{key}.json"
    if p.exists():
        return json.loads(p.read_text())
    seg = np.load(BENCH / f"{ID}.npz")["segment"].astype(int)
    keys = ("raw8", "ladu", "dos")
    zb = plant_series(ID, "none", 0.0, 0)
    Fb = {k: zb[k] for k in keys}
    res = []
    for s in settings_:
        Du = plant_score(det, s, Fb, seg)
        pos, neg, ties = [], [], 0
        for sd in range(10):
            zp = plant_series(ID, mode, 0.5, sd)
            for o in offsets:
                t0s = P.block_onsets(seg, o)
                F = {k: P.splice(Fb[k], zp[k], t0s) for k in keys}
                Dp = plant_score(det, s, F, seg)
                a, b = P.win_max(Dp, t0s), P.win_max(Du, t0s)
                pos.append(a)
                neg.append(b)
                ties += int(np.sum(a == b))
        pos, neg = np.concatenate(pos), np.concatenate(neg)
        res.append({"setting": s, "auc": P.auc(pos, neg), "n": len(pos), "equal_share": ties / len(pos),
                    "median_abs_diff_over_sd": float(np.median(np.abs(pos - neg)) / (np.nanstd(neg) + 1e-300))})
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, p)
    return res


def cusum_argmax(ID, mode) -> dict:
    """Which raw feature attains the CUSUM maximum (fixed setting), original series, after startup."""
    st = settings()["cusum"]
    zb = plant_series(ID, "none", 0.0, 0)
    seg = np.load(BENCH / f"{ID}.npz")["segment"].astype(int)
    cnt = np.zeros(8)
    for s0, s1 in ev.seg_ranges(seg):
        X = zb["raw8"][s0:s1]
        k = st["startup_period"]
        if len(X) <= k:
            continue
        mu, sd = X[:k].mean(0), X[:k].std(0)
        sd = np.where(sd < 1e-10, 1.0, sd)
        Sp, Sn = np.zeros(8), np.zeros(8)
        last = -st["cooldown"]
        for t in range(k, len(X)):
            z = (X[t] - mu) / sd
            Sp, Sn = np.maximum(0, Sp + z - st["kappa"]), np.maximum(0, Sn - z - st["kappa"])
            m = np.maximum(Sp, Sn)
            cnt[np.argmax(m)] += 1
            if m.max() >= st["threshold"] and t - last >= st["cooldown"]:
                last, Sp, Sn = t, np.zeros(8), np.zeros(8)
    return {"share_by_feature": (cnt / cnt.sum()).round(4).tolist(),
            "features": ["n", "E", "mean degree", "sd degree", "density", "spectral radius", "max singular value",
                         "components"]}


def section_g(g, workers) -> dict:
    from multiprocessing import get_context
    pg = plant_grid(g)
    jobs = []
    for ID in ("PS", "HT09", "PM", "MIT"):
        for mode in P.STRUCT:
            for det in pg:
                jobs += [(ID, mode, det, [x], plant_offsets(ID, det)) for x in pg[det]]
    c4 = [("PS", "rand", det, [g[det]["fixed"]], tuple(range(40))) for det in ("mmd", "cusum")]
    order = sorted(jobs + c4, key=lambda j: (j[0] != "PM", j[2] != "bocpd"))       # longest first
    with get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        done = pool.map(plant_cell, order, chunksize=1)
    res = {}
    for j, r in zip(order, done):
        if j not in c4:
            res.setdefault(j[:3], []).extend(r)
    pj = json.loads((RESULTS / "plants.json").read_text())["backgrounds"]
    cell = lambda ID, m: next(c for c in pj[ID]["e1"] if c["mode"] == m and c["rho"] == 0.5)
    out = {"design": f"seeds 0-9, offsets {PL_OFFSETS} (40 trials per cell; BOCPD on PM offsets 0, 20), rho 0.5",
           "grids": {d: len(v) for d, v in pg.items()}, "cells": {}}
    for (ID, mode, det), r in res.items():
        best = max(r, key=lambda x: x["auc"])
        fx = g["scpd" if det == "dos" else det]["fixed"]
        fixed = next((x for x in r if json.dumps(x["setting"], sort_keys=True) == json.dumps(fx, sort_keys=True)),
                     None)
        out["cells"].setdefault(f"{ID}/{mode}", {})[det] = {
            "fixed_auc_step14": cell(ID, mode)["rows"][det]["auc"],
            "fixed_auc_here": fixed["auc"] if fixed else None, "oracle_auc": best["auc"],
            "oracle_setting": best["setting"], "grid_median_auc": float(np.median([x["auc"] for x in r])),
            "n_settings": len(r), "fixed_equal_share": fixed["equal_share"] if fixed else None,
            "fixed_median_abs_diff_over_sd": fixed["median_abs_diff_over_sd"] if fixed else None}
    out["C4"] = {j[2]: {"here": r[0]["auc"], "step14": cell("PS", "rand")["rows"][j[2]]["auc"]}
                 for j, r in zip(order, done) if j in c4}
    out["cusum_argmax_feature"] = {ID: cusum_argmax(ID, "rand") for ID in ("PS", "HT09", "PM", "MIT")}
    return out


# ------------------------------------------------------------------------------------------------ I: s-GNN
SGNN_GRID = [{"lr": lr, "weight_decay": wd, "dropout": dr, "hidden": h} for lr in (1e-5, 1e-4, 1e-3, 1e-2)
             for wd in (1e-6, 1e-5) for dr in (0.05, 0.2, 0.4) for h in (32, 64, 128)]


def sgnn_configs() -> list[dict]:
    rng = np.random.default_rng(SEED + 2)
    return [{}] + [SGNN_GRID[i] for i in sorted(rng.choice(len(SGNN_GRID), 12, replace=False))]


def section_i(run: bool, workers: int) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    from methods.sgnn import flags
    z = np.load(BENCH / "PM.npz", allow_pickle=False)
    seg, events, off = z["segment"], z["events"], z["offsets"]
    cfgs = sgnn_configs()
    key = sha(BENCH / "PM.npz", REPO / "methods" / "sgnn.py", src_sha(sgnn_worker))
    jobs, cells = [], {}
    for s, u in enumerate(np.unique(seg)):
        idx = np.nonzero(seg == u)[0]
        a, b = int(idx[0]), int(idx[-1]) + 1
        cps = [int(e) - a for e in events if a <= e < b]
        tr, va, te = ev.individual_split(b - a, cps, np.random.default_rng([20261023, s, 0]))
        for c, cfg in enumerate(cfgs):
            p = CACHE / f"sgnn_s{s}_c{c}_{sha(key, json.dumps(cfg, sort_keys=True))}.npz"
            cells[(s, c)] = (p, cps, va, te, b - a)
            if not p.exists() and run:
                jp = CACHE / f"sgnn_job_s{s}_c{c}.npz"
                np.savez(jp, n=int(z["n"]), indptr=z["indptr"][a:b], indices=z["indices"][off[a]:off[b]],
                         offsets=off[a:b + 1] - off[a], cps=np.array(cps), train=tr, val=va, test=te, rep=0,
                         cfg=json.dumps(cfg))
                jobs.append((jp, p))
    if jobs:
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   CUDA_VISIBLE_DEVICES="")
        t0 = time.time()

        def one(j):
            r = subprocess.run([SGNN_PY, "-I", str(Path(__file__)), "_sgnn", str(j[0]), str(j[1])], env=env,
                               check=False, capture_output=True, text=True)
            if r.returncode:
                print(f"s-GNN {j[1].name} failed: {r.stderr[-300:]}", flush=True)
            j[0].unlink(missing_ok=True)
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(one, jobs))
        print(f"s-GNN: {len(jobs)} trainings, {time.time() - t0:.0f} s", flush=True)
    TH = np.round(np.linspace(0.05, 0.95, 19), 2)
    cnt = table2("PM")["S"]["counts"]
    res = {"grid_source": "Sulem et al. (MLJ 2024) Sec. 4.4 (S&P 500): lr {1e-5..1e-2}, weight decay {1e-6, 1e-5}, "
                          "dropout {0.05, 0.2, 0.4}, hidden {32, 64, 128}; Sort-k {50, 100, 200} exceeds the 27 "
                          "vertices of PAMAP2, so DyNNet's default n // 4 is kept. The PAMAP2 section states no grid. "
                          "Selection as in the paper: the highest F1 of the validation pairs (DyNNet results.json).",
           "configs": cfgs, "subjects": {}}
    for s in range(8):
        rows = []
        for c in range(len(cfgs)):
            p, cps, va, te, T = cells[(s, c)]
            if not p.exists():
                continue
            Zr = np.load(p)
            flg = np.zeros(T, int)
            flg[cps] = 1
            lab = np.cumsum(flg)
            yv, yt = ev.cp_targets(lab[va], 5), ev.cp_targets(lab[te], 5)
            fv = {float(t): ev.adjusted_f1(flags(Zr["Zv"], 20, t), yv, 0.5) for t in TH}
            tb = max(TH, key=lambda t: (fv[float(t)], -abs(t - 0.5)))
            a = int(np.flatnonzero(seg == np.unique(seg)[s])[0])
            pos = a + te[Zr["it"]]                     # global index of each test Z
            x, mk = np.full(len(seg), np.nan), np.zeros(len(seg), bool)
            x[pos], mk[pos] = 1 - Zr["Zt"], True
            evs = [e for e in events if a + te[0] <= e <= a + te[-1]]
            e_s, _ = ep_local(x, mk, seg, evs, 5)
            e_c, _ = ep_local(cnt, mk & np.isfinite(cnt), seg, evs, 5)
            rows.append({"config": c, "val_pair_f1": float(Zr["val_pair_f1"]), "val_f1_0.5": fv[0.5],
                         "test_f1_0.5": ev.adjusted_f1(flags(Zr["Zt"], 20, .5), yt, 0.5),
                         "theta_val": float(tb), "test_f1_theta": ev.adjusted_f1(flags(Zr["Zt"], 20, tb), yt, 0.5),
                         "test_ep": float(np.mean(list(e_s.values()))) if e_s else None,
                         "counts_ep_same_steps": float(np.mean(list(e_c.values()))) if e_c else None,
                         "n_test_events": len(e_s), "seconds": float(Zr["seconds"])})
        if not rows:
            continue
        b = max(rows, key=lambda r: (r["val_pair_f1"], -r["config"]))
        res["subjects"][s] = {"runs": rows, "best_val": b}
    S = res["subjects"]
    if S:
        d0 = [next(r for r in v["runs"] if r["config"] == 0) for v in S.values() if any(r["config"] == 0
                                                                                     for r in v["runs"])]
        bv = [v["best_val"] for v in S.values()]
        mean = lambda xs: float(np.mean([x for x in xs if x is not None])) if xs else None
        res["mean_test"] = {"defaults_f1_0.5": mean([r["test_f1_0.5"] for r in d0]),
                            "best_val_f1_0.5": mean([r["test_f1_0.5"] for r in bv]),
                            "best_val_f1_theta_val": mean([r["test_f1_theta"] for r in bv]),
                            "oracle_config_f1": mean([max(r["test_f1_0.5"] for r in v["runs"]) for v in S.values()]),
                            "best_val_test_ep": mean([r["test_ep"] for r in bv]),
                            "counts_ep_same_steps": mean([r["counts_ep_same_steps"] for r in bv]),
                            "n_subjects": len(S), "n_runs": sum(len(v["runs"]) for v in S.values()),
                            "paper_f1": 0.77625}
    return res


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    import argparse
    from multiprocessing import get_context
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="ABCDEFGH")
    ap.add_argument("--sgnn", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        print(summary(json.loads(OUT.read_text())))
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = Timer()
    for ID in REPO_VIEW:                               # the Table 1 bundles once, in this process
        inputs(ID)
    g = grids()
    dss, big = pools()
    need = set()
    if set(a.sections) & set("ABCDF"):
        for d in DETS:
            for s in g[d]["settings"]:
                for ID in B:
                    need.add((d, json.dumps(s), ID))
            for ID in B:
                if DEFAULT[d] is not None:
                    need.add((d, json.dumps(default_of(d, ID)), ID))
            for ID in SHORT:
                need.add((d, json.dumps(short_setting(d, ID, g)), ID))
        for s in itertools.product(W, WR, MODES):
            for ID in B:
                need.add(("counts", json.dumps(list(s) + ["online"]), ID))
    jobs = sorted({(d, s, ID) for d, s, ID in need}, key=lambda j: (j[0] != "tire", j[2] not in ("PM", "EEG")))
    todo = [(d, json.loads(s), ID) for d, s, ID in jobs]
    todo = [(d, tuple(s) if isinstance(s, list) else s, ID) for d, s, ID in todo]
    with get_context("fork").Pool(a.workers, maxtasksperchild=20) as pool:
        for i, _ in enumerate(pool.imap_unordered(_job, todo, chunksize=1)):
            if i % 500 == 0:
                print(f"series {i}/{len(todo)} {t0():.0f} s", flush=True)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out["spec"] = __doc__
    out["checks"] = checks()
    print("checks", out["checks"]["pass"], flush=True)
    sec = {"A": lambda: section_a(g, dss, big), "B": lambda: section_b(g, dss), "C": lambda: section_c(dss, big),
           "D": lambda: section_d(dss, g), "E": lambda: section_e(dss, big), "F": lambda: section_f(dss),
           "H": lambda: section_h(dss), "G": lambda: section_g(g, a.workers),
           "J": lambda: section_j(dss, big, a.workers), "K": lambda: section_k(dss, big, a.workers),
           "L": lambda: section_l(dss, big, a.workers), "N": section_n,
           "M": lambda: section_m(dss, big, a.workers), "R": lambda: section_r(dss, big)}
    for k in a.sections:
        out[k] = sec[k]()
        write_json(OUT, out)
        print(f"section {k} done {t0():.0f} s", flush=True)
    if a.sgnn:
        out["I"] = section_i(True, min(a.workers, 6))
        write_json(OUT, out)
    out["_runtime_s"] = round(t0(), 1)
    write_json(OUT, out)
    print(f"step 22: {t0():.0f} s -> {OUT}")


def summary(R) -> str:
    f = lambda v: "n/a" if v is None else f"{v:.3f}"
    md = ["## checks", json.dumps(R["checks"]["pass"])]
    if "A" in R:
        A = R["A"]
        md.append("\n## A. pooled EP (ten datasets) by setting rule")
        md.append("| detector | grid n | default [CI] | Table 2 fixed [CI] | grid median (IQR) | oracle | LODO [CI] |")
        md.append("|---|---|---|---|---|---|---|")
        for d in DETS:
            r = A[d]
            dfl = f"{f(r['default']['mean'])} [{f(r['default']['lo'])}, {f(r['default']['hi'])}]" if "default" in r \
                else "n/a"
            md.append(f"| {d} | {r['grid']['n']} | {dfl} | {f(r['fixed']['mean'])} [{f(r['fixed']['lo'])}, "
                      f"{f(r['fixed']['hi'])}] | {f(r['grid']['median'])} ({f(r['grid']['q25'])}-"
                      f"{f(r['grid']['q75'])}) | {f(r['oracle']['pooled'])} | {f(r['lodo']['mean'])} "
                      f"[{f(r['lodo']['lo'])}, {f(r['lodo']['hi'])}] |")
        md.append("\n>= 10-event pool (EEG, HT09, PM, SCC): fixed / default / grid median / oracle / LODO")
        for d in DETS:
            r = A[d]
            md.append(f"{d}: {f(r['fixed']['pooled_10plus'])} / {f(r.get('default', {}).get('pooled_10plus'))} / "
                      f"{f(r['grid']['median_10plus'])} / {f(r['oracle']['pooled_10plus'])} / "
                      f"{f(r['lodo']['pooled_10plus'])}")
        md.append("counts lead per rule: " + json.dumps(A["counts_lead"]))
    if "B" in R:
        Bb = R["B"]
        md.append("\n## B. short series (B2) and source datasets scored")
        for ID in SHORT:
            md.append(f"{ID}: " + json.dumps({d: {k: v for k, v in r.items() if k in ("ep", f"hits@{HITS_N[ID]}",
                                                                                     "hits@2_secondary",
                                                                                     "n_events_scored")}
                                               for d, r in Bb["B2"][ID].items()}))
        md.append("scored source datasets: " + json.dumps(Bb["source_datasets_scored"]))
        md.append("B1 (own burn-in), ten datasets: " + json.dumps(
            {d: [round(v["mean"], 3), round(v["lo"], 3), round(v["hi"], 3)] for d, v in Bb["pooled_B1_ten"].items()}))
        md.append("pooled EP, every dataset a detector scores: " + json.dumps(
            {d: [round(v["pooled"], 3), len(v["datasets"])] for d, v in Bb["pooled_all_scored"].items()}))
    if "C" in R:
        C = R["C"]
        md.append("\n## C. pooled EP vs tau (common tau)")
        rows = list(C["tau"]["0"].keys()) if "0" in C["tau"] else list(C["tau"][0].keys())
        md.append("| tau | " + " | ".join(rows) + " |")
        md.append("|---|" + "---|" * len(rows))
        for tau, blk in C["tau"].items():
            md.append(f"| {tau} | " + " | ".join(f(blk[r]["pooled"]) for r in rows) + " |")
        for k in ("tau_own", "guard17", "guard_own"):
            md.append(f"{k}: " + json.dumps({r: round(v["pooled"], 3) for r, v in C[k].items()}))
    if "D" in R:
        md.append("\n## D. alignment (pooled)")
        md.append(json.dumps({r: {k: round(v, 3) for k, v in x.items()} for r, x in R["D"]["pooled"].items()}))
        md.append("UCI: " + json.dumps(R["D"]["per_ds"]))
    if "E" in R:
        md.append("\n## E. counts variants")
        md.append(json.dumps({k: round(v["mean"], 3) for k, v in R["E"]["variants"].items()}))
        md.append("online grid: " + json.dumps(R["E"]["online_grid"]))
    if "F" in R:
        md.append("\n## F. own constructions")
        for ID, r in R["F"].items():
            md.append(f"{ID}: " + json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("ep", "ep_common_mask",
                                                                                           "n_events_scored")
                                                  or kk.startswith("hits")} for k, v in r.items()
                                              if isinstance(v, dict) and k != "same_events"}))
            md.append("  same events (a = source detector, b = counts): " + json.dumps(
                {k: v for k, v in r.get("same_events", {}).items() if k != "rows"}))
    if "G" in R:
        md.append("\n## G. plants (rho 0.5): fixed / oracle AUC")
        md.append("C4: " + json.dumps(R["G"]["C4"]))
        for c, r in R["G"]["cells"].items():
            md.append(f"{c}: " + json.dumps({d: [round(v["fixed_auc_step14"], 3), round(v["oracle_auc"], 3)]
                                             for d, v in r.items()}))
    if "J" in R:
        J = R["J"]
        md.append("\n## J. identity rows: fixed [CI] / grid median (IQR) / oracle / LODO [CI]; >= 10 events")
        for r in ID_GRID_ROWS:
            x = J[r]
            md.append(f"{r}: {f(x['fixed']['mean'])} [{f(x['fixed']['lo'])}, {f(x['fixed']['hi'])}] / "
                      f"{f(x['grid']['median'])} ({f(x['grid']['q25'])}-{f(x['grid']['q75'])}) / "
                      f"{f(x['oracle']['pooled'])} {x['oracle']['setting']} / {f(x['lodo']['mean'])} "
                      f"[{f(x['lodo']['lo'])}, {f(x['lodo']['hi'])}]; >=10: {f(x['fixed']['pooled_10plus'])} / "
                      f"{f(x['grid']['median_10plus'])} / {f(x['oracle']['pooled_10plus'])} / "
                      f"{f(x['lodo']['pooled_10plus'])}")
        md.append("check fixed = step 16: " + json.dumps(J["check_fixed_equals_step16"]))
        for r in ID_ROWS:
            md.append(f"{r}: tau 0..10 " + " ".join(f(v["pooled"]) for v in J["tau"][r].values())
                      + f"; own span {J['tau_own'][r]['tau']}: {f(J['tau_own'][r]['pooled'])}; guard17 "
                      f"{f(J['guard17'][r]['mean'])} [{f(J['guard17'][r]['lo'])}, {f(J['guard17'][r]['hi'])}]; "
                      f"guard own {f(J['guard_own'][r]['mean'])}")
    if "K" in R:
        md.append("\n## K. score tests (EP [CI]; >= 10 events; counts - row [CI])")
        for r, x in R["K"].items():
            if r != "counts_minus":
                cm = R["K"]["counts_minus"].get(r)
                md.append(f"{r}: {f(x['mean'])} [{f(x['lo'])}, {f(x['hi'])}]; {f(x['pooled_10plus'])}"
                          + (f"; counts - row {cm['mean']:+.3f} [{cm['lo']:+.3f}, {cm['hi']:+.3f}]" if cm else ""))
    if "N" in R:
        md.append("\n## N. grid sizes: F1 grid / settings that change the score")
        md.extend(f"{r}: {v['f1_grid']} / {v['score_settings']} ({v['score_parameters']})" for r, v in R["N"].items())
    if "M" in R:
        M = R["M"]
        md.append(f"\n## M. LAD with k eigenvalues, windows {M['windows']}: pooled EP all ten / >= 10")
        md.append(json.dumps({r: {k: round(v, 3) for k, v in x.items()} for r, x in M["pooled"].items()}))
        md.append("per dataset: " + json.dumps({d: {r: round(v, 3) for r, v in x.items()} for d, x in M["ep"].items()}))
        md.append("M0: " + json.dumps(M["M0"]))
    if "R" in R:
        Rr = R["R"]
        md.append(f"\n## R. residual design, calibration: {json.dumps(Rr['pass'])}")
        for k in ("C1_constant", "C2_counts"):
            md.append(f"{k}: " + json.dumps({p: {q: round(v, 4) for q, v in x.items()} for p, x in Rr[k].items()
                                             if isinstance(x, dict)}))
        md.append(f"C3_noise: all {Rr['C3_noise']['all']:.4f}, >= 10 {Rr['C3_noise']['10plus']:.4f}")
        for k in ("residual_ep", "residual_ep_rank"):
            if k in Rr:
                md.append(f"{k}: " + json.dumps({r: {p: [round(x['mean'], 3), round(x['lo'], 3), round(x['hi'], 3)]
                                                     for p, x in v.items()} for r, v in Rr[k].items()}))
    if "I" in R and R["I"].get("mean_test"):
        md.append("\n## I. s-GNN: " + json.dumps(R["I"]["mean_test"]))
    return "\n".join(md)


if __name__ == "__main__":
    main()
