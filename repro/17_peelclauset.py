"""Step 17: Peel & Clauset's own constructions and events (MIT_PC, ENRON_PC), their GHRG detector, their Gaussian
baselines, and the event percentile of every benchmark row on them.

Data   repro/data/bench/{MIT_PC, ENRON_PC}.npz (step 10, data/downloaders/peel_clauset.py) and the step-11 rows
       (scores_<ID>.npz; run `11_bench_scores.py MIT_PC ENRON_PC` first).
GHRG   methods/ghrg.py: per window of w weeks the authors' fitHRG + consensusHRG (compiled from
       data/raw/peel_code/cpdetectr1.3.tar.gz into repro/data/pc/ghrg_bin), the authors' likelihood-ratio test and
       parametric bootstrap (1,000 draws). w = 4 (P&C on MIT and in their synthetic tests). One consensus tree per
       window is cached (the MCMC is not seeded).
Base   P&C's three baselines: Gaussian GLR + bootstrap on weekly mean degree, mean local clustering (both over the
       vertices with an edge that week, which is what P&C plot: check C4) and mean geodesic distance (connected
       pairs), w = 4.
Scores D_t from the window t-w .. t-1: the authors' level (share of null draws <= the best split gain; change when
       > 0.95); ghrg_lr is the best split gain itself. P&C's sequential protocol is replayed from the same windows.
EP     repro/evaluate.ep, tolerance 1 week (primary) and 2; (a) step-12 protocol: one common mask over every row,
       common burn-in = the largest row burn-in; (b) burn-in 4, each row where finite; (b') each row from its own
       burn-in. ENRON_PC also with the events at the dates of Peel's timeline file.
Checks C1 event weeks; C2 port = authors' Python (`--validate`, runs their changepointDetection.py under Python 3.8
       with dendropy 4.0.2 after a lib2to3 conversion); C3 null windows sampled from fitted HRGs; C4 our weekly
       summaries against the values P&C plot in Fig. 8 (blue steps read from the vector PDF with pymupdf).
Pooled re-report of results/bench.json (step 12) with the two series added, and with MIT / ENRON replaced by them.
Spec written before the run: prereg/specs/PEELCLAUSET.md, reproduced in results/peelclauset.json ("spec").
Output: results/peelclauset.json; `--summary` prints the tables.

Why (spec v5 P; --why, from results/peelclauset.json and the step-11 caches): P1 null windows per series; P0 the week
       mapping of our Enron to P&C's (correlation of weekly edge counts at offsets); P2 every step-12 row on our Enron
       and on P&C's, each with our events and with P&C's, at tol 1 and 4; P3 the count components and the 48 count
       settings on ENRON_PC; P4 per event on ENRON_PC.

Usage: 17_peelclauset.py [--validate] [--why] [--summary]
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import evaluate as ev
import networkx as nx
import numpy as np
from common import DATA, RESULTS, ROOT, Timer, sha, write_json
from rows import BENCH, PRIMARY, settings, stored

from methods import ghrg

IDS = ["MIT_PC", "ENRON_PC"]
PC = DATA / "pc"
BIN = PC / "ghrg_bin"
CACHE = PC / "cache"
SRC = ROOT.parent / "data" / "raw" / "peel_code"
BASE = {"pc_degree": "mean degree", "pc_clust": "mean local clustering", "pc_geo": "mean geodesic distance"}
PDF = SRC / "1403.0989v2.pdf"                                       # P&C, arXiv v2 (Fig. 8 on page 9)
TOLS = (1, 2)
NSHIFT = 1000
REPO_ROWS = PRIMARY                                                  # the pooled rows of step 12
# P&C Fig. 8, green bars (x in pt of the vector PDF, page 9): GHRG and the three baselines. Week = (x - x0) / dx;
# MIT weeks count from Mon 2004-09-06 (the copy's week 0 is the next one), Enron weeks from Mon 1999-05-10.
FIG8 = {"MIT_PC": {"x0": 88.36, "dx": 13.655, "offset": -1,
                   "ghrg": [88.36, 102.02, 115.67, 156.63, 170.29, 183.94, 197.59, 224.9, 252.21, 265.86, 279.51,
                            293.17, 320.48, 347.78, 375.09, 443.36, 470.67, 497.98],
                   "pc_clust": [375.09, 470.67], "pc_degree": [170.29, 320.48],
                   "pc_geo": [115.67, 129.32, 170.29, 361.44]},
        "ENRON_PC": {"x0": 72.33, "dx": 2.9499, "offset": 0,
                     "ghrg": [157.88, 261.13, 293.59, 317.19, 328.99, 367.34, 382.09, 390.94, 408.64, 426.34, 458.79,
                              467.64, 491.25, 494.2, 497.15, 503.05],
                     "pc_clust": [113.63, 151.98, 184.43, 193.28, 275.89, 317.19, 331.94, 361.44, 390.94, 429.29,
                                  467.64, 476.5, 511.9],
                     "pc_degree": [107.73, 146.08, 181.48, 293.59, 331.94, 390.94, 432.24, 464.69, 514.85],
                     "pc_geo": [101.83, 181.48, 184.43, 240.48, 346.69, 361.44, 405.69, 494.2, 532.55]}}


def fig8_weeks(ID: str, key: str) -> list[int]:
    f = FIG8[ID]
    return sorted({round((x - f["x0"]) / f["dx"]) + f["offset"] for x in f[key]})


# ------------------------------------------------------------------------------------------------ inputs
def series(ID: str) -> dict:
    with np.load(BENCH / f"{ID}.npz", allow_pickle=False) as z:
        d = {"T": int(z["T"]), "n": int(z["n"]), "seg": z["segment"].astype(int), "events": [int(e) for e in z["events"]],
             "events_timeline": [int(e) for e in z["events_timeline"]] if "events_timeline" in z.files else None,
             "meta": json.loads(str(z["meta_json"]))}
    d["edges"] = [np.argwhere(np.triu(A.toarray(), 1)) for A in stored(ID)]
    return d


# ------------------------------------------------------------------------------------------------ GHRG
def tree_of(job) -> tuple[str, str]:
    key, graphs, n = job
    p = CACHE / f"ghrg_{key}.tree"
    if not p.exists():
        if sum(len(E) for E in graphs) == 0:
            p.write_text("")
        else:
            p.write_text(ghrg.fit_consensus(graphs, n, BIN))
    return key, p.read_text()


def build_bin() -> None:
    """Compile the authors' two C++ programs once (their Makefile flags are obsolete for current g++)."""
    if all((BIN / d / d.split("_")[0]).exists() for d in ("fitHRG_GPL_Bayes", "consensusHRG_GPL_Bayes")):
        return
    with tempfile.TemporaryDirectory() as t:
        subprocess.run(["tar", "-xzf", str(SRC / "cpdetectr1.3.tar.gz"), "-C", t], check=True)
        for d in ("fitHRG_GPL_Bayes", "consensusHRG_GPL_Bayes"):
            (BIN / d).mkdir(parents=True, exist_ok=True)
            for f in list((Path(t) / "release" / d).glob("*.cc")) + list((Path(t) / "release" / d).glob("*.h")):
                if not f.name.startswith("._"):
                    shutil.copy(f, BIN / d / f.name)
            exe = d.split("_")[0]
            subprocess.run(["g++", "-O3", "-o", exe, f"{exe}.cc"], cwd=BIN / d, check=True, capture_output=True)


def ghrg_windows(ID: str, d: dict, w: int) -> dict:
    """{s: test of the window s .. s+w-1} for s = 0 .. T-w-1 (the windows of the score series and of the replay)."""
    T, n = d["T"], d["n"]
    jobs = {s: (ghrg.window_key(d["edges"][s:s + w], n, w, BIN), d["edges"][s:s + w], n) for s in range(T - w)}
    todo = [j for j in jobs.values() if not (CACHE / f"ghrg_{j[0]}.tree").exists()]
    if todo:
        t0 = Timer()
        with Pool(4) as pool:
            for i, _ in enumerate(pool.imap_unordered(tree_of, todo)):
                if i % 20 == 0:
                    print(f"{ID} w={w}: {i + 1}/{len(todo)} consensus trees, {t0():.0f} s", flush=True)
    out = {}
    for s, (key, graphs, _) in jobs.items():
        txt = (CACHE / f"ghrg_{key}.tree").read_text()
        if not txt.strip():
            out[s] = {"level": np.nan, "stat": np.nan, "change": False, "levels": np.array([np.nan])}
            continue
        lp, npar = ghrg.parse_tree(txt, n)
        lca, ni = ghrg.lca_and_pairs(lp, npar)
        e = ghrg.node_counts(graphs, lca, len(ni))
        out[s] = ghrg.window_test(e, ni, np.random.default_rng([ev.SEED, IDS.index(ID), w, s]))
        out[s]["n_internal"] = len(ni)
    return out


# ------------------------------------------------------------------------------------------------ baselines
def summaries(d: dict) -> dict:
    """Weekly summaries over the vertices with an edge that week (what P&C plot, check C4)."""
    x = {k: np.full(d["T"], np.nan) for k in BASE}
    for t, E in enumerate(d["edges"]):
        if not len(E):
            continue
        G = nx.Graph(list(map(tuple, E)))
        x["pc_degree"][t] = 2 * len(E) / G.number_of_nodes()
        x["pc_clust"][t] = nx.average_clustering(G)
        tot = cnt = 0
        for _, dist in nx.all_pairs_shortest_path_length(G):
            tot += sum(dist.values())
            cnt += len(dist) - 1
        x["pc_geo"][t] = tot / cnt if cnt else np.nan
    return x


def base_windows(ID: str, x: np.ndarray, k: int, w: int = 4) -> dict:
    return {s: ghrg.gaussian_window(x[s:s + w], np.random.default_rng([ev.SEED, IDS.index(ID), 100 + k, s]))
            for s in range(len(x) - w) if np.isfinite(x[s:s + w]).all()}


# ------------------------------------------------------------------------------------------------ evaluation
def hits(det, events, tol) -> dict:
    """Event recall (a detection in [e, e + tol]) and precision (a detection with an event in [d - tol, d])."""
    det = sorted(det)
    rec = [e for e in events if any(e <= x <= e + tol for x in det)]
    prec = [x for x in det if any(e <= x <= e + tol for e in events)]
    return {"n_detections": len(det), "recall": len(rec) / len(events) if events else None,
            "precision": len(prec) / len(det) if det else None, "events_hit": rec}


def ep_block(S: dict, rows, seg, events, all_events, tol, mask_of, shifts: bool) -> dict:
    out = {}
    for m in rows:
        mask = mask_of(m)
        v, nnull = ev.ep(S[m], mask, seg, events, tol, all_events=all_events)
        out[m] = {"ep_mean": float(np.mean(list(v.values()))) if v and nnull else None,
                  "ep_per_event": {int(e): round(x, 4) for e, x in v.items()} if nnull else {},
                  "n_events": len(v), "n_null_windows": nnull}
    if shifts:
        mask = mask_of(rows[0])
        null = ev.shift_null(S, rows, mask, seg, events, tol, np.random.default_rng([ev.SEED, len(events), tol]),
                             NSHIFT)
        for m in rows:
            if out[m]["ep_mean"] is not None:
                out[m]["shift_p"] = ev.shift_p(null[m], out[m]["ep_mean"])
    return out


def evaluate_id(ID: str, d: dict, S: dict, burn: dict) -> dict:
    seg = d["seg"]
    rows = list(S)
    common = ev.common_mask(S, burn, rows, seg)
    res = {"T": d["T"], "n": d["n"], "events": d["events"], "common_burn": max(burn[m] for m in rows),
           "n_common_mask": int(common.sum())}
    variants = {"fig8": d["events"]}
    if d["events_timeline"]:
        variants["timeline"] = d["events_timeline"]
    for name, evs in variants.items():
        for tol in TOLS:
            k = f"{name}_tol{tol}"
            res[f"a_{k}"] = ep_block(S, rows, seg, evs, evs, tol, lambda m: common, shifts=True)
            res[f"b_{k}"] = ep_block(S, rows, seg, evs, evs, tol, lambda m: ev.scored_mask(seg, S[m], 4), False)
            res[f"b_own_{k}"] = ep_block(S, rows, seg, evs, evs, tol,
                                         lambda m: ev.scored_mask(seg, S[m], max(4, burn[m])), False)
    return res


def pooled(bench: dict, pc: dict) -> dict:
    """Pooled EP (equal weight per dataset, bootstrap CI of step 12) of the step-12 rows: as published, plus the two
    P&C series (protocol a, Fig. 8 events, tol 1), and with MIT / ENRON replaced by them."""
    base = {d: {m: np.array(bench["datasets"][d]["m2"]["ep_per_event"][m]) for m in PRIMARY}
            for d in bench["pooled_datasets"]}
    new = {}
    for ID in IDS:
        blk = pc[ID]["a_fig8_tol1"]
        if all(blk[m]["ep_mean"] is not None for m in PRIMARY):
            evs = sorted(blk["counts"]["ep_per_event"], key=int)
            new[ID] = {m: np.array([blk[m]["ep_per_event"][e] for e in evs]) for m in PRIMARY}
    sets = {"published": base, "plus_pc": {**base, **new},
            "pc_replaces_mit_enron": {**{k: v for k, v in base.items() if k not in ("MIT", "ENRON")}, **new}}
    out = {}
    for name, ds in sets.items():
        blk = {"datasets": sorted(ds), "ep": {}, "counts_minus": {}}
        for m in PRIMARY:
            b = ev.boot_pooled({k: ds[k][m] - 0.5 for k in ds})
            blk["ep"][m] = {"mean": b["mean"] + 0.5, "lo": b["lo"] + 0.5, "hi": b["hi"] + 0.5}
            if m != "counts":
                c = ev.boot_pooled({k: ds[k]["counts"] - ds[k][m] for k in ds})
                blk["counts_minus"][m] = {k: c[k] for k in ("mean", "lo", "hi")}
        out[name] = blk
    return out


# ------------------------------------------------------------------------------------------------ checks
FIG8_SCRIPT = """
import json, sys, fitz
page = fitz.open(sys.argv[1])[8]
out = []
for d in page.get_drawings():
    c = d.get("color")
    if c and tuple(round(v, 2) for v in c) == (0.0, 0.0, 1.0):
        out += [[it[1].x, it[2].x, it[1].y] for it in d["items"] if it[0] == "l"]
json.dump(out, open(sys.argv[2], "w"))
"""
FIG8_PANELS = {"MIT_PC": (185, 305), "ENRON_PC": (525, 645)}         # y range (pt) of the three summary panels


def fig8_values(ID: str, offset: int) -> dict:
    """P&C's plotted weekly clustering / mean degree / geodesic (blue steps of Fig. 8), as -y, by our week index."""
    p = PC / "fig8_blue.json"
    if not p.exists():
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "x.py").write_text(FIG8_SCRIPT)
            subprocess.run(["uv", "run", "--no-project", "--with", "pymupdf", "python", "-I", str(Path(t) / "x.py"),
                            str(PDF), str(p)], check=True, capture_output=True)
    f, (lo, hi) = FIG8[ID], FIG8_PANELS[ID]
    by: dict = {}
    for a, b, y in json.loads(p.read_text()):
        if lo <= y <= hi and abs((b - a) - f["dx"]) < 0.3 * f["dx"]:
            by.setdefault(round((a - f["x0"]) / f["dx"]) + offset, []).append(y)
    return {k: -np.sort(v) for k, v in by.items() if len(v) == 3}   # top to bottom: clustering, degree, geodesic


def fidelity(ID: str, x: dict, T: int) -> dict:
    """C4: Pearson r between our weekly summaries and P&C's plotted ones, at plot offsets -2 .. +2 weeks."""
    out = {}
    for off in range(-2, 3):
        v = fig8_values(ID, FIG8[ID]["offset"] + off)
        ks = [k for k in sorted(v) if 0 <= k < T]
        r = {}
        for j, m in enumerate(("pc_clust", "pc_degree", "pc_geo")):
            a, b = np.array([v[k][j] for k in ks]), x[m][ks]
            ok = np.isfinite(b)
            r[m] = round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3)
        out[f"offset_{off:+d}"] = {"weeks": len(ks), **r}
    best = max(out, key=lambda k: out[k]["pc_degree"])
    return {"by_offset": out, "best_offset_mean_degree": best, "pass": out[best]["pc_degree"] > 0.9}


def null_check(ID: str, d: dict, nwin: int = 20) -> dict:
    """C3: windows of w = 4 graphs drawn from the HRG fitted to an observed window, refitted and tested."""
    n, w = d["n"], 4
    starts = np.linspace(0, d["T"] - w - 1, nwin).astype(int)
    jobs, meta = [], []
    for s in starts:
        txt = (CACHE / f"ghrg_{ghrg.window_key(d['edges'][s:s + w], n, w, BIN)}.tree").read_text()
        if not txt.strip():
            continue
        lp, npar = ghrg.parse_tree(txt, n)
        lca, ni = ghrg.lca_and_pairs(lp, npar)
        E = ghrg.node_counts(d["edges"][s:s + w], lca, len(ni)).sum(0)
        rng = np.random.default_rng([ev.SEED, IDS.index(ID), 999, s])
        q = np.clip(2 * E / (w * np.maximum(ni, 1)), 0, 1)   # fitted density per unordered pair at each node
        iu = np.triu_indices(n, 1)
        graphs = [np.c_[iu[0], iu[1]][rng.random(len(iu[0])) < q[lca[iu]]] for _ in range(w)]
        key = "null_" + ghrg.window_key(graphs, n, w, BIN)
        jobs.append((key, graphs, n))
        meta.append(int(s))
    with Pool(4) as pool:
        trees = dict(pool.imap_unordered(tree_of, jobs))
    lv = []
    for (key, graphs, _), s in zip(jobs, meta):
        lp, npar = ghrg.parse_tree(trees[key], n)
        lca, ni = ghrg.lca_and_pairs(lp, npar)
        r = ghrg.window_test(ghrg.node_counts(graphs, lca, len(ni)), ni, np.random.default_rng([ev.SEED, 7, s]))
        lv.append(r["level"])
    lv = np.array(lv)
    return {"windows": meta, "levels": lv.round(4).tolist(), "false_detection_rate": float(np.mean(lv > ghrg.ALPHA)),
            "pass": bool(np.mean(lv > ghrg.ALPHA) <= 0.10)}


def validate() -> None:
    """C2: the authors' changepointDetection.py (lib2to3 under Python 3.8, dendropy 4.0.2) on two cached windows."""
    out = {}
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        subprocess.run(["tar", "-xzf", str(SRC / "cpdetectr1.3.tar.gz"), "-C", str(t)], check=True)
        code = t / "release" / "changepointDetection.py"
        uv = ["uv", "run", "--no-project", "--python", "3.8", "--with", "dendropy==4.0.2", "--with", "numpy",
              "--with", "scipy", "python"]
        subprocess.run(uv[:-1] + ["python", "-m", "lib2to3", "-w", "-n", str(code)], check=True, capture_output=True)
        code.write_text(code.read_text().replace("leafGenerator.next()", "next(leafGenerator)"))
        driver = t / "drive.py"
        driver.write_text(
            "import sys, os, numpy as np\nsys.path.insert(0, sys.argv[1])\nimport changepointDetection as cd\n"
            "wd, nsamp, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]\nos.chdir(wd)\n"
            "open('nProcessors.txt', 'w').write('1\\n')\nad = cd.anomalyDetection()\nad.importNames('names.lut')\n"
            "ad.nsamples = nsamp\nfiles = ['g%02d.pairs' % k for k in range(4)]\n"
            "ad.consensusFilename = 'tree.tree'\nad.currentNormalNetworks = files\nad.createNullModel('.')\n"
            "ad.testNetwork('.', files)\nnp.savez(out, test=ad.test_model, null=ad.null_model)\n")
        for ID, s, nsamp in (("MIT_PC", 4, 200), ("ENRON_PC", 120, 100)):
            d = series(ID)
            graphs = d["edges"][s:s + 4]
            txt = (CACHE / f"ghrg_{ghrg.window_key(graphs, d['n'], 4, BIN)}.tree").read_text()
            wd = t / ID
            wd.mkdir()
            (wd / "names.lut").write_text("virtual\treal\n" + "".join(f"{i}\t{i}\n" for i in range(d["n"])))
            for k, E in enumerate(graphs):
                (wd / f"g{k:02d}.pairs").write_text("".join(f"{a}\t{b}\n" for a, b in E))
            (wd / "tree.tree").write_text(txt)
            t0 = Timer()
            subprocess.run(uv + [str(driver), str(code.parent), str(wd), str(nsamp), str(wd / "a.npz")], check=True,
                           capture_output=True)
            a = np.load(wd / "a.npz")
            lp, npar = ghrg.parse_tree(txt, d["n"])
            lca, ni = ghrg.lca_and_pairs(lp, npar)
            e = ghrg.node_counts(graphs, lca, len(ni))
            g = ghrg.split_gains(e, ni)
            nl = ghrg.null_draws(e, ni, np.random.default_rng([ev.SEED, 5, s]), 2000)
            from scipy.stats import ks_2samp
            ks = ks_2samp(a["null"], nl)
            out[ID] = {"window_start": s, "authors_test": a["test"].tolist(), "port_test": g.tolist(),
                       "max_abs_diff_test": float(np.max(np.abs(a["test"] - g))), "authors_null_draws": nsamp,
                       "port_null_draws": 2000, "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
                       "authors_seconds": round(t0(), 1),
                       "pass": bool(np.max(np.abs(a["test"] - g)) < 1e-6 and ks.pvalue > 0.01)}
            print(ID, out[ID], flush=True)
    write_json(PC / "validation.json", out)


# ------------------------------------------------------------------------------------------------ main
def run() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    build_bin()
    st = settings()
    burn = dict(st["burn"])
    out = {"spec": "prereg/specs/PEELCLAUSET.md (written before the run); summary in the module docstring",
           "datasets": {}, "detections": {}, "summaries": {}, "checks": {}}
    for ID in IDS:
        d = series(ID)
        T = d["T"]
        z = np.load(BENCH / f"scores_{ID}.npz")
        S = {r: z[f"s_{r}"] for r in REPO_ROWS}
        det = {}
        res = ghrg_windows(ID, d, 4)
        S["ghrg"] = ghrg.score_series(res, T, 4)
        S["ghrg_lr"] = ghrg.score_series(res, T, 4, "stat")
        burn["ghrg"] = burn["ghrg_lr"] = 4
        det["ghrg"] = ghrg.sequential(res, T, 4)
        out["summaries"][f"{ID}_ghrg_levels"] = [round(float(res[s]["level"]), 4) for s in sorted(res)]
        x = summaries(d)
        for k, m in enumerate(BASE):
            res = base_windows(ID, x[m], k)
            S[m] = ghrg.score_series(res, T, 4)
            burn[m] = 4
            det[m] = ghrg.sequential({s: res.get(s, {"change": False}) for s in range(T - 4)}, T, 4)
            out["summaries"][f"{ID}_{m}"] = np.round(x[m], 4).tolist()
        out["checks"][f"C4_{ID}"] = fidelity(ID, x, T)
        evs = d["events"]
        out["detections"][ID] = {
            "replay": {m: {"weeks": v, **{f"tol{t}": hits(v, evs, t) for t in TOLS}} for m, v in det.items()},
            "published_fig8": {m: {"weeks": fig8_weeks(ID, m), **{f"tol{t}": hits(fig8_weeks(ID, m), evs, t)
                                                                  for t in TOLS}}
                               for m in ("ghrg", "pc_degree", "pc_clust", "pc_geo")}}
        if ID == "ENRON_PC":                         # C4: P&C's Enron steps sit one week left of our weeks
            out["detections"][ID]["published_fig8_plus1"] = {
                m: {"weeks": [k + 1 for k in fig8_weeks(ID, m)],
                    **{f"tol{t}": hits([k + 1 for k in fig8_weeks(ID, m)], evs, t) for t in TOLS}}
                for m in ("ghrg", "pc_degree", "pc_clust", "pc_geo")}
        r = evaluate_id(ID, d, S, burn)
        r["meta"] = {k: d["meta"][k] for k in d["meta"] if k not in ("week_start_dates",)}
        r["week_start_dates"] = d["meta"]["week_start_dates"]
        out["datasets"][ID] = r
        out["checks"][f"C1_{ID}"] = {"events_stated": d["meta"]["n_events_stated_by_pc"],
                                     "events_listed": len(d["meta"]["events"]), "distinct_weeks_in_series": len(evs),
                                     "pass": len(d["meta"]["events"]) == d["meta"]["n_events_stated_by_pc"]}
        print(ID, {m: round(v["ep_mean"], 3) for m, v in r["a_fig8_tol1"].items() if v["ep_mean"] is not None},
              flush=True)
        out["checks"][f"C3_{ID}"] = null_check(ID, d)
    vp = PC / "validation.json"
    out["checks"]["C2"] = json.loads(vp.read_text()) if vp.exists() else "not run (17_peelclauset.py --validate)"
    bp = RESULTS / "bench.json"
    if bp.exists():
        out["pooled"] = pooled(json.loads(bp.read_text()), out["datasets"])
        out["pooled"]["bench_json_sha"] = sha(bp)
    return out


# ------------------------------------------------------------------------------------------------ why (spec v5 P)
OFFSET = 32          # our ENRON index t = P&C week t + 32 (the index-to-date assumed in data/downloaders/repo_view.py)


def _fill(x, mask):
    x = np.asarray(x, float)
    return np.where(np.isfinite(x), x, np.min(x[mask & np.isfinite(x)]))


def why(R: dict) -> dict:
    """Why the counts trail on P&C's series (spec v5 P): P1 resolution, P0 the week mapping, P2 event list x
    construction x tolerance on Enron, P3 count components and windows on ENRON_PC, P4 per event on ENRON_PC."""
    from common import MODES, WR, W, load, mad_scale
    from rows import meta
    burn = settings()["burn"]
    out = {"P1": {ID: {k: {"n_events": R["datasets"][ID][k]["counts"]["n_events"],
                           "n_null_windows": R["datasets"][ID][k]["counts"]["n_null_windows"],
                           "ep_counts": R["datasets"][ID][k]["counts"]["ep_mean"]}
                       for k in ("a_fig8_tol1", "a_fig8_tol2")} | {"T": R["datasets"][ID]["T"],
                                                                   "event_weeks": len(R["datasets"][ID]["events"])}
                  for ID in IDS}}
    pc = series("ENRON_PC")
    with np.load(BENCH / "ENRON.npz", allow_pickle=False) as z:
        pc_on_ours = [int(e) for e in z["events_secondary"]]
    ours = meta("ENRON")
    e_ours = np.log1p(np.asarray(load("enron")["f"]["edges_noloop"], float))
    e_pc = np.log1p(np.array([len(E) for E in pc["edges"]], float))
    out["P0"] = {}
    for o in range(-3, 4):
        t = np.arange(len(e_ours))
        k = t - OFFSET + o
        ok = (k >= 0) & (k < len(e_pc))
        out["P0"][o] = {"r": float(np.corrcoef(e_ours[t[ok]], e_pc[k[ok]])[0, 1]), "weeks": int(ok.sum())}
    out["P0_best_offset"] = max(out["P0"], key=lambda o: out["P0"][o]["r"])
    con = {"ours": (np.load(BENCH / "scores_ENRON.npz"), ours["seg"], {"ours": ours["events"], "pc": pc_on_ours}),
           "pc": (np.load(BENCH / "scores_ENRON_PC.npz"), pc["seg"],
                  {"pc": pc["events"], "ours": [e - OFFSET for e in ours["events"] if 0 <= e - OFFSET < pc["T"]]})}
    out["P2"] = {}
    for c, (z, seg, evsets) in con.items():
        S = {r: z[f"s_{r}"] for r in PRIMARY}
        mask = ev.common_mask(S, burn, PRIMARY, seg)
        for en, evs in evsets.items():
            for tol in (1, 4):
                res = {r: ev.ep(S[r], mask, seg, evs, tol) for r in PRIMARY}
                ep = {r: float(np.mean(list(v.values()))) if v and n else None for r, (v, n) in res.items()}
                oth = {r: x for r, x in ep.items() if r not in ("counts", "constant") and x is not None}
                best = max(oth, key=oth.get)
                out["P2"][f"construction_{c}__events_{en}__tol{tol}"] = {
                    "ep": ep, "n_events": len(res["counts"][0]), "n_null_windows": res["counts"][1],
                    "counts_rank": 1 + sum(x > ep["counts"] for x in oth.values()), "best_other": [best, oth[best]]}
    S = {r: con["pc"][0][f"s_{r}"] for r in PRIMARY}
    seg = pc["seg"]
    mask = ev.common_mask(S, burn, PRIMARY, seg)
    C = np.array([[np.unique(E).size, len(E)] for E in pc["edges"]], float)
    Z = mad_scale(C, seg)
    epm = lambda x, evs=pc["events"]: float(np.mean(list(ev.ep(_fill(x, mask), mask, seg, evs, 1)[0].values())))
    grid = {f"{w},{wr},{m}": epm(ev.window_scores(Z, seg, w, wr, m, start=w + wr)) for w in W for wr in WR for m in MODES}
    g = np.array(list(grid.values()))
    out["P3"] = {"nplus_only": epm(ev.window_scores(Z[:, 0], seg, 2, 5, "cen", start=7)),
                 "edges_only": epm(ev.window_scores(Z[:, 1], seg, 2, 5, "cen", start=7)),
                 "counts_fixed_recomputed": epm(ev.window_scores(Z, seg, 2, 5, "gam", start=7)),
                 "grid_48": {"median": float(np.median(g)), "min": float(g.min()), "max": float(g.max()),
                             "argmax": max(grid, key=grid.get)},
                 "share_active": {"median": float(np.median(C[:, 0] / pc["n"])),
                                  "first20": float(np.median(C[:20, 0] / pc["n"])),
                                  "last20": float(np.median(C[-20:, 0] / pc["n"]))}}
    desc = {}
    for e in pc["meta"]["events"]:
        desc.setdefault(int(e["index"]), []).append(e["desc"])
    v = {r: ev.ep(S[r], mask, seg, pc["events"], 1)[0] for r in ("counts", "cusum", "scpd", "tire", "lad")}
    out["P4"] = [{"week": e, "desc": "; ".join(desc.get(e, [])), **{r: round(v[r][e], 3) for r in v}}
                 for e in sorted(v["counts"])]
    return out


# ------------------------------------------------------------------------------------------------ summary
def f(v, d=2):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def summary(R: dict) -> str:
    md = []
    for ID, r in R["datasets"].items():
        md.append(f"## {ID}: T {r['T']}, n {r['n']}, event weeks {len(r['events'])}, common burn-in "
                  f"{r['common_burn']}\n")
        keys = [k for k in r if k.startswith(("a_", "b_"))]
        rows = list(r[keys[0]])
        md.append("| row | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 1))
        for m in rows:
            md.append(f"| {m} | " + " | ".join(
                f"{f(r[k][m]['ep_mean'])} (n={r[k][m]['n_events']}, null={r[k][m]['n_null_windows']}"
                + (f", p={f(r[k][m].get('shift_p'), 3)}" if "shift_p" in r[k][m] else "") + ")" for k in keys) + " |")
        md.append("")
        D = R["detections"][ID]
        md.append("| detector | replay weeks | recall/precision tol1 | P&C Fig. 8 weeks | recall/precision tol1 |\n"
                  "|---|---|---|---|---|")
        for m in D["replay"]:
            p = D["published_fig8"].get(m, {})
            md.append(f"| {m} | {D['replay'][m]['weeks']} | {f(D['replay'][m]['tol1']['recall'])} / "
                      f"{f(D['replay'][m]['tol1']['precision'])} | {p.get('weeks', '-')} | "
                      f"{f(p.get('tol1', {}).get('recall'))} / {f(p.get('tol1', {}).get('precision'))} |")
        md.append("")
    for k, v in R["checks"].items():
        md.append(f"- {k}: {json.dumps(v)[:400]}")
    if "why" in R:
        Y = R["why"]
        md.append("\n### Why the counts trail (spec v5 P)\n")
        md.append("P1 resolution: " + json.dumps(Y["P1"]))
        md.append(f"P0 week mapping (best offset {Y['P0_best_offset']}): "
                  + json.dumps({o: round(v["r"], 3) for o, v in Y["P0"].items()}))
        md.append("| cell | n events | null windows | counts | rank of counts | best other | " + " | ".join(PRIMARY)
                  + " |\n|---|---|---|---|---|---|" + "---|" * len(PRIMARY))
        for k, v in Y["P2"].items():
            md.append(f"| {k} | {v['n_events']} | {v['n_null_windows']} | {f(v['ep']['counts'], 3)} | "
                      f"{v['counts_rank']} | {v['best_other'][0]} {f(v['best_other'][1], 3)} | "
                      + " | ".join(f(v["ep"][r], 3) for r in PRIMARY) + " |")
        md.append("P3 (ENRON_PC, P&C events, tol 1): " + json.dumps(Y["P3"]))
        md.append("P4 per event (ENRON_PC, tol 1): " + "; ".join(
            f"w{x['week']} {x['desc'][:40]}: " + " ".join(f"{r} {x[r]:.2f}" for r in ("counts", "cusum", "scpd",
                                                                                        "tire", "lad"))
            for x in Y["P4"]))
    if "pooled" in R:
        for name, blk in R["pooled"].items():
            if name == "bench_json_sha":
                continue
            md.append(f"\n### pooled EP, {name}: {', '.join(blk['datasets'])}\n")
            md.append("| row | EP [95% CI] | counts - row [95% CI] |\n|---|---|---|")
            for m, v in blk["ep"].items():
                c = blk["counts_minus"].get(m)
                md.append(f"| {m} | {f(v['mean'], 3)} [{f(v['lo'], 3)}, {f(v['hi'], 3)}] | "
                          + (f"{f(c['mean'], 3)} [{f(c['lo'], 3)}, {f(c['hi'], 3)}]" if c else "-") + " |")
    return "\n".join(md)


def main() -> None:
    p = RESULTS / "peelclauset.json"
    if "--validate" in sys.argv:
        validate()
        return
    if "--why" in sys.argv:                       # reads results/peelclauset.json and the step-11 caches only
        R = json.loads(p.read_text())
        R["why"] = why(R)
        write_json(p, R)
        print(summary(R))
        return
    if "--summary" in sys.argv and p.exists():
        print(summary(json.loads(p.read_text())))
        return
    t0 = Timer()
    out = run()
    write_json(p, out)
    print(f"step 17: {t0():.0f} s -> {p}")
    print(summary(json.loads(p.read_text())))


if __name__ == "__main__":
    main()
