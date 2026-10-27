"""Step 13: gate G0 of benchmark B: the source papers' own pipelines on their own datasets.

* LAD (Huang et al., KDD 2020): the authors' code (data/raw/lad) on the authors' files and construction (weighted;
  directed Chung Laplacian on UCI and canVote), probe seed 0, the authors' two-window scoring; Hits@n = share of the
  events among the top round(p T) steps (exact index match).
* SCPD (Huang et al., PAKDD 2023; methods/scpd.py) on its own construction of the stored graphs: weighted, active
  vertices only (MAG, SCC), the fixed 3,000 airports with the MATLAB density-of-states path (FL); probe seed 0.
* s-GNN-I (Sulem et al., MLJ 2024; methods/sgnn.py) on PM with --sgnn: individual split per subject, reps 0-1,
  alarm rule of DyNNet's detect.py, adjusted F1 (+-5). Needs the DGL environment (repro/README.md); without --sgnn
  the cached result is reported and flagged.
Output: results/bench_g0.json and data/bench/g0_rows.npz (the score series of the G0 rows for step 12).

Usage: 13_bench_g0.py [--sgnn] [--python PATH] [--workers 4]
"""
from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LAD_DIR = REPO / "data" / "raw" / "lad"
BENCH = HERE / "data" / "bench"
CACHE = HERE / "data" / "bench_g0"
SGNN_PY = os.environ.get("SGNN_PYTHON", str(REPO / "data" / "raw" / "sgnn" / "venv" / "bin" / "python"))

# LAD Anomaly_Detection.py: USLegis(), UCI_Message(), canVote(); n = round(p T) ranked steps
LAD = {"USLegis": {"T": 12, "p": 0.20, "w1": 1, "w2": 2, "init": 2, "events": [3, 7],
                   "file": "USLegis_processed/LegisEdgelist.txt"},
       "UCI": {"T": 196, "p": 0.05, "w1": 7, "w2": 14, "init": 14, "events": [65, 158],
               "file": "UCI_processed/OCnodeslinks_chars.txt"},
       "canVote": {"T": 14, "p": 0.154, "w1": 2, "w2": 4, "init": 4, "events": [7, 9],
                   "file": "canVote_processed/canVote_edgelist.txt"}}
# SCPD own construction: (active vertices only, MATLAB DOS path, windows, percent ranked)
SCPD = {"MAG": (True, False, (5, 10), 0.05), "FL": (False, True, (2, 4), 0.07), "SCC": (True, False, (5, 10), 0.05)}
MAG_PAPER = [1864, 1900, 1934, 1938, 1944, 1945, 1947, 1950, 1958]       # SCPD Fig. 4 (years)
FL_PAPER = [11, 12]                                                      # SCPD Fig. 5 (week index)
SCC_PAPER = [33, 69, 76, 115, 121, 178, 183, 205]                        # SCPD Fig. 6 (day index from 2022-04-01)
SGNN_PAPER = [0.76, 0.91, 0.60, 0.73, 0.85, 0.74, 0.90, 0.72]           # s-GNN-I, individual split, subjects 101-108
TOL_PM, L_SGNN, SGNN_REPS = 5, 20, 2
# UCI: svds misses copies of the highly repeated eigenvalue 1.95 depending on round-off, so the spectrum depends on
# the BLAS thread count (one thread or all cores: ARPACK does not converge); two threads, the repository default
LAD_THREADS = "2"


# ------------------------------------------------------------------------------------------------ LAD (worker)
def lad_worker(ID: str, out: str) -> None:
    """Runs under `uv run --with scikit-learn python -I` (the authors' code needs scikit-learn). No file of the
    authors' repository is edited; three shims make it run on current libraries."""
    import types
    sys.dont_write_bytecode = True
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(LAD_DIR))
    sys.modules["datasets.SBM_loader"] = types.ModuleType("datasets.SBM_loader")     # file missing in the repo
    import datasets
    datasets.SBM_loader = sys.modules["datasets.SBM_loader"]
    try:
        import pylab  # noqa: F401
    except ImportError:                                                              # plotting only
        for m in ["pylab", "mpl_toolkits", "mpl_toolkits.axes_grid1"]:
            sys.modules[m] = types.ModuleType(m)
        sys.modules["mpl_toolkits.axes_grid1"].make_axes_locatable = None
    import compute_SVD
    import networkx as nx
    import scipy.sparse as sps
    from sklearn.preprocessing import normalize

    class CSR(sps.csr_matrix):                                                       # scipy >= 1.14 dropped asfptype
        def asfptype(self):
            return sps.csr_matrix(self.astype(np.float64))
    import Anomaly_Detection as AD  # after compute_SVD and scikit-learn: in the other order svds fails on UCI
    lap = nx.laplacian_matrix
    compute_SVD.nx = types.SimpleNamespace(**{k: getattr(nx, k) for k in dir(nx) if not k.startswith("__")})
    compute_SVD.nx.laplacian_matrix = lambda G, *a, **k: CSR(lap(G, *a, **k))
    from datasets import UCI_loader, USLegis_loader, canVote_loader

    np.random.seed(0)                                                                # svds start vector
    f = str(LAD_DIR / "datasets" / LAD[ID]["file"])
    if ID == "USLegis":
        eig, _ = compute_SVD.SVD_perSlice(USLegis_loader.load_legis_temporarl_edgelist(f), directed=False,
                                          num_eigen=6, top=True, max_size=102)
    elif ID == "UCI":
        eig, _ = compute_SVD.SVD_perSlice(UCI_loader.load_temporarl_edgelist(f, max_nodes=1901), directed=True,
                                          num_eigen=6, top=True, max_size=1901)
    else:
        G = canVote_loader.load_canVote_temporarl_edgelist(f)
        eig, _ = compute_SVD.SVD_perSlice(G, directed=True, num_eigen=338, top=True, max_size=max(len(g) for g in G))
    c = LAD[ID]
    spec = np.asarray(eig).real
    s = normalize(spec.reshape((c["T"], -1)), norm="l2")
    z = AD.change_detection_two_windows(s, principal=True, percent_ranked=c["p"], window1=c["w1"], window2=c["w2"],
                                        initial_window=c["init"], difference=True)[2]
    np.savez(out, spec=spec, z=np.asarray(z, float))


def lad_run(ID: str) -> dict:
    from common import sha  # main process only: the workers run in other environments
    src = [LAD_DIR / "datasets" / LAD[ID]["file"]] + sorted(LAD_DIR.glob("*.py")) + sorted(LAD_DIR.glob("*/*.py"))
    code = Path(__file__).read_text().split("def lad_worker")[1].split("def lad_run")[0]
    p = CACHE / f"lad_{ID}_{sha(*src, code, LAD[ID], LAD_THREADS)}.npz"
    if not p.exists():
        t0 = time.time()
        env = dict(os.environ, OMP_NUM_THREADS=LAD_THREADS, OPENBLAS_NUM_THREADS=LAD_THREADS,
                   MKL_NUM_THREADS=LAD_THREADS)
        subprocess.run(["uv", "run", "--with", "scikit-learn", "python", "-I", str(Path(__file__)), "_lad", ID,
                        str(p)], cwd=REPO, env=env, check=True)
        print(f"LAD {ID}: {time.time() - t0:.0f} s", flush=True)
    return dict(np.load(p))


def top_n(z, n: int) -> list[int]:
    return sorted(int(i) for i in np.asarray(z).argsort()[-n:][::-1])


# ------------------------------------------------------------------------------------------------ SCPD
def scpd_dos(ID: str, lo: int = 0, hi: int = 0):
    """Seed-0 DOS of snapshots [lo, hi) on SCPD's own construction. MAG / FL: one probe generator RandomState(0)
    over the series; SCC (processed in day chunks): RandomState(t) on day t."""
    import scipy.sparse as sp

    from methods import scpd
    active, matlab = SCPD[ID][:2]
    z = np.load(BENCH / f"{ID}.npz", allow_pickle=False)
    n, off = int(z["n"]), z["offsets"]
    rng = np.random.RandomState(0)
    out = []
    if ID == "SCC":
        a0, b0 = off[lo], off[hi]
        rows, cols, wts = z["rows"][a0:b0], z["indices"][a0:b0], z["weights"][a0:b0].astype(float)
        for t in range(lo, hi):
            a, b = off[t] - a0, off[t + 1] - a0
            r, c, w = rows[a:b], cols[a:b], wts[a:b]
            act, inv = np.unique(np.concatenate([r, c]), return_inverse=True)
            A = sp.csr_matrix((w, (inv[:len(r)], inv[len(r):])), shape=(len(act), len(act)))
            out.append(scpd.dos(A, rng=np.random.RandomState(t)))
        return np.array(out)
    ip, ix, wt = z["indptr"], z["indices"], z["weights"]
    for t in range(int(z["T"])):
        a, b = off[t], off[t + 1]
        A = sp.csr_matrix((wt[a:b].astype(float), ix[a:b], ip[t]), shape=(n, n))
        if active:
            keep = np.flatnonzero(np.diff(A.indptr))
            A = A[keep][:, keep]
        out.append(scpd.dos(A, rng=rng, matlab=matlab))
    return np.array(out)


def scpd_series(ID: str, workers: int):
    from multiprocessing import Pool

    from common import sha
    code = Path(__file__).read_text().split("def scpd_dos")[1].split("def scpd_series")[0]
    p = CACHE / f"dos_{ID}_{sha(BENCH / f'{ID}.npz', REPO / 'methods' / 'scpd.py', code)}.npy"
    if not p.exists():
        t0 = time.time()
        if ID == "SCC":                                   # independent days: 4 processes, < 2 GB each
            T = int(np.load(BENCH / "SCC.npz")["T"])
            cut = np.linspace(0, T, workers + 1).astype(int)
            with Pool(workers) as pool:
                D = np.concatenate(pool.starmap(scpd_dos, [(ID, int(a), int(b)) for a, b in itertools.pairwise(cut)]))
        else:
            D = scpd_dos(ID)
        np.save(p, D)
        print(f"SCPD DOS {ID}: {time.time() - t0:.0f} s", flush=True)
    return np.load(p)


# ------------------------------------------------------------------------------------------------ s-GNN (worker)
def sgnn_worker(job: str, out: str) -> None:
    """Runs under the DGL environment: one training on the train part, Z on the test part."""
    import scipy.sparse as sp
    sys.path.insert(0, str(REPO))
    from methods.sgnn import SGNN
    j = np.load(job)
    n, ip, ix, off = int(j["n"]), j["indptr"], j["indices"], j["offsets"]
    mats = [sp.csr_matrix((np.ones(off[t + 1] - off[t]), ix[off[t]:off[t + 1]], ip[t]), shape=(n, n))
            for t in range(len(ip))]
    cpf = np.zeros(len(mats), int)
    cpf[j["cps"]] = 1
    tr, va, te = j["train"], j["val"], j["test"]
    t0 = time.time()
    m = SGNN(seed=int(j["rep"]), workdir=out + ".work").fit([mats[i] for i in tr], cpf[tr], [mats[i] for i in va],
                                                         cpf[va])
    Z, idx = m.similarity([mats[i] for i in te])
    np.savez(out, Z=np.asarray(Z, float), idx=np.asarray(idx, int), seconds=time.time() - t0)
    shutil.rmtree(out + ".work", ignore_errors=True)                 # DyNNet's pickled graphs and checkpoints


def sgnn_runs(run: bool, python: str, workers: int) -> dict:
    """Adjusted F1 of s-GNN-I per (subject, rep). The split rng is default_rng([20261023, subject, rep])."""
    from concurrent.futures import ThreadPoolExecutor

    import evaluate as ev
    from common import sha

    from methods.sgnn import flags
    z = np.load(BENCH / "PM.npz", allow_pickle=False)
    seg, events, off = z["segment"], z["events"], z["offsets"]
    key = sha(BENCH / "PM.npz", REPO / "methods" / "sgnn.py",
              REPO / "data" / "raw" / "sgnn" / "DyNNet" / "src" / "train.py",
              Path(__file__).read_text().split("def sgnn_worker")[1].split("def sgnn_runs")[0])
    jobs, F1, Zs = [], np.full((8, SGNN_REPS), np.nan), {}
    for s, u in enumerate(np.unique(seg)):
        idx = np.nonzero(seg == u)[0]
        a, b = int(idx[0]), int(idx[-1]) + 1
        cps = [int(e) - a for e in events if a <= e < b]
        for r in range(SGNN_REPS):
            tr, va, te = ev.individual_split(b - a, cps, np.random.default_rng([20261023, s, r]))
            p = CACHE / f"sgnn_s{s}_r{r}_{key}.npz"
            if not p.exists() and run:
                jp = CACHE / f"sgnn_job_s{s}_r{r}.npz"
                np.savez(jp, n=int(z["n"]), indptr=z["indptr"][a:b], indices=z["indices"][off[a]:off[b]],
                         offsets=off[a:b + 1] - off[a], cps=np.array(cps), train=tr, val=va, test=te, rep=r)
                jobs.append((jp, p))
            Zs[(s, r)] = (p, cps, te, b - a)
    if jobs:
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   CUDA_VISIBLE_DEVICES="")
        t0 = time.time()
        with ThreadPoolExecutor(min(workers, 3)) as pool:            # three trainings at a time, one thread each
            list(pool.map(lambda jp: subprocess.run([python, "-I", str(Path(__file__)), "_sgnn", str(jp[0]),
                                                     str(jp[1])], env=env, check=True), jobs))
        print(f"s-GNN: {len(jobs)} trainings, {time.time() - t0:.0f} s", flush=True)
        for jp, _ in jobs:
            jp.unlink()
    seconds = []
    for (s, r), (p, cps, te, T) in Zs.items():
        if not p.exists():
            continue
        Zr = np.load(p)
        flg = np.zeros(T, int)
        flg[cps] = 1
        tgt = ev.cp_targets(np.cumsum(flg)[te], TOL_PM)
        F1[s, r] = ev.adjusted_f1(flags(Zr["Z"], L_SGNN), tgt, 0.5)
        seconds.append(float(Zr["seconds"]))
    return {"F1": F1, "seconds": seconds, "key": key}


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    import argparse

    import common
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgnn", action="store_true", help="train s-GNN-I (DGL environment) where no cache exists")
    ap.add_argument("--python", default=SGNN_PY, help="python of the DGL environment")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = common.Timer()
    out, rows = {}, {}

    lad = {}
    for ID, c in LAD.items():
        r = lad_run(ID)
        zr = np.round(r["z"], 6)                          # the G0 row keeps 6 decimals (score file of the G0 run)
        n = round(c["T"] * c["p"])
        top = top_n(r["z"], n)
        ev_sec = [int(e) for e in np.load(BENCH / f"{ID}.npz")["events_secondary"]] if ID == "canVote" else None
        lad[ID] = {"n": n, "events": c["events"], "top": top,
                   "hits": sum(e in top for e in c["events"]) / len(c["events"]),
                   "rank_of_events": [int((r["z"] > r["z"][e]).sum()) + 1 for e in c["events"]],
                   "top_rounded_z": top_n(zr, n)}
        if ev_sec:
            lad[ID]["events_secondary"] = ev_sec
            lad[ID]["hits_secondary"] = sum(e in top for e in ev_sec) / len(ev_sec)
        rows[f"lad_own_{ID}"] = zr
    out["lad"] = {"setting": "authors' code (github.com/shenyangHuang/LAD @ 58330238) on the authors' files; svds "
                             "start vector from np.random.seed(0); scoring change_detection_two_windows",
                  "datasets": lad}

    from methods import scpd
    sc = {}
    for ID, (_, _, win, pr) in SCPD.items():
        rows[f"scpd_own_{ID}"] = scpd.score(scpd_series(ID, a.workers), win, pr)[0]
    D = scpd_series("MAG", a.workers)
    an = scpd.score(D[0:-1][0:180], [5, 10], 0.05)[1]          # SCPD scores years 1837-2016 (spectrums[0:-1])
    yrs = [int(1837 + x) for x in an]
    sc["MAG"] = {"years": yrs, "paper_years": MAG_PAPER, "overlap_with_paper": len(set(yrs) & set(MAG_PAPER))}
    an = scpd.score(scpd_series("FL", a.workers)[0:-1], [2, 4], 0.07)[1].tolist()
    sc["FL"] = {"weeks": an, "paper_weeks": FL_PAPER, "overlap_with_paper": len(set(an) & set(FL_PAPER))}
    D = scpd_series("SCC", a.workers)
    an = scpd.score(D, [5, 10], 8 / len(D))[1].tolist()
    an_drop = scpd.score(D[:-1], [5, 10], 8 / (len(D) - 1))[1].tolist()
    sc["SCC"] = {"days": an, "paper_days": SCC_PAPER, "exact_overlap": len(set(an) & set(SCC_PAPER)),
                 "overlap_day_minus_1": len({x - 1 for x in an} & set(SCC_PAPER)),
                 "days_last_day_dropped": an_drop,
                 "overlap_day_minus_1_last_day_dropped": len({x - 1 for x in an_drop} & set(SCC_PAPER)),
                 "note": "SCPD states no windows or percent for SCC: windows 5/10 and its 8 detections are used"}
    out["scpd"] = {"setting": "methods/scpd.py, probe seed 0, own construction (weighted; active vertices on MAG "
                              "and SCC; fixed 3,000 airports and MATLAB DOS path on FL)", "datasets": sc}

    g = sgnn_runs(a.sgnn, a.python, a.workers)
    F1 = g["F1"]
    out["sgnn"] = {"setting": "s-GNN-I, DyNNet @ 2cd2035 defaults, individual split, reps 0-1, adjusted F1 (+-5)",
                   "ran_now": bool(a.sgnn), "n_runs": int(np.isfinite(F1).sum()),
                   "mean": float(np.nanmean(F1)) if np.isfinite(F1).any() else None,
                   "per_subject": np.nanmean(F1, 1).round(4).tolist() if np.isfinite(F1).any() else None,
                   "per_run": F1.round(4).tolist(), "paper_per_subject": SGNN_PAPER,
                   "paper_mean": float(np.mean(SGNN_PAPER)), "seconds_per_training": g["seconds"]}
    np.savez(BENCH / "g0_rows.npz", **{k: np.asarray(v, float) for k, v in rows.items()})
    out["_runtime_s"] = round(t0(), 1)
    common.write_json(common.RESULTS / "bench_g0.json", out)
    print("LAD hits", {k: v["hits"] for k, v in lad.items()}, "canVote secondary", lad["canVote"].get("hits_secondary"))
    print("SCPD", {k: v.get("overlap_with_paper", v.get("exact_overlap")) for k, v in sc.items()},
          "SCC day-1 overlap", sc["SCC"]["overlap_day_minus_1_last_day_dropped"])
    print("s-GNN-I", out["sgnn"]["n_runs"], "runs, mean adjusted F1", out["sgnn"]["mean"], out["sgnn"]["per_subject"])


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "_lad":
        lad_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "_sgnn":
        sgnn_worker(sys.argv[2], sys.argv[3])
    else:
        sys.path.insert(0, str(HERE))
        main()
