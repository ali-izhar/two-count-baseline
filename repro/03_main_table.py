"""Step 3: tuned F1 on the selection series of the counts and constant (5,040 settings) and SCPD, LADdos (41,400).

The fixed setting of the counts (the highest mean F1 over MIT, Enron and AskUbuntu) is the count detector of the
benchmark (rows.settings). The constant score alarms every q steps; its F1 is the floor of the alarm schedule.

SCPD and LADdos turn the same KPM moments into a Jackson-damped density histogram and score how far each snapshot's
histogram is from the principal direction of the recent ones (1 - |cos|); SCPD takes the first difference of the
larger of the two context scores, LADdos the larger positive first difference.
Output: repro/results/main_table.json and a printed table.

Events: the selection events (config.SELECTION_CPS for MIT); --corrected uses the corrected MIT events and writes
results/main_table_corrected.json.

Usage: 03_main_table.py [--summary] [--corrected]
"""
from __future__ import annotations

import itertools
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from common import (
    CD,
    LABEL,
    LW,
    NAMES,
    NB,
    NM,
    RESULTS,
    SEEDS,
    SW,
    THR_SCPD,
    TOL,
    Timer,
    count_vectors,
    f1_score,
    grid,
    load,
    run_grid,
    summarize,
    write_json,
)

from config import selection_events
from methods.scpd import jackson

SUFFIX = "_corrected" if "--corrected" in sys.argv else ""



PAPER = {  # stored check values of this step's tuned F1
    "counts": {"mit_reality": .71, "askubuntu": 1.00, "enron": .57, "mean_best": .76, "fixed": .55, "transfer": .28},
    "constant": {"mit_reality": .32, "askubuntu": 1.00, "enron": .27, "mean_best": .53, "fixed": .49,
                 "transfer": .30},
    "scpd": {"mit_reality": .54, "askubuntu": .70, "enron": .59, "mean_best": .61, "fixed": .42, "transfer": .12},
    "laddos": {"mit_reality": .52, "askubuntu": .80, "enron": .67, "mean_best": .66, "fixed": .42,
               "transfer": None},
}


def grid_scpd() -> list[tuple]:
    return [(nm, nb, sw, lw, q, k, v) for nm, nb, sw, lw in itertools.product(NM, NB, SW, LW)
            for q in CD for k, v in THR_SCPD]


def signatures(M: np.ndarray, nm: int, nb: int) -> np.ndarray:
    N = nm + 1
    mu = M[:, :N] * jackson(np.ones(N))
    edges = np.linspace(-1, 1, nb + 1)
    x = np.clip(0.5 * (edges[:-1] + edges[1:]), -0.999, 0.999)
    Tk = np.cos(np.outer(np.arange(N), np.arccos(x)))
    h = (mu * np.r_[1.0, 2.0 * np.ones(N - 1)]) @ Tk / (np.pi * np.sqrt(1 - x ** 2))
    h = np.maximum(h, 0)
    s = h.sum(1, keepdims=True)
    h = np.where(s > 1e-12, h / np.where(s > 1e-12, s, 1), h)
    nrm = np.linalg.norm(h, axis=1, keepdims=True)
    return np.where(nrm > 1e-12, h / np.where(nrm > 1e-12, nrm, 1), h)


def zwin(sig: np.ndarray, c: int, first: int) -> np.ndarray:
    Z = np.zeros(len(sig))
    for t in range(first, len(sig)):
        _, S, Vt = np.linalg.svd(sig[t - c:t], full_matrices=False)
        if S[0] < 1e-12:
            continue
        ns = np.linalg.norm(sig[t])
        Z[t] = 1 - (abs(sig[t] @ Vt[0]) / ns if ns > 1e-12 else 0.0)
    return Z


def scpd_alarms(s: np.ndarray, th: float, q: int, start: int) -> list[int]:
    out, last = [], -10 ** 9
    for t in np.nonzero(s >= th)[0]:
        if t >= start and t - last >= q:
            out.append(int(t))
            last = t
    return out


def task_scpd(name: str, seed: str) -> tuple:
    d = load(name)
    M, cps, tol = d["mom"][seed], selection_events(name, d["cps"], bool(SUFFIX)), TOL[name]
    F = {"scpd": [], "laddos": []}
    for nm, nb in itertools.product(NM, NB):
        sig = signatures(M, nm, nb)
        cache = {}
        for sw, lw in itertools.product(SW, LW):
            first = max(sw, lw)
            for c in (sw, lw):
                if (c, first) not in cache:
                    cache[(c, first)] = zwin(sig, c, first)
            zs, zl = cache[(sw, first)], cache[(lw, first)]
            sc = {"scpd": np.r_[0, np.diff(np.maximum(zs, zl))],
                  "laddos": np.maximum(np.r_[0, np.maximum(np.diff(zs), 0)], np.r_[0, np.maximum(np.diff(zl), 0)])}
            start = lw + sw
            for m, s in sc.items():
                pos = s[s > 0]
                for q in CD:
                    for kind, v in THR_SCPD:
                        th = v if kind == "abs" else (np.quantile(pos, v) if len(pos) else np.inf)
                        F[m].append(f1_score(scpd_alarms(s, th, q, start), cps, tol)[0])
    return ("scpd", name, seed, {m: np.array(f) for m, f in F.items()})


def task_grid(name: str) -> tuple:
    d = load(name)
    cps, tol = selection_events(name, d["cps"], bool(SUFFIX)), TOL[name]
    C = count_vectors(d["f"])
    return ("grid", name, None, {"counts": run_grid(C, cps, tol)[None], "constant": run_grid(C, cps, tol, "constant")[None]})


def run(task):
    t = Timer()
    out = task_grid(task[1]) if task[0] == "grid" else task_scpd(task[1], task[2])
    print(f"  {task} done in {t():.1f} s", flush=True)
    return out


def describe(s: dict, G: list) -> dict:
    """Attach the settings to the selected indices."""
    for n in NAMES:
        s["best"][n]["setting"] = G[s["best"][n]["index"]]
        s["transfer"]["per_dataset"][n]["setting"] = G[s["transfer"]["per_dataset"][n]["index"]]
    s["fixed"]["setting"] = G[s["fixed"]["index"]]
    return s


def main() -> None:
    t = Timer()
    tasks = [("grid", n, None) for n in NAMES] + [("scpd", n, s) for n in NAMES for s in SEEDS]
    with ProcessPoolExecutor(max_workers=4) as ex:
        outs = list(ex.map(run, tasks))
    F = {det: {} for det in ("counts", "constant", "scpd", "laddos")}
    for kind, name, seed, res in outs:
        if kind == "grid":
            for det in ("counts", "constant"):
                F[det][name] = res[det]
    for det in ("scpd", "laddos"):
        for n in NAMES:
            F[det][n] = np.stack([r[det] for k, nm, s, r in outs if k == "scpd" and nm == n])
    G41, Gc = grid_scpd(), grid()
    table = {}
    for det, Fd in F.items():
        G = G41 if det in ("scpd", "laddos") else Gc
        table[det] = describe(summarize(Fd), G)
        table[det]["n_settings"] = len(G)
        table[det]["n_seeds"] = int(Fd["enron"].shape[0])
    table["_runtime_s"] = round(t(), 1)
    out = RESULTS / f"main_table{SUFFIX}.json"
    write_json(out, table)

    print("\nTable 1 (repro vs paper)")
    hdr = f"{'detector':10s} " + " ".join(f"{LABEL[n]:>15s}" for n in NAMES) + \
        f" {'mean (best)':>15s} {'mean (fixed)':>15s} {'transfer':>15s}"
    print(hdr)
    for det in ("scpd", "laddos", "counts", "constant"):
        s, p = table[det], PAPER[det]
        cells = [(s["best"][n]["f1"], p[n]) for n in NAMES] + [(s["mean_best"], p["mean_best"]),
                                                                (s["fixed"]["f1"], p["fixed"]),
                                                                (s["transfer"]["f1"], p["transfer"])]
        print(f"{det:10s} " + " ".join(f"{a:6.3f} ({'  -- ' if b is None else f'{b:.2f}'})".rjust(15)
                                       for a, b in cells))
    print("\nselected settings (best per dataset; fixed):")
    for det in ("counts", "scpd", "laddos"):
        s = table[det]
        print(f"  {det}: " + "; ".join(f"{LABEL[n]} {s['best'][n]['setting']} (ties {s['best'][n]['n_ties']}, "
                                       f"seed sd {s['best'][n]['sd_over_seeds']:.3f})" for n in NAMES)
              + f"; fixed {s['fixed']['setting']}")
    print("transfer tie ranges (min / mean / max held-out F1 over tied settings):")
    for det in ("counts", "constant", "scpd", "laddos"):
        tr = table[det]["transfer"]["per_dataset"]
        print(f"  {det}: " + "; ".join(f"{LABEL[n]} {tr[n]['f1']:.3f} [{tr[n]['tie_min']:.3f}/{tr[n]['tie_mean']:.3f}/"
                                       f"{tr[n]['tie_max']:.3f}, {tr[n]['n_ties']} ties]" for n in NAMES))
    print(f"\nstep 3 total {t():.1f} s -> {out}")


def summary() -> None:
    """Tuned F1 (MIT + Enron mean), seed SD, and the per-dataset fixed / transfer F1 of every detector."""
    import json
    J = json.loads((RESULTS / f"main_table{SUFFIX}.json").read_text())
    for f in ("baselines", "tire"):
        if (RESULTS / f"{f}{SUFFIX}.json").exists():   # written by steps 8 and 9
            j = json.loads((RESULTS / f"{f}{SUFFIX}.json").read_text())
            J.update({k: v for k, v in j.items() if not k.startswith("_")} if f == "baselines" else {"tire": j})
    print("\nper detector: MIT+Enron mean of best F1 | seed sd of best (MIT AskU Enron) | fixed F1 | transfer F1")
    for det in ("cusum", "ewma", "bocpd", "mmd", "lad", "tire", "scpd", "laddos", "counts", "constant"):
        if det not in J:
            continue
        s = J[det]
        me = np.mean([s["best"][n]["f1"] for n in ("mit_reality", "enron")])
        sd = " ".join(f"{s['best'][n].get('sd_over_seeds', 0.0):.3f}" for n in NAMES)
        fx = " ".join(f"{s['fixed']['per_dataset'][n]:.2f}" for n in NAMES)
        tr = " ".join(f"{s['transfer']['per_dataset'][n]['f1']:.2f}" for n in NAMES)
        print(f"  {det:9s} {me:.2f} | {sd} | {fx} | {tr}")
    # false alarms at the best settings
    from common import alarms, f1_score, score_series, theta_of
    print("\nalarms of the counts at the best settings: correct / total")
    for n in ("mit_reality", "enron"):
        d = load(n)
        w, wr, md, q, kind, v = J["counts"]["best"][n]["setting"]
        D = score_series(count_vectors(d["f"]), w, wr, md)
        al = alarms(D, theta_of(D, kind, v), q, w + wr + 5)
        tp = round(f1_score(al, selection_events(n, d["cps"], bool(SUFFIX)), TOL[n])[1] * len(al))
        print(f"  {LABEL[n]}: counts {tp}/{len(al)}")


if __name__ == "__main__":
    if "--summary" not in sys.argv:                   # --summary: print from the JSON files only
        main()
    summary()
