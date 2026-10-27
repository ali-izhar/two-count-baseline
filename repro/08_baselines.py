"""Step 8: tuned F1 on the selection series of CUSUM, EWMA, BOCPD, MMD and LAD over the grids of utils/param_grids.py.

The feature detectors read the eight raw features of each snapshot (vertex and edge counts among them); LAD reads the
top eigenvalues of D - A. All are deterministic, so one run per setting. Selection as in step 3.
Output: repro/results/baselines.json and a printed table.

Usage: 08_baselines.py [--corrected]   (--corrected: corrected MIT events -> results/baselines_corrected.json)
"""
from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np
from common import NAMES, RESULTS, TOL, Timer, bundle, f1_score, summarize, write_json

from config import selection_events
from data.registry import DataBundle, DataRegistry
from utils.param_grids import get_grid

METHODS = ("cusum", "ewma", "bocpd", "mmd", "lad")
SUFFIX = "_corrected" if "--corrected" in sys.argv else ""
_B: dict = {}



def slim(name: str) -> DataBundle:
    """The bundle without its graphs: the workers need only features, eigenvalues and events."""
    b = DataRegistry().load(name)
    cps = selection_events(name, b.true_cps, bool(SUFFIX))
    return bundle(b.features_raw, b.lad_eigenvalues, cps, b.tolerance, b.max_nodes, name)


def _init(bundles: dict) -> None:
    _B.update(bundles)


def run(job: tuple[str, str]) -> tuple[str, str, np.ndarray]:
    method, name = job
    detect = __import__(f"methods.{method}", fromlist=["detect"]).detect
    b = _B[name]
    return method, name, np.array([f1_score(detect(b, p).change_points, b.true_cps, TOL[name])[0]
                                   for p in get_grid(method)])


def main() -> None:
    t0 = Timer()
    bundles = {n: slim(n) for n in NAMES}            # loaded one at a time: the Enron pickle needs ~5 GB
    with Pool(4, initializer=_init, initargs=(bundles,)) as pool:
        res = pool.map(run, [(m, n) for m in METHODS for n in NAMES], chunksize=1)
    F = {m: {n: f[None, :] for mm, n, f in res if mm == m} for m in METHODS}
    out = {}
    print(f"{'detector':8s} {'MIT':>6s} {'AskU':>6s} {'Enron':>6s} {'best':>6s} {'fixed':>6s} {'transfer':>8s}")
    for m in METHODS:
        grid = get_grid(m)
        s = summarize(F[m])
        for n in NAMES:
            s["best"][n]["setting"] = grid[s["best"][n]["index"]]
        s["fixed"]["setting"] = grid[s["fixed"]["index"]]
        s["n_settings"] = len(grid)
        out[m] = s
        b = [s["best"][n]["f1"] for n in NAMES]
        print(f"{m:8s} {b[0]:6.3f} {b[1]:6.3f} {b[2]:6.3f} {s['mean_best']:6.3f} {s['fixed']['f1']:6.3f} "
              f"{s['transfer']['f1']:8.3f}")
    out["_runtime_s"] = round(t0(), 1)
    write_json(RESULTS / f"baselines{SUFFIX}.json", out)
    print(f"step 8 total {t0():.1f} s -> {RESULTS / f'baselines{SUFFIX}.json'}")


if __name__ == "__main__":
    main()
