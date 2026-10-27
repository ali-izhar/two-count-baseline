"""Step 9: tuned F1 on the selection series of TIRE (offline autoencoder trained on each whole feature sequence).

360 training settings, each scored at 9 thresholds x 5 cooldowns, alarms from t = 2N for window N (the burn-in of
methods/tire.py). Training is deterministic (seed 42). Selection as in step 3. Output: repro/results/tire.json.

Usage: 09_tire.py [--corrected]   (--corrected: corrected MIT events -> results/tire_corrected.json)
"""
from __future__ import annotations

import sys
from itertools import product
from multiprocessing import Pool

import numpy as np
from common import NAMES, RESULTS, TOL, Timer, bundle, f1_score, summarize, write_json

from config import selection_events
from data.registry import DataRegistry
from methods.interface import apply_cooldown

# (window N, parallel windows K, latent dim, shared dim <= latent dim, penalty lambda, domain, epochs)
TRAIN = [c for c in product([3, 5, 7], [1, 2], [1, 2, 3], [1, 2], [0.1, 1.0, 10.0], ["td", "both"], [100, 200])
         if c[3] <= c[2]]
THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
COOLDOWNS = [2, 3, 5, 7, 10]
SUFFIX = "_corrected" if "--corrected" in sys.argv else ""
_B: dict = {}



def _init(bundles: dict) -> None:
    _B.update(bundles)


def run(job: tuple[str, int]) -> tuple[str, int, np.ndarray]:
    """Train once, then sweep threshold and cooldown on the scores (training does not depend on them)."""
    from methods.tire import detect
    name, i = job
    ws, pk, tdl, tds, lam, dom, ep = TRAIN[i]
    p = {"window_size": ws, "parallel_k": pk, "td_latent_dim": tdl, "td_shared_dim": tds, "lambda_td": lam,
         "lambda_fd": lam, "domain": dom, "epochs": ep, "threshold": 0.0, "cooldown": 1, "seed": 42}
    b = _B[name]
    s = detect(b, p).scores
    return name, i, np.array([f1_score(apply_cooldown(s, th, cd, start=2 * ws), b.true_cps, TOL[name])[0]
                              for th in THRESHOLDS for cd in COOLDOWNS])


def main() -> None:
    t0 = Timer()
    bundles = {}
    for n in NAMES:
        b = DataRegistry().load(n)
        cps = selection_events(n, b.true_cps, bool(SUFFIX))
        bundles[n] = bundle(b.features_raw, cps=cps, tol=b.tolerance, n=b.max_nodes, name=n)
        del b
    F = {n: np.zeros((len(TRAIN), len(THRESHOLDS) * len(COOLDOWNS))) for n in NAMES}
    with Pool(4, initializer=_init, initargs=(bundles,)) as pool:
        for n, i, f in pool.imap_unordered(run, [(n, i) for n in NAMES for i in range(len(TRAIN))]):
            F[n][i] = f
    settings = [(t, th, cd) for t in TRAIN for th in THRESHOLDS for cd in COOLDOWNS]
    out = summarize({n: F[n].reshape(1, -1) for n in NAMES})
    for n in NAMES:
        out["best"][n]["setting"] = settings[out["best"][n]["index"]]
    out["fixed"]["setting"] = settings[out["fixed"]["index"]]
    out["n_settings"] = len(settings)
    out["_runtime_s"] = round(t0(), 1)
    write_json(RESULTS / f"tire{SUFFIX}.json", out)
    b = [out["best"][n]["f1"] for n in NAMES]
    print(f"TIRE: MIT {b[0]:.3f} AskU {b[1]:.3f} Enron {b[2]:.3f}; best {out['mean_best']:.3f}, "
          f"fixed {out['fixed']['f1']:.3f}, transfer {out['transfer']['f1']:.3f}")
    print(f"step 9 total {t0():.1f} s -> {RESULTS / f'tire{SUFFIX}.json'}")


if __name__ == "__main__":
    main()
