"""Grids of the baselines CUSUM, EWMA, BOCPD, MMD and LAD (repro/08_baselines.py); thresholds are absolute."""

from itertools import product

import numpy as np


def _make_grid(param_lists: dict) -> list[dict]:
    """Create grid from dict of parameter lists."""
    keys = list(param_lists.keys())
    values = list(param_lists.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _subsample(grid: list[dict], target: int, seed: int = 42) -> list[dict]:
    """Subsample grid to target size if too large."""
    if len(grid) <= target:
        return grid
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(grid), size=target, replace=False)
    return [grid[i] for i in sorted(indices)]


def lad_grid() -> list[dict]:
    """LAD: thresholds × num_eigen × windows × cooldowns → subsample to 3000."""
    grid = _make_grid({
        "threshold": [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01,
                       0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        "num_eigen": [5, 8, 10, 15, 20],
        "short_window": [3, 5, 7],
        "long_window": [10, 15, 20, 25],
        "cooldown": [3, 5, 7, 10, 15],
    })
    valid = [p for p in grid if p["long_window"] > p["short_window"]]
    return _subsample(valid, 3000)


def cusum_grid() -> list[dict]:
    """CUSUM: 19 thresholds × 6 kappas × 4 startups × 6 cooldowns = 2,736."""
    return _make_grid({
        "threshold": [float(x) for x in np.concatenate([np.arange(1, 10, 1), np.arange(10, 60, 5)])],
        "kappa": [0.2, 0.3, 0.5, 0.7, 1.0, 1.5],
        "startup_period": [5, 8, 10, 15],
        "cooldown": [2, 3, 5, 7, 10, 15],
    })


def ewma_grid() -> list[dict]:
    """EWMA: 23 thresholds × 9 lambdas × 3 startups × 6 cooldowns = 3,726."""
    return _make_grid({
        "threshold": [float(x) for x in np.concatenate([
            np.arange(0.5, 5, 0.5), np.arange(5, 15, 1), np.arange(15, 35, 5)])],
        "lambda_param": [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
        "startup_period": [5, 10, 15],
        "cooldown": [2, 3, 5, 7, 10, 15],
    })


def bocpd_grid() -> list[dict]:
    """BOCPD: 17 thresholds × 12 hazard_rates × 14 cooldowns = 2,856."""
    return _make_grid({
        "threshold": [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.12,
                       0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "hazard_rate": [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07,
                        0.1, 0.15, 0.2, 0.3, 0.5],
        "cooldown": [2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50],
    })


def mmd_grid() -> list[dict]:
    """MMD: 19 thresholds × 7 windows × 22 cooldowns = 2,926."""
    return _make_grid({
        "threshold": [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.008, 0.01,
                       0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2,
                       0.3, 0.5, 0.7, 1.0],
        "window_size": [3, 5, 7, 10, 12, 15, 20],
        "cooldown": [2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40, 45,
                     50, 55, 60, 65, 70, 75, 80, 85, 90],
    })


# Registry
GRIDS = {
    "lad": lad_grid,
    "cusum": cusum_grid,
    "ewma": ewma_grid,
    "bocpd": bocpd_grid,
    "mmd": mmd_grid,
}


def get_grid(method: str) -> list[dict]:
    """Get parameter grid for a method."""
    return GRIDS[method]()
