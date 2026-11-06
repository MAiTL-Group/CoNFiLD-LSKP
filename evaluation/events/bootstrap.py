from __future__ import annotations

from typing import Dict

import numpy as np


def bootstrap_core_atoms(core_atoms: Dict[str, np.ndarray], n_bootstrap: int = 200, seed: int = 42) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, float]] = {}
    for key, arr in core_atoms.items():
        x = np.asarray(arr, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            out[key] = {"mean": float("nan"), "median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
            continue
        stats = np.empty(int(n_bootstrap), dtype=np.float64)
        for b in range(int(n_bootstrap)):
            idx = rng.integers(0, x.size, size=x.size)
            stats[b] = float(np.mean(x[idx]))
        out[key] = {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "ci_low": float(np.quantile(stats, 0.025)),
            "ci_high": float(np.quantile(stats, 0.975)),
            "n": int(x.size),
        }
    return out
