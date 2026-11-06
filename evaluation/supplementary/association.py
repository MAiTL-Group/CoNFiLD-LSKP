"""Numerical kernels preserved from the research implementation."""
from __future__ import annotations
import argparse
import json
import math
import re
import statistics as stats
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

PRIMARY = ["w1_time_lag_1", "w1_time_lag_16", "Tint_gap"]
BOOTSTRAPS = 1000


def corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    if method == "pearson":
        result = pearsonr(x, y)
    else:
        result = spearmanr(x, y)
    return float(getattr(result, "statistic", result[0]))


def bootstrap_summary(rows: List[Dict], metric: str, rng: np.random.Generator) -> Dict:
    selected = [row for row in rows if row["metric"] == metric]
    x = np.asarray([row["delta_tube"] for row in selected])
    y = np.asarray([row["delta_metric"] for row in selected])
    aligned = np.asarray([
        row["alignment"] in {"both_lower_lskp", "both_higher_lskp"} for row in selected
    ], dtype=float)
    draws = {"pearson": [], "spearman": [], "alignment_fraction": []}
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, len(selected), size=len(selected))
        draws["pearson"].append(corr(x[idx], y[idx], "pearson"))
        draws["spearman"].append(corr(x[idx], y[idx], "spearman"))
        draws["alignment_fraction"].append(float(aligned[idx].mean()))
    output = {
        "metric": metric,
        "n": len(selected),
        "pearson": corr(x, y, "pearson"),
        "spearman": corr(x, y, "spearman"),
        "alignment_fraction": float(aligned.mean()),
    }
    for key, values in draws.items():
        array = np.asarray(values, dtype=float)
        output[key + "_bootstrap_low"] = float(np.nanquantile(array, 0.025))
        output[key + "_bootstrap_high"] = float(np.nanquantile(array, 0.975))
    return output


def stratified_rows(rows: List[Dict]) -> List[Dict]:
    output = []
    cases = sorted({row["case"] for row in rows})
    for metric in PRIMARY:
        selected = [row for row in rows if row["metric"] == metric]
        for case in cases:
            subset = [row for row in selected if row["case"] == case]
            x = np.asarray([row["delta_tube"] for row in subset])
            y = np.asarray([row["delta_metric"] for row in subset])
            output.append({
                "scope": f"case_only:{case}",
                "metric": metric,
                "n": len(subset),
                "pearson": corr(x, y, "pearson"),
                "spearman": corr(x, y, "spearman"),
            })
        for omitted in cases:
            subset = [row for row in selected if row["case"] != omitted]
            x = np.asarray([row["delta_tube"] for row in subset])
            y = np.asarray([row["delta_metric"] for row in subset])
            output.append({
                "scope": f"leave_case_out:{omitted}",
                "metric": metric,
                "n": len(subset),
                "pearson": corr(x, y, "pearson"),
                "spearman": corr(x, y, "spearman"),
            })
    return output
