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

from .tube import as_tensor, build_transition_descriptor, align_and_project, generated_paths

def get_alpha(c_seq: torch.Tensor, artifact: Dict[str, object]) -> torch.Tensor:
    prior = artifact["local_stochastic_prior"]
    centers = as_tensor(prior["centers"])
    tau = float(prior["tau"])
    q = build_transition_descriptor(c_seq, dict(artifact["descriptor_cfg"]))
    if prior.get("descriptor_mean") is not None and prior.get("descriptor_std") is not None:
        q = (q - as_tensor(prior["descriptor_mean"])) / as_tensor(prior["descriptor_std"]).clamp(min=1.0e-6)
    q_flat = q.reshape(-1, q.shape[-1])
    dist2 = (q_flat[:, None, :] - centers[None, :, :]).pow(2).sum(dim=-1)
    return torch.softmax(-tau * dist2, dim=-1).reshape(q.shape[0], q.shape[1], -1)


def summarize_amplification_for_root(
    case: str,
    model: str,
    horizon: int,
    root: Path,
    artifact: Dict[str, object],
    hs: Tuple[int, ...] = (1, 4, 8, 16),
    max_starts: int = 2048,
) -> List[Dict[str, object]]:
    prior = artifact["local_stochastic_prior"]
    k_stack = as_tensor(prior["koopman_k"])
    dim = int(k_stack.shape[-1])
    num_regimes = int(k_stack.shape[0])
    if num_regimes == 1:
        out = []
        k = k_stack[0]
        for h in hs:
            kh = torch.linalg.matrix_power(k, int(h))
            norm = float(torch.linalg.svdvals(kh).max().item())
            out.append(
                {
                    "case": case,
                    "model": model,
                    "horizon": int(horizon),
                    "product_h": int(h),
                    "n_products": 1,
                    "amp_norm_mean": norm,
                    "amp_norm_median": norm,
                    "amp_norm_q95": norm,
                    "amp_rate_mean": math.log(max(norm, 1e-12)) / float(h),
                    "note": "single-regime exact K^h",
                }
            )
        return out

    products_by_h: Dict[int, List[float]] = {h: [] for h in hs}
    rng = np.random.default_rng(42)
    for path in generated_paths(str(root)):
        x = torch.from_numpy(np.load(path).astype(np.float32, copy=True))
        c = align_and_project(x, artifact, full_sequence=True)
        alpha = get_alpha(c, artifact)
        kbar = torch.einsum("btr,rij->btij", alpha, k_stack)[0]
        max_h = max(hs)
        n_start = max(0, int(kbar.shape[0]) - max_h + 1)
        if n_start <= 0:
            continue
        starts = np.arange(n_start)
        if starts.size > max_starts:
            starts = rng.choice(starts, size=max_starts, replace=False)
        for start in starts:
            prod = torch.eye(dim)
            for offset in range(max_h):
                prod = kbar[int(start) + offset] @ prod
                h = offset + 1
                if h in products_by_h:
                    products_by_h[h].append(float(torch.linalg.svdvals(prod).max().item()))
    out = []
    for h, vals in products_by_h.items():
        arr = np.asarray(vals, dtype=float)
        out.append(
            {
                "case": case,
                "model": model,
                "horizon": int(horizon),
                "product_h": int(h),
                "n_products": int(arr.size),
                "amp_norm_mean": float(np.mean(arr)) if arr.size else float("nan"),
                "amp_norm_median": float(np.median(arr)) if arr.size else float("nan"),
                "amp_norm_q95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
                "amp_rate_mean": float(np.mean(np.log(np.maximum(arr, 1e-12)) / float(h))) if arr.size else float("nan"),
                "note": f"sampled regime-switching products; max_starts_per_seed={max_starts}",
            }
        )
    return out
