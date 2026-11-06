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

from .common import load_artifact, load_aligned_split, pca_from_dict, prior_from_dict
MAX_TRANSITIONS = 4096


def correlations(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    pearson_result = pearsonr(x, y)
    spearman_result = spearmanr(x, y)
    return {
        "pearson": float(getattr(pearson_result, "statistic", pearson_result[0])),
        "spearman": float(getattr(spearman_result, "statistic", spearman_result[0])),
    }


def cosine_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_flat = x.flatten(1)
    y_flat = y.flatten(1)
    denom = torch.linalg.norm(x_flat, dim=1) * torch.linalg.norm(y_flat, dim=1)
    return (x_flat * y_flat).sum(dim=1) / denom.clamp_min(1e-12)


def objective_and_grad(
    windows: torch.Tensor,
    basis: torch.Tensor,
    pca_mean: torch.Tensor,
    operator: torch.Tensor,
    sigma2: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = windows.clone().requires_grad_(True)
    if mode == "lskp":
        coarse = (x - pca_mean) @ basis
        pred = coarse[:, 1] @ operator.T
        residual2 = (coarse[:, 2] - pred).pow(2) / sigma2
        score = residual2.mean(dim=1)
    elif mode == "velocity":
        score = (x[:, 2] - x[:, 1]).pow(2).mean(dim=1)
    elif mode == "acceleration":
        score = (x[:, 2] - 2.0 * x[:, 1] + x[:, 0]).pow(2).mean(dim=1)
    else:
        raise ValueError(mode)
    grad = torch.autograd.grad(score.sum(), x)[0]
    return score.detach(), grad.detach()


def audit_case(case: str, prior_dir: Path, split_paths: Dict) -> Dict:
    artifact, cfg = load_artifact(prior_dir, split_paths)
    aligned = load_aligned_split(artifact, cfg, "valid")
    windows = torch.stack(
        [aligned[:, :-2], aligned[:, 1:-1], aligned[:, 2:]], dim=2
    ).reshape(-1, 3, aligned.shape[-1])
    indices = torch.linspace(
        0, windows.shape[0] - 1, steps=min(MAX_TRANSITIONS, windows.shape[0])
    ).round().long().unique()
    windows = windows.index_select(0, indices)
    pca = pca_from_dict(artifact["pca_coarse_stats"])
    prior = prior_from_dict(artifact["local_stochastic_prior"])
    if prior.koopman_k.shape[0] != 1:
        raise ValueError(f"{case} is not single-regime")

    scores = {}
    grads = {}
    for mode in ["lskp", "velocity", "acceleration"]:
        scores[mode], grads[mode] = objective_and_grad(
            windows,
            pca.basis,
            pca.mean.reshape(1, 1, -1),
            prior.koopman_k[0],
            prior.sigma2[0].clamp_min(1e-6),
            mode,
        )

    rows = []
    for mode in ["velocity", "acceleration"]:
        corr = correlations(scores["lskp"].numpy(), scores[mode].numpy())
        cosine = cosine_rows(grads["lskp"], grads[mode])
        norm_ratio = (
            torch.linalg.norm(grads[mode].flatten(1), dim=1)
            / torch.linalg.norm(grads["lskp"].flatten(1), dim=1).clamp_min(1e-12)
        )
        rows.append({
            "case": case,
            "comparator": mode,
            "n": int(windows.shape[0]),
            "score_pearson": corr["pearson"],
            "score_spearman": corr["spearman"],
            "gradient_cosine_mean": float(cosine.mean().item()),
            "gradient_cosine_median": float(cosine.median().item()),
            "gradient_cosine_q05": float(torch.quantile(cosine, 0.05).item()),
            "gradient_cosine_q95": float(torch.quantile(cosine, 0.95).item()),
            "gradient_norm_ratio_median": float(norm_ratio.median().item()),
        })
    return {
        "case": case,
        "ambient_dim": int(aligned.shape[-1]),
        "coarse_dim": int(pca.basis.shape[1]),
        "validation_windows": int(windows.shape[0]),
        "rows": rows,
    }
