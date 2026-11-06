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

from Koopman.lskp.data import build_transition_descriptor, normalize_descriptor, soft_assign_transition_regime
from .common import load_artifact, load_coarse_split, prior_from_dict
SEEDS = [3407, 3408, 3409, 3410, 3411]
RIDGE = 1e-3
SIGMA_FLOOR = 1e-6


def nearest_labels(q: torch.Tensor, centers: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    labels = []
    flat = q.reshape(-1, q.shape[-1])
    for start in range(0, flat.shape[0], chunk):
        block = flat[start : start + chunk]
        dist2 = (block[:, None, :] - centers[None, :, :]).pow(2).sum(dim=-1)
        labels.append(dist2.argmin(dim=1))
    return torch.cat(labels)


def fit_models(z0: torch.Tensor, z1: torch.Tensor, labels: torch.Tensor, regimes: int):
    operators = []
    sigma2 = []
    counts = []
    eye = torch.eye(z0.shape[1], dtype=z0.dtype)
    for regime in range(regimes):
        mask = labels == regime
        count = int(mask.sum().item())
        counts.append(count)
        if count <= z0.shape[1]:
            operators.append(torch.full((z0.shape[1], z0.shape[1]), float("nan")))
            sigma2.append(torch.full((z0.shape[1],), float("nan")))
            continue
        x = z0[mask]
        y = z1[mask]
        w = torch.linalg.solve(x.T @ x + RIDGE * eye, x.T @ y)
        k = w.T.contiguous()
        residual = y - x @ k.T
        operators.append(k)
        sigma2.append(residual.pow(2).mean(dim=0).clamp_min(SIGMA_FLOOR))
    return torch.stack(operators), torch.stack(sigma2), counts


def evaluate(
    z0: torch.Tensor,
    z1: torch.Tensor,
    labels: torch.Tensor,
    operators: torch.Tensor,
    sigma2: torch.Tensor,
) -> Dict[str, float]:
    valid_regime = torch.isfinite(operators).all(dim=(1, 2))
    covered = valid_regime[labels]
    if not covered.any():
        return {"coverage": 0.0, "mse": float("nan"), "nll": float("nan")}
    active_labels = labels[covered]
    x = z0[covered]
    y = z1[covered]
    pred = torch.bmm(operators[active_labels], x.unsqueeze(-1)).squeeze(-1)
    residual2 = (y - pred).pow(2)
    scale = sigma2[active_labels]
    nll = 0.5 * (torch.log(scale) + residual2 / scale)
    return {
        "coverage": float(covered.float().mean().item()),
        "mse": float(residual2.mean().item()),
        "nll": float(nll.mean().item()),
    }


def operator_separation(operators: torch.Tensor) -> float:
    finite = operators[torch.isfinite(operators).all(dim=(1, 2))]
    if finite.shape[0] < 2:
        return 0.0
    values = []
    for i in range(finite.shape[0]):
        for j in range(i + 1, finite.shape[0]):
            denom = 0.5 * (torch.linalg.norm(finite[i]) + torch.linalg.norm(finite[j]))
            values.append(float((torch.linalg.norm(finite[i] - finite[j]) / denom.clamp_min(1e-12)).item()))
    return float(np.mean(values))


def evaluate_soft_prior(
    coarse: torch.Tensor,
    descriptor: torch.Tensor,
    prior,
) -> Dict[str, float]:
    alpha, _ = soft_assign_transition_regime(descriptor, prior)
    z0 = coarse[:, :-1]
    z1 = coarse[:, 1:]
    pred_r = torch.einsum("rdi,bti->btrd", prior.koopman_k, z0)
    pred = (alpha.unsqueeze(-1) * pred_r).sum(dim=2)
    scale = (alpha.unsqueeze(-1) * prior.sigma2[None, None]).sum(dim=2).clamp_min(SIGMA_FLOOR)
    residual2 = (z1 - pred).pow(2)
    return {
        "coverage": 1.0,
        "mse": float(residual2.mean().item()),
        "nll": float((0.5 * (torch.log(scale) + residual2 / scale)).mean().item()),
    }


def random_histogram_labels(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return reference[torch.randperm(reference.numel(), generator=generator)]


def random_valid_labels(train_labels: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    probs = torch.bincount(train_labels).float()
    probs /= probs.sum()
    return torch.multinomial(probs, count, replacement=True, generator=generator)


def shifted_labels(labels_2d: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    result = labels_2d.clone()
    shifts = torch.randint(1, labels_2d.shape[1], (labels_2d.shape[0],), generator=generator)
    for row, shift in enumerate(shifts.tolist()):
        result[row] = torch.roll(labels_2d[row], shifts=shift)
    return result.reshape(-1)


def summarize_variant(
    name: str,
    replicate: int,
    z0_train: torch.Tensor,
    z1_train: torch.Tensor,
    train_labels: torch.Tensor,
    z0_valid: torch.Tensor,
    z1_valid: torch.Tensor,
    valid_labels: torch.Tensor,
    z0_test: torch.Tensor,
    z1_test: torch.Tensor,
    test_labels: torch.Tensor,
    regimes: int,
) -> Dict[str, object]:
    operators, sigma2, counts = fit_models(z0_train, z1_train, train_labels, regimes)
    row = {
        "variant": name,
        "replicate": replicate,
        "regimes": regimes,
        "train_min_count": min(counts),
        "train_median_count": float(np.median(counts)),
        "valid_active_regimes": int(torch.unique(valid_labels).numel()),
        "test_active_regimes": int(torch.unique(test_labels).numel()),
        "operator_separation": operator_separation(operators),
    }
    row.update({"train_" + key: value for key, value in evaluate(
        z0_train, z1_train, train_labels, operators, sigma2
    ).items()})
    row.update({"valid_" + key: value for key, value in evaluate(
        z0_valid, z1_valid, valid_labels, operators, sigma2
    ).items()})
    row.update({"test_" + key: value for key, value in evaluate(
        z0_test, z1_test, test_labels, operators, sigma2
    ).items()})
    return row


def audit_case(prior_dir: Path, split_paths: Dict) -> Dict:
    artifact, cfg = load_artifact(prior_dir, split_paths)
    _, c_train = load_coarse_split(artifact, cfg, "train")
    _, c_valid = load_coarse_split(artifact, cfg, "valid")
    _, c_test = load_coarse_split(artifact, cfg, "test")
    prior = prior_from_dict(artifact["local_stochastic_prior"])
    q_train_raw = build_transition_descriptor(c_train, **artifact["descriptor_cfg"])
    q_valid_raw = build_transition_descriptor(c_valid, **artifact["descriptor_cfg"])
    q_test_raw = build_transition_descriptor(c_test, **artifact["descriptor_cfg"])
    q_train = normalize_descriptor(
        q_train_raw,
        prior.descriptor_mean,
        prior.descriptor_std,
    )
    q_valid = normalize_descriptor(
        q_valid_raw,
        prior.descriptor_mean,
        prior.descriptor_std,
    )
    q_test = normalize_descriptor(
        q_test_raw,
        prior.descriptor_mean,
        prior.descriptor_std,
    )
    train_labels_2d = nearest_labels(q_train, prior.centers).reshape(q_train.shape[:2])
    valid_labels_2d = nearest_labels(q_valid, prior.centers).reshape(q_valid.shape[:2])
    test_labels_2d = nearest_labels(q_test, prior.centers).reshape(q_test.shape[:2])
    train_labels = train_labels_2d.reshape(-1)
    valid_labels = valid_labels_2d.reshape(-1)
    test_labels = test_labels_2d.reshape(-1)
    z0_train = c_train[:, :-1].reshape(-1, c_train.shape[-1])
    z1_train = c_train[:, 1:].reshape(-1, c_train.shape[-1])
    z0_valid = c_valid[:, :-1].reshape(-1, c_valid.shape[-1])
    z1_valid = c_valid[:, 1:].reshape(-1, c_valid.shape[-1])
    z0_test = c_test[:, :-1].reshape(-1, c_test.shape[-1])
    z1_test = c_test[:, 1:].reshape(-1, c_test.shape[-1])

    rows: List[Dict[str, object]] = []
    soft_row = {
        "variant": "submitted_soft_prior",
        "replicate": 0,
        "regimes": int(prior.centers.shape[0]),
        "train_min_count": int(torch.bincount(train_labels).min().item()),
        "train_median_count": float(torch.bincount(train_labels).float().median().item()),
        "valid_active_regimes": int(torch.unique(valid_labels).numel()),
        "test_active_regimes": int(torch.unique(test_labels).numel()),
        "operator_separation": operator_separation(prior.koopman_k),
    }
    soft_row.update({"train_" + key: value for key, value in evaluate_soft_prior(
        c_train, q_train_raw, prior
    ).items()})
    soft_row.update({"valid_" + key: value for key, value in evaluate_soft_prior(
        c_valid, q_valid_raw, prior
    ).items()})
    soft_row.update({"test_" + key: value for key, value in evaluate_soft_prior(
        c_test, q_test_raw, prior
    ).items()})
    rows.append(soft_row)
    rows.append(summarize_variant(
        "descriptor_lloyd", 0, z0_train, z1_train, train_labels,
        z0_valid, z1_valid, valid_labels, z0_test, z1_test, test_labels,
        prior.centers.shape[0]
    ))
    rows.append(summarize_variant(
        "global_r1", 0, z0_train, z1_train, torch.zeros_like(train_labels),
        z0_valid, z1_valid, torch.zeros_like(valid_labels),
        z0_test, z1_test, torch.zeros_like(test_labels), 1
    ))
    for replicate, seed in enumerate(SEEDS):
        rows.append(summarize_variant(
            "random_frequency_matched", replicate, z0_train, z1_train,
            random_histogram_labels(train_labels, seed), z0_valid, z1_valid,
            random_valid_labels(train_labels, valid_labels.numel(), seed + 10000),
            z0_test, z1_test,
            random_valid_labels(train_labels, test_labels.numel(), seed + 20000),
            prior.centers.shape[0],
        ))
        rows.append(summarize_variant(
            "time_shifted_labels", replicate, z0_train, z1_train,
            shifted_labels(train_labels_2d, seed), z0_valid, z1_valid,
            shifted_labels(valid_labels_2d, seed + 10000), z0_test, z1_test,
            shifted_labels(test_labels_2d, seed + 20000), prior.centers.shape[0],
        ))

    grouped = {}
    for variant in sorted({str(row["variant"]) for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        grouped[variant] = {}
        for key in [
            "valid_mse", "valid_nll", "valid_coverage",
            "test_mse", "test_nll", "test_coverage", "operator_separation"
        ]:
            values = np.asarray([float(row[key]) for row in selected], dtype=float)
            grouped[variant][key + "_mean"] = float(np.nanmean(values))
            grouped[variant][key + "_std"] = float(np.nanstd(values))
        grouped[variant]["replicates"] = len(selected)
    return {"rows": rows, "grouped": grouped, "train_transitions": int(train_labels.numel()),
            "valid_transitions": int(valid_labels.numel()), "test_transitions": int(test_labels.numel())}

