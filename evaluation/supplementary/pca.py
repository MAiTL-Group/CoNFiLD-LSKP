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

from .common import load_artifact, load_aligned_split
CANDIDATE_DIMS = [8, 16, 32, 48, 64, 96]
RIDGE = 1e-3


def covariance_eigensystem(flat: torch.Tensor, chunk: int = 131072):
    mean = flat.mean(dim=0)
    covariance = torch.zeros((flat.shape[1], flat.shape[1]), dtype=torch.float64)
    for start in range(0, flat.shape[0], chunk):
        block = flat[start : start + chunk].double() - mean.double()
        covariance += block.T @ block
    covariance /= max(1, flat.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    return mean.float(), eigenvalues[order].clamp_min(0.0), eigenvectors[:, order].float()


def audit_case(case: str, prior_dir: Path, split_paths: Dict) -> Dict:
    artifact, cfg = load_artifact(prior_dir, split_paths)
    train = load_aligned_split(artifact, cfg, "train")
    valid = load_aligned_split(artifact, cfg, "valid")
    flat_train = train.reshape(-1, train.shape[-1])
    flat_valid = valid.reshape(-1, valid.shape[-1])
    mean, eigenvalues, basis_all = covariance_eigensystem(flat_train)
    total_train_variance = float(eigenvalues.sum().item())
    centered_valid = flat_valid - mean
    valid_mean = flat_valid.mean(dim=0)
    valid_covariance_centered = flat_valid - valid_mean
    valid_total_energy = float(centered_valid.pow(2).sum().item())
    valid_total_variance = float(valid_covariance_centered.pow(2).sum().item())
    submitted_dim = int(artifact["dim_coarse"])
    max_dim = max(dim for dim in CANDIDATE_DIMS if dim <= train.shape[-1])
    basis_max = basis_all[:, :max_dim]
    c_train_max = ((train - mean) @ basis_max).contiguous()
    c_valid_max = ((valid - mean) @ basis_max).contiguous()

    rows: List[Dict] = []
    for dim in [value for value in CANDIDATE_DIMS if value <= train.shape[-1]]:
        c_train = c_train_max[..., :dim]
        c_valid = c_valid_max[..., :dim]
        z0 = c_train[:, :-1].reshape(-1, dim)
        z1 = c_train[:, 1:].reshape(-1, dim)
        gram = z0.T @ z0
        system = gram + RIDGE * torch.eye(dim)
        operator = torch.linalg.solve(system, z0.T @ z1).T
        v0 = c_valid[:, :-1].reshape(-1, dim)
        v1 = c_valid[:, 1:].reshape(-1, dim)
        residual = v1 - v0 @ operator.T
        projected_valid_energy = float((centered_valid @ basis_all[:, :dim]).pow(2).sum().item())
        projected_valid_variance = float(
            (valid_covariance_centered @ basis_all[:, :dim]).pow(2).sum().item()
        )
        rows.append({
            "case": case,
            "dim": dim,
            "is_submitted_dim": int(dim == submitted_dim),
            "train_variance_retained": float(eigenvalues[:dim].sum().item() / total_train_variance),
            # Standard held-out covariance retention centers validation by its
            # own mean. The train-centered energy readout additionally exposes
            # mean shift relative to the fitted PCA origin.
            "valid_variance_retained": projected_valid_variance / max(valid_total_variance, 1e-12),
            "valid_train_centered_energy_retained": projected_valid_energy / max(valid_total_energy, 1e-12),
            "valid_projection_residual_fraction": 1.0 - projected_valid_energy / max(valid_total_energy, 1e-12),
            "valid_mean_shift_energy_fraction": max(
                0.0,
                1.0 - valid_total_variance / max(valid_total_energy, 1e-12),
            ),
            "valid_one_step_mse": float(residual.pow(2).mean().item()),
            "regularized_normal_condition": float(torch.linalg.cond(system.double()).item()),
        })
    return {
        "case": case,
        "train_samples": int(train.shape[0]),
        "valid_samples": int(valid.shape[0]),
        "ambient_dim": int(train.shape[-1]),
        "submitted_dim": submitted_dim,
        "rows": rows,
    }
