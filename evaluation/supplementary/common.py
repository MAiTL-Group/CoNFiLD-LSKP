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

from Koopman.lskp.data import LocalStochasticKoopmanPrior
from Koopman.pkcp.data import AlignmentStats, PcaCoarseStats, build_aligned_sequences, load_npy, project_pca_coarse_observable

def load_artifact(prior_dir: Path, split_paths: Dict):
    """Load a trusted local prior and explicit split paths; never use archived paths."""
    artifact = torch.load(Path(prior_dir) / "lskp_prior.pt", map_location="cpu")
    if not split_paths or any(not Path(value).is_file() for value in split_paths.values()):
        raise FileNotFoundError("Supply existing latent split files explicitly")
    return artifact, {key + "_npy": str(value) for key, value in split_paths.items()}


def alignment_from_dict(payload: Dict) -> AlignmentStats:
    """Reconstruct alignment statistics from a serialized prior artifact."""
    return AlignmentStats(
        data_mean=payload["data_mean"],
        data_std=payload["data_std"],
        repr_mean=payload.get("repr_mean"),
        repr_std=payload.get("repr_std"),
    )


def pca_from_dict(payload: Dict) -> PcaCoarseStats:
    """Reconstruct PCA statistics from a serialized prior artifact."""
    return PcaCoarseStats(
        mean=payload["mean"],
        basis=payload["basis"],
        rank=int(torch.as_tensor(payload["rank"]).item()),
        energy_ratio=float(torch.as_tensor(payload["energy_ratio"]).item()),
    )


def prior_from_dict(payload: Dict) -> LocalStochasticKoopmanPrior:
    """Reconstruct a local stochastic prior from its serialized payload."""
    return LocalStochasticKoopmanPrior(
        centers=payload["centers"],
        koopman_k=payload["koopman_k"],
        sigma2=payload["sigma2"],
        tau=float(torch.as_tensor(payload["tau"]).item()),
        descriptor_mean=payload.get("descriptor_mean"),
        descriptor_std=payload.get("descriptor_std"),
        burst_proxy_threshold=(
            float(torch.as_tensor(payload["burst_proxy_threshold"]).item())
            if payload.get("burst_proxy_threshold") is not None
            else None
        ),
    )


def load_aligned_split(artifact: Dict, cfg: Dict, split: str) -> torch.Tensor:
    """Load and align one latent split exactly as in offline prior fitting."""
    key = f"{split}_npy"
    if key in cfg:
        path = Path(cfg[key])
    elif split == "test":
        path = Path(cfg["valid_npy"]).with_name("test_data.npy")
    else:
        raise KeyError(key)
    raw = load_npy(path)
    stats = alignment_from_dict(artifact["alignment_stats"])
    return build_aligned_sequences(
        raw,
        prior_variant=artifact["prior_variant"],
        repr_mode=artifact["repr_mode"],
        window_size=int(artifact["window_size"]),
        temporal_stride=int(artifact["temporal_stride"]),
        stats=stats,
    )


def load_coarse_split(artifact: Dict, cfg: Dict, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return aligned and frozen-PCA coarse sequences for one split."""
    aligned = load_aligned_split(artifact, cfg, split)
    coarse = project_pca_coarse_observable(aligned, pca_from_dict(artifact["pca_coarse_stats"]))
    return aligned, coarse
