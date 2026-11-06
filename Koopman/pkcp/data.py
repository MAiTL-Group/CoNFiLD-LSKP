"""Case-aware data loading and alignment for PKCP."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class AlignmentStats:
    data_mean: torch.Tensor
    data_std: torch.Tensor
    repr_mean: Optional[torch.Tensor]
    repr_std: Optional[torch.Tensor]

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "data_mean": self.data_mean,
            "data_std": self.data_std,
            "repr_mean": self.repr_mean,
            "repr_std": self.repr_std,
        }


@dataclass
class PcaCoarseStats:
    mean: torch.Tensor
    basis: torch.Tensor
    rank: int
    energy_ratio: float

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "mean": self.mean,
            "basis": self.basis,
            "rank": torch.tensor(int(self.rank), dtype=torch.int64),
            "energy_ratio": torch.tensor(float(self.energy_ratio), dtype=torch.float32),
        }



def validate_prior_variant(
    prior_variant: str,
    repr_mode: str,
    window_size: int,
    temporal_stride: int,
    raw_time_length: Optional[int] = None,
) -> None:
    variant = str(prior_variant)
    mode = str(repr_mode)
    stride = int(temporal_stride)
    win = int(window_size)
    if variant == "absolute_contiguous":
        if mode != "absolute":
            raise ValueError("absolute_contiguous requires repr_mode=absolute")
        if stride != 1:
            raise ValueError("absolute_contiguous requires temporal_stride=1")
        return
    if variant == "delta_from_first_stride4":
        if mode != "delta_from_first":
            raise ValueError("delta_from_first_stride4 requires repr_mode=delta_from_first")
        if stride != 4:
            raise ValueError("delta_from_first_stride4 requires temporal_stride=4")
        min_required = 1 + (win - 1) * stride
        if raw_time_length is not None and int(raw_time_length) < min_required:
            raise ValueError(f"raw_time_length={raw_time_length} < required {min_required}")
        return
    raise ValueError(f"Unknown prior_variant: {prior_variant}")



def load_npy(npy_path: Path) -> torch.Tensor:
    data = torch.tensor(np.load(npy_path), dtype=torch.float32)
    if data.ndim != 3:
        raise ValueError(f"expected [N,T,D], got {tuple(data.shape)}")
    return data



def select_case_window(x_seq: torch.Tensor, window_size: int, temporal_stride: int) -> torch.Tensor:
    if x_seq.ndim != 3:
        raise ValueError(f"expected [N,T,D], got {tuple(x_seq.shape)}")
    n, t, d = x_seq.shape
    win = int(window_size)
    stride = int(temporal_stride)
    needed = 1 + (win - 1) * stride
    if t < needed:
        raise ValueError(f"sequence length {t} shorter than required {needed}")
    indices = torch.arange(0, needed, stride)
    return x_seq[:, indices, :]



def apply_repr_mode(x_seq: torch.Tensor, repr_mode: str) -> torch.Tensor:
    mode = str(repr_mode)
    if mode == "absolute":
        return x_seq
    if mode == "delta_from_first":
        return x_seq - x_seq[:, :1, :]
    raise ValueError(f"Unknown repr_mode: {repr_mode}")



def compute_base_stats(x_seq: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x_seq.mean(dim=(0, 1), keepdim=True)
    std = x_seq.std(dim=(0, 1), unbiased=False, keepdim=True)
    return mean, std + float(eps)



def compute_repr_stats(x_seq: torch.Tensor, repr_mode: str, eps: float = 1e-8) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if repr_mode == "absolute":
        return None, None
    x_repr = apply_repr_mode(x_seq, repr_mode)
    mean = x_repr.mean(dim=(0, 1), keepdim=True)
    std = x_repr.std(dim=(0, 1), unbiased=False, keepdim=True)
    return mean, std + float(eps)



def apply_alignment(x_seq: torch.Tensor, repr_mode: str, stats: AlignmentStats) -> torch.Tensor:
    x_norm = (x_seq - stats.data_mean.to(x_seq.device)) / stats.data_std.to(x_seq.device)
    x_repr = apply_repr_mode(x_norm, repr_mode)
    if stats.repr_mean is not None and stats.repr_std is not None:
        x_repr = (x_repr - stats.repr_mean.to(x_seq.device)) / stats.repr_std.to(x_seq.device)
    return x_repr



def fit_alignment_stats(
    x_train_raw: torch.Tensor,
    prior_variant: str,
    repr_mode: str,
    window_size: int,
    temporal_stride: int,
) -> AlignmentStats:
    validate_prior_variant(prior_variant, repr_mode, window_size, temporal_stride, raw_time_length=x_train_raw.shape[1])
    train_win = select_case_window(x_train_raw, window_size=window_size, temporal_stride=temporal_stride)
    data_mean, data_std = compute_base_stats(train_win)
    train_norm = (train_win - data_mean) / data_std
    repr_mean, repr_std = compute_repr_stats(train_norm, repr_mode)
    return AlignmentStats(data_mean=data_mean, data_std=data_std, repr_mean=repr_mean, repr_std=repr_std)



def build_aligned_sequences(
    x_raw: torch.Tensor,
    prior_variant: str,
    repr_mode: str,
    window_size: int,
    temporal_stride: int,
    stats: AlignmentStats,
) -> torch.Tensor:
    validate_prior_variant(prior_variant, repr_mode, window_size, temporal_stride, raw_time_length=x_raw.shape[1])
    x_win = select_case_window(x_raw, window_size=window_size, temporal_stride=temporal_stride)
    return apply_alignment(x_win, repr_mode=repr_mode, stats=stats)


def build_causal_context_windows(aligned_sequences: torch.Tensor, context_length: int) -> torch.Tensor:
    """Build left-padded causal context windows [N, T, L, D]."""
    if aligned_sequences.ndim != 3:
        raise ValueError(f"expected [N,T,D], got {tuple(aligned_sequences.shape)}")
    ctx = int(context_length)
    if ctx <= 0:
        raise ValueError(f"context_length must be >= 1, got {ctx}")
    if ctx == 1:
        return aligned_sequences.unsqueeze(2)
    pad = aligned_sequences[:, :1, :].expand(-1, ctx - 1, -1)
    padded = torch.cat([pad, aligned_sequences], dim=1)
    windows = []
    tlen = aligned_sequences.shape[1]
    for offset in range(ctx):
        windows.append(padded[:, offset : offset + tlen, :])
    return torch.stack(windows, dim=2)


@torch.no_grad()
def fit_pca_coarse_stats(
    x_seq: torch.Tensor,
    energy_ratio: float = 0.95,
    dim_coarse: Optional[int] = None,
) -> PcaCoarseStats:
    """Fit a fixed PCA basis used as the Stage A/B coarse observable target.

    If dim_coarse is provided, it is treated as the final hard dimension.
    Otherwise, the rank is chosen from the target energy_ratio.
    """
    if x_seq.ndim != 3:
        raise ValueError(f"expected [N,T,D], got {tuple(x_seq.shape)}")
    flat = x_seq.reshape(-1, x_seq.shape[-1])
    mean = flat.mean(dim=0, keepdim=True)
    xc = flat - mean
    _, s, vh = torch.linalg.svd(xc, full_matrices=False)
    if s.numel() == 0:
        raise ValueError("cannot fit PCA on empty input")
    if dim_coarse not in {None, "", 0}:
        rank = int(dim_coarse)
        if rank <= 0 or rank > int(vh.shape[0]):
            raise ValueError(f"invalid dim_coarse={rank}; expected in [1, {int(vh.shape[0])}]")
    else:
        energy = torch.cumsum(s.pow(2), dim=0) / torch.clamp(s.pow(2).sum(), min=1e-12)
        threshold = torch.tensor(float(energy_ratio), device=energy.device, dtype=energy.dtype)
        rank = int(torch.searchsorted(energy, threshold).item()) + 1
        rank = max(1, min(rank, int(vh.shape[0])))
    basis = vh[:rank].t().contiguous()
    return PcaCoarseStats(mean=mean, basis=basis, rank=rank, energy_ratio=float(energy_ratio))


@torch.no_grad()
def apply_pca_coarse_target(x_seq: torch.Tensor, pca_stats: PcaCoarseStats) -> torch.Tensor:
    """Project aligned sequences to the fixed low-rank PCA coarse target and reconstruct."""
    if x_seq.ndim != 3:
        raise ValueError(f"expected [N,T,D], got {tuple(x_seq.shape)}")
    flat = x_seq.reshape(-1, x_seq.shape[-1])
    mean = pca_stats.mean.to(flat.device, flat.dtype)
    basis = pca_stats.basis.to(flat.device, flat.dtype)
    xc = flat - mean
    recon = (xc @ basis @ basis.t()) + mean
    return recon.reshape_as(x_seq)


@torch.no_grad()
def project_pca_coarse_observable(x_seq: torch.Tensor, pca_stats: PcaCoarseStats) -> torch.Tensor:
    """Project aligned sequences to low-dimensional PCA coefficients."""
    if x_seq.ndim != 3:
        raise ValueError(f"expected [N,T,D], got {tuple(x_seq.shape)}")
    flat = x_seq.reshape(-1, x_seq.shape[-1])
    mean = pca_stats.mean.to(flat.device, flat.dtype)
    basis = pca_stats.basis.to(flat.device, flat.dtype)
    coeff = (flat - mean) @ basis
    return coeff.reshape(x_seq.shape[0], x_seq.shape[1], basis.shape[1])


class TimeLaggedPairDataset(Dataset):
    """Flatten aligned sequences into (u_t, u_{t+1}) pairs."""

    def __init__(self, aligned_sequences: torch.Tensor):
        if aligned_sequences.ndim != 3:
            raise ValueError(f"expected [N,T,D], got {tuple(aligned_sequences.shape)}")
        if aligned_sequences.shape[1] < 2:
            raise ValueError("sequence length must be >= 2")
        self.u0 = aligned_sequences[:, :-1, :].reshape(-1, aligned_sequences.shape[-1])
        self.u1 = aligned_sequences[:, 1:, :].reshape(-1, aligned_sequences.shape[-1])

    def __len__(self) -> int:
        return self.u0.shape[0]

    def __getitem__(self, idx: int):
        return self.u0[idx], self.u1[idx]


class ContextTimeLaggedPairDataset(Dataset):
    """Causal time-lagged context pairs [(u_{t-L+1:t}), (u_{t-L+2:t+1})]."""

    def __init__(self, aligned_sequences: torch.Tensor, context_length: int):
        if aligned_sequences.ndim != 3:
            raise ValueError(f"expected [N,T,D], got {tuple(aligned_sequences.shape)}")
        if aligned_sequences.shape[1] < 2:
            raise ValueError("sequence length must be >= 2")
        contexts = build_causal_context_windows(aligned_sequences, context_length=context_length)
        self.u0 = contexts[:, :-1, :, :].reshape(-1, contexts.shape[2], contexts.shape[3])
        self.u1 = contexts[:, 1:, :, :].reshape(-1, contexts.shape[2], contexts.shape[3])

    def __len__(self) -> int:
        return self.u0.shape[0]

    def __getitem__(self, idx: int):
        return self.u0[idx], self.u1[idx]


class ContextRolloutDataset(Dataset):
    """Causal context pairs plus future rollout targets for Stage A multi-step forecasting."""

    def __init__(self, aligned_sequences: torch.Tensor, context_length: int, rollout_horizon: int):
        if aligned_sequences.ndim != 3:
            raise ValueError(f"expected [N,T,D], got {tuple(aligned_sequences.shape)}")
        if aligned_sequences.shape[1] < 2:
            raise ValueError("sequence length must be >= 2")
        horizon = int(rollout_horizon)
        if horizon <= 0:
            raise ValueError(f"rollout_horizon must be >= 1, got {horizon}")
        contexts = build_causal_context_windows(aligned_sequences, context_length=context_length)
        n, tlen, ctx, dim = contexts.shape
        usable = tlen - horizon
        if usable <= 0:
            raise ValueError(
                f"sequence length {tlen} too short for rollout_horizon={horizon}"
            )
        self.u0 = contexts[:, :usable, :, :].reshape(-1, ctx, dim)
        self.u1 = contexts[:, 1 : usable + 1, :, :].reshape(-1, ctx, dim)
        future = []
        for h in range(1, horizon + 1):
            future.append(aligned_sequences[:, h : h + usable, :])
        self.future = torch.stack(future, dim=2).reshape(-1, horizon, dim)

    def __len__(self) -> int:
        return self.u0.shape[0]

    def __getitem__(self, idx: int):
        return self.u0[idx], self.u1[idx], self.future[idx]


class SequenceDataset(Dataset):
    """Aligned sequence dataset for Stage B chart fitting.

    Optionally carries a second tensor target with the same leading dimension.
    """

    def __init__(self, aligned_sequences: torch.Tensor, target_sequences: Optional[torch.Tensor] = None):
        if aligned_sequences.ndim != 3:
            raise ValueError(f"expected [N,T,D], got {tuple(aligned_sequences.shape)}")
        if target_sequences is not None:
            if target_sequences.shape != aligned_sequences.shape:
                raise ValueError(
                    f"target shape mismatch: {tuple(target_sequences.shape)} vs {tuple(aligned_sequences.shape)}"
                )
        self.data = aligned_sequences
        self.target = target_sequences

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        if self.target is None:
            return self.data[idx]
        return self.data[idx], self.target[idx]
