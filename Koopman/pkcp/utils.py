"""Utility functions for PKCP offline prior training."""

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

try:  # pragma: no cover - optional dependency
    import swanlab
except ImportError:  # pragma: no cover - optional dependency
    swanlab = None



def parse_hidden_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    return [int(x) for x in str(value).split(",") if str(x).strip()]



def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def save_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)



def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)



def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)



def ensure_tensor_stats(stats: Dict[str, torch.Tensor], device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in stats.items():
        if value is None:
            out[key] = None
            continue
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device)
        out[key] = tensor
    return out



def offdiag_cov_loss(z: torch.Tensor) -> torch.Tensor:
    """Return squared off-diagonal covariance penalty.

    Args:
        z: [B, D]
    """
    if z.shape[0] <= 1:
        return z.new_tensor(0.0)
    zc = z - z.mean(dim=0, keepdim=True)
    cov = zc.t().mm(zc) / float(z.shape[0] - 1)
    offdiag = cov - torch.diag_embed(torch.diagonal(cov))
    return offdiag.pow(2).mean()


def cross_cov_loss(z_left: torch.Tensor, z_right: torch.Tensor) -> torch.Tensor:
    """Squared cross-covariance penalty between two representation blocks."""
    if z_left.shape[0] <= 1:
        return z_left.new_tensor(0.0)
    zl = z_left - z_left.mean(dim=0, keepdim=True)
    zr = z_right - z_right.mean(dim=0, keepdim=True)
    cov = zl.t().mm(zr) / float(z_left.shape[0] - 1)
    return cov.pow(2).mean()



def variance_floor_loss(z: torch.Tensor, gamma: float) -> torch.Tensor:
    if z.shape[0] <= 1:
        return z.new_tensor(0.0)
    std = z.std(dim=0, unbiased=False)
    return torch.clamp(float(gamma) - std, min=0.0).mean()


def variance_balance_loss(z: torch.Tensor, target_std: Optional[float] = None) -> torch.Tensor:
    """Penalize uneven usage of dyn dimensions to resist effective-rank shrinkage."""
    if z.shape[0] <= 1:
        return z.new_tensor(0.0)
    std = z.std(dim=0, unbiased=False)
    if target_std is None:
        target = std.mean().detach()
    else:
        target = z.new_tensor(float(target_std))
    return (std - target).pow(2).mean()



def centered_covariance(z0: torch.Tensor, z1: Optional[torch.Tensor] = None, unbiased: bool = True) -> torch.Tensor:
    if z1 is None:
        z1 = z0
    z0c = z0 - z0.mean(dim=0, keepdim=True)
    z1c = z1 - z1.mean(dim=0, keepdim=True)
    denom = max(1, z0.shape[0] - 1) if unbiased else max(1, z0.shape[0])
    return z0c.t().mm(z1c) / float(denom)



def _sym_matrix_inv_sqrt(cov: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    cov = cov + float(eps) * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=float(eps))
    inv_sqrt = evecs @ torch.diag(evals.rsqrt()) @ evecs.t()
    return inv_sqrt



def vamp2_loss(z0: torch.Tensor, z1: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, Dict[str, float]]:
    """VAMP-2 style loss with numerical safeguards.

    Returns the loss plus logging diagnostics.
    """
    if z0.ndim != 2 or z1.ndim != 2:
        raise ValueError("vamp2_loss expects 2D tensors [B, D]")
    if z0.shape != z1.shape:
        raise ValueError(f"shape mismatch: {tuple(z0.shape)} vs {tuple(z1.shape)}")
    if z0.shape[0] <= z0.shape[1]:
        raise ValueError("batch too small for stable covariance estimate")

    c00 = centered_covariance(z0, unbiased=True)
    c11 = centered_covariance(z1, unbiased=True)
    c01 = centered_covariance(z0, z1, unbiased=True)

    try:
        c00_inv_sqrt = _sym_matrix_inv_sqrt(c00, eps=eps)
        c11_inv_sqrt = _sym_matrix_inv_sqrt(c11, eps=eps)
        vamp_mat = c00_inv_sqrt @ c01 @ c11_inv_sqrt
        score = vamp_mat.pow(2).sum()
    except RuntimeError:
        # Fallback: whiten z0/z1 directly and use a CCA-style surrogate.
        z0w = (z0 - z0.mean(dim=0, keepdim=True)) / (z0.std(dim=0, unbiased=False, keepdim=True) + eps)
        z1w = (z1 - z1.mean(dim=0, keepdim=True)) / (z1.std(dim=0, unbiased=False, keepdim=True) + eps)
        corr = centered_covariance(z0w, z1w, unbiased=True)
        score = corr.pow(2).sum()

    std = z0.std(dim=0, unbiased=False)
    offdiag = c00 - torch.diag_embed(torch.diagonal(c00))
    metrics = {
        "std_mean": float(std.mean().item()),
        "std_min": float(std.min().item()),
        "cov_offdiag": float(offdiag.abs().mean().item()),
        "vamp_score": float(score.item()),
    }
    return -score, metrics



def fit_ridge_koopman(z0: torch.Tensor, z1: torch.Tensor, ridge: float = 1e-4) -> torch.Tensor:
    """Closed-form ridge fit for one-step Koopman operator.

    Args:
        z0: [N, D]
        z1: [N, D]
    """
    if z0.ndim != 2 or z1.ndim != 2:
        raise ValueError("fit_ridge_koopman expects 2D tensors")
    if z0.shape != z1.shape:
        raise ValueError(f"shape mismatch: {tuple(z0.shape)} vs {tuple(z1.shape)}")
    xtx = z0.t().mm(z0)
    reg = float(ridge) * torch.eye(z0.shape[1], device=z0.device, dtype=z0.dtype)
    xty = z0.t().mm(z1)
    # Solve (X^T X + λI) W = X^T Y, returning W; use row-vector convention later.
    w = torch.linalg.solve(xtx + reg, xty)
    return w.t().contiguous()



def compute_pairwise_dyn_residual(z_seq: torch.Tensor, koopman_k: torch.Tensor) -> torch.Tensor:
    z0 = z_seq[:, :-1, :]
    z1 = z_seq[:, 1:, :]
    z1_pred = torch.einsum("ij,btj->bti", koopman_k, z0)
    return (z1 - z1_pred).pow(2).mean()



def charbonnier_loss(diff: torch.Tensor, eps: float = 1.0e-3) -> torch.Tensor:
    eps = float(eps)
    return (torch.sqrt(diff.pow(2) + eps * eps) - eps).mean()



def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)



def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def init_swanlab_run(cfg: Dict[str, Any], default_project: str, default_run_name: str):
    """Initialize an optional SwanLab run from a config dict."""
    if not bool(cfg.get("enable_swanlab", False)):
        return None
    if swanlab is None:
        print("swanlab is not installed, ignore enable_swanlab flag", flush=True)
        return None

    swanlab_kwargs = {
        "project": cfg.get("swanlab_project", default_project),
        "experiment_name": cfg.get("swanlab_run_name") or default_run_name,
        "tags": cfg.get("swanlab_tags"),
        "description": cfg.get("swanlab_notes"),
        "resume": cfg.get("swanlab_resume", False),
        "id": cfg.get("swanlab_resume_id"),
    }
    swanlab_kwargs = {k: v for k, v in swanlab_kwargs.items() if v is not None}
    try:
        return swanlab.init(**swanlab_kwargs)
    except TypeError:
        if "experiment_name" in swanlab_kwargs:
            swanlab_kwargs["name"] = swanlab_kwargs.pop("experiment_name")
        return swanlab.init(**swanlab_kwargs)


def swanlab_log(run, payload: Dict[str, Any], step: int) -> None:
    if run is None:
        return
    log_payload = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = float(value.item())
            else:
                continue
        if isinstance(value, (int, float, bool)):
            log_payload[key] = value
    if log_payload:
        swanlab.log(log_payload, step=step)


def finish_swanlab_run(run, step: Optional[int] = None) -> None:
    if run is None:
        return
    try:
        if step is not None:
            run.log({"epoch": step}, step=step)
    except Exception:
        pass
    finish_fn = getattr(run, "finish", None)
    if callable(finish_fn):
        finish_fn()
