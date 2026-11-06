"""Data utilities for the Local Stochastic Koopman Prior (LSKP)."""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass
class LocalStochasticKoopmanPrior:
    """Local linear transition cores with regime-specific residual variances."""

    centers: torch.Tensor  # [R, Q]
    koopman_k: torch.Tensor  # [R, D, D]
    sigma2: torch.Tensor  # [R, D]
    tau: float
    descriptor_mean: Optional[torch.Tensor] = None  # [Q]
    descriptor_std: Optional[torch.Tensor] = None  # [Q]
    burst_proxy_threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, torch.Tensor]:
        payload = {
            "centers": self.centers,
            "koopman_k": self.koopman_k,
            "sigma2": self.sigma2,
            "tau": torch.tensor(float(self.tau), dtype=torch.float32),
        }
        if self.descriptor_mean is not None:
            payload["descriptor_mean"] = self.descriptor_mean
        if self.descriptor_std is not None:
            payload["descriptor_std"] = self.descriptor_std
        if self.burst_proxy_threshold is not None:
            payload["burst_proxy_threshold"] = torch.tensor(
                float(self.burst_proxy_threshold), dtype=torch.float32
            )
        return payload


def fit_descriptor_stats(
    q_seq: torch.Tensor,
    *,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate mean/std for descriptor normalization from [B, T-1, Q]."""
    if q_seq.ndim != 3:
        raise ValueError(f"expected [B,T,Q], got {tuple(q_seq.shape)}")
    q_flat = q_seq.reshape(-1, q_seq.shape[-1])
    mean = q_flat.mean(dim=0)
    std = q_flat.std(dim=0, unbiased=False).clamp(min=float(eps))
    return mean, std


def normalize_descriptor(
    q_seq: torch.Tensor,
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
    *,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Normalize transition descriptors with broadcastable mean/std."""
    if mean is None or std is None:
        return q_seq
    mean_t = mean.to(device=q_seq.device, dtype=q_seq.dtype)
    std_t = std.to(device=q_seq.device, dtype=q_seq.dtype).clamp(min=float(eps))
    return (q_seq - mean_t) / std_t


def build_transition_descriptor(
    c_seq: torch.Tensor,
    *,
    include_level: bool = True,
    include_delta: bool = True,
    include_energy_delta: bool = True,
    include_prev_delta: bool = False,
    include_delta2: bool = False,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Build short-lag transition descriptors [B, T-1, Q] from coarse observables [B, T, C]."""
    if c_seq.ndim != 3 or c_seq.shape[1] < 2:
        raise ValueError(f"expected [B,T,C] with T>=2, got {tuple(c_seq.shape)}")
    c_now = c_seq[:, :-1, :]
    c_next = c_seq[:, 1:, :]
    delta = c_next - c_now
    zero_delta = torch.zeros_like(delta[:, :1, :])
    delta_prev = torch.cat([zero_delta, delta[:, :-1, :]], dim=1)
    delta2 = delta - delta_prev
    feats = []
    if include_level:
        feats.append(c_now)
    if include_delta:
        feats.append(delta)
    if include_prev_delta:
        feats.append(delta_prev)
    if include_delta2:
        feats.append(delta2)
    if include_energy_delta:
        e_now = c_now.pow(2).sum(dim=-1, keepdim=True)
        e_next = c_next.pow(2).sum(dim=-1, keepdim=True)
        de = e_next - e_now
        delta_norm = torch.sqrt(delta.pow(2).sum(dim=-1, keepdim=True) + float(eps))
        feats.extend([de, delta_norm])
    if not feats:
        raise ValueError("transition descriptor must include at least one block")
    return torch.cat(feats, dim=-1)


@torch.no_grad()
def run_kmeans(
    x: torch.Tensor,
    num_clusters: int,
    *,
    iters: int = 25,
    seed: int = 3407,
    enable_tqdm: bool = False,
    tqdm_desc: str = "lskp kmeans",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple Lloyd k-means for [N, D] data."""
    if x.ndim != 2:
        raise ValueError(f"expected [N,D], got {tuple(x.shape)}")
    n = int(x.shape[0])
    k = int(num_clusters)
    if k <= 0 or k > n:
        raise ValueError(f"invalid num_clusters={k} for N={n}")
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    init_idx = torch.randperm(n, generator=gen, device=x.device)[:k]
    centers = x[init_idx].clone()
    assign = torch.zeros(n, dtype=torch.long, device=x.device)
    iterator = range(int(iters))
    if enable_tqdm and tqdm is not None:
        iterator = tqdm(iterator, desc=tqdm_desc, leave=False, dynamic_ncols=True)
    for _ in iterator:
        dist2 = (x[:, None, :] - centers[None, :, :]).pow(2).sum(dim=-1)
        assign = dist2.argmin(dim=1)
        new_centers = centers.clone()
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centers[j] = x[mask].mean(dim=0)
        if torch.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return centers, assign


def _fit_ridge_koopman_local(
    z0: torch.Tensor,
    z1: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    if z0.ndim != 2 or z1.ndim != 2 or z0.shape != z1.shape:
        raise ValueError("expected matched [N,D] tensors")
    xtx = z0.t().mm(z0)
    xty = z0.t().mm(z1)
    reg = float(ridge) * torch.eye(z0.shape[1], device=z0.device, dtype=z0.dtype)
    w = torch.linalg.solve(xtx + reg, xty)
    return w.t().contiguous()


def fit_local_stochastic_koopman_prior(
    z_seq: torch.Tensor,
    q_seq: torch.Tensor,
    *,
    num_regimes: int,
    ridge: float = 1.0e-3,
    tau: float = 5.0,
    kmeans_iters: int = 20,
    seed: int = 3407,
    enable_tqdm: bool = False,
    burst_scores: Optional[torch.Tensor] = None,
    burst_quantile: float = 0.75,
    burst_boost: float = 1.0,
    burst_regime_count: int = 0,
) -> LocalStochasticKoopmanPrior:
    """Fit regime-specific local Koopman cores and residual variances."""
    if z_seq.ndim != 3 or z_seq.shape[1] < 2:
        raise ValueError("z_seq must be [B,T,D] with T>=2")
    if q_seq.ndim != 3 or q_seq.shape[:2] != (z_seq.shape[0], z_seq.shape[1] - 1):
        raise ValueError("q_seq must be [B,T-1,Q]")

    z0 = z_seq[:, :-1, :].reshape(-1, z_seq.shape[-1])
    z1 = z_seq[:, 1:, :].reshape(-1, z_seq.shape[-1])
    q_flat = q_seq.reshape(-1, q_seq.shape[-1])
    burst_threshold = None
    q_cluster = q_flat
    explicit_centers = None
    if burst_scores is not None:
        burst_flat = burst_scores.reshape(-1).to(device=q_flat.device, dtype=q_flat.dtype)
        if burst_flat.shape[0] != q_flat.shape[0]:
            raise ValueError("burst_scores must match q_seq leading flattened dimension")
        burst_quantile = float(burst_quantile)
        burst_boost = float(burst_boost)
        burst_regime_count = int(burst_regime_count)
        if 0.0 < burst_quantile < 1.0:
            burst_threshold = float(torch.quantile(burst_flat, burst_quantile).item())
            burst_idx = (burst_flat >= burst_threshold).nonzero(as_tuple=False).view(-1)
            non_burst_idx = (burst_flat < burst_threshold).nonzero(as_tuple=False).view(-1)
            if 0 < burst_regime_count < int(num_regimes):
                num_normal = int(num_regimes) - burst_regime_count
                min_burst = max(int(z_seq.shape[-1]) + 1, burst_regime_count * 4)
                min_normal = max(int(z_seq.shape[-1]) + 1, num_normal * 4)
                if burst_idx.numel() >= min_burst and non_burst_idx.numel() >= min_normal:
                    burst_centers, _ = run_kmeans(
                        q_flat.index_select(0, burst_idx),
                        num_clusters=burst_regime_count,
                        iters=int(kmeans_iters),
                        seed=int(seed) + 17,
                        enable_tqdm=enable_tqdm,
                        tqdm_desc="lskp burst regimes",
                    )
                    normal_centers, _ = run_kmeans(
                        q_flat.index_select(0, non_burst_idx),
                        num_clusters=num_normal,
                        iters=int(kmeans_iters),
                        seed=int(seed),
                        enable_tqdm=enable_tqdm,
                        tqdm_desc="lskp normal regimes",
                    )
                    explicit_centers = torch.cat([normal_centers, burst_centers], dim=0)
            if explicit_centers is None and burst_boost > 1.0 and burst_idx.numel() > 0:
                extra_count = int(round((burst_boost - 1.0) * float(burst_idx.numel())))
                if extra_count > 0:
                    repeat_idx = burst_idx.repeat((extra_count + burst_idx.numel() - 1) // burst_idx.numel())[
                        :extra_count
                    ]
                    q_cluster = torch.cat([q_flat, q_flat.index_select(0, repeat_idx)], dim=0)
    if explicit_centers is not None:
        centers = explicit_centers
    else:
        centers, _ = run_kmeans(
            q_cluster,
            num_clusters=int(num_regimes),
            iters=int(kmeans_iters),
            seed=int(seed),
            enable_tqdm=enable_tqdm,
            tqdm_desc="lskp regimes",
        )
    dist2_assign = (q_flat[:, None, :] - centers[None, :, :]).pow(2).sum(dim=-1)
    labels = dist2_assign.argmin(dim=1)

    k_list = []
    sigma2_list = []
    iterator = range(int(num_regimes))
    if enable_tqdm and tqdm is not None:
        iterator = tqdm(iterator, desc="lskp fit", leave=False, dynamic_ncols=True)
    for ridx in iterator:
        mask = labels == ridx
        if int(mask.sum().item()) < max(4, z_seq.shape[-1] + 1):
            dist2 = (q_flat - centers[ridx : ridx + 1]).pow(2).sum(dim=-1)
            topk = min(max(16, z_seq.shape[-1] + 1), dist2.shape[0])
            nn_idx = dist2.topk(topk, largest=False).indices
            sel_z0 = z0.index_select(0, nn_idx)
            sel_z1 = z1.index_select(0, nn_idx)
        else:
            sel_z0 = z0[mask]
            sel_z1 = z1[mask]
        k_r = _fit_ridge_koopman_local(sel_z0, sel_z1, ridge=float(ridge))
        resid = sel_z1 - torch.einsum("ij,nj->ni", k_r, sel_z0)
        sigma2 = resid.pow(2).mean(dim=0).clamp(min=1.0e-6)
        k_list.append(k_r)
        sigma2_list.append(sigma2)

    return LocalStochasticKoopmanPrior(
        centers=centers,
        koopman_k=torch.stack(k_list, dim=0),
        sigma2=torch.stack(sigma2_list, dim=0),
        tau=float(tau),
        burst_proxy_threshold=burst_threshold,
    )


def compute_regime_confidence(alpha: torch.Tensor, *, eps: float = 1.0e-8) -> Dict[str, torch.Tensor]:
    """Compute richer confidence statistics from soft regime assignments [B,T,R]."""
    if alpha.ndim != 3:
        raise ValueError(f"expected [B,T,R], got {tuple(alpha.shape)}")
    num_regimes = int(alpha.shape[-1])
    topk = min(2, num_regimes)
    top_vals = alpha.topk(topk, dim=-1).values
    conf_max = top_vals[..., 0]
    if topk >= 2:
        conf_margin = top_vals[..., 0] - top_vals[..., 1]
    else:
        conf_margin = top_vals[..., 0]
    entropy = -(alpha * torch.log(alpha.clamp(min=float(eps)))).sum(dim=-1)
    if num_regimes > 1:
        entropy_norm = entropy / float(math.log(num_regimes))
    else:
        entropy_norm = torch.zeros_like(entropy)
    conf_compound = (conf_max * conf_margin * (1.0 - entropy_norm)).clamp(min=0.0, max=1.0)
    return {
        "conf_max": conf_max,
        "conf_margin": conf_margin,
        "conf_entropy": entropy,
        "conf_entropy_norm": entropy_norm,
        "conf_compound": conf_compound,
    }


def soft_assign_transition_regime(
    q_seq: torch.Tensor,
    prior: LocalStochasticKoopmanPrior,
    *,
    return_confidence: bool = False,
    return_details: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Soft-assign transition descriptors to local regimes."""
    q_norm = normalize_descriptor(q_seq, prior.descriptor_mean, prior.descriptor_std)
    q_flat = q_norm.reshape(-1, q_norm.shape[-1])
    centers = prior.centers.to(q_flat.device, q_flat.dtype)
    dist2 = (q_flat[:, None, :] - centers[None, :, :]).pow(2).sum(dim=-1)
    alpha = torch.softmax(-float(prior.tau) * dist2, dim=-1)
    alpha = alpha.reshape(q_seq.shape[0], q_seq.shape[1], -1)
    if not return_confidence and not return_details:
        return alpha, None
    details = compute_regime_confidence(alpha)
    dist2 = dist2.reshape(q_seq.shape[0], q_seq.shape[1], -1)
    details["dist2_min"] = dist2.min(dim=-1).values
    if return_details:
        return alpha, details
    return alpha, details["conf_max"]


def compute_lskp_whitened_residual(
    z_seq: torch.Tensor,
    q_seq: torch.Tensor,
    prior: LocalStochasticKoopmanPrior,
    *,
    loss_mode: str = "mse",
    huber_delta: float = 2.0,
    return_aux: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute regime-aware whitened one-step residual loss per sample."""
    if z_seq.ndim != 3 or z_seq.shape[1] < 2:
        raise ValueError("z_seq must be [B,T,D] with T>=2")
    alpha, conf_details = soft_assign_transition_regime(q_seq, prior, return_details=True)
    z0 = z_seq[:, :-1, :]
    z1 = z_seq[:, 1:, :]
    k_stack = prior.koopman_k.to(z_seq.device, z_seq.dtype)
    sigma2 = prior.sigma2.to(z_seq.device, z_seq.dtype)
    z_pred_r = torch.einsum("rdi,bti->btrd", k_stack, z0)
    z_pred = (alpha.unsqueeze(-1) * z_pred_r).sum(dim=2)
    sigma2_mix = (alpha.unsqueeze(-1) * sigma2.unsqueeze(0).unsqueeze(0)).sum(dim=2)
    resid_white = (z1 - z_pred) / torch.sqrt(sigma2_mix + 1.0e-6)
    loss_mode = str(loss_mode).lower()
    if loss_mode == "mse":
        loss_term = resid_white.pow(2)
    elif loss_mode == "huber":
        abs_white = resid_white.abs()
        delta = float(huber_delta)
        loss_term = torch.where(
            abs_white <= delta,
            0.5 * abs_white.pow(2),
            delta * (abs_white - 0.5 * delta),
        )
    else:
        raise ValueError(f"unsupported lskp loss_mode: {loss_mode}")
    loss_vec = loss_term.mean(dim=(1, 2))
    conf_vec = conf_details["conf_max"].mean(dim=1)
    if not return_aux:
        return loss_vec, conf_vec
    mse_step = resid_white.pow(2).mean(dim=-1)
    abs_white = resid_white.abs()
    delta = float(huber_delta)
    huber_term = torch.where(
        abs_white <= delta,
        0.5 * abs_white.pow(2),
        delta * (abs_white - 0.5 * delta),
    )
    huber_step = huber_term.mean(dim=-1)
    aux = {
        "conf_max_vec": conf_details["conf_max"].mean(dim=1),
        "conf_margin_vec": conf_details["conf_margin"].mean(dim=1),
        "conf_entropy_vec": conf_details["conf_entropy"].mean(dim=1),
        "conf_entropy_norm_vec": conf_details["conf_entropy_norm"].mean(dim=1),
        "conf_compound_vec": conf_details["conf_compound"].mean(dim=1),
        "dist2_min_vec": conf_details["dist2_min"].mean(dim=1),
        "conf_max_step": conf_details["conf_max"],
        "conf_margin_step": conf_details["conf_margin"],
        "conf_entropy_step": conf_details["conf_entropy"],
        "conf_entropy_norm_step": conf_details["conf_entropy_norm"],
        "conf_compound_step": conf_details["conf_compound"],
        "dist2_min_step": conf_details["dist2_min"],
        "loss_mse_step": mse_step,
        "loss_huber_step": huber_step,
    }
    return loss_vec, conf_vec, aux
