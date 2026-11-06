"""Explicit observable lifting and manifold projection utilities for HOKP."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass
class LocalPcaCharts:
    centers: torch.Tensor  # [M, C]
    means: torch.Tensor  # [M, D]
    bases: torch.Tensor  # [M, D, R]
    tau: float

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "centers": self.centers,
            "means": self.means,
            "bases": self.bases,
            "tau": torch.tensor(float(self.tau), dtype=torch.float32),
        }


@dataclass
class LocalKoopmanCharts:
    centers: torch.Tensor  # [M, K]
    koopman_k: torch.Tensor  # [M, D, D]
    tau: float

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "centers": self.centers,
            "koopman_k": self.koopman_k,
            "tau": torch.tensor(float(self.tau), dtype=torch.float32),
        }


def build_hankel_states(c_seq: torch.Tensor, hankel_length: int) -> torch.Tensor:
    """Build left-padded Hankel states [B, T, L*C] from coarse observables [B, T, C]."""
    if c_seq.ndim != 3:
        raise ValueError(f"expected [B,T,C], got {tuple(c_seq.shape)}")
    length = int(hankel_length)
    if length <= 0:
        raise ValueError(f"hankel_length must be >= 1, got {length}")
    if length == 1:
        return c_seq
    pad = c_seq[:, :1, :].expand(-1, length - 1, -1)
    padded = torch.cat([pad, c_seq], dim=1)
    states = []
    tlen = c_seq.shape[1]
    for start in range(length):
        states.append(padded[:, start : start + tlen, :])
    return torch.cat(states, dim=-1)


def build_augmented_observables(
    c_seq: torch.Tensor,
    *,
    use_level: bool = True,
    use_delta: bool = True,
    use_accel: bool = True,
) -> torch.Tensor:
    """Build temporal observables [B,T,C_aug] from coarse coefficients [B,T,C]."""
    if c_seq.ndim != 3:
        raise ValueError(f"expected [B,T,C], got {tuple(c_seq.shape)}")
    feats = []
    if use_level:
        feats.append(c_seq)
    delta = None
    if use_delta or use_accel:
        prev = torch.cat([c_seq[:, :1, :], c_seq[:, :-1, :]], dim=1)
        delta = c_seq - prev
    if use_delta:
        feats.append(delta)
    if use_accel:
        prev_delta = torch.cat([delta[:, :1, :], delta[:, :-1, :]], dim=1)
        accel = delta - prev_delta
        feats.append(accel)
    if not feats:
        raise ValueError("at least one observable block must be enabled")
    return torch.cat(feats, dim=-1)


def build_transition_observables(
    c_seq: torch.Tensor,
    *,
    use_level: bool = True,
    use_delta: bool = True,
    use_accel: bool = True,
    use_delta_norm: bool = True,
    use_energy_drift: bool = True,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Build transition-centered observables [B,T,C_aug] from coarse coefficients."""
    if c_seq.ndim != 3:
        raise ValueError(f"expected [B,T,C], got {tuple(c_seq.shape)}")
    feats = [
        build_augmented_observables(
            c_seq,
            use_level=use_level,
            use_delta=use_delta,
            use_accel=use_accel,
        )
    ]
    prev = torch.cat([c_seq[:, :1, :], c_seq[:, :-1, :]], dim=1)
    delta = c_seq - prev
    if use_delta_norm:
        delta_norm = torch.sqrt(delta.pow(2).sum(dim=-1, keepdim=True) + float(eps))
        feats.append(delta_norm)
    if use_energy_drift:
        energy = c_seq.pow(2).sum(dim=-1, keepdim=True)
        prev_energy = torch.cat([energy[:, :1, :], energy[:, :-1, :]], dim=1)
        feats.append(energy - prev_energy)
    return torch.cat(feats, dim=-1)


def build_chart_keys(
    c_seq: torch.Tensor,
    *,
    use_level: bool = True,
    use_delta: bool = True,
    use_accel: bool = False,
) -> torch.Tensor:
    """Build chart assignment keys [B,T,K] from coarse coefficients [B,T,C]."""
    return build_augmented_observables(
        c_seq,
        use_level=use_level,
        use_delta=use_delta,
        use_accel=use_accel,
    )


def build_transition_chart_keys(
    c_seq: torch.Tensor,
    *,
    use_level: bool = True,
    use_delta: bool = True,
    use_accel: bool = False,
    use_delta_norm: bool = True,
    use_energy_drift: bool = True,
    residual_proxy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build transition-centered chart keys [B,T,K] from coarse coefficients."""
    feats = [
        build_chart_keys(
            c_seq,
            use_level=use_level,
            use_delta=use_delta,
            use_accel=use_accel,
        )
    ]
    prev = torch.cat([c_seq[:, :1, :], c_seq[:, :-1, :]], dim=1)
    delta = c_seq - prev
    if use_delta_norm:
        feats.append(torch.sqrt(delta.pow(2).sum(dim=-1, keepdim=True) + 1.0e-6))
    if use_energy_drift:
        energy = c_seq.pow(2).sum(dim=-1, keepdim=True)
        prev_energy = torch.cat([energy[:, :1, :], energy[:, :-1, :]], dim=1)
        feats.append(energy - prev_energy)
    if residual_proxy is not None:
        if residual_proxy.ndim == 2:
            residual_proxy = residual_proxy.unsqueeze(-1)
        if residual_proxy.ndim != 3 or residual_proxy.shape[:2] != c_seq.shape[:2]:
            raise ValueError("residual_proxy must broadcast to [B,T,1]")
        feats.append(residual_proxy.to(device=c_seq.device, dtype=c_seq.dtype))
    return torch.cat(feats, dim=-1)


def clip_spectral_radius(koopman_k: torch.Tensor, rho_max: float = 0.99, eps: float = 1.0e-12) -> torch.Tensor:
    """Scale K if its spectral radius exceeds rho_max."""
    rho = float(torch.linalg.eigvals(koopman_k).abs().max().item())
    if rho <= float(rho_max):
        return koopman_k
    scale = float(rho_max) / max(rho, float(eps))
    return koopman_k * scale


def _fit_ridge_koopman_local(
    z0: torch.Tensor,
    z1: torch.Tensor,
    ridge: float = 1.0e-4,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if z0.ndim != 2 or z1.ndim != 2:
        raise ValueError("expected [N,D] and [N,D]")
    if z0.shape != z1.shape:
        raise ValueError(f"shape mismatch: {tuple(z0.shape)} vs {tuple(z1.shape)}")
    if sample_weight is not None:
        if sample_weight.ndim != 1 or sample_weight.shape[0] != z0.shape[0]:
            raise ValueError("sample_weight must be [N]")
        sw = torch.sqrt(sample_weight.clamp(min=0.0) + 1.0e-12).to(z0.device, z0.dtype)
        z0 = z0 * sw[:, None]
        z1 = z1 * sw[:, None]
    xtx = z0.t().mm(z0)
    reg = float(ridge) * torch.eye(z0.shape[1], device=z0.device, dtype=z0.dtype)
    xty = z0.t().mm(z1)
    w = torch.linalg.solve(xtx + reg, xty)
    return w.t().contiguous()


@torch.no_grad()
def estimate_transition_residual_proxy(
    phi_seq: torch.Tensor,
    *,
    ridge: float = 1.0e-3,
    koopman_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate a simple transition residual proxy [B,T,1] from a global Koopman fit."""
    if phi_seq.ndim != 3 or phi_seq.shape[1] < 2:
        raise ValueError("estimate_transition_residual_proxy expects [B,T,D] with T>=2")
    phi_now = phi_seq[:, :-1, :].reshape(-1, phi_seq.shape[-1])
    phi_next = phi_seq[:, 1:, :].reshape(-1, phi_seq.shape[-1])
    if koopman_k is None:
        koopman_k = _fit_ridge_koopman_local(phi_now, phi_next, ridge=float(ridge))
    pred = torch.einsum("ij,btj->bti", koopman_k, phi_seq[:, :-1, :])
    resid = torch.sqrt((phi_seq[:, 1:, :] - pred).pow(2).mean(dim=-1, keepdim=True) + 1.0e-12)
    pad = torch.zeros_like(resid[:, :1, :])
    proxy = torch.cat([pad, resid], dim=1)
    return proxy, koopman_k


@torch.no_grad()
def run_kmeans(
    x: torch.Tensor,
    num_clusters: int,
    iters: int = 25,
    seed: int = 3407,
    enable_tqdm: bool = False,
    tqdm_desc: str = "hokp kmeans",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple Lloyd k-means on [N, C] coarse observables."""
    if x.ndim != 2:
        raise ValueError(f"expected [N,C], got {tuple(x.shape)}")
    n, dim = x.shape
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
            else:
                refill = torch.randint(0, n, size=(1,), generator=gen, device=x.device).item()
                new_centers[j] = x[refill]
        if torch.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return centers, assign


@torch.no_grad()
def fit_local_pca_charts(
    u_flat: torch.Tensor,
    c_flat: torch.Tensor,
    num_charts: int,
    manifold_rank: int,
    tau: float,
    kmeans_iters: int = 25,
    seed: int = 3407,
    enable_tqdm: bool = False,
) -> LocalPcaCharts:
    """Fit local affine PCA charts in aligned latent space, clustered in coarse space."""
    if u_flat.ndim != 2 or c_flat.ndim != 2:
        raise ValueError("fit_local_pca_charts expects [N,D] and [N,C]")
    if u_flat.shape[0] != c_flat.shape[0]:
        raise ValueError("sample count mismatch between u_flat and c_flat")
    centers, assign = run_kmeans(
        c_flat,
        num_clusters=num_charts,
        iters=kmeans_iters,
        seed=seed,
        enable_tqdm=enable_tqdm,
    )
    global_mean = u_flat.mean(dim=0, keepdim=True)
    global_x = u_flat - global_mean
    _, _, global_vh = torch.linalg.svd(global_x, full_matrices=False)
    rank = max(1, min(int(manifold_rank), int(global_vh.shape[0])))
    global_basis = global_vh[:rank].t().contiguous()
    means = []
    bases = []
    for j in range(int(num_charts)):
        mask = assign == j
        if int(mask.sum().item()) <= rank:
            means.append(global_mean[0])
            bases.append(global_basis)
            continue
        u_local = u_flat[mask]
        mean = u_local.mean(dim=0, keepdim=True)
        xc = u_local - mean
        _, _, vh = torch.linalg.svd(xc, full_matrices=False)
        basis = vh[:rank].t().contiguous()
        means.append(mean[0])
        bases.append(basis)
    return LocalPcaCharts(
        centers=centers,
        means=torch.stack(means, dim=0),
        bases=torch.stack(bases, dim=0),
        tau=float(tau),
    )


@torch.no_grad()
def fit_local_koopman_charts(
    phi_seq: torch.Tensor,
    chart_key_seq: torch.Tensor,
    *,
    num_charts: int,
    ridge: float = 1.0e-3,
    rho_max: float = 0.99,
    kmeans_iters: int = 25,
    seed: int = 3407,
    enable_tqdm: bool = False,
    sample_weight_seq: Optional[torch.Tensor] = None,
) -> LocalKoopmanCharts:
    """Fit local Koopman operators on chart-conditioned transition pairs."""
    if phi_seq.ndim != 3 or chart_key_seq.ndim != 3:
        raise ValueError("fit_local_koopman_charts expects [B,T,D] and [B,T,K]")
    if phi_seq.shape[:2] != chart_key_seq.shape[:2]:
        raise ValueError("sequence shape mismatch between phi_seq and chart_key_seq")
    if phi_seq.shape[1] < 2:
        raise ValueError("phi_seq must have T>=2")

    phi_now = phi_seq[:, :-1, :].reshape(-1, phi_seq.shape[-1])
    phi_next = phi_seq[:, 1:, :].reshape(-1, phi_seq.shape[-1])
    key_now = chart_key_seq[:, :-1, :].reshape(-1, chart_key_seq.shape[-1])
    weight_now = None
    if sample_weight_seq is not None:
        if sample_weight_seq.ndim == 3 and sample_weight_seq.shape[-1] == 1:
            sample_weight_seq = sample_weight_seq[..., 0]
        if sample_weight_seq.ndim != 2 or sample_weight_seq.shape[:2] != phi_seq.shape[:2]:
            raise ValueError("sample_weight_seq must be [B,T] or [B,T,1]")
        weight_now = sample_weight_seq[:, :-1].reshape(-1).to(phi_now.device, phi_now.dtype)

    global_k = clip_spectral_radius(
        _fit_ridge_koopman_local(phi_now, phi_next, ridge=float(ridge), sample_weight=weight_now),
        rho_max=float(rho_max),
    )
    centers, assign = run_kmeans(
        key_now,
        num_clusters=num_charts,
        iters=kmeans_iters,
        seed=seed,
        enable_tqdm=enable_tqdm,
        tqdm_desc="hokp local K kmeans",
    )
    ks = []
    dim = int(phi_now.shape[-1])
    min_samples = max(dim + 1, 8)
    for j in range(int(num_charts)):
        mask = assign == j
        if int(mask.sum().item()) < min_samples:
            ks.append(global_k)
            continue
        local_weight = weight_now[mask] if weight_now is not None else None
        local_k = _fit_ridge_koopman_local(
            phi_now[mask],
            phi_next[mask],
            ridge=float(ridge),
            sample_weight=local_weight,
        )
        local_k = clip_spectral_radius(local_k, rho_max=float(rho_max))
        ks.append(local_k)
    return LocalKoopmanCharts(
        centers=centers,
        koopman_k=torch.stack(ks, dim=0),
        tau=10.0,
    )


def project_local_pca_manifold(
    u_flat: torch.Tensor,
    c_flat: torch.Tensor,
    charts: LocalPcaCharts,
    *,
    return_confidence: bool = False,
) -> torch.Tensor:
    """Soft local PCA manifold projection for aligned latents [N, D]."""
    if u_flat.ndim != 2 or c_flat.ndim != 2:
        raise ValueError("project_local_pca_manifold expects [N,D] and [N,C]")
    centers = charts.centers.to(u_flat.device, u_flat.dtype)
    means = charts.means.to(u_flat.device, u_flat.dtype)
    bases = charts.bases.to(u_flat.device, u_flat.dtype)
    tau = float(charts.tau)
    dist2 = (c_flat[:, None, :] - centers[None, :, :]).pow(2).sum(dim=-1)
    alpha = torch.softmax(-tau * dist2, dim=1)
    diff = u_flat[:, None, :] - means[None, :, :]
    coeff = torch.einsum("nmd,mdr->nmr", diff, bases)
    recon = means[None, :, :] + torch.einsum("nmr,mdr->nmd", coeff, bases)
    proj = (alpha[..., None] * recon).sum(dim=1)
    if not return_confidence:
        return proj
    confidence = alpha.max(dim=1).values
    return proj, confidence


def mix_local_koopman_k(
    key_flat: torch.Tensor,
    charts: LocalKoopmanCharts,
    *,
    return_confidence: bool = False,
):
    """Mix local Koopman operators from chart keys [N,K]."""
    if key_flat.ndim != 2:
        raise ValueError("mix_local_koopman_k expects [N,K]")
    centers = charts.centers.to(key_flat.device, key_flat.dtype)
    koopman_k = charts.koopman_k.to(key_flat.device, key_flat.dtype)
    tau = float(charts.tau)
    dist2 = (key_flat[:, None, :] - centers[None, :, :]).pow(2).sum(dim=-1)
    alpha = torch.softmax(-tau * dist2, dim=1)
    k_mix = torch.einsum("nm,mij->nij", alpha, koopman_k)
    if not return_confidence:
        return k_mix
    confidence = alpha.max(dim=1).values
    return k_mix, confidence


def compute_hokp_dyn_residual(phi_seq: torch.Tensor, koopman_k: torch.Tensor) -> torch.Tensor:
    if phi_seq.ndim != 3 or phi_seq.shape[1] < 2:
        raise ValueError("compute_hokp_dyn_residual expects [B,T,D] with T>=2")
    phi_now = phi_seq[:, :-1, :]
    phi_next = phi_seq[:, 1:, :]
    phi_pred = torch.einsum("ij,btj->bti", koopman_k, phi_now)
    return (phi_next - phi_pred).pow(2).mean()


def compute_local_hokp_dyn_residual(
    phi_seq: torch.Tensor,
    chart_key_seq: torch.Tensor,
    local_charts: LocalKoopmanCharts,
) -> torch.Tensor:
    """Compute one-step dyn residual with chart-conditioned mixed Koopman operators."""
    if phi_seq.ndim != 3 or chart_key_seq.ndim != 3:
        raise ValueError("compute_local_hokp_dyn_residual expects [B,T,D] and [B,T,K]")
    if phi_seq.shape[:2] != chart_key_seq.shape[:2] or phi_seq.shape[1] < 2:
        raise ValueError("sequence mismatch or T<2")
    phi_now = phi_seq[:, :-1, :]
    phi_next = phi_seq[:, 1:, :]
    key_now = chart_key_seq[:, :-1, :].reshape(-1, chart_key_seq.shape[-1])
    k_mix = mix_local_koopman_k(key_now, local_charts).reshape(
        phi_now.shape[0], phi_now.shape[1], phi_now.shape[-1], phi_now.shape[-1]
    )
    phi_pred = torch.einsum("btij,btj->bti", k_mix, phi_now)
    return (phi_next - phi_pred).pow(2).mean()
