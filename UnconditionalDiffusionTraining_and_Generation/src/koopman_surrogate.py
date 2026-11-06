import math
from typing import Dict, Optional

import numpy as np
import torch as th
import torch.nn.functional as F


class DistilledSkeletonRegularizer:
    """
    Lightweight online surrogate for distilled Koopman skeleton regularization.

    The artifact is expected to provide either:
    - `A_eff`, `b_eff` for absolute mode, or
    - `A_dyn`, `b_dyn` plus representation statistics for general mode.
    """

    def __init__(self, artifact_path: str, koopman_cfg: dict, diffusion, device: th.device):
        self.artifact_path = artifact_path
        self.koopman_cfg = koopman_cfg or {}
        self.diffusion = diffusion
        self.device = device

        artifact = th.load(artifact_path, map_location="cpu")
        if not isinstance(artifact, dict):
            raise ValueError(f"invalid surrogate artifact: {artifact_path}")
        self.artifact = artifact

        self.case = artifact.get("case", None)
        self.dim_x = int(artifact.get("dim_x", 0) or 0)
        self.dim_z_dyn = int(artifact.get("dim_z_dyn", 0) or 0)
        self.window_size = int(artifact.get("window_size", 0) or 0)
        self.delay_l = int(artifact.get("delay_L", 0) or 0)
        self.repr_mode = str(artifact.get("repr_mode", self.koopman_cfg.get("repr_mode", "absolute"))).lower()
        self.audit_info = artifact.get("audit", {}) or {}

        if self.delay_l != 0:
            raise ValueError(
                f"distilled_skeleton first version only supports delay_L=0, got {self.delay_l}"
            )
        if self.repr_mode not in ("absolute", "delta_from_first"):
            raise ValueError(f"unsupported repr_mode in artifact: {self.repr_mode}")

        self.loss_cfg = self.koopman_cfg.get("loss", {}) or {}
        self.noise_gate_cfg = self.koopman_cfg.get("noise_gate", {}) or {}
        self.reliable_cfg = self.koopman_cfg.get("reliable", {}) or {}
        self.audit_cfg = self.koopman_cfg.get("audit", {}) or {}

        self.interval = max(1, int(self.koopman_cfg.get("interval", 1)))
        self.start_step = max(0, int(self.koopman_cfg.get("start_step", 0)))
        self.ramp_steps = max(0, int(self.koopman_cfg.get("ramp_steps", 0)))
        self.lambda_base = float(self.koopman_cfg.get("lambda_base", 0.0))
        self.ratio_cap_vs_diff = float(self.koopman_cfg.get("ratio_cap_vs_diff", 0.0))
        self.charbonnier_eps = float(self.loss_cfg.get("charbonnier_eps", 1.0e-3))
        self.scale_eps = float(self.loss_cfg.get("scale_eps", 1.0e-8))
        self.beta = float(self.noise_gate_cfg.get("beta", 8.0))
        self.sigma_max = float(self.noise_gate_cfg.get("sigma_max", 0.60))
        self.use_mask = bool(self.reliable_cfg.get("use_mask", True))
        self.audit_enabled = bool(self.audit_cfg.get("enabled", True))
        self.audit_subset_size = max(1, int(self.audit_cfg.get("subset_size", 128)))

        self._alphas_cumprod = None
        if diffusion is not None and hasattr(diffusion, "alphas_cumprod"):
            self._alphas_cumprod = th.from_numpy(diffusion.alphas_cumprod).float().to(device)

        self.K = self._to_tensor(artifact.get("K", None), "K", expected_ndim=2)
        self.resid_scale = self._to_tensor(
            artifact.get("resid_scale_per_dim", None),
            "resid_scale_per_dim",
            expected_ndim=1,
        )
        self.reliable_mask_train = self._to_bool_tensor(artifact.get("reliable_mask_train", None), "reliable_mask_train")
        self.reliable_mask_valid = self._to_bool_tensor(artifact.get("reliable_mask_valid", None), "reliable_mask_valid")

        self.A_eff = None
        self.b_eff = None
        if self.repr_mode == "absolute" and ("A_eff" in artifact and "b_eff" in artifact):
            self.A_eff = self._to_tensor(artifact.get("A_eff"), "A_eff", expected_ndim=2)
            self.b_eff = self._to_tensor(artifact.get("b_eff"), "b_eff", expected_ndim=1)

        self.A_dyn = self._to_tensor(artifact.get("A_dyn", None), "A_dyn", expected_ndim=2)
        self.b_dyn = self._to_tensor(artifact.get("b_dyn", None), "b_dyn", expected_ndim=1)

        self.ugc_min = self._to_tensor(artifact.get("ugc_min", None), "ugc_min")
        self.ugc_max = self._to_tensor(artifact.get("ugc_max", None), "ugc_max")
        self.data_mean = self._to_tensor(artifact.get("data_mean", None), "data_mean")
        self.data_std = self._to_tensor(artifact.get("data_std", None), "data_std")
        self.repr_mean = self._to_tensor(artifact.get("repr_mean", None), "repr_mean")
        self.repr_std = self._to_tensor(artifact.get("repr_std", None), "repr_std")

        if self.A_eff is None and (self.A_dyn is None or self.b_dyn is None):
            raise ValueError("artifact must provide either A_eff/b_eff or A_dyn/b_dyn")
        if self.dim_x <= 0 or self.dim_z_dyn <= 0:
            raise ValueError("artifact missing dim_x or dim_z_dyn")
        if self.window_size <= 0:
            raise ValueError("artifact missing window_size")
        if self.K.shape != (self.dim_z_dyn, self.dim_z_dyn):
            raise ValueError(f"invalid K shape: {tuple(self.K.shape)}")
        if self.resid_scale.shape[0] != self.dim_z_dyn:
            raise ValueError(
                f"resid_scale_per_dim shape mismatch: {tuple(self.resid_scale.shape)} vs dim_z_dyn={self.dim_z_dyn}"
            )

    def _to_tensor(self, value, name: str, expected_ndim: Optional[int] = None):
        if value is None:
            return None
        t = th.as_tensor(value, dtype=th.float32, device=self.device)
        if expected_ndim is not None and t.ndim != expected_ndim:
            raise ValueError(f"{name} expected ndim={expected_ndim}, got shape={tuple(t.shape)}")
        return t

    def _to_bool_tensor(self, value, name: str):
        if value is None:
            raise ValueError(f"artifact missing {name}")
        t = th.as_tensor(value, dtype=th.bool, device=self.device).flatten()
        return t

    @staticmethod
    def _to_sequence(x0_hat: th.Tensor) -> th.Tensor:
        if x0_hat.dim() == 4 and x0_hat.shape[1] == 1:
            return x0_hat[:, 0]
        if x0_hat.dim() == 3:
            return x0_hat
        raise ValueError(f"unexpected x0_hat shape: {tuple(x0_hat.shape)}")

    def _canonicalize_1x1xd(self, tensor: Optional[th.Tensor], dim_x: int) -> Optional[th.Tensor]:
        if tensor is None:
            return None
        t = tensor.to(device=self.device, dtype=th.float32)
        if t.ndim == 0:
            return t.view(1, 1, 1).expand(1, 1, dim_x)
        if t.ndim == 1:
            if t.shape[0] == dim_x:
                return t.view(1, 1, dim_x)
            if t.shape[0] == 1:
                return t.view(1, 1, 1).expand(1, 1, dim_x)
        if t.ndim == 2:
            if t.shape == (1, dim_x):
                return t.view(1, 1, dim_x)
            if t.shape == (1, 1):
                return t.view(1, 1, 1).expand(1, 1, dim_x)
        if t.ndim == 3:
            if t.shape[0] == 1 and t.shape[1] == 1 and t.shape[2] in (1, dim_x):
                if t.shape[2] == 1:
                    return t.expand(1, 1, dim_x)
                return t
        raise ValueError(f"cannot canonicalize tensor to [1,1,{dim_x}]: shape={tuple(t.shape)}")

    def _build_repr_input(self, x_seq: th.Tensor) -> th.Tensor:
        dim_x = x_seq.shape[-1]
        ugc_min = self._canonicalize_1x1xd(self.ugc_min, dim_x)
        ugc_max = self._canonicalize_1x1xd(self.ugc_max, dim_x)
        if ugc_min is None or ugc_max is None:
            raise ValueError("artifact missing ugc_min/ugc_max")
        raw = (x_seq + 1.0) * (ugc_max - ugc_min) / 2.0 + ugc_min

        data_mean = self._canonicalize_1x1xd(self.data_mean, dim_x)
        data_std = self._canonicalize_1x1xd(self.data_std, dim_x)
        if data_mean is not None and data_std is not None:
            raw = (raw - data_mean) / (data_std + self.scale_eps)

        if self.repr_mode == "delta_from_first":
            raw = raw - raw[:, :1, :]

        repr_mean = self._canonicalize_1x1xd(self.repr_mean, dim_x)
        repr_std = self._canonicalize_1x1xd(self.repr_std, dim_x)
        if repr_mean is not None and repr_std is not None:
            raw = (raw - repr_mean) / (repr_std + self.scale_eps)
        return raw

    def _project_dyn(self, x_seq: th.Tensor) -> th.Tensor:
        if self.A_eff is not None and self.b_eff is not None and self.repr_mode == "absolute":
            return F.linear(x_seq, self.A_eff, self.b_eff)
        x_repr = self._build_repr_input(x_seq)
        return F.linear(x_repr, self.A_dyn, self.b_dyn)

    def _one_step_residual(self, z_dyn: th.Tensor) -> th.Tensor:
        if z_dyn.shape[1] < 2:
            raise ValueError("sequence length must be >=2 for one-step residual")
        z_now = z_dyn[:, :-1, :]
        z_next = z_dyn[:, 1:, :]
        z_pred = th.einsum("ij,btj->bti", self.K, z_now)
        return z_next - z_pred

    def _noise_gate(self, t: th.Tensor) -> th.Tensor:
        if self._alphas_cumprod is None:
            return th.ones_like(t, dtype=th.float32, device=self.device)
        alpha_bar = self._alphas_cumprod.to(device=t.device)[t.long()]
        sigma = th.sqrt(th.clamp(1.0 - alpha_bar, min=0.0))
        gate = th.exp(-self.beta * sigma)
        gate = th.where(sigma <= self.sigma_max, gate, th.zeros_like(gate))
        return gate

    def _lookup_reliable(self, batch_indices: th.Tensor, split: str) -> th.Tensor:
        if not self.use_mask:
            return th.ones_like(batch_indices, dtype=th.bool, device=self.device)
        mask_table = self.reliable_mask_train if split == "train" else self.reliable_mask_valid
        idx = batch_indices.long().to(self.device)
        if idx.numel() == 0:
            return th.zeros_like(idx, dtype=th.bool)
        if int(idx.max().item()) >= int(mask_table.numel()):
            raise ValueError(
                f"batch index out of range for {split} reliable mask: max_idx={int(idx.max().item())}, size={int(mask_table.numel())}"
            )
        return mask_table[idx]

    def _ramp(self, global_step: int) -> float:
        if global_step < self.start_step:
            return 0.0
        if self.ramp_steps <= 0:
            return 1.0
        progress = float(global_step - self.start_step + 1) / float(self.ramp_steps)
        return max(0.0, min(1.0, progress))

    def _lambda_eff(self, diff_loss_scalar: th.Tensor, loss_raw: th.Tensor, global_step: int) -> th.Tensor:
        if loss_raw.detach().item() <= 0.0:
            return loss_raw.new_tensor(0.0)
        ramp = self._ramp(global_step)
        if ramp <= 0.0:
            return loss_raw.new_tensor(0.0)
        base = loss_raw.new_tensor(self.lambda_base * ramp)
        if self.ratio_cap_vs_diff <= 0.0:
            return base
        cap = self.ratio_cap_vs_diff * diff_loss_scalar.detach() / (loss_raw.detach() + self.scale_eps)
        cap = th.clamp(cap, min=0.0)
        return th.minimum(base, cap)

    def _compute_per_sample(self, x0_hat: th.Tensor, t: th.Tensor, batch_indices: th.Tensor, split: str):
        x_seq = self._to_sequence(x0_hat)
        if x_seq.shape[-1] != self.dim_x:
            raise ValueError(f"artifact dim_x={self.dim_x} but batch has D={x_seq.shape[-1]}")
        z_dyn = self._project_dyn(x_seq)
        resid = self._one_step_residual(z_dyn)
        resid_norm = resid / (self.resid_scale.view(1, 1, -1) + self.scale_eps)
        robust = th.sqrt(resid_norm.pow(2) + self.charbonnier_eps ** 2) - self.charbonnier_eps
        per_sample = robust.mean(dim=(1, 2))
        gate = self._noise_gate(t).to(per_sample.dtype)
        reliable = self._lookup_reliable(batch_indices, split=split)
        active = reliable & (gate > 0)
        weighted = per_sample * gate
        resid_abs_median = float(resid.detach().abs().median().item()) if resid.numel() > 0 else 0.0
        return {
            "per_sample": per_sample,
            "weighted": weighted,
            "gate": gate,
            "reliable": reliable,
            "active": active,
            "resid_abs_median": resid_abs_median,
        }

    def compute_train_loss(
        self,
        x0_hat: th.Tensor,
        t: th.Tensor,
        batch_indices: th.Tensor,
        global_step: int,
        diff_loss_scalar: th.Tensor,
    ) -> Dict[str, object]:
        if global_step < self.start_step or (global_step % self.interval) != 0:
            zero = diff_loss_scalar.new_tensor(0.0)
            return {
                "active": False,
                "loss_raw": zero,
                "loss": zero,
                "lambda_eff": 0.0,
                "active_frac": 0.0,
                "reliable_frac": 0.0,
                "noise_gate_mean": 0.0,
                "resid_abs_median": 0.0,
            }

        stats = self._compute_per_sample(x0_hat, t, batch_indices, split="train")
        active = stats["active"]
        reliable = stats["reliable"]
        gate = stats["gate"]
        weighted = stats["weighted"]
        active_frac = float(active.float().mean().item()) if active.numel() > 0 else 0.0
        reliable_frac = float(reliable.float().mean().item()) if reliable.numel() > 0 else 0.0
        noise_gate_mean = float(gate.mean().item()) if gate.numel() > 0 else 0.0
        if active.sum().item() <= 0:
            zero = diff_loss_scalar.new_tensor(0.0)
            return {
                "active": False,
                "loss_raw": zero,
                "loss": zero,
                "lambda_eff": 0.0,
                "active_frac": active_frac,
                "reliable_frac": reliable_frac,
                "noise_gate_mean": noise_gate_mean,
                "resid_abs_median": float(stats["resid_abs_median"]),
            }

        loss_raw = weighted[active].mean()
        lambda_eff = self._lambda_eff(diff_loss_scalar, loss_raw, global_step)
        loss = lambda_eff * loss_raw
        return {
            "active": True,
            "loss_raw": loss_raw,
            "loss": loss,
            "lambda_eff": float(lambda_eff.detach().item()),
            "active_frac": active_frac,
            "reliable_frac": reliable_frac,
            "noise_gate_mean": noise_gate_mean,
            "resid_abs_median": float(stats["resid_abs_median"]),
        }

    def compute_valid_audit(
        self,
        x0_hat: th.Tensor,
        t: th.Tensor,
        batch_indices: th.Tensor,
    ) -> Dict[str, float]:
        stats = self._compute_per_sample(x0_hat, t, batch_indices, split="valid")
        active = stats["active"]
        reliable = stats["reliable"]
        gate = stats["gate"]
        weighted = stats["weighted"]
        count = int(batch_indices.shape[0])
        active_count = int(active.sum().item())
        loss_raw = float(weighted[active].mean().item()) if active_count > 0 else 0.0
        return {
            "count": float(count),
            "active_count": float(active_count),
            "loss_raw": loss_raw,
            "active_frac": float(active.float().mean().item()) if count > 0 else 0.0,
            "reliable_frac": float(reliable.float().mean().item()) if count > 0 else 0.0,
            "noise_gate_mean": float(gate.mean().item()) if count > 0 else 0.0,
            "resid_abs_median": float(stats["resid_abs_median"]),
        }
