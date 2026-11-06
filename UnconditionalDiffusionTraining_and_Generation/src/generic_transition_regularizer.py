from typing import Dict, Optional

import torch as th


class GenericTransitionRegularizer:
    """
    Config-gated generic transition penalties for post-review baselines.

    This module intentionally avoids DNS-calibrated regimes, residual scales, or
    fitted priors. It only penalizes simple predicted-clean latent smoothness,
    making it a generic baseline for the LSKP transition prior.
    """

    def __init__(self, diffusion, cfg=None, device: Optional[th.device] = None):
        self.diffusion = diffusion
        self.cfg = dict(cfg or {})
        self.device = device if device is not None else th.device("cpu")
        self.mode = str(self.cfg.get("mode", "velocity_l2")).lower()
        self.warmup_steps = max(0, int(self.cfg.get("warmup_steps", 0) or 0))
        self.interval = max(1, int(self.cfg.get("interval", 1) or 1))
        self.subbatch = max(1, int(self.cfg.get("subbatch", 1) or 1))
        self.window_size = max(0, int(self.cfg.get("window_size", 0) or 0))
        self.sigma_max = float(self.cfg.get("sigma_max", 1.0))
        self.lambda_base = float(self.cfg.get("lambda_base", 0.0))
        self.ratio_cap_vs_diff = float(self.cfg.get("ratio_cap_vs_diff", 0.0))
        self.ramp_steps = max(0, int(self.cfg.get("ramp_steps", 0) or 0))
        self.normalize_by_signal = bool(self.cfg.get("normalize_by_signal", False))
        self.scale_eps = float(self.cfg.get("scale_eps", 1.0e-8))

        if self.mode not in {"velocity_l2", "acceleration_l2"}:
            raise ValueError(
                "generic_regularizer.mode must be one of "
                "{velocity_l2, acceleration_l2}; got %s" % self.mode
            )
        if self.lambda_base < 0.0:
            raise ValueError("generic_regularizer.lambda_base must be non-negative")
        if self.ratio_cap_vs_diff < 0.0:
            raise ValueError("generic_regularizer.ratio_cap_vs_diff must be non-negative")

        alphas = getattr(diffusion, "alphas_cumprod", None)
        if alphas is None:
            raise ValueError("diffusion object must expose alphas_cumprod")
        self._alphas_cumprod = th.as_tensor(alphas, dtype=th.float32, device=self.device)

    @staticmethod
    def _zero_like_ref(ref: th.Tensor) -> th.Tensor:
        return th.zeros((), device=ref.device, dtype=ref.dtype)

    def _inactive_result(self, ref: th.Tensor) -> Dict[str, object]:
        zero = self._zero_like_ref(ref)
        return {
            "loss": zero,
            "loss_raw": zero,
            "lambda_eff": 0.0,
            "active_frac": 0.0,
            "subbatch_frac": 0.0,
            "noise_gate_mean": 0.0,
            "num_selected": 0,
        }

    @staticmethod
    def _to_sequence(x0_hat: th.Tensor) -> th.Tensor:
        if x0_hat.ndim == 4 and x0_hat.shape[1] == 1:
            return x0_hat[:, 0]
        if x0_hat.ndim == 3:
            return x0_hat
        raise ValueError(
            "generic transition regularizer expects x0_hat shaped "
            "[B, 1, T, D] or [B, T, D], got %s" % (tuple(x0_hat.shape),)
        )

    def _noise_gate(self, t: th.Tensor) -> th.Tensor:
        alpha_bar = self._alphas_cumprod[t.long()].to(device=t.device, dtype=th.float32)
        sigma = th.sqrt(th.clamp(1.0 - alpha_bar, min=0.0))
        return sigma <= float(self.sigma_max)

    def _select_subbatch(self, x_seq: th.Tensor, t: th.Tensor):
        active = self._noise_gate(t)
        active_idx = th.nonzero(active, as_tuple=False).flatten()
        if active_idx.numel() == 0:
            return None, active
        if active_idx.numel() > self.subbatch:
            active_idx = active_idx[: self.subbatch]
        return x_seq.index_select(0, active_idx), active

    def _window(self, x_seq: th.Tensor) -> th.Tensor:
        if self.window_size <= 0 or self.window_size >= x_seq.shape[1]:
            return x_seq
        return x_seq[:, : self.window_size]

    def _raw_loss_per_sample(self, x_seq: th.Tensor) -> th.Tensor:
        x_win = self._window(x_seq)
        if self.mode == "velocity_l2":
            if x_win.shape[1] < 2:
                return th.zeros((x_win.shape[0],), device=x_win.device, dtype=x_win.dtype)
            residual = x_win[:, 1:] - x_win[:, :-1]
        else:
            if x_win.shape[1] < 3:
                return th.zeros((x_win.shape[0],), device=x_win.device, dtype=x_win.dtype)
            residual = x_win[:, 2:] - 2.0 * x_win[:, 1:-1] + x_win[:, :-2]

        loss = residual.pow(2).flatten(1).mean(dim=1)
        if self.normalize_by_signal:
            denom = x_win.pow(2).flatten(1).mean(dim=1).clamp_min(self.scale_eps)
            loss = loss / denom
        return loss

    def _ramp(self, global_step: int) -> float:
        if global_step < self.warmup_steps:
            return 0.0
        if self.ramp_steps <= 0:
            return 1.0
        progress = (global_step - self.warmup_steps) / float(self.ramp_steps)
        return max(0.0, min(1.0, progress))

    def _effective_lambda(self, loss_raw: th.Tensor, diff_loss_scalar: th.Tensor, global_step: int) -> float:
        base = self.lambda_base * self._ramp(global_step)
        if base <= 0.0:
            return 0.0
        if self.ratio_cap_vs_diff <= 0.0:
            return float(base)
        raw = float(loss_raw.detach().item())
        diff = float(diff_loss_scalar.detach().item())
        if raw <= self.scale_eps or diff <= 0.0:
            return 0.0
        cap = self.ratio_cap_vs_diff * diff / (raw + self.scale_eps)
        return float(min(base, cap))

    def compute_train_loss(
        self,
        *,
        x0_hat: th.Tensor,
        t: th.Tensor,
        global_step: int,
        diff_loss_scalar: th.Tensor,
    ) -> Dict[str, object]:
        if self.lambda_base <= 0.0:
            return self._inactive_result(diff_loss_scalar)
        if global_step < self.warmup_steps:
            return self._inactive_result(diff_loss_scalar)
        if self.interval > 1 and (global_step % self.interval) != 0:
            return self._inactive_result(diff_loss_scalar)

        x_seq = self._to_sequence(x0_hat)
        selected, active = self._select_subbatch(x_seq, t)
        if selected is None:
            result = self._inactive_result(diff_loss_scalar)
            result["active_frac"] = float(active.float().mean().detach().item())
            result["noise_gate_mean"] = result["active_frac"]
            return result

        loss_per_sample = self._raw_loss_per_sample(selected)
        loss_raw = loss_per_sample.mean()
        lambda_eff = self._effective_lambda(loss_raw, diff_loss_scalar, global_step)
        loss = loss_raw * float(lambda_eff)
        active_frac = float(active.float().mean().detach().item())
        subbatch_frac = float(selected.shape[0]) / max(1.0, float(x_seq.shape[0]))
        return {
            "loss": loss,
            "loss_raw": loss_raw,
            "lambda_eff": float(lambda_eff),
            "active_frac": active_frac,
            "subbatch_frac": subbatch_frac,
            "noise_gate_mean": active_frac,
            "num_selected": int(selected.shape[0]),
        }
