import torch as th


class LegacyKoopmanRegularizer:
    """
    Koopman regularization with an energy-normalized Huber loss.

    Undo min-max normalization of x0_hat, then standardize using the Koopman
    training statistics. Normalize the Huber numerator by z_true energy and
    weight the result by exp(-beta * noise_level).
    """

    def __init__(self, diffusion, koopman_model, koopman_cfg=None, koopman_norm=None):
        self.diffusion = diffusion
        self.koopman_model = koopman_model
        self.koopman_cfg = koopman_cfg or {}
        self.koopman_norm = koopman_norm or {}
        self.repr_mode = str(self.koopman_cfg.get("repr_mode", "absolute")).lower()
        if self.repr_mode not in ("absolute", "delta_from_first"):
            raise ValueError(f"Unknown koopman repr_mode: {self.repr_mode}")
        self._alphas_cumprod = None
        if diffusion is not None and hasattr(diffusion, "alphas_cumprod"):
            self._alphas_cumprod = th.from_numpy(diffusion.alphas_cumprod).float()

    def _to_sequence(self, x0_hat):
        if x0_hat.dim() == 4 and x0_hat.shape[1] == 1:
            return x0_hat[:, 0]
        return x0_hat

    def _apply_repr_mode(self, x_windows):
        if self.repr_mode == "absolute":
            return x_windows
        if self.repr_mode == "delta_from_first":
            return x_windows - x_windows[:, :1, :]
        raise ValueError(f"Unknown koopman repr_mode: {self.repr_mode}")

    def _build_windows(self, x_seq, window_size):
        bsz, tlen, _ = x_seq.shape
        if tlen < window_size:
            return None, bsz, 0, 0
        # Construct sliding windows explicitly to preserve their order.
        windows = []
        for start in range(0, tlen - window_size + 1):
            windows.append(x_seq[:, start : start + window_size])
        x_windows = th.stack(windows, dim=1)
        bsz, num_win, lp1, dim_x = x_windows.shape
        x_windows = x_windows.reshape(bsz * num_win, lp1, dim_x)
        return x_windows, bsz, num_win, lp1

    def _denorm_minmax(self, x_windows, lp1, dim_x):
        del lp1, dim_x
        min_val = self.koopman_norm.get("min", None)
        max_val = self.koopman_norm.get("max", None)
        if min_val is None or max_val is None:
            return x_windows
        # Undo the [-1,1] normalization used in diffusion training.
        min_t = th.tensor(min_val, device=x_windows.device, dtype=x_windows.dtype)
        max_t = th.tensor(max_val, device=x_windows.device, dtype=x_windows.dtype)
        return (x_windows + 1.0) * (max_t - min_t) / 2.0 + min_t

    def _norm_standardize(self, x_windows, lp1, dim_x):
        del dim_x
        mean = self.koopman_norm.get("mean", None)
        std = self.koopman_norm.get("std", None)
        if mean is None or std is None:
            return x_windows
        # Use the standardization statistics saved during KoopmanAE training.
        mean_t = th.tensor(mean, device=x_windows.device, dtype=x_windows.dtype)
        std_t = th.tensor(std, device=x_windows.device, dtype=x_windows.dtype)
        if mean_t.ndim == 3 and mean_t.shape[1] == 1:
            mean_t = mean_t.expand(-1, lp1, -1)
            std_t = std_t.expand(-1, lp1, -1)
        return (x_windows - mean_t) / (std_t + 1e-8)

    def _repr_standardize(self, x_windows, lp1, dim_x):
        del dim_x
        mean = self.koopman_norm.get("repr_mean", None)
        std = self.koopman_norm.get("repr_std", None)
        if mean is None or std is None:
            return x_windows
        mean_t = th.tensor(mean, device=x_windows.device, dtype=x_windows.dtype)
        std_t = th.tensor(std, device=x_windows.device, dtype=x_windows.dtype)
        if mean_t.ndim == 3 and mean_t.shape[1] == 1:
            mean_t = mean_t.expand(-1, lp1, -1)
            std_t = std_t.expand(-1, lp1, -1)
        return (x_windows - mean_t) / (std_t + 1e-8)

    def per_sample_loss(self, x0_hat, window_size):
        x_seq = self._to_sequence(x0_hat)
        bsz = x_seq.shape[0]
        zeros = th.zeros(bsz, device=x_seq.device, dtype=x_seq.dtype)
        delay_l = int(getattr(self.koopman_model, "delay_L", 0))
        if window_size <= delay_l + 1:
            raise ValueError(
                f"window_size={window_size} is too small; require window_size >= delay_L+2 (delay_L={delay_l})"
            )

        x_windows, bsz, num_win, lp1 = self._build_windows(x_seq, window_size)
        if x_windows is None:
            return {
                "loss_per_sample": zeros,
                "num_per_sample": zeros,
                "den_per_sample": zeros,
                "stats": {},
            }

        dim_x = x_windows.shape[-1]
        x_windows = self._denorm_minmax(x_windows, lp1, dim_x)
        x_windows = self._norm_standardize(x_windows, lp1, dim_x)
        x_windows = self._apply_repr_mode(x_windows)
        x_windows = self._repr_standardize(x_windows, lp1, dim_x)

        _, z_pred, z_true = self.koopman_model(x_windows)
        if z_pred.shape[1] == 0:
            raise ValueError(
                "Koopman supervision has no time steps; increase window_size or decrease delay_L."
            )
        z_pred = z_pred.reshape(bsz, num_win, z_pred.shape[1], z_pred.shape[2])
        z_true = z_true.reshape(bsz, num_win, z_true.shape[1], z_true.shape[2])

        diff = z_pred - z_true
        delta = float(self.koopman_cfg.get("huber_delta", 1.0))
        abs_diff = diff.abs()
        quadratic = 0.5 * diff.pow(2)
        linear = delta * (abs_diff - 0.5 * delta)
        huber = th.where(abs_diff <= delta, quadratic, linear)
        num = huber.mean(dim=(1, 2, 3))
        den = z_true.pow(2).mean(dim=(1, 2, 3))
        # Normalize the Huber numerator by target energy.
        loss_per_sample = num / (den + 1e-8)

        return {
            "loss_per_sample": loss_per_sample,
            "num_per_sample": num,
            "den_per_sample": den,
            "stats": {},
        }

    def sample_weights(self, t, schedule_weights=None):
        del schedule_weights
        beta = float(self.koopman_cfg.get("lambda_k_beta", 4.0))
        alpha_bar = None
        if self._alphas_cumprod is not None and self._alphas_cumprod.numel() > 1:
            alpha_bar = self._alphas_cumprod.to(t.device)[t].float()
            noise_level = th.sqrt(th.clamp(1.0 - alpha_bar, min=0.0))
        else:
            noise_level = th.zeros_like(t, dtype=th.float32, device=t.device)
        # Apply a monotone noise gate.
        weight_koop = th.exp(-beta * noise_level)
        return {
            "weight": weight_koop,
            "beta": beta,
            "alpha_bar": alpha_bar,
            "stats": {},
        }

    @staticmethod
    def reduce_weighted(weight_koop, loss_per_sample, num_per_sample, den_per_sample):
        weight_sum = weight_koop.sum()
        koop_loss = (weight_koop * loss_per_sample).sum() / (weight_sum + 1e-8)
        koop_num = (weight_koop * num_per_sample).sum() / (weight_sum + 1e-8)
        koop_den = (weight_koop * den_per_sample).sum() / (weight_sum + 1e-8)

        n_eff = (weight_sum * weight_sum) / (weight_koop.pow(2).sum() + 1e-8)
        active_frac = (weight_koop > 1e-12).float().mean()
        return {
            "koop_loss": koop_loss,
            "koop_num": koop_num,
            "koop_den": koop_den,
            "stats": {
                "n_eff": n_eff,
                "active_frac": active_frac,
                "weight_sum": weight_sum,
            },
        }

    def state_dict(self):
        """
        This regularizer has no dynamic state to persist.
        """
        return {"class": "LegacyKoopmanRegularizer"}

    def load_state_dict(self, state):
        del state
        return

    def set_lambda_overrides(self, **kwargs):
        del kwargs
        return

    def clear_lambda_overrides(self):
        return


class ModernKoopmanRegularizer(LegacyKoopmanRegularizer):
    """
    Koopman regularization with configurable robustness and noise gating.

    Supports clamp/tanh bounds on x0_hat, shape-checked standardization,
    normalized_huber / normalized_charbonnier / log_charbonnier penalties,
    and bump-shaped noise gates with weight diagnostics.
    """

    def __init__(self, diffusion, koopman_model, koopman_cfg=None, koopman_norm=None):
        super().__init__(diffusion, koopman_model, koopman_cfg=koopman_cfg, koopman_norm=koopman_norm)
        self.preprocess_cfg = (self.koopman_cfg.get("preprocess", {}) or {})
        self.loss_cfg = (self.koopman_cfg.get("loss", {}) or {})
        self.weight_cfg = (self.koopman_cfg.get("weight", {}) or {})
        self.consistency_cfg = (self.koopman_cfg.get("consistency", {}) or {})
        self.dyn_cfg = (self.koopman_cfg.get("dyn", {}) or {})

        self.loss_mode = str(self.loss_cfg.get("mode", "normalized_charbonnier")).lower()
        self.scale_mode = str(self.loss_cfg.get("scale_mode", "ema_median")).lower()
        self.scale_momentum = float(self.loss_cfg.get("scale_momentum", 0.01))
        self.scale_clamp_min = float(self.loss_cfg.get("scale_clamp_min", 1e-4))
        self.scale_clamp_max = float(self.loss_cfg.get("scale_clamp_max", 1e4))
        self.scale_eps = float(self.loss_cfg.get("scale_eps", 1e-8))

        self.charbonnier_eps = float(self.loss_cfg.get("charbonnier_eps", 1e-3))
        self.huber_delta = float(self.loss_cfg.get("huber_delta", 1.0))
        self.huber_delta_mode = str(self.loss_cfg.get("huber_delta_mode", "fixed")).lower()
        self.huber_delta_mult = float(self.loss_cfg.get("huber_delta_mult", 1.5))

        self.clamp_mode = str(self.preprocess_cfg.get("clamp_x0_hat", "none")).lower()
        self.clip_min = float(self.preprocess_cfg.get("clip_min", -1.0))
        self.clip_max = float(self.preprocess_cfg.get("clip_max", 1.0))
        self.validate_norm_shape = bool(self.preprocess_cfg.get("validate_norm_shape", True))

        self.weight_mode = str(self.weight_cfg.get("mode", "exp_only")).lower()
        self.bump_type = str(self.weight_cfg.get("bump_type", "raised_cosine")).lower()
        self.bump_s1 = float(self.weight_cfg.get("s1", 0.3))
        self.bump_s2 = float(self.weight_cfg.get("s2", 0.7))
        self.bump_k = float(self.weight_cfg.get("logistic_k", 0.05))
        self.weight_w_min = float(self.weight_cfg.get("w_min", 0.0))
        self.weight_w_max = float(self.weight_cfg.get("w_max", 1e6))
        self.include_schedule_sampler_weight = bool(self.weight_cfg.get("include_schedule_sampler_weight", False))
        self.lambda_k1 = float(self.dyn_cfg.get("lambda_k1", 1.0))
        self.lambda_k_multi = float(self.dyn_cfg.get("lambda_k_multi", 0.0))
        dyn_multi_cfg = (self.dyn_cfg.get("multi_step", {}) or {})
        self.dyn_multi_enabled = bool(dyn_multi_cfg.get("enabled", False))
        self.dyn_multi_horizon = int(dyn_multi_cfg.get("horizon", 1))

        self.consistency_enabled = bool(self.consistency_cfg.get("enabled", False))
        self.consistency_mode = str(self.consistency_cfg.get("mode", "teacher_forced_aux")).lower()
        legacy_cons_enabled = any(k in self.consistency_cfg for k in ("lambda_cons", "horizon"))
        new_cons_enabled = any(k in self.consistency_cfg for k in ("lambda_cons1", "lambda_cons_multi", "multi_step"))
        if new_cons_enabled:
            self.lambda_cons1 = float(self.consistency_cfg.get("lambda_cons1", self.consistency_cfg.get("lambda_cons", 0.0)))
            self.lambda_cons_multi = float(self.consistency_cfg.get("lambda_cons_multi", 0.0))
            cons_multi_cfg = (self.consistency_cfg.get("multi_step", {}) or {})
            self.cons_multi_enabled = bool(cons_multi_cfg.get("enabled", False))
            self.cons_multi_horizon = int(cons_multi_cfg.get("horizon", self.consistency_cfg.get("horizon", 1)))
        elif legacy_cons_enabled:
            # A combined consistency term averages over h=1..H.
            self.lambda_cons1 = 0.0
            self.lambda_cons_multi = float(self.consistency_cfg.get("lambda_cons", 0.0))
            self.cons_multi_enabled = True
            self.cons_multi_horizon = int(self.consistency_cfg.get("horizon", 1))
        else:
            self.lambda_cons1 = 0.0
            self.lambda_cons_multi = 0.0
            self.cons_multi_enabled = False
            self.cons_multi_horizon = 1
        self.lambda_cons = self.lambda_cons1 + self.lambda_cons_multi
        self.consistency_horizon = self.cons_multi_horizon
        self._lambda_k1_override = None
        self._lambda_k_multi_override = None
        self._lambda_cons1_override = None
        self._lambda_cons_multi_override = None

        self._scale_ema = None
        self._delta_ema = None
        self._noise_rank = None
        if self._alphas_cumprod is not None and self._alphas_cumprod.numel() > 0:
            noise = th.sqrt(th.clamp(1.0 - self._alphas_cumprod, min=0.0))
            order = th.argsort(noise)
            rank = th.empty_like(noise)
            if noise.numel() > 1:
                rank[order] = th.linspace(0.0, 1.0, steps=noise.numel())
            else:
                rank.fill_(0.0)
            self._noise_rank = rank

    @staticmethod
    def _state_to_tensor(value, ref_tensor):
        if value is None:
            return None
        if th.is_tensor(value):
            return value.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        return th.tensor(float(value), device=ref_tensor.device, dtype=ref_tensor.dtype)

    def _canonicalize_norm_tensor(self, value, lp1, dim_x, device, dtype, name):
        t = th.as_tensor(value, device=device, dtype=dtype)

        if t.ndim == 0:
            return t.view(1, 1, 1).expand(1, lp1, dim_x)

        if t.ndim == 1:
            if t.shape[0] == 1:
                return t.view(1, 1, 1).expand(1, lp1, dim_x)
            if t.shape[0] == dim_x:
                return t.view(1, 1, dim_x).expand(1, lp1, dim_x)
            raise ValueError(f"Invalid {name} shape: {tuple(t.shape)}; expected length 1 or dim_x={dim_x}")

        if t.ndim == 2:
            if t.shape == (1, 1):
                return t.view(1, 1, 1).expand(1, lp1, dim_x)
            if t.shape == (1, dim_x):
                return t.view(1, 1, dim_x).expand(1, lp1, dim_x)
            if t.shape == (lp1, dim_x):
                return t.view(1, lp1, dim_x)
            raise ValueError(
                f"Invalid {name} shape: {tuple(t.shape)}; expected (1,1)/(1,{dim_x})/({lp1},{dim_x})"
            )

        if t.ndim == 3:
            if t.shape[0] != 1:
                raise ValueError(f"{name} first dimension must be 1; got {tuple(t.shape)}")
            if t.shape[2] not in (1, dim_x):
                raise ValueError(f"{name} last dimension must be 1 or dim_x={dim_x}; got {tuple(t.shape)}")
            if t.shape[1] not in (1, lp1):
                raise ValueError(f"{name} second dimension must be 1 or lp1={lp1}; got {tuple(t.shape)}")
            if t.shape[2] == 1:
                t = t.expand(1, t.shape[1], dim_x)
            if t.shape[1] == 1:
                t = t.expand(1, lp1, dim_x)
            return t

        raise ValueError(f"Unsupported {name} dimensions: ndim={t.ndim}, shape={tuple(t.shape)}")

    def _denorm_minmax(self, x_windows, lp1, dim_x):
        min_val = self.koopman_norm.get("min", None)
        max_val = self.koopman_norm.get("max", None)
        if min_val is None or max_val is None:
            return x_windows

        if self.validate_norm_shape:
            min_t = self._canonicalize_norm_tensor(min_val, lp1, dim_x, x_windows.device, x_windows.dtype, "min")
            max_t = self._canonicalize_norm_tensor(max_val, lp1, dim_x, x_windows.device, x_windows.dtype, "max")
        else:
            min_t = th.as_tensor(min_val, device=x_windows.device, dtype=x_windows.dtype)
            max_t = th.as_tensor(max_val, device=x_windows.device, dtype=x_windows.dtype)

        return (x_windows + 1.0) * (max_t - min_t) / 2.0 + min_t

    def _norm_standardize(self, x_windows, lp1, dim_x):
        mean = self.koopman_norm.get("mean", None)
        std = self.koopman_norm.get("std", None)
        if mean is None or std is None:
            return x_windows

        if self.validate_norm_shape:
            mean_t = self._canonicalize_norm_tensor(mean, lp1, dim_x, x_windows.device, x_windows.dtype, "mean")
            std_t = self._canonicalize_norm_tensor(std, lp1, dim_x, x_windows.device, x_windows.dtype, "std")
        else:
            mean_t = th.as_tensor(mean, device=x_windows.device, dtype=x_windows.dtype)
            std_t = th.as_tensor(std, device=x_windows.device, dtype=x_windows.dtype)
            if mean_t.ndim == 3 and mean_t.shape[1] == 1:
                mean_t = mean_t.expand(-1, lp1, -1)
                std_t = std_t.expand(-1, lp1, -1)

        return (x_windows - mean_t) / (std_t + self.scale_eps)

    def _repr_standardize(self, x_windows, lp1, dim_x):
        mean = self.koopman_norm.get("repr_mean", None)
        std = self.koopman_norm.get("repr_std", None)
        if mean is None or std is None:
            return x_windows

        if self.validate_norm_shape:
            mean_t = self._canonicalize_norm_tensor(mean, lp1, dim_x, x_windows.device, x_windows.dtype, "repr_mean")
            std_t = self._canonicalize_norm_tensor(std, lp1, dim_x, x_windows.device, x_windows.dtype, "repr_std")
        else:
            mean_t = th.as_tensor(mean, device=x_windows.device, dtype=x_windows.dtype)
            std_t = th.as_tensor(std, device=x_windows.device, dtype=x_windows.dtype)
            if mean_t.ndim == 3 and mean_t.shape[1] == 1:
                mean_t = mean_t.expand(-1, lp1, -1)
                std_t = std_t.expand(-1, lp1, -1)

        return (x_windows - mean_t) / (std_t + self.scale_eps)

    def _apply_x0_guard(self, x_seq):
        if self.clamp_mode == "clip":
            x_seq_safe = x_seq.clamp(self.clip_min, self.clip_max)
        elif self.clamp_mode == "tanh":
            x_seq_safe = th.tanh(x_seq)
        else:
            x_seq_safe = x_seq
        overflow_rate = ((x_seq < -1.0) | (x_seq > 1.0)).float().mean()
        return x_seq_safe, overflow_rate

    def _get_scale(self, diff):
        abs_diff_med = diff.detach().abs().median()
        if self.scale_mode == "batch_median":
            scale = abs_diff_med
        elif self.scale_mode == "ema_median":
            self._scale_ema = self._state_to_tensor(self._scale_ema, abs_diff_med)
            if self._scale_ema is None:
                self._scale_ema = abs_diff_med
            else:
                self._scale_ema = (1.0 - self.scale_momentum) * self._scale_ema + self.scale_momentum * abs_diff_med
            scale = self._scale_ema
        else:
            scale = abs_diff_med
        scale = th.clamp(scale, min=self.scale_clamp_min, max=self.scale_clamp_max)
        return scale

    def _get_huber_delta(self, diff_norm):
        if self.huber_delta_mode == "adaptive_batch_median":
            return th.clamp(self.huber_delta_mult * diff_norm.detach().abs().median(), min=1e-6)

        if self.huber_delta_mode == "adaptive_ema_median":
            cur = diff_norm.detach().abs().median()
            self._delta_ema = self._state_to_tensor(self._delta_ema, cur)
            if self._delta_ema is None:
                self._delta_ema = cur
            else:
                self._delta_ema = (1.0 - self.scale_momentum) * self._delta_ema + self.scale_momentum * cur
            return th.clamp(self.huber_delta_mult * self._delta_ema, min=1e-6)

        return th.tensor(self.huber_delta, device=diff_norm.device, dtype=diff_norm.dtype)

    @staticmethod
    def _huber(diff_norm, delta):
        abs_diff = diff_norm.abs()
        quadratic = 0.5 * diff_norm.pow(2)
        linear = delta * (abs_diff - 0.5 * delta)
        return th.where(abs_diff <= delta, quadratic, linear)

    def _robust_per_sample(self, diff, ref_for_ratio=None):
        """
        Compute per-sample losses from a residual tensor.
        Returns: per_sample, num_proxy, den_proxy, scale.
        """
        mode = self.loss_mode
        if mode == "normalized_huber":
            scale = self._get_scale(diff)
            diff_norm = diff / (scale.detach() + self.scale_eps)
            delta = self._get_huber_delta(diff_norm)
            robust = self._huber(diff_norm, delta)
            per = robust.mean(dim=tuple(range(1, robust.dim())))
            den = th.ones_like(per) * scale.detach()
            return per, per, den, scale.detach()

        if mode == "normalized_charbonnier":
            scale = self._get_scale(diff)
            diff_norm = diff / (scale.detach() + self.scale_eps)
            robust = th.sqrt(diff_norm.pow(2) + self.charbonnier_eps ** 2) - self.charbonnier_eps
            per = robust.mean(dim=tuple(range(1, robust.dim())))
            den = th.ones_like(per) * scale.detach()
            return per, per, den, scale.detach()

        if mode == "log_charbonnier":
            scale = self._get_scale(diff)
            diff_norm = diff / (scale.detach() + self.scale_eps)
            log_comp = th.log1p(diff_norm.abs())
            robust = th.sqrt(log_comp.pow(2) + self.charbonnier_eps ** 2) - self.charbonnier_eps
            per = robust.mean(dim=tuple(range(1, robust.dim())))
            den = th.ones_like(per) * scale.detach()
            return per, per, den, scale.detach()

        if mode == "legacy_ratio_huber":
            delta = self._get_huber_delta(diff)
            huber = self._huber(diff, delta)
            num = huber.mean(dim=tuple(range(1, huber.dim())))
            if ref_for_ratio is None:
                # Use normalized Charbonnier without a ratio for consistency.
                scale = self._get_scale(diff)
                diff_norm = diff / (scale.detach() + self.scale_eps)
                robust = th.sqrt(diff_norm.pow(2) + self.charbonnier_eps ** 2) - self.charbonnier_eps
                per = robust.mean(dim=tuple(range(1, robust.dim())))
                den = th.ones_like(per) * scale.detach()
                return per, per, den, scale.detach()
            den = ref_for_ratio.pow(2).mean(dim=tuple(range(1, ref_for_ratio.dim()))).detach()
            per = num / (den + self.scale_eps)
            scale = self._get_scale(diff)
            return per, num, den, scale.detach()

        raise ValueError(f"Unknown koopman.loss.mode: {mode}")

    @staticmethod
    def _aggregate_branch_terms(per_list, num_list, den_list, scale_list, ref_tensor):
        if not per_list:
            zeros = th.zeros(ref_tensor.shape[0], device=ref_tensor.device, dtype=ref_tensor.dtype)
            zero_scalar = th.tensor(0.0, device=ref_tensor.device, dtype=ref_tensor.dtype)
            return zeros, zeros, zeros, zero_scalar
        per = th.stack(per_list, dim=0).mean(dim=0)
        num = th.stack(num_list, dim=0).mean(dim=0)
        den = th.stack(den_list, dim=0).mean(dim=0)
        scale = th.stack(scale_list, dim=0).mean()
        return per, num, den, scale

    @staticmethod
    def _mean_over_windows(per_bnw, bsz, num_win):
        if num_win <= 1:
            return per_bnw.reshape(bsz)
        return per_bnw.reshape(bsz, num_win).mean(dim=1)

    def _dyn_multi_per_sample(self, z_seq_full_flat, K):
        bsz = z_seq_full_flat.shape[0]
        zeros = th.zeros(bsz, device=z_seq_full_flat.device, dtype=z_seq_full_flat.dtype)
        zero_scalar = th.tensor(0.0, device=z_seq_full_flat.device, dtype=z_seq_full_flat.dtype)
        lambda_k_multi = self._current_lambda("lambda_k_multi", self.lambda_k_multi, self._lambda_k_multi_override)
        if (not self.dyn_multi_enabled) or lambda_k_multi == 0.0:
            return zeros, zeros, zeros, zero_scalar, zero_scalar

        z_dyn_seq, _ = self.koopman_model.split_z(z_seq_full_flat)
        teff = z_dyn_seq.shape[1]
        h_max = min(int(self.dyn_multi_horizon), teff - 1)
        if h_max <= 0:
            return zeros, zeros, zeros, zero_scalar, zero_scalar

        z_roll = z_dyn_seq[:, :-1, :]
        per_list = []
        num_list = []
        den_list = []
        scale_list = []
        raw_abs_values = []
        for h in range(1, h_max + 1):
            z_roll = th.einsum("ij,btj->bti", K, z_roll)
            valid_len = teff - h
            z_pred_h = z_roll[:, :valid_len, :]
            z_true_h = z_dyn_seq[:, h : h + valid_len, :]
            raw_abs_values.append((z_pred_h - z_true_h).detach().abs().reshape(-1))
            per_h, num_h, den_h, scale_h = self._robust_per_sample(
                z_pred_h - z_true_h, ref_for_ratio=z_true_h
            )
            per_list.append(per_h)
            num_list.append(num_h)
            den_list.append(den_h)
            scale_list.append(scale_h)
        raw_abs_median = (
            th.cat(raw_abs_values, dim=0).median()
            if raw_abs_values
            else zero_scalar
        )
        per, num, den, scale = self._aggregate_branch_terms(per_list, num_list, den_list, scale_list, z_seq_full_flat)
        return per, num, den, scale, raw_abs_median

    def _consistency_one_step_per_sample(self, x_windows, z_seq_full, z_next_pred):
        bsz = z_seq_full.shape[0]
        zeros = th.zeros(bsz, device=z_seq_full.device, dtype=z_seq_full.dtype)
        zero_scalar = th.tensor(0.0, device=z_seq_full.device, dtype=z_seq_full.dtype)
        lambda_cons1 = self._current_lambda("lambda_cons1", self.lambda_cons1, self._lambda_cons1_override)
        if (not self.consistency_enabled) or lambda_cons1 == 0.0:
            return zeros, zeros, zeros, zero_scalar
        if self.consistency_mode != "teacher_forced_aux":
            raise NotImplementedError(f"Unknown consistency mode: {self.consistency_mode}")
        if not hasattr(self.koopman_model, "split_z"):
            raise ValueError("Consistency requires a split Koopman model with split_z")
        if not hasattr(self.koopman_model, "decode"):
            raise ValueError("Consistency requires a Koopman model with decode")

        _, z_aux_seq = self.koopman_model.split_z(z_seq_full)
        if z_aux_seq.shape[-1] > 0:
            z_next_full = th.cat([z_next_pred, z_aux_seq[:, 1:, :]], dim=-1)
        else:
            z_next_full = z_next_pred

        bnw, t_eff_minus1, _ = z_next_full.shape
        dim_x = x_windows.shape[-1]
        x_next_pred = self.koopman_model.decode(z_next_full.reshape(-1, z_next_full.shape[-1])).reshape(
            bnw, t_eff_minus1, dim_x
        )
        delay_l = int(getattr(self.koopman_model, "delay_L", 0))
        x_next_true = x_windows[:, delay_l + 1 :, :]
        if x_next_true.shape[1] != t_eff_minus1:
            raise ValueError(
                f"One-step consistency time alignment failed: pred_len={t_eff_minus1}, true_len={x_next_true.shape[1]}, "
                f"delay_L={delay_l}"
            )
        return self._robust_per_sample(x_next_pred - x_next_true, ref_for_ratio=None)

    def _consistency_multi_step_per_sample(self, x_windows, z_seq_full, K):
        bsz = z_seq_full.shape[0]
        zeros = th.zeros(bsz, device=z_seq_full.device, dtype=z_seq_full.dtype)
        zero_scalar = th.tensor(0.0, device=z_seq_full.device, dtype=z_seq_full.dtype)
        lambda_cons_multi = self._current_lambda("lambda_cons_multi", self.lambda_cons_multi, self._lambda_cons_multi_override)
        if (not self.consistency_enabled) or (not self.cons_multi_enabled) or lambda_cons_multi == 0.0:
            return zeros, zeros, zeros, zero_scalar
        if self.consistency_mode != "teacher_forced_aux":
            raise NotImplementedError(f"Unknown consistency mode: {self.consistency_mode}")
        if not hasattr(self.koopman_model, "split_z"):
            raise ValueError("Consistency requires a split Koopman model with split_z")
        if not hasattr(self.koopman_model, "decode"):
            raise ValueError("Consistency requires a Koopman model with decode")

        z_dyn_seq, z_aux_seq = self.koopman_model.split_z(z_seq_full)
        teff = z_dyn_seq.shape[1]
        h_max = min(int(self.cons_multi_horizon), teff - 1)
        if h_max <= 0:
            return zeros, zeros, zeros, zero_scalar

        base_len = teff - h_max
        z_roll = z_dyn_seq[:, :base_len, :]
        bnw = z_seq_full.shape[0]
        dim_x = x_windows.shape[-1]
        delay_l = int(getattr(self.koopman_model, "delay_L", 0))
        per_list = []
        num_list = []
        den_list = []
        scale_list = []

        for h in range(1, h_max + 1):
            z_roll = th.einsum("ij,btj->bti", K, z_roll)
            if z_aux_seq.shape[-1] > 0:
                z_aux_h = z_aux_seq[:, h : h + base_len, :]
                if z_aux_h.shape[1] != base_len:
                    raise ValueError(
                        f"Consistency auxiliary alignment failed: h={h}, aux_len={z_aux_h.shape[1]}, base_len={base_len}"
                    )
                z_full_h = th.cat([z_roll, z_aux_h], dim=-1)
            else:
                z_full_h = z_roll

            x_pred_h = self.koopman_model.decode(z_full_h.reshape(-1, z_full_h.shape[-1])).reshape(
                bnw, base_len, dim_x
            )
            x_true_h = x_windows[:, delay_l + h : delay_l + h + base_len, :]
            if x_true_h.shape[1] != base_len:
                raise ValueError(
                    f"Consistency alignment failed: h={h}, x_true_len={x_true_h.shape[1]}, base_len={base_len}, delay_L={delay_l}"
                )
            per_h, num_h, den_h, scale_h = self._robust_per_sample(x_pred_h - x_true_h, ref_for_ratio=None)
            per_list.append(per_h)
            num_list.append(num_h)
            den_list.append(den_h)
            scale_list.append(scale_h)
        return self._aggregate_branch_terms(per_list, num_list, den_list, scale_list, z_seq_full)

    def _consistency_per_sample(self, x_windows, z_seq_full, z_next_pred):
        """
        Return the combined consistency term.
        """
        lambda_cons1 = self._current_lambda("lambda_cons1", self.lambda_cons1, self._lambda_cons1_override)
        lambda_cons_multi = self._current_lambda("lambda_cons_multi", self.lambda_cons_multi, self._lambda_cons_multi_override)
        if (not self.consistency_enabled) or (lambda_cons1 == 0.0 and lambda_cons_multi == 0.0):
            zeros = th.zeros(z_seq_full.shape[0], device=z_seq_full.device, dtype=z_seq_full.dtype)
            zero_scalar = th.tensor(0.0, device=z_seq_full.device, dtype=z_seq_full.dtype)
            return zeros, zeros, zeros, zero_scalar
        cons1, num1, den1, scale1 = self._consistency_one_step_per_sample(x_windows, z_seq_full, z_next_pred)
        if self.cons_multi_enabled and lambda_cons_multi != 0.0:
            if not hasattr(self.koopman_model, "_spectral_norm_K"):
                raise ValueError("Consistency requires a Koopman model with _spectral_norm_K")
            K = self.koopman_model._spectral_norm_K()
            consm, numm, denm, scalem = self._consistency_multi_step_per_sample(x_windows, z_seq_full, K)
        else:
            consm, numm, denm = th.zeros_like(cons1), th.zeros_like(num1), th.zeros_like(den1)
            scalem = th.tensor(0.0, device=z_seq_full.device, dtype=z_seq_full.dtype)
        total = lambda_cons1 * cons1 + lambda_cons_multi * consm
        num = lambda_cons1 * num1 + lambda_cons_multi * numm
        den = lambda_cons1 * den1 + lambda_cons_multi * denm
        scale = 0.5 * (scale1 + scalem)
        return total, num, den, scale

    def per_sample_loss(self, x0_hat, window_size):
        x_seq = self._to_sequence(x0_hat)
        bsz = x_seq.shape[0]
        zeros = th.zeros(bsz, device=x_seq.device, dtype=x_seq.dtype)
        lambda_k1 = self._current_lambda("lambda_k1", self.lambda_k1, self._lambda_k1_override)
        lambda_k_multi = self._current_lambda("lambda_k_multi", self.lambda_k_multi, self._lambda_k_multi_override)
        lambda_cons1 = self._current_lambda("lambda_cons1", self.lambda_cons1, self._lambda_cons1_override)
        lambda_cons_multi = self._current_lambda("lambda_cons_multi", self.lambda_cons_multi, self._lambda_cons_multi_override)
        lambda_cons = lambda_cons1 + lambda_cons_multi
        delay_l = int(getattr(self.koopman_model, "delay_L", 0))
        if window_size <= delay_l + 1:
            raise ValueError(
                f"window_size={window_size} is too small; require window_size >= delay_L+2 (delay_L={delay_l})"
            )

        x_seq_safe, overflow_rate = self._apply_x0_guard(x_seq)
        x_windows, bsz, num_win, lp1 = self._build_windows(x_seq_safe, window_size)
        if x_windows is None:
            return {
                "loss_per_sample": zeros,
                "num_per_sample": zeros,
                "den_per_sample": zeros,
                "loss_dyn_per_sample": zeros,
                "loss_dyn1_per_sample": zeros,
                "loss_dyn_multi_per_sample": zeros,
                "loss_cons_per_sample": zeros,
                "loss_cons1_per_sample": zeros,
                "loss_cons_multi_per_sample": zeros,
                "lambda_cons": lambda_cons,
                "lambda_cons1": lambda_cons1,
                "lambda_cons_multi": lambda_cons_multi,
                "stats": {
                    "overflow_rate": overflow_rate,
                    "scale": th.tensor(0.0, device=x_seq.device, dtype=x_seq.dtype),
                    "scale_dyn": th.tensor(0.0, device=x_seq.device, dtype=x_seq.dtype),
                    "scale_dyn1": th.tensor(0.0, device=x_seq.device, dtype=x_seq.dtype),
                    "scale_dyn_multi": th.tensor(0.0, device=x_seq.device, dtype=x_seq.dtype),
                    "raw_dyn1_abs_median": th.tensor(0.0, device=x_seq.device, dtype=x_seq.dtype),
                    "raw_dyn_multi_abs_median": th.tensor(0.0, device=x_seq.device, dtype=x_seq.dtype),
                    "scale_cons": th.tensor(0.0, device=x_seq.device, dtype=x_seq.dtype),
                    "scale_cons1": th.tensor(0.0, device=x_seq.device, dtype=x_seq.dtype),
                    "scale_cons_multi": th.tensor(0.0, device=x_seq.device, dtype=x_seq.dtype),
                },
            }

        dim_x = x_windows.shape[-1]
        x_windows = self._denorm_minmax(x_windows, lp1, dim_x)
        x_windows = self._norm_standardize(x_windows, lp1, dim_x)
        x_windows = self._apply_repr_mode(x_windows)
        x_windows = self._repr_standardize(x_windows, lp1, dim_x)

        z_seq_full, z_pred, z_true, K = self.koopman_model(x_windows, return_k=True)
        if z_pred.shape[1] == 0:
            raise ValueError(
                "Koopman supervision has no time steps; increase window_size or decrease delay_L."
            )
        bnw = x_windows.shape[0]
        z_pred_flat = z_pred.reshape(bnw, z_pred.shape[1], z_pred.shape[2])
        z_true_flat = z_true.reshape(bnw, z_true.shape[1], z_true.shape[2])
        z_seq_full_flat = z_seq_full.reshape(bnw, z_seq_full.shape[1], z_seq_full.shape[2])
        loss_dyn1_bnw, num_dyn1_bnw, den_dyn1_bnw, scale_dyn1 = self._robust_per_sample(
            z_pred_flat - z_true_flat, ref_for_ratio=z_true_flat
        )
        loss_dyn1_per_sample = self._mean_over_windows(loss_dyn1_bnw, bsz, num_win)
        num_dyn1 = self._mean_over_windows(num_dyn1_bnw, bsz, num_win)
        den_dyn1 = self._mean_over_windows(den_dyn1_bnw, bsz, num_win)

        loss_dyn_multi_bnw, num_dyn_multi_bnw, den_dyn_multi_bnw, scale_dyn_multi, raw_dyn_multi_abs_median = self._dyn_multi_per_sample(
            z_seq_full_flat, K
        )
        raw_dyn1_abs_median = (z_pred_flat - z_true_flat).detach().abs().median()
        loss_dyn_multi_per_sample = self._mean_over_windows(loss_dyn_multi_bnw, bsz, num_win)
        num_dyn_multi = self._mean_over_windows(num_dyn_multi_bnw, bsz, num_win)
        den_dyn_multi = self._mean_over_windows(den_dyn_multi_bnw, bsz, num_win)

        loss_dyn_per_sample = lambda_k1 * loss_dyn1_per_sample + lambda_k_multi * loss_dyn_multi_per_sample
        num_dyn = lambda_k1 * num_dyn1 + lambda_k_multi * num_dyn_multi
        den_dyn = lambda_k1 * den_dyn1 + lambda_k_multi * den_dyn_multi

        x_windows_flat = x_windows
        z_next_pred_flat = z_pred_flat
        loss_cons1_bnw, _, _, scale_cons1 = self._consistency_one_step_per_sample(
            x_windows_flat, z_seq_full_flat, z_next_pred_flat
        )
        loss_cons_multi_bnw, _, _, scale_cons_multi = self._consistency_multi_step_per_sample(
            x_windows_flat, z_seq_full_flat, K
        )
        loss_cons1_per_sample = self._mean_over_windows(loss_cons1_bnw, bsz, num_win)
        loss_cons_multi_per_sample = self._mean_over_windows(loss_cons_multi_bnw, bsz, num_win)
        loss_cons_per_sample = lambda_cons1 * loss_cons1_per_sample + lambda_cons_multi * loss_cons_multi_per_sample

        scale_dyn = lambda_k1 * scale_dyn1 + lambda_k_multi * scale_dyn_multi
        scale_cons = lambda_cons1 * scale_cons1 + lambda_cons_multi * scale_cons_multi
        loss_per_sample = loss_dyn_per_sample + loss_cons_per_sample

        return {
            "loss_per_sample": loss_per_sample,
            "num_per_sample": num_dyn,
            "den_per_sample": den_dyn,
            "loss_dyn_per_sample": loss_dyn_per_sample,
            "loss_dyn1_per_sample": loss_dyn1_per_sample,
            "loss_dyn_multi_per_sample": loss_dyn_multi_per_sample,
            "loss_cons_per_sample": loss_cons_per_sample,
            "loss_cons1_per_sample": loss_cons1_per_sample,
            "loss_cons_multi_per_sample": loss_cons_multi_per_sample,
            "lambda_cons": lambda_cons,
            "lambda_cons1": lambda_cons1,
            "lambda_cons_multi": lambda_cons_multi,
            "stats": {
                "overflow_rate": overflow_rate,
                "scale": scale_dyn,
                "scale_dyn": scale_dyn,
                "scale_dyn1": scale_dyn1,
                "scale_dyn_multi": scale_dyn_multi,
                "raw_dyn1_abs_median": raw_dyn1_abs_median,
                "raw_dyn_multi_abs_median": raw_dyn_multi_abs_median,
                "scale_cons": scale_cons,
                "scale_cons1": scale_cons1,
                "scale_cons_multi": scale_cons_multi,
            },
        }

    def _build_bump(self, s):
        s1 = self.bump_s1
        s2 = self.bump_s2
        if s2 <= s1:
            return th.ones_like(s)

        if self.bump_type == "logistic":
            k = max(self.bump_k, 1e-6)
            left = th.sigmoid((s - s1) / k)
            right = th.sigmoid((s2 - s) / k)
            return left * right

        # Raised cosine, exactly zero outside the interval.
        bump = th.zeros_like(s)
        mask = (s > s1) & (s < s2)
        ratio = (s - s1) / (s2 - s1 + 1e-12)
        bump_val = 0.5 * (1.0 - th.cos(2.0 * th.pi * ratio))
        bump = th.where(mask, bump_val, bump)
        return bump

    def sample_weights(self, t, schedule_weights=None):
        beta = float(self.weight_cfg.get("beta", self.koopman_cfg.get("lambda_k_beta", 4.0)))
        alpha_bar = None

        if self._alphas_cumprod is not None and self._alphas_cumprod.numel() > 1:
            alpha_bar = self._alphas_cumprod.to(t.device)[t].float()
            noise_level = th.sqrt(th.clamp(1.0 - alpha_bar, min=0.0))
        else:
            noise_level = th.zeros_like(t, dtype=th.float32, device=t.device)

        if self._noise_rank is not None:
            s = self._noise_rank.to(t.device)[t]
        else:
            if noise_level.numel() > 1:
                order = th.argsort(noise_level)
                s = th.empty_like(noise_level)
                s[order] = th.linspace(0.0, 1.0, steps=noise_level.numel(), device=noise_level.device)
            else:
                s = th.zeros_like(noise_level)

        exp_w = th.exp(-beta * noise_level)
        bump = self._build_bump(s)

        if self.weight_mode == "uniform":
            w = th.ones_like(exp_w)
        elif self.weight_mode == "bump_only":
            w = bump
        elif self.weight_mode == "exp_bump":
            w = exp_w * bump
        else:
            w = exp_w

        if self.include_schedule_sampler_weight and schedule_weights is not None:
            w = w * schedule_weights.detach()

        w = th.clamp(w, min=self.weight_w_min, max=self.weight_w_max)

        return {
            "weight": w,
            "beta": beta,
            "alpha_bar": alpha_bar,
            "stats": {
                "noise_level": noise_level.detach(),
                "noise_rank": s.detach(),
            },
        }

    def state_dict(self):
        """
        Save dynamic regularizer state for training resumption.
        """
        return {
            "class": "ModernKoopmanRegularizer",
            "scale_ema": None if self._scale_ema is None else float(self._scale_ema.detach().item()),
            "delta_ema": None if self._delta_ema is None else float(self._delta_ema.detach().item()),
            "loss_mode": str(self.loss_mode),
            "scale_mode": str(self.scale_mode),
            "consistency_mode": str(self.consistency_mode),
            "consistency_horizon": int(self.consistency_horizon),
            "dyn_multi_horizon": int(self.dyn_multi_horizon),
            "lambda_k1": float(self.lambda_k1),
            "lambda_k_multi": float(self.lambda_k_multi),
            "lambda_cons1": float(self.lambda_cons1),
            "lambda_cons_multi": float(self.lambda_cons_multi),
        }

    def load_state_dict(self, state):
        if not state:
            return
        self._scale_ema = state.get("scale_ema", self._scale_ema)
        self._delta_ema = state.get("delta_ema", self._delta_ema)

    @staticmethod
    def _current_lambda(name, base_value, override_value):
        del name
        return float(base_value if override_value is None else override_value)

    def set_lambda_overrides(
        self,
        *,
        lambda_k1=None,
        lambda_k_multi=None,
        lambda_cons1=None,
        lambda_cons_multi=None,
    ):
        self._lambda_k1_override = None if lambda_k1 is None else float(lambda_k1)
        self._lambda_k_multi_override = None if lambda_k_multi is None else float(lambda_k_multi)
        self._lambda_cons1_override = None if lambda_cons1 is None else float(lambda_cons1)
        self._lambda_cons_multi_override = None if lambda_cons_multi is None else float(lambda_cons_multi)

    def clear_lambda_overrides(self):
        self._lambda_k1_override = None
        self._lambda_k_multi_override = None
        self._lambda_cons1_override = None
        self._lambda_cons_multi_override = None
