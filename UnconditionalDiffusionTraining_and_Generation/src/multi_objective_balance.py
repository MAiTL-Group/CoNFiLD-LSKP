import math

import torch as th


class FixedLambdaBalancer:
    """
    Fixed-weight balance: L = L_diff + lambda_k * L_koop.
    """

    def __init__(self, base_lambda: float):
        self.base_lambda = float(base_lambda)
        self.needs_grad_norm = False
        self.uses_pcgrad = False

    def compute_lambda(
        self,
        *,
        base_loss,
        koop_loss,
        params,
        step,
        base_lambda_override=None,
        target_ratio_override=None,
        n_eff=None,
        active_frac=None,
    ):
        del base_loss, koop_loss, params, step, target_ratio_override, n_eff, active_frac
        lam = self.base_lambda if base_lambda_override is None else float(base_lambda_override)
        return lam, {"method": "fixed", "updated": 0.0}

    def state_dict(self):
        return {
            "class": "FixedLambdaBalancer",
            "base_lambda": float(self.base_lambda),
        }

    def load_state_dict(self, state):
        if not state:
            return
        if "base_lambda" in state:
            self.base_lambda = float(state["base_lambda"])


class GradNormLiteBalancer:
    """
    Lightweight GradNorm: adjust lambda_k using gradient-norm ratios.

    Target relation:
        ||lambda * grad(L_koop)|| ~= target_ratio * ||grad(L_diff)||
    """

    def __init__(
        self,
        *,
        base_lambda: float,
        target_ratio: float = 0.3,
        ema: float = 0.05,
        update_interval: int = 20,
        lambda_min: float = 0.0,
        lambda_max: float = 10.0,
        eps: float = 1e-8,
        target_clip_min: float = 0.0,
        target_clip_max: float = float("inf"),
        lambda_init=None,
        g_koop_floor: float = 0.0,
        skip_on_low_support: bool = False,
        min_n_eff: float = 0.0,
        min_active_frac: float = 0.0,
    ):
        self.base_lambda = float(base_lambda)
        self.needs_grad_norm = True
        self.uses_pcgrad = False
        self.target_ratio = float(target_ratio)
        self.ema = float(ema)
        self.update_interval = max(1, int(update_interval))
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.eps = float(eps)
        self.target_clip_min = float(target_clip_min)
        self.target_clip_max = float(target_clip_max)
        if lambda_init is None:
            lambda_init = float(base_lambda)
            if math.isfinite(self.target_clip_max):
                lambda_init = min(lambda_init, self.target_clip_max)
        self.lambda_init = float(lambda_init)
        self.g_koop_floor = float(g_koop_floor)
        self.skip_on_low_support = bool(skip_on_low_support)
        self.min_n_eff = float(min_n_eff)
        self.min_active_frac = float(min_active_frac)

        self.lambda_state = max(self.lambda_min, min(self.lambda_max, float(self.lambda_init)))
        self.last_g_diff = 0.0
        self.last_g_koop = 0.0
        self.last_target = float(self.lambda_state)
        self.last_raw_target = float(self.lambda_state)
        self.last_skip_low_g_koop = 0.0
        self.last_skip_low_support = 0.0

    @staticmethod
    def _grad_norm(loss, params):
        if not params:
            return loss.new_tensor(0.0)
        grads = th.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
        sq = loss.new_tensor(0.0)
        for g in grads:
            if g is None:
                continue
            sq = sq + g.float().pow(2).sum()
        return sq.sqrt()

    def compute_lambda(
        self,
        *,
        base_loss,
        koop_loss,
        params,
        step,
        base_lambda_override=None,
        target_ratio_override=None,
        n_eff=None,
        active_frac=None,
    ):
        lam_base = self.base_lambda if base_lambda_override is None else float(base_lambda_override)
        target_ratio = self.target_ratio if target_ratio_override is None else float(target_ratio_override)
        n_eff_val = None if n_eff is None else float(n_eff.detach().item() if th.is_tensor(n_eff) else n_eff)
        active_frac_val = (
            None if active_frac is None else float(active_frac.detach().item() if th.is_tensor(active_frac) else active_frac)
        )

        stats = {
            "method": "gradnorm_lite",
            "updated": 0.0,
            "g_diff": float(self.last_g_diff),
            "g_koop": float(self.last_g_koop),
            "target": float(self.last_target),
            "raw_target": float(self.last_raw_target),
            "skipped_low_g_koop": float(self.last_skip_low_g_koop),
            "skipped_low_support": float(self.last_skip_low_support),
            "n_eff": float(n_eff_val) if n_eff_val is not None else float("nan"),
            "active_frac": float(active_frac_val) if active_frac_val is not None else float("nan"),
        }

        # Update on scheduled steps; otherwise reuse the smoothed lambda.
        if (int(step) % self.update_interval) != 0:
            lam = max(self.lambda_min, min(self.lambda_max, self.lambda_state))
            return lam, stats

        g_diff = self._grad_norm(base_loss, params)
        g_koop = self._grad_norm(koop_loss, params)

        g_diff_val = float(g_diff.detach().item())
        g_koop_val = float(g_koop.detach().item())
        raw_target = lam_base
        skip_low_g_koop = 0.0
        skip_low_support = 0.0

        if math.isfinite(g_diff_val) and math.isfinite(g_koop_val):
            raw_target = target_ratio * g_diff_val / (g_koop_val + self.eps)
        if not math.isfinite(raw_target):
            raw_target = lam_base

        if (not math.isfinite(g_diff_val)) or (not math.isfinite(g_koop_val)) or (g_koop_val < self.g_koop_floor):
            skip_low_g_koop = 1.0

        if self.skip_on_low_support:
            if n_eff_val is not None and n_eff_val < self.min_n_eff:
                skip_low_support = 1.0
            if active_frac_val is not None and active_frac_val < self.min_active_frac:
                skip_low_support = 1.0

        if skip_low_g_koop or skip_low_support:
            target = self.lambda_state
        else:
            target = raw_target
            if math.isfinite(self.target_clip_min):
                target = max(self.target_clip_min, target)
            if math.isfinite(self.target_clip_max):
                target = min(self.target_clip_max, target)

            self.lambda_state = (1.0 - self.ema) * self.lambda_state + self.ema * target
            self.lambda_state = max(self.lambda_min, min(self.lambda_max, self.lambda_state))

        self.last_g_diff = g_diff_val
        self.last_g_koop = g_koop_val
        self.last_target = float(target)
        self.last_raw_target = float(raw_target)
        self.last_skip_low_g_koop = float(skip_low_g_koop)
        self.last_skip_low_support = float(skip_low_support)

        stats["updated"] = 0.0 if (skip_low_g_koop or skip_low_support) else 1.0
        stats["g_diff"] = self.last_g_diff
        stats["g_koop"] = self.last_g_koop
        stats["target"] = self.last_target
        stats["raw_target"] = self.last_raw_target
        stats["skipped_low_g_koop"] = self.last_skip_low_g_koop
        stats["skipped_low_support"] = self.last_skip_low_support

        return self.lambda_state, stats

    def state_dict(self):
        return {
            "class": "GradNormLiteBalancer",
            "base_lambda": float(self.base_lambda),
            "target_ratio": float(self.target_ratio),
            "ema": float(self.ema),
            "update_interval": int(self.update_interval),
            "lambda_min": float(self.lambda_min),
            "lambda_max": float(self.lambda_max),
            "eps": float(self.eps),
            "target_clip_min": float(self.target_clip_min),
            "target_clip_max": float(self.target_clip_max),
            "lambda_init": float(self.lambda_init),
            "g_koop_floor": float(self.g_koop_floor),
            "skip_on_low_support": bool(self.skip_on_low_support),
            "min_n_eff": float(self.min_n_eff),
            "min_active_frac": float(self.min_active_frac),
            "lambda_state": float(self.lambda_state),
            "last_g_diff": float(self.last_g_diff),
            "last_g_koop": float(self.last_g_koop),
            "last_target": float(self.last_target),
            "last_raw_target": float(self.last_raw_target),
            "last_skip_low_g_koop": float(self.last_skip_low_g_koop),
            "last_skip_low_support": float(self.last_skip_low_support),
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.lambda_init = float(state.get("lambda_init", self.lambda_init))
        self.lambda_state = float(state.get("lambda_state", self.lambda_state))
        self.last_g_diff = float(state.get("last_g_diff", self.last_g_diff))
        self.last_g_koop = float(state.get("last_g_koop", self.last_g_koop))
        self.last_target = float(state.get("last_target", self.last_target))
        self.last_raw_target = float(state.get("last_raw_target", self.last_raw_target))
        self.last_skip_low_g_koop = float(state.get("last_skip_low_g_koop", self.last_skip_low_g_koop))
        self.last_skip_low_support = float(state.get("last_skip_low_support", self.last_skip_low_support))


class PCGradBalancer:
    """
    PCGrad task-conflict projection.
    Task weights use lambda_k (fixed by default). When gradients conflict,
    project the Koopman component relative to the diffusion gradient.
    """

    def __init__(
        self,
        *,
        base_lambda: float,
        eps: float = 1e-8,
        protect_diffusion: bool = True,
        lambda_min: float = 0.0,
        lambda_max: float = 10.0,
    ):
        self.base_lambda = float(base_lambda)
        self.eps = float(eps)
        self.protect_diffusion = bool(protect_diffusion)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.lambda_state = float(base_lambda)

        self.needs_grad_norm = False
        self.uses_pcgrad = True

        self.last_cos = 0.0
        self.last_conflict = 0.0
        self.last_proj_coef = 0.0

    @staticmethod
    def _dot_and_norm(grads_a, grads_b, eps):
        dot = None
        norm_a = None
        norm_b = None
        for ga, gb in zip(grads_a, grads_b):
            if ga is None or gb is None:
                continue
            ga_f = ga.float()
            gb_f = gb.float()
            prod = (ga_f * gb_f).sum()
            na = ga_f.pow(2).sum()
            nb = gb_f.pow(2).sum()
            dot = prod if dot is None else dot + prod
            norm_a = na if norm_a is None else norm_a + na
            norm_b = nb if norm_b is None else norm_b + nb
        if dot is None:
            zero = th.tensor(0.0)
            return zero, zero, zero, 0.0
        norm_a = th.sqrt(norm_a + eps)
        norm_b = th.sqrt(norm_b + eps)
        cos = float((dot / (norm_a * norm_b + eps)).detach().item())
        return dot, norm_a, norm_b, cos

    def project_koop_grads(self, grads_diff, grads_koop):
        """
        Args:
            grads_diff: List of diffusion-branch gradients.
            grads_koop: List of lambda-weighted Koopman gradients.
        Returns:
            proj_koop: Koopman gradients after conflict projection.
            stats: Diagnostics (cos/conflict/proj_coef).
        """
        eps_t = th.tensor(self.eps)
        dot, norm_d, norm_k, cos = self._dot_and_norm(grads_diff, grads_koop, eps_t)
        conflict = bool(float(dot.detach().item()) < 0.0)
        proj_coef = 0.0

        proj_koop = list(grads_koop)
        if conflict:
            if self.protect_diffusion:
                # Remove the Koopman component that conflicts with diffusion.
                denom = norm_d.pow(2) + eps_t
                alpha = dot / denom
                proj_coef = float(alpha.detach().item())
                new_list = []
                for gd, gk in zip(grads_diff, grads_koop):
                    if gk is None:
                        new_list.append(None)
                    elif gd is None:
                        new_list.append(gk)
                    else:
                        new_list.append(gk - alpha.to(gk.dtype) * gd)
                proj_koop = new_list
            else:
                # Symmetric mode also projects the diffusion gradient.
                denom = norm_k.pow(2) + eps_t
                alpha = dot / denom
                proj_coef = float(alpha.detach().item())
                new_diff = []
                for gd, gk in zip(grads_diff, grads_koop):
                    if gd is None:
                        new_diff.append(None)
                    elif gk is None:
                        new_diff.append(gd)
                    else:
                        new_diff.append(gd - alpha.to(gd.dtype) * gk)
                grads_diff[:] = new_diff

        self.last_cos = float(cos)
        self.last_conflict = 1.0 if conflict else 0.0
        self.last_proj_coef = float(proj_coef)

        return proj_koop, {
            "method": "pcgrad",
            "cos": self.last_cos,
            "conflict": self.last_conflict,
            "proj_coef": self.last_proj_coef,
        }

    def compute_lambda(
        self,
        *,
        base_loss,
        koop_loss,
        params,
        step,
        base_lambda_override=None,
        target_ratio_override=None,
        n_eff=None,
        active_frac=None,
    ):
        del base_loss, koop_loss, params, step, target_ratio_override, n_eff, active_frac
        lam = self.base_lambda if base_lambda_override is None else float(base_lambda_override)
        self.lambda_state = max(self.lambda_min, min(self.lambda_max, lam))
        return self.lambda_state, {
            "method": "pcgrad",
            "updated": 0.0,
            "cos": float(self.last_cos),
            "conflict": float(self.last_conflict),
            "proj_coef": float(self.last_proj_coef),
        }

    def state_dict(self):
        return {
            "class": "PCGradBalancer",
            "base_lambda": float(self.base_lambda),
            "eps": float(self.eps),
            "protect_diffusion": bool(self.protect_diffusion),
            "lambda_min": float(self.lambda_min),
            "lambda_max": float(self.lambda_max),
            "lambda_state": float(self.lambda_state),
            "last_cos": float(self.last_cos),
            "last_conflict": float(self.last_conflict),
            "last_proj_coef": float(self.last_proj_coef),
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.lambda_state = float(state.get("lambda_state", self.lambda_state))
        self.last_cos = float(state.get("last_cos", self.last_cos))
        self.last_conflict = float(state.get("last_conflict", self.last_conflict))
        self.last_proj_coef = float(state.get("last_proj_coef", self.last_proj_coef))


def create_lambda_balancer(balance_cfg, base_lambda: float):
    cfg = balance_cfg or {}
    method = str(cfg.get("method", "fixed")).lower()
    if method == "gradnorm_lite":
        return GradNormLiteBalancer(
            base_lambda=base_lambda,
            target_ratio=float(cfg.get("target_ratio", 0.3)),
            ema=float(cfg.get("ema", 0.05)),
            update_interval=int(cfg.get("update_interval", 20)),
            lambda_min=float(cfg.get("lambda_min", 0.0)),
            lambda_max=float(cfg.get("lambda_max", 10.0)),
            eps=float(cfg.get("eps", 1e-8)),
            target_clip_min=float(cfg.get("target_clip_min", 0.0)),
            target_clip_max=float(cfg.get("target_clip_max", float("inf"))),
            lambda_init=cfg.get("lambda_init", None),
            g_koop_floor=float(cfg.get("g_koop_floor", 0.0)),
            skip_on_low_support=bool(cfg.get("skip_on_low_support", False)),
            min_n_eff=float(cfg.get("min_n_eff", 0.0)),
            min_active_frac=float(cfg.get("min_active_frac", 0.0)),
        )
    if method == "pcgrad":
        return PCGradBalancer(
            base_lambda=base_lambda,
            eps=float(cfg.get("eps", 1e-8)),
            protect_diffusion=bool(cfg.get("protect_diffusion", True)),
            lambda_min=float(cfg.get("lambda_min", 0.0)),
            lambda_max=float(cfg.get("lambda_max", 10.0)),
        )
    return FixedLambdaBalancer(base_lambda=base_lambda)
