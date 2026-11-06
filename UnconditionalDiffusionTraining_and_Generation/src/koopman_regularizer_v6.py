import json
import os
from typing import Dict, Optional

import numpy as np
import torch as th
import torch.nn.functional as F

from Koopman.pkcp.data import build_causal_context_windows
from Koopman.pkcp.models import AuxEncoder, ChartDecoder, PredictiveDynEncoder
from Koopman.hokp.data import (
    LocalKoopmanCharts,
    LocalPcaCharts,
    build_augmented_observables,
    build_chart_keys,
    build_hankel_states,
    build_transition_chart_keys,
    build_transition_observables,
    estimate_transition_residual_proxy,
    mix_local_koopman_k,
    project_local_pca_manifold,
)
from Koopman.lskp.data import (
    LocalStochasticKoopmanPrior,
    build_transition_descriptor,
    compute_lskp_whitened_residual,
)


def validate_path1_v6_recipe(
    koopman_cfg: dict,
    *,
    time_length: int,
    checkpoint_repr_mode: str,
    checkpoint_delay_l: int,
) -> Dict[str, object]:
    """
    Validate the regularizer configuration against the Koopman checkpoint.

    Args:
        koopman_cfg:
            The recipe's koopman configuration dictionary.
        time_length:
            Top-level time_length from the diffusion recipe.
        checkpoint_repr_mode:
            train_config.repr_mode from the Koopman checkpoint.
        checkpoint_delay_l:
            model_config.delay_L from the Koopman checkpoint.

    Returns:
        Normalized configuration values for logging or reuse by train.py.
    """
    cfg = dict(koopman_cfg or {})
    prior_variant = str(cfg.get("prior_variant", "")).lower().strip()
    repr_mode = str(cfg.get("repr_mode", "absolute")).lower().strip()
    window_size = int(cfg.get("window_size", 0) or 0)
    temporal_stride = int(cfg.get("temporal_stride", 1) or 1)
    checkpoint_repr_mode = str(checkpoint_repr_mode or "absolute").lower().strip()
    checkpoint_delay_l = int(checkpoint_delay_l or 0)
    time_length = int(time_length or 0)

    if checkpoint_delay_l != 0:
        raise ValueError(
            f"path1_v6_calibrated first version only supports delay_L=0, got checkpoint delay_L={checkpoint_delay_l}"
        )
    if window_size <= 1:
        raise ValueError(f"path1_v6_calibrated requires window_size > 1, got {window_size}")
    if temporal_stride <= 0:
        raise ValueError(f"path1_v6_calibrated requires temporal_stride >= 1, got {temporal_stride}")
    if time_length <= 0:
        raise ValueError(f"path1_v6_calibrated requires time_length > 0, got {time_length}")
    if prior_variant not in {"absolute_contiguous", "delta_from_first_stride4"}:
        raise ValueError(
            "path1_v6_calibrated requires prior_variant in "
            "{absolute_contiguous, delta_from_first_stride4}"
        )

    if prior_variant == "absolute_contiguous":
        if checkpoint_repr_mode != "absolute":
            raise ValueError(
                "absolute_contiguous requires checkpoint repr_mode=absolute, "
                f"got {checkpoint_repr_mode}"
            )
        if repr_mode not in {"auto", "absolute"}:
            raise ValueError(
                f"absolute_contiguous only supports repr_mode=auto/absolute, got {repr_mode}"
            )
        if temporal_stride != 1:
            raise ValueError(
                f"absolute_contiguous requires temporal_stride=1, got {temporal_stride}"
            )
        if time_length < window_size:
            raise ValueError(
                f"absolute_contiguous requires time_length >= window_size, got "
                f"time_length={time_length}, window_size={window_size}"
            )
    elif prior_variant == "delta_from_first_stride4":
        required = 1 + (window_size - 1) * temporal_stride
        if checkpoint_repr_mode != "delta_from_first":
            raise ValueError(
                "delta_from_first_stride4 requires checkpoint repr_mode=delta_from_first, "
                f"got {checkpoint_repr_mode}"
            )
        if repr_mode not in {"auto", "delta_from_first"}:
            raise ValueError(
                "delta_from_first_stride4 only supports repr_mode=auto/delta_from_first, "
                f"got {repr_mode}"
            )
        if temporal_stride != 4:
            raise ValueError(
                f"delta_from_first_stride4 requires temporal_stride=4, got {temporal_stride}"
            )
        if time_length < required:
            raise ValueError(
                "delta_from_first_stride4 requires "
                f"time_length >= 1 + (window_size-1)*temporal_stride = {required}, "
                f"got time_length={time_length}"
            )

    return {
        "prior_variant": prior_variant,
        "repr_mode": checkpoint_repr_mode if repr_mode == "auto" else repr_mode,
        "window_size": window_size,
        "temporal_stride": temporal_stride,
        "time_length": time_length,
        "required_time_length": 1 + (window_size - 1) * temporal_stride,
        "checkpoint_delay_L": checkpoint_delay_l,
    }


class Path1V6CalibratedRegularizer:
    """
    Noise-gated Koopman regularization with contribution caps.

    Supports one-step dynamics (raw MSE) and manifold projection (raw
    Charbonnier), with a low-noise gate, warmup, update intervals and
    subbatch evaluation.
    """

    def __init__(
        self,
        diffusion,
        koopman_model=None,
        koopman_cfg=None,
        koopman_norm=None,
        prior_artifact: Optional[dict] = None,
        device: Optional[th.device] = None,
    ):
        self.diffusion = diffusion
        self.koopman_model = koopman_model
        self.koopman_cfg = dict(koopman_cfg or {})
        self.koopman_norm = dict(koopman_norm or {})
        self.prior_artifact = dict(prior_artifact or {})
        self.prior_backend = str(self.koopman_cfg.get("prior_backend", "split_ae")).lower()
        if device is not None:
            self.device = device
        elif koopman_model is not None:
            self.device = next(koopman_model.parameters()).device
        else:
            self.device = th.device("cpu")

        self.mode = str(self.koopman_cfg.get("mode", "")).lower()
        self.prior_variant = str(self.koopman_cfg.get("prior_variant", "absolute_contiguous")).lower()
        self.repr_mode = str(self.koopman_cfg.get("repr_mode", "absolute")).lower()
        self.window_size = int(self.koopman_cfg.get("window_size", 0) or 0)
        self.temporal_stride = int(self.koopman_cfg.get("temporal_stride", 1) or 1)
        self.warmup_steps = max(0, int(self.koopman_cfg.get("warmup_steps", 0) or 0))
        self.interval = max(1, int(self.koopman_cfg.get("koop_interval", 1) or 1))
        self.koop_subbatch = max(1, int(self.koopman_cfg.get("koop_subbatch", 1) or 1))
        subbatch_rng_cfg = dict(self.koopman_cfg.get("subbatch_rng", {}) or {})
        self.subbatch_rng_enabled = bool(subbatch_rng_cfg.get("enabled", False))
        self.subbatch_rng_seed = int(subbatch_rng_cfg.get("seed", 0))
        self.subbatch_generator = None
        if self.subbatch_rng_enabled:
            if "seed" not in subbatch_rng_cfg:
                raise ValueError("koopman.subbatch_rng.seed is required when enabled")
            self.subbatch_generator = th.Generator(device=self.device)
            self.subbatch_generator.manual_seed(self.subbatch_rng_seed)
        self.sigma_max = float(self.koopman_cfg.get("sigma_max", 1.0))
        self.lambda_base = float(self.koopman_cfg.get("lambda_base", 0.0))
        self.ratio_cap_vs_diff = float(self.koopman_cfg.get("ratio_cap_vs_diff", 0.0))
        self.lambda_dyn_base = float(self.koopman_cfg.get("lambda_dyn_base", self.lambda_base))
        self.lambda_man_base = float(self.koopman_cfg.get("lambda_man_base", 0.0))
        self.ratio_cap_dyn_vs_diff = float(
            self.koopman_cfg.get("ratio_cap_dyn_vs_diff", self.ratio_cap_vs_diff)
        )
        self.ratio_cap_man_vs_diff = float(
            self.koopman_cfg.get("ratio_cap_man_vs_diff", self.ratio_cap_vs_diff)
        )
        self.gamma_man = float(self.koopman_cfg.get("gamma_man", 0.0))
        self.ramp_steps = max(0, int(self.koopman_cfg.get("ramp_steps", 0) or 0))
        self.chart_confidence_threshold = float(
            self.koopman_cfg.get("chart_confidence_threshold", 0.0)
        )
        self.chart_confidence_power = float(self.koopman_cfg.get("chart_confidence_power", 1.0))
        self.dyn_confidence_threshold = float(
            self.koopman_cfg.get("dyn_confidence_threshold", 0.0)
        )
        self.dyn_confidence_power = float(self.koopman_cfg.get("dyn_confidence_power", 1.0))
        self.unknown_conf_threshold = float(self.koopman_cfg.get("unknown_conf_threshold", -1.0))
        self.unknown_weight_scale = float(self.koopman_cfg.get("unknown_weight_scale", 1.0))
        self.selective_conf_threshold = float(
            self.koopman_cfg.get("selective_conf_threshold", self.unknown_conf_threshold)
        )
        self.selective_weight_scale = float(
            self.koopman_cfg.get("selective_weight_scale", self.unknown_weight_scale)
        )
        self.selective_burst_scale = float(self.koopman_cfg.get("selective_burst_scale", 1.0))
        self.selective_burst_conf_threshold = float(
            self.koopman_cfg.get("selective_burst_conf_threshold", self.selective_conf_threshold)
        )
        self.burst_base_scale = float(self.koopman_cfg.get("burst_base_scale", 1.0))
        self.burst_extra_scale = float(
            self.koopman_cfg.get("burst_extra_scale", self.selective_burst_scale)
        )
        self.burst_huber_on_all = bool(self.koopman_cfg.get("burst_huber_on_all", False))

        loss_cfg = self.koopman_cfg.get("loss", {}) or {}
        self.dyn_mode = str(loss_cfg.get("dyn_mode", "mse")).lower()
        self.man_mode = str(loss_cfg.get("man_mode", "charbonnier")).lower()
        self.huber_delta = float(loss_cfg.get("huber_delta", 2.0))
        self.conditional_huber_delta = float(
            loss_cfg.get("conditional_huber_delta", self.huber_delta)
        )
        self.charbonnier_eps = float(loss_cfg.get("charbonnier_eps", 1.0e-3))
        self.scale_eps = float(loss_cfg.get("scale_eps", 1.0e-8))

        if self.mode != "path1_v6_calibrated":
            raise ValueError(f"unexpected v6 mode: {self.mode}")
        if self.window_size <= 1:
            raise ValueError(f"path1_v6_calibrated requires window_size > 1, got {self.window_size}")
        if self.temporal_stride <= 0:
            raise ValueError(
                f"path1_v6_calibrated requires temporal_stride >= 1, got {self.temporal_stride}"
            )
        if self.dyn_mode not in {"mse", "huber"}:
            raise ValueError(
                f"path1_v6_calibrated only supports dyn_mode in {{mse,huber}}, got {self.dyn_mode}"
            )
        if self.prior_backend != "lskp" and self.dyn_mode != "mse":
            raise ValueError(
                "path1_v6_calibrated only supports dyn_mode=huber for lskp backend"
            )
        if self.man_mode != "charbonnier":
            raise ValueError(
                f"path1_v6_calibrated only supports man_mode=charbonnier, got {self.man_mode}"
            )
        if self.charbonnier_eps <= 0:
            raise ValueError(
                f"path1_v6_calibrated requires charbonnier_eps > 0, got {self.charbonnier_eps}"
            )
        if self.prior_backend not in {"split_ae", "pkcp", "hokp", "lskp"}:
            raise ValueError(f"path1_v6_calibrated unsupported prior_backend: {self.prior_backend}")

        if self.prior_variant == "absolute_contiguous":
            if self.repr_mode != "absolute":
                raise ValueError(
                    f"absolute_contiguous requires repr_mode=absolute, got {self.repr_mode}"
                )
            if self.temporal_stride != 1:
                raise ValueError(
                    f"absolute_contiguous requires temporal_stride=1, got {self.temporal_stride}"
                )
        elif self.prior_variant == "delta_from_first_stride4":
            if self.repr_mode != "delta_from_first":
                raise ValueError(
                    "delta_from_first_stride4 requires repr_mode=delta_from_first, "
                    f"got {self.repr_mode}"
                )
            if self.temporal_stride != 4:
                raise ValueError(
                    f"delta_from_first_stride4 requires temporal_stride=4, got {self.temporal_stride}"
                )
        else:
            raise ValueError(
                "path1_v6_calibrated prior_variant must be one of "
                "{absolute_contiguous, delta_from_first_stride4}"
            )

        self._alphas_cumprod = None
        if diffusion is not None and hasattr(diffusion, "alphas_cumprod"):
            self._alphas_cumprod = th.from_numpy(diffusion.alphas_cumprod).float().to(self.device)

        self.data_mean = self._to_tensor(self.koopman_norm.get("mean", None), "mean")
        self.data_std = self._to_tensor(self.koopman_norm.get("std", None), "std")
        self.repr_mean = self._to_tensor(self.koopman_norm.get("repr_mean", None), "repr_mean")
        self.repr_std = self._to_tensor(self.koopman_norm.get("repr_std", None), "repr_std")
        self.ugc_min = self._to_tensor(self.koopman_norm.get("min", None), "min")
        self.ugc_max = self._to_tensor(self.koopman_norm.get("max", None), "max")

        if self.ugc_min is None or self.ugc_max is None:
            raise ValueError("path1_v6_calibrated requires koopman_norm min/max from UCG training data")
        if self.data_mean is None or self.data_std is None:
            raise ValueError("path1_v6_calibrated requires checkpoint data_mean/data_std")
        if self.repr_mode != "absolute" and (self.repr_mean is None or self.repr_std is None):
            raise ValueError(
                "non-absolute repr_mode requires checkpoint repr_mean/repr_std for v6 exact alignment"
            )

        if self.prior_backend == "split_ae":
            if self.koopman_model is None:
                raise ValueError("split_ae backend requires a frozen Koopman model")
            if int(getattr(self.koopman_model, "delay_L", 0)) != 0:
                raise ValueError(
                    "path1_v6_calibrated first version only supports delay_L=0 "
                    f"(got {int(getattr(self.koopman_model, 'delay_L', 0))})"
                )
            self.pkcp_phi = None
            self.pkcp_psi = None
            self.pkcp_decoder = None
            self.pkcp_k = None
            self.hokp_k = None
            self.hokp_pca_mean = None
            self.hokp_pca_basis = None
            self.hokp_hankel_length = None
            self.hokp_local_charts = None
            self.hokp_artifact_version = "hokp1"
            self.hokp_observable_cfg = None
            self.hokp_chart_key_cfg = None
            self.hokp_weight_cfg = None
            self.hokp_local_koopman = None
        elif self.prior_backend == "pkcp":
            self.koopman_model = None
            self._init_pkcp_prior()
            self.hokp_k = None
            self.hokp_pca_mean = None
            self.hokp_pca_basis = None
            self.hokp_hankel_length = None
            self.hokp_local_charts = None
            self.hokp_artifact_version = "hokp1"
            self.hokp_observable_cfg = None
            self.hokp_chart_key_cfg = None
            self.hokp_weight_cfg = None
            self.hokp_local_koopman = None
        elif self.prior_backend == "hokp":
            self.koopman_model = None
            self.pkcp_phi = None
            self.pkcp_psi = None
            self.pkcp_decoder = None
            self.pkcp_k = None
            self._init_hokp_prior()
            self.lskp_prior = None
            self.lskp_pca_mean = None
            self.lskp_pca_basis = None
            self.lskp_descriptor_cfg = None
        else:
            self.koopman_model = None
            self.pkcp_phi = None
            self.pkcp_psi = None
            self.pkcp_decoder = None
            self.pkcp_k = None
            self.hokp_k = None
            self.hokp_pca_mean = None
            self.hokp_pca_basis = None
            self.hokp_hankel_length = None
            self.hokp_local_charts = None
            self.hokp_artifact_version = "hokp1"
            self.hokp_observable_cfg = None
            self.hokp_chart_key_cfg = None
            self.hokp_weight_cfg = None
            self.hokp_local_koopman = None
            self._init_lskp_prior()

        self.reference_metrics = self._load_reference_metrics()
        if self.prior_backend == "lskp":
            self.dyn_ref = self._resolve_ref_metric(
                ("dyn_whitened_val", "tail_dyn_whitened_val", "lag1_residual_val")
            )
        else:
            self.dyn_ref = self._resolve_ref_metric(("dyn1_val_mse", "dyn_val_mse", "dyn_ref_mse"))
        self.man_ref = self._resolve_ref_metric(
            ("manifold_val_charbonnier", "manifold_charbonnier", "man_ref_charb")
        )

    def _init_pkcp_prior(self) -> None:
        artifact = self.prior_artifact
        if not artifact:
            raise ValueError("pkcp backend requires prior_artifact")
        if str(artifact.get("prior_backend", "")).lower() != "pkcp":
            raise ValueError("prior_artifact is not a PKCP artifact")

        dim_input = int(artifact["dim_input"])
        dim_dyn = int(artifact["dim_dyn"])
        dim_aux = int(artifact["dim_aux"])
        stage_a_cfg = artifact.get("stage_a_config", {}) or {}
        stage_b_cfg = artifact.get("stage_b_config", {}) or {}
        self.pkcp_context_length = int(stage_a_cfg.get("context_length", 1))

        self.pkcp_phi = PredictiveDynEncoder(
            dim_input=dim_input,
            dim_dyn=dim_dyn,
            hidden_dims=stage_a_cfg.get("hidden_dims", None),
            context_length=self.pkcp_context_length,
            temporal_hidden_dim=stage_a_cfg.get("temporal_hidden_dim", None),
        ).to(self.device)
        self.pkcp_phi.load_state_dict(artifact["phi_state_dict"])
        self.pkcp_phi.eval()
        self.pkcp_phi.requires_grad_(False)

        self.pkcp_psi = AuxEncoder(
            dim_input=dim_input,
            dim_dyn=dim_dyn,
            dim_aux=dim_aux,
            hidden_dims=stage_b_cfg.get("hidden_dims_aux", None),
        ).to(self.device)
        self.pkcp_psi.load_state_dict(artifact["psi_state_dict"])
        self.pkcp_psi.eval()
        self.pkcp_psi.requires_grad_(False)

        hidden_dims_dec = stage_b_cfg.get("hidden_dims_dec", None)
        self.pkcp_decoder = ChartDecoder(
            dim_input=dim_input,
            dim_dyn=dim_dyn,
            dim_aux=dim_aux,
            hidden_dims_coarse=hidden_dims_dec,
            hidden_dims_res=hidden_dims_dec,
        ).to(self.device)
        self.pkcp_decoder.load_state_dict(artifact["chart_decoder_state_dict"])
        self.pkcp_decoder.eval()
        self.pkcp_decoder.requires_grad_(False)

        self.pkcp_k = th.as_tensor(artifact["koopman_k"], dtype=th.float32, device=self.device)

    def _load_reference_metrics(self) -> Dict[str, float]:
        log_cfg = self.koopman_cfg.get("logging", {}) or {}
        ref_path = log_cfg.get("reference_metrics_path", None)
        if not ref_path:
            return {}
        if not isinstance(ref_path, str):
            return {}
        if not ref_path.strip():
            return {}
        if not os.path.exists(ref_path):
            return {}
        try:
            with open(ref_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _init_hokp_prior(self) -> None:
        artifact = self.prior_artifact
        if not artifact:
            raise ValueError("hokp backend requires prior_artifact")
        if str(artifact.get("prior_backend", "")).lower() != "hokp":
            raise ValueError("prior_artifact is not a HOKP artifact")
        self.hokp_artifact_version = str(artifact.get("artifact_version", "hokp1")).lower()
        self.hokp_k = th.as_tensor(artifact["koopman_k"], dtype=th.float32, device=self.device)
        pca_stats = artifact["pca_coarse_stats"]
        self.hokp_pca_mean = th.as_tensor(pca_stats["mean"], dtype=th.float32, device=self.device)
        self.hokp_pca_basis = th.as_tensor(pca_stats["basis"], dtype=th.float32, device=self.device)
        self.hokp_hankel_length = int(artifact["hankel_length"])
        self.hokp_observable_cfg = dict(artifact.get("observable_cfg", {}) or {})
        self.hokp_chart_key_cfg = dict(artifact.get("chart_key_cfg", {}) or {})
        self.hokp_weight_cfg = dict(artifact.get("weight_cfg", {}) or {})
        chart_dict = artifact["local_pca_charts"]
        self.hokp_local_charts = LocalPcaCharts(
            centers=th.as_tensor(chart_dict["centers"], dtype=th.float32, device=self.device),
            means=th.as_tensor(chart_dict["means"], dtype=th.float32, device=self.device),
            bases=th.as_tensor(chart_dict["bases"], dtype=th.float32, device=self.device),
            tau=float(chart_dict["tau"]),
        )
        local_koopman = artifact.get("local_koopman_charts", None)
        if local_koopman is not None:
            self.hokp_local_koopman = LocalKoopmanCharts(
                centers=th.as_tensor(local_koopman["centers"], dtype=th.float32, device=self.device),
                koopman_k=th.as_tensor(local_koopman["koopman_k"], dtype=th.float32, device=self.device),
                tau=float(local_koopman["tau"]),
            )
        else:
            self.hokp_local_koopman = None

    def _init_lskp_prior(self) -> None:
        artifact = self.prior_artifact
        if not artifact:
            raise ValueError("lskp backend requires prior_artifact")
        if str(artifact.get("prior_backend", "")).lower() != "lskp":
            raise ValueError("prior_artifact is not a LSKP artifact")
        self.lskp_artifact_version = str(artifact.get("artifact_version", "lskp1")).lower()
        pca_stats = artifact["pca_coarse_stats"]
        self.lskp_pca_mean = th.as_tensor(pca_stats["mean"], dtype=th.float32, device=self.device)
        self.lskp_pca_basis = th.as_tensor(pca_stats["basis"], dtype=th.float32, device=self.device)
        self.lskp_descriptor_cfg = dict(artifact.get("descriptor_cfg", {}) or {})
        prior_dict = artifact["local_stochastic_prior"]
        self.lskp_prior = LocalStochasticKoopmanPrior(
            centers=th.as_tensor(prior_dict["centers"], dtype=th.float32, device=self.device),
            koopman_k=th.as_tensor(prior_dict["koopman_k"], dtype=th.float32, device=self.device),
            sigma2=th.as_tensor(prior_dict["sigma2"], dtype=th.float32, device=self.device),
            tau=float(prior_dict["tau"]),
            descriptor_mean=(
                th.as_tensor(prior_dict["descriptor_mean"], dtype=th.float32, device=self.device)
                if "descriptor_mean" in prior_dict
                else None
            ),
            descriptor_std=(
                th.as_tensor(prior_dict["descriptor_std"], dtype=th.float32, device=self.device)
                if "descriptor_std" in prior_dict
                else None
            ),
            burst_proxy_threshold=(
                float(prior_dict["burst_proxy_threshold"])
                if "burst_proxy_threshold" in prior_dict
                else (
                    float(artifact["burst_proxy_threshold"])
                    if artifact.get("burst_proxy_threshold", None) is not None
                    else None
                )
            ),
        )
        self.lskp_burst_proxy_threshold = self.lskp_prior.burst_proxy_threshold

    def _project_hokp_coarse(self, u_seq: th.Tensor) -> th.Tensor:
        flat = u_seq.reshape(-1, u_seq.shape[-1])
        coeff = (flat - self.hokp_pca_mean.to(flat.device, flat.dtype)) @ self.hokp_pca_basis.to(flat.device, flat.dtype)
        return coeff.reshape(u_seq.shape[0], u_seq.shape[1], -1)

    def _project_lskp_coarse(self, u_seq: th.Tensor) -> th.Tensor:
        flat = u_seq.reshape(-1, u_seq.shape[-1])
        coeff = (flat - self.lskp_pca_mean.to(flat.device, flat.dtype)) @ self.lskp_pca_basis.to(flat.device, flat.dtype)
        return coeff.reshape(u_seq.shape[0], u_seq.shape[1], -1)

    def _build_lskp_descriptor(self, c_seq: th.Tensor) -> th.Tensor:
        return build_transition_descriptor(
            c_seq,
            include_level=bool(self.lskp_descriptor_cfg.get("include_level", True)),
            include_delta=bool(self.lskp_descriptor_cfg.get("include_delta", True)),
            include_energy_delta=bool(self.lskp_descriptor_cfg.get("include_energy_delta", True)),
            include_prev_delta=bool(self.lskp_descriptor_cfg.get("include_prev_delta", False)),
            include_delta2=bool(self.lskp_descriptor_cfg.get("include_delta2", False)),
        )

    def _build_hokp_observable(self, c_seq: th.Tensor) -> th.Tensor:
        if self.hokp_artifact_version == "hokp2":
            return build_augmented_observables(
                c_seq,
                use_level=bool(self.hokp_observable_cfg.get("use_level", True)),
                use_delta=bool(self.hokp_observable_cfg.get("use_delta", True)),
                use_accel=bool(self.hokp_observable_cfg.get("use_accel", False)),
            )
        if self.hokp_artifact_version == "hokp3":
            return build_transition_observables(
                c_seq,
                use_level=bool(self.hokp_observable_cfg.get("use_level", True)),
                use_delta=bool(self.hokp_observable_cfg.get("use_delta", True)),
                use_accel=bool(self.hokp_observable_cfg.get("use_accel", False)),
                use_delta_norm=bool(self.hokp_observable_cfg.get("use_delta_norm", True)),
                use_energy_drift=bool(self.hokp_observable_cfg.get("use_energy_drift", True)),
            )
        return c_seq

    def _build_hokp_chart_key(
        self,
        c_seq: th.Tensor,
        *,
        residual_proxy: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        if self.hokp_artifact_version == "hokp2":
            return build_chart_keys(
                c_seq,
                use_level=bool(self.hokp_chart_key_cfg.get("use_level", True)),
                use_delta=bool(self.hokp_chart_key_cfg.get("use_delta", True)),
                use_accel=bool(self.hokp_chart_key_cfg.get("use_accel", False)),
            )
        if self.hokp_artifact_version == "hokp3":
            return build_transition_chart_keys(
                c_seq,
                use_level=bool(self.hokp_chart_key_cfg.get("use_level", True)),
                use_delta=bool(self.hokp_chart_key_cfg.get("use_delta", True)),
                use_accel=bool(self.hokp_chart_key_cfg.get("use_accel", False)),
                use_delta_norm=bool(self.hokp_chart_key_cfg.get("use_delta_norm", True)),
                use_energy_drift=bool(self.hokp_chart_key_cfg.get("use_energy_drift", True)),
                residual_proxy=(
                    residual_proxy
                    if bool(self.hokp_chart_key_cfg.get("use_residual_proxy", False))
                    else None
                ),
            )
        return c_seq

    def _project_hokp_manifold(
        self, u_seq: th.Tensor, chart_key_seq: th.Tensor, *, return_confidence: bool = False
    ):
        flat_u = u_seq.reshape(-1, u_seq.shape[-1])
        flat_key = chart_key_seq.reshape(-1, chart_key_seq.shape[-1])
        proj = project_local_pca_manifold(
            flat_u,
            flat_key,
            self.hokp_local_charts,
            return_confidence=return_confidence,
        )
        if not return_confidence:
            return proj.reshape_as(u_seq)
        proj_flat, conf_flat = proj
        return proj_flat.reshape_as(u_seq), conf_flat.reshape(u_seq.shape[0], u_seq.shape[1])

    def _resolve_ref_metric(self, keys) -> Optional[float]:
        for key in keys:
            value = self.reference_metrics.get(key, None)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0.0:
                return value
        return None

    def _to_tensor(self, value, name: str) -> Optional[th.Tensor]:
        if value is None:
            return None
        tensor = th.as_tensor(value, dtype=th.float32, device=self.device)
        if tensor.numel() == 0:
            raise ValueError(f"invalid empty tensor for {name}")
        return tensor

    @staticmethod
    def _to_sequence(x0_hat: th.Tensor) -> th.Tensor:
        if x0_hat.dim() == 4 and x0_hat.shape[1] == 1:
            return x0_hat[:, 0]
        if x0_hat.dim() == 3:
            return x0_hat
        raise ValueError(f"unexpected x0_hat shape: {tuple(x0_hat.shape)}")

    def _canonicalize_stat(self, tensor: Optional[th.Tensor], dim_x: int) -> Optional[th.Tensor]:
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
        if t.ndim == 3 and t.shape[0] == 1 and t.shape[1] == 1 and t.shape[2] in (1, dim_x):
            if t.shape[2] == 1:
                return t.expand(1, 1, dim_x)
            return t
        raise ValueError(f"cannot canonicalize stat tensor to [1,1,{dim_x}], got {tuple(t.shape)}")

    def _build_single_window(self, x_seq: th.Tensor) -> th.Tensor:
        bsz, tlen, _ = x_seq.shape
        del bsz
        if self.prior_variant == "absolute_contiguous":
            if tlen < self.window_size:
                raise ValueError(
                    f"absolute_contiguous requires tlen >= window_size, got tlen={tlen}, "
                    f"window_size={self.window_size}"
                )
            return x_seq[:, : self.window_size, :]
        if self.prior_variant == "delta_from_first_stride4":
            required = 1 + (self.window_size - 1) * self.temporal_stride
            if tlen < required:
                raise ValueError(
                    "delta_from_first_stride4 requires "
                    f"tlen >= {required}, got {tlen}"
                )
            indices = th.arange(self.window_size, device=x_seq.device, dtype=th.long) * self.temporal_stride
            return x_seq.index_select(1, indices)
        raise ValueError(f"unsupported prior_variant: {self.prior_variant}")

    def _align_to_koopman_repr(self, x_window: th.Tensor) -> th.Tensor:
        dim_x = x_window.shape[-1]
        ugc_min = self._canonicalize_stat(self.ugc_min, dim_x)
        ugc_max = self._canonicalize_stat(self.ugc_max, dim_x)
        data_mean = self._canonicalize_stat(self.data_mean, dim_x)
        data_std = self._canonicalize_stat(self.data_std, dim_x)

        u = (x_window + 1.0) * (ugc_max - ugc_min) / 2.0 + ugc_min
        u = (u - data_mean) / (data_std + self.scale_eps)

        if self.repr_mode == "delta_from_first":
            u = u - u[:, :1, :]

        repr_mean = self._canonicalize_stat(self.repr_mean, dim_x)
        repr_std = self._canonicalize_stat(self.repr_std, dim_x)
        if repr_mean is not None and repr_std is not None:
            u = (u - repr_mean) / (repr_std + self.scale_eps)
        return u

    def _noise_gate(self, t: th.Tensor) -> th.Tensor:
        if self._alphas_cumprod is None:
            return th.ones_like(t, dtype=th.bool, device=self.device)
        alpha_bar = self._alphas_cumprod[t.long()].to(device=t.device)
        sigma = th.sqrt(th.clamp(1.0 - alpha_bar, min=0.0))
        return sigma <= self.sigma_max

    def _select_subbatch(self, x_seq: th.Tensor, t: th.Tensor):
        gate = self._noise_gate(t)
        gate_idx = gate.nonzero(as_tuple=False).view(-1)
        active_frac = float(gate.float().mean().item()) if gate.numel() > 0 else 0.0
        noise_gate_mean = active_frac
        if gate_idx.numel() == 0:
            return None, None, {
                "active_frac": active_frac,
                "subbatch_frac": 0.0,
                "noise_gate_mean": noise_gate_mean,
            }
        if gate_idx.numel() > self.koop_subbatch:
            perm = th.randperm(
                gate_idx.numel(),
                device=gate_idx.device,
                generator=self.subbatch_generator,
            )
            select_idx = gate_idx.index_select(0, perm[: self.koop_subbatch])
        else:
            select_idx = gate_idx
        subbatch_frac = float(select_idx.numel()) / float(max(int(x_seq.shape[0]), 1))
        return (
            x_seq.index_select(0, select_idx.to(device=x_seq.device)),
            t.index_select(0, select_idx.to(device=t.device)),
            {
                "active_frac": active_frac,
                "subbatch_frac": subbatch_frac,
                "noise_gate_mean": noise_gate_mean,
            },
        )

    def _encode_window(self, u_window: th.Tensor):
        bsz, tlen, dim_x = u_window.shape
        if self.prior_backend == "split_ae":
            z_full = self.koopman_model.encode(u_window.reshape(bsz * tlen, dim_x)).reshape(
                bsz, tlen, self.koopman_model.dim_z
            )
            z_dyn, _ = self.koopman_model.split_z(z_full)
            u_recon = self.koopman_model.decode(z_full.reshape(-1, z_full.shape[-1])).reshape(bsz, tlen, dim_x)
            K = self.koopman_model._spectral_norm_K()
            return z_full, z_dyn, u_recon, K, {}

        if self.prior_backend == "pkcp":
            flat = u_window.reshape(bsz * tlen, dim_x)
            if self.pkcp_context_length > 1:
                contexts = build_causal_context_windows(u_window, context_length=self.pkcp_context_length)
                phi_in = contexts.reshape(-1, contexts.shape[2], contexts.shape[3])
            else:
                phi_in = flat
            z_dyn = self.pkcp_phi(phi_in)
            z_aux = self.pkcp_psi(flat, z_dyn)
            u_recon, _, _ = self.pkcp_decoder(z_dyn, z_aux)
            z_dyn = z_dyn.reshape(bsz, tlen, -1)
            u_recon = u_recon.reshape(bsz, tlen, dim_x)
            return None, z_dyn, u_recon, self.pkcp_k, {}

        if self.prior_backend == "lskp":
            c_seq = self._project_lskp_coarse(u_window)
            q_seq = self._build_lskp_descriptor(c_seq)
            aux = {"descriptor_seq": q_seq, "coarse_seq": c_seq}
            return None, c_seq, u_window, self.lskp_prior, aux

        c_seq = self._project_hokp_coarse(u_window)
        observable_seq = self._build_hokp_observable(c_seq)
        z_dyn = build_hankel_states(observable_seq, hankel_length=self.hokp_hankel_length)
        residual_proxy = None
        if self.hokp_artifact_version == "hokp3":
            residual_proxy, _ = estimate_transition_residual_proxy(
                z_dyn,
                koopman_k=self.hokp_k,
            )
        chart_key_seq = self._build_hokp_chart_key(c_seq, residual_proxy=residual_proxy)
        if self.hokp_artifact_version in {"hokp2", "hokp3"}:
            u_recon, man_conf = self._project_hokp_manifold(
                u_window,
                chart_key_seq,
                return_confidence=True,
            )
            K_obj = self.hokp_local_koopman if self.hokp_local_koopman is not None else self.hokp_k
            aux = {
                "chart_key_seq": chart_key_seq,
                "man_conf_seq": man_conf,
                "dyn_residual_proxy_seq": residual_proxy,
            }
        else:
            u_recon = self._project_hokp_manifold(u_window, chart_key_seq)
            K_obj = self.hokp_k
            aux = {}
        return None, z_dyn, u_recon, K_obj, aux

    def _compute_dyn_raw(self, z_dyn: th.Tensor, K, aux: Optional[dict] = None):
        if isinstance(K, LocalStochasticKoopmanPrior):
            if aux is None or "descriptor_seq" not in aux:
                raise ValueError("lskp dyn requires descriptor_seq")
            loss_vec, conf_vec, extra = compute_lskp_whitened_residual(
                z_dyn,
                aux["descriptor_seq"],
                K,
                loss_mode=self.dyn_mode,
                huber_delta=self.huber_delta,
                return_aux=True,
            )
            return loss_vec, conf_vec, extra
        if z_dyn.shape[1] < 2:
            raise ValueError("v6 exact prior requires window_size >= 2")
        z_now = z_dyn[:, :-1, :]
        z_next = z_dyn[:, 1:, :]
        dyn_conf_vec = th.ones(z_dyn.shape[0], device=z_dyn.device, dtype=z_dyn.dtype)
        if isinstance(K, LocalKoopmanCharts):
            if aux is None or "chart_key_seq" not in aux:
                raise ValueError("local HOKP dyn requires chart_key_seq")
            key_now = aux["chart_key_seq"][:, :-1, :]
            k_mix, dyn_conf = mix_local_koopman_k(
                key_now.reshape(-1, key_now.shape[-1]),
                K,
                return_confidence=True,
            )
            k_mix = k_mix.reshape(z_now.shape[0], z_now.shape[1], z_now.shape[-1], z_now.shape[-1])
            dyn_conf_vec = dyn_conf.reshape(z_now.shape[0], z_now.shape[1]).mean(dim=1)
            z_pred = th.einsum("btij,btj->bti", k_mix, z_now)
        else:
            z_pred = th.einsum("ij,btj->bti", K, z_now)
        residual = z_next - z_pred
        return residual.pow(2).mean(dim=(1, 2)), dyn_conf_vec, {}

    def _compute_man_raw(self, u_recon: th.Tensor, u_target: th.Tensor) -> th.Tensor:
        diff = u_recon - u_target
        charb = th.sqrt(diff.pow(2) + self.charbonnier_eps ** 2) - self.charbonnier_eps
        return charb.mean(dim=(1, 2))

    def _ramp(self, global_step: int) -> float:
        if global_step < self.warmup_steps:
            return 0.0
        if self.ramp_steps <= 0:
            return 1.0
        progress = float(global_step - self.warmup_steps + 1) / float(self.ramp_steps)
        return max(0.0, min(1.0, progress))

    def _lambda_eff(self, diff_loss_scalar: th.Tensor, prior_raw: th.Tensor, global_step: int) -> th.Tensor:
        if prior_raw.detach().item() <= 0.0:
            return prior_raw.new_tensor(0.0)
        ramp = self._ramp(global_step)
        if ramp <= 0.0:
            return prior_raw.new_tensor(0.0)
        base = prior_raw.new_tensor(self.lambda_base * ramp)
        if self.ratio_cap_vs_diff <= 0.0:
            return base
        cap = self.ratio_cap_vs_diff * diff_loss_scalar.detach() / (prior_raw.detach() + self.scale_eps)
        return th.minimum(base, th.clamp(cap, min=0.0))

    def _lambda_eff_component(
        self,
        diff_loss_scalar: th.Tensor,
        loss_raw: th.Tensor,
        global_step: int,
        *,
        base_value: float,
        ratio_cap: float,
    ) -> th.Tensor:
        if loss_raw.detach().item() <= 0.0:
            return loss_raw.new_tensor(0.0)
        ramp = self._ramp(global_step)
        if ramp <= 0.0:
            return loss_raw.new_tensor(0.0)
        base = loss_raw.new_tensor(float(base_value) * ramp)
        if ratio_cap <= 0.0:
            return base
        cap = float(ratio_cap) * diff_loss_scalar.detach() / (loss_raw.detach() + self.scale_eps)
        return th.minimum(base, th.clamp(cap, min=0.0))

    def compute_train_loss(
        self,
        x0_hat: th.Tensor,
        t: th.Tensor,
        global_step: int,
        diff_loss_scalar: th.Tensor,
    ) -> Dict[str, object]:
        """
        Compute the configured training-time Koopman regularization.

        Args:
            x0_hat:
                Predicted-clean latents, shaped [B, 1, T, D] or [B, T, D].
            t:
                Diffusion timesteps for the microbatch, shaped [B].
            global_step:
                Global training step for warmup, interval and ramp.
            diff_loss_scalar:
                Scalar diffusion loss for the microbatch contribution cap.

        Returns:
            Total regularization loss, raw dynamics/manifold losses and
            diagnostic statistics.
        """
        zero = diff_loss_scalar.new_tensor(0.0)
        if global_step < self.warmup_steps or (global_step % self.interval) != 0:
            return {
                "active": False,
                "loss": zero,
                "loss_dyn_raw": zero,
                "loss_man_raw": zero,
                "loss_prior_raw": zero,
                "lambda_eff": 0.0,
                "lambda_dyn_eff": 0.0,
                "lambda_man_eff": 0.0,
                "active_frac": 0.0,
                "subbatch_frac": 0.0,
                "dyn_conf_mean": 0.0,
                "dyn_conf_active_frac": 0.0,
                "dyn_conf_max_mean": 0.0,
                "dyn_conf_margin_mean": 0.0,
                "dyn_conf_entropy_mean": 0.0,
                "dyn_conf_compound_mean": 0.0,
                "dyn_unknown_frac": 0.0,
                "man_conf_mean": 0.0,
                "man_conf_active_frac": 0.0,
                "dyn_rel": 0.0,
                "man_rel": 0.0,
            }

        x_seq = self._to_sequence(x0_hat)
        x_sel, t_sel, sel_stats = self._select_subbatch(x_seq, t)
        if x_sel is None or t_sel is None or x_sel.shape[0] == 0:
            return {
                "active": False,
                "loss": zero,
                "loss_dyn_raw": zero,
                "loss_man_raw": zero,
                "loss_prior_raw": zero,
                "lambda_eff": 0.0,
                "lambda_dyn_eff": 0.0,
                "lambda_man_eff": 0.0,
                "active_frac": float(sel_stats["active_frac"]),
                "subbatch_frac": float(sel_stats["subbatch_frac"]),
                "dyn_conf_mean": 0.0,
                "dyn_conf_active_frac": 0.0,
                "dyn_conf_max_mean": 0.0,
                "dyn_conf_margin_mean": 0.0,
                "dyn_conf_entropy_mean": 0.0,
                "dyn_conf_compound_mean": 0.0,
                "dyn_unknown_frac": 0.0,
                "man_conf_mean": 0.0,
                "man_conf_active_frac": 0.0,
                "dyn_rel": 0.0,
                "man_rel": 0.0,
            }

        x_window = self._build_single_window(x_sel)
        u_window = self._align_to_koopman_repr(x_window)
        _, z_dyn, u_recon, K, aux = self._encode_window(u_window)

        loss_dyn_vec, dyn_conf_vec, dyn_aux = self._compute_dyn_raw(z_dyn, K, aux=aux)
        loss_man_vec = self._compute_man_raw(u_recon, u_window)
        loss_dyn_raw = loss_dyn_vec.mean()
        loss_man_raw = loss_man_vec.mean()
        loss_prior_raw = loss_dyn_raw + self.gamma_man * loss_man_raw

        if self.prior_backend == "lskp" and getattr(self, "lskp_artifact_version", "lskp1") == "lskp13":
            conf_step = dyn_aux.get("conf_compound_step", None)
            if conf_step is None:
                raise ValueError("lskp13 requires step-level confidence stats")
            mse_step = dyn_aux.get("loss_mse_step", None)
            huber_step = dyn_aux.get("loss_huber_step", None)
            if mse_step is None or huber_step is None:
                raise ValueError("lskp13 requires step-level mse/huber loss stats")
            dyn_weight_step = th.ones_like(mse_step)
            low_conf_mask = None
            if self.selective_conf_threshold >= 0.0:
                low_conf_mask = conf_step < self.selective_conf_threshold
                dyn_weight_step = th.where(
                    low_conf_mask,
                    dyn_weight_step * self.selective_weight_scale,
                    dyn_weight_step,
                )
            burst_mask = None
            burst_conf_mask = None
            if aux is not None and "coarse_seq" in aux and self.selective_burst_scale < 1.0:
                c_seq = aux["coarse_seq"]
                burst_proxy_step = th.sqrt(
                    (c_seq[:, 1:, :] - c_seq[:, :-1, :]).pow(2).sum(dim=-1) + self.scale_eps
                )
                if self.lskp_burst_proxy_threshold is not None:
                    burst_mask = burst_proxy_step >= float(self.lskp_burst_proxy_threshold)
                    burst_conf_mask = conf_step < self.selective_burst_conf_threshold
                    burst_relax_mask = burst_mask & burst_conf_mask
                    dyn_weight_step = th.where(
                        burst_relax_mask,
                        dyn_weight_step * self.selective_burst_scale,
                        dyn_weight_step,
                    )
                else:
                    burst_proxy_step = None
            else:
                burst_proxy_step = None
            conditional_huber_mask = low_conf_mask
            if burst_mask is not None and burst_conf_mask is not None:
                conditional_huber_mask = (
                    conditional_huber_mask | (burst_mask & burst_conf_mask)
                    if conditional_huber_mask is not None
                    else (burst_mask & burst_conf_mask)
                )
            if conditional_huber_mask is None:
                step_loss = mse_step
            else:
                step_loss = th.where(conditional_huber_mask, huber_step, mse_step)
            dyn_weight = dyn_weight_step.mean(dim=1).clamp(min=0.0, max=1.0)
            dyn_conf_mean = float(dyn_aux.get("conf_max_vec", dyn_conf_vec).mean().detach().item())
            dyn_conf_max_mean = float(dyn_aux.get("conf_max_vec", dyn_conf_vec).mean().detach().item())
            dyn_conf_margin_mean = float(
                dyn_aux.get("conf_margin_vec", th.zeros_like(dyn_conf_vec)).mean().detach().item()
            )
            dyn_conf_entropy_mean = float(
                dyn_aux.get("conf_entropy_vec", th.zeros_like(dyn_conf_vec)).mean().detach().item()
            )
            dyn_conf_compound_mean = float(
                dyn_aux.get("conf_compound_vec", dyn_conf_vec).mean().detach().item()
            )
            dyn_unknown_frac = (
                float(low_conf_mask.float().mean().detach().item())
                if low_conf_mask is not None
                else 0.0
            )
            loss_dyn_term = (step_loss * dyn_weight_step).mean()
        elif self.prior_backend == "lskp" and getattr(self, "lskp_artifact_version", "lskp1") == "lskp13b":
            conf_step = dyn_aux.get("conf_compound_step", None)
            if conf_step is None:
                raise ValueError("lskp13b requires step-level confidence stats")
            mse_step = dyn_aux.get("loss_mse_step", None)
            huber_step = dyn_aux.get("loss_huber_step", None)
            if mse_step is None or huber_step is None:
                raise ValueError("lskp13b requires step-level mse/huber loss stats")
            dyn_weight_step = th.ones_like(mse_step)
            low_conf_mask = None
            if self.selective_conf_threshold >= 0.0:
                low_conf_mask = conf_step < self.selective_conf_threshold
                dyn_weight_step = th.where(
                    low_conf_mask,
                    dyn_weight_step * self.selective_weight_scale,
                    dyn_weight_step,
                )
            burst_mask = None
            burst_conf_mask = None
            if aux is not None and "coarse_seq" in aux:
                c_seq = aux["coarse_seq"]
                burst_proxy_step = th.sqrt(
                    (c_seq[:, 1:, :] - c_seq[:, :-1, :]).pow(2).sum(dim=-1) + self.scale_eps
                )
                if self.lskp_burst_proxy_threshold is not None:
                    burst_mask = burst_proxy_step >= float(self.lskp_burst_proxy_threshold)
                    if self.burst_base_scale < 1.0:
                        dyn_weight_step = th.where(
                            burst_mask,
                            dyn_weight_step * self.burst_base_scale,
                            dyn_weight_step,
                        )
                    if low_conf_mask is not None:
                        burst_conf_mask = burst_mask & (conf_step < self.selective_burst_conf_threshold)
                        if self.burst_extra_scale < 1.0:
                            dyn_weight_step = th.where(
                                burst_conf_mask,
                                dyn_weight_step * self.burst_extra_scale,
                                dyn_weight_step,
                            )
                else:
                    burst_proxy_step = None
            else:
                burst_proxy_step = None
            conditional_huber_mask = None
            if self.burst_huber_on_all and burst_mask is not None:
                conditional_huber_mask = burst_mask
            if low_conf_mask is not None:
                conditional_huber_mask = (
                    low_conf_mask if conditional_huber_mask is None else (conditional_huber_mask | low_conf_mask)
                )
            if burst_conf_mask is not None:
                conditional_huber_mask = (
                    burst_conf_mask
                    if conditional_huber_mask is None
                    else (conditional_huber_mask | burst_conf_mask)
                )
            if conditional_huber_mask is None:
                step_loss = mse_step
            else:
                step_loss = th.where(conditional_huber_mask, huber_step, mse_step)
            dyn_weight = dyn_weight_step.mean(dim=1).clamp(min=0.0, max=1.0)
            dyn_conf_mean = float(dyn_aux.get("conf_max_vec", dyn_conf_vec).mean().detach().item())
            dyn_conf_max_mean = float(dyn_aux.get("conf_max_vec", dyn_conf_vec).mean().detach().item())
            dyn_conf_margin_mean = float(
                dyn_aux.get("conf_margin_vec", th.zeros_like(dyn_conf_vec)).mean().detach().item()
            )
            dyn_conf_entropy_mean = float(
                dyn_aux.get("conf_entropy_vec", th.zeros_like(dyn_conf_vec)).mean().detach().item()
            )
            dyn_conf_compound_mean = float(
                dyn_aux.get("conf_compound_vec", dyn_conf_vec).mean().detach().item()
            )
            dyn_unknown_frac = (
                float(low_conf_mask.float().mean().detach().item())
                if low_conf_mask is not None
                else 0.0
            )
            dyn_conf_active_frac = float((dyn_weight > 0).float().mean().detach().item())
            loss_dyn_term = (step_loss * dyn_weight_step).mean()
        elif self.prior_backend == "lskp" and getattr(self, "lskp_artifact_version", "lskp1") != "lskp1":
            dyn_weight = dyn_aux.get("conf_compound_vec", dyn_conf_vec).clamp(min=0.0, max=1.0)
            dyn_conf_mean = float(dyn_weight.mean().detach().item())
            dyn_conf_max_mean = float(dyn_aux.get("conf_max_vec", dyn_conf_vec).mean().detach().item())
            dyn_conf_margin_mean = float(
                dyn_aux.get("conf_margin_vec", th.zeros_like(dyn_weight)).mean().detach().item()
            )
            dyn_conf_entropy_mean = float(
                dyn_aux.get("conf_entropy_vec", th.zeros_like(dyn_weight)).mean().detach().item()
            )
            dyn_conf_compound_mean = float(dyn_weight.mean().detach().item())
            unknown_mask = None
            if self.unknown_conf_threshold >= 0.0:
                unknown_mask = dyn_weight < self.unknown_conf_threshold
                dyn_weight = th.where(
                    unknown_mask,
                    dyn_weight * self.unknown_weight_scale,
                    dyn_weight,
                )
            dyn_unknown_frac = (
                float(unknown_mask.float().mean().detach().item())
                if unknown_mask is not None
                else 0.0
            )
            if self.dyn_confidence_threshold > 0.0:
                dyn_keep = dyn_weight >= self.dyn_confidence_threshold
                dyn_weight = th.where(dyn_keep, dyn_weight, th.zeros_like(dyn_weight))
            if self.dyn_confidence_power != 1.0:
                dyn_weight = dyn_weight.pow(self.dyn_confidence_power)
            dyn_conf_active_frac = float((dyn_weight > 0).float().mean().detach().item())
            loss_dyn_term = (loss_dyn_vec * dyn_weight).mean()
        else:
            dyn_weight = dyn_conf_vec.clamp(min=0.0, max=1.0)
            dyn_conf_mean = float(dyn_conf_vec.mean().detach().item())
            dyn_conf_max_mean = dyn_conf_mean
            dyn_conf_margin_mean = 0.0
            dyn_conf_entropy_mean = 0.0
            dyn_conf_compound_mean = dyn_conf_mean
            dyn_unknown_frac = 0.0
            if self.dyn_confidence_threshold > 0.0:
                dyn_keep = dyn_weight >= self.dyn_confidence_threshold
                dyn_weight = th.where(dyn_keep, dyn_weight, th.zeros_like(dyn_weight))
            if self.dyn_confidence_power != 1.0:
                dyn_weight = dyn_weight.pow(self.dyn_confidence_power)
            dyn_conf_active_frac = float((dyn_weight > 0).float().mean().detach().item())
            loss_dyn_term = (loss_dyn_vec * dyn_weight).mean()
        if self.prior_backend == "lskp" and getattr(self, "lskp_artifact_version", "lskp1") == "lskp13":
            dyn_conf_active_frac = float((dyn_weight > 0).float().mean().detach().item())

        man_conf_seq = aux.get("man_conf_seq", None)
        if man_conf_seq is not None:
            man_conf_vec = man_conf_seq.mean(dim=1)
            man_weight = man_conf_vec.clamp(min=0.0, max=1.0)
            if self.chart_confidence_threshold > 0.0:
                keep = man_weight >= self.chart_confidence_threshold
                man_weight = th.where(keep, man_weight, th.zeros_like(man_weight))
            if self.chart_confidence_power != 1.0:
                man_weight = man_weight.pow(self.chart_confidence_power)
            man_conf_mean = float(man_conf_vec.mean().detach().item())
            man_conf_active_frac = float((man_weight > 0).float().mean().detach().item())
        else:
            man_weight = th.ones_like(loss_man_vec)
            man_conf_mean = 1.0
            man_conf_active_frac = 1.0
        loss_man_term = (loss_man_vec * man_weight).mean()

        use_split_lambdas = ("lambda_dyn_base" in self.koopman_cfg) or ("lambda_man_base" in self.koopman_cfg)
        if use_split_lambdas:
            lambda_dyn_eff = self._lambda_eff_component(
                diff_loss_scalar,
                loss_dyn_term,
                global_step,
                base_value=self.lambda_dyn_base,
                ratio_cap=self.ratio_cap_dyn_vs_diff,
            )
            lambda_man_eff = self._lambda_eff_component(
                diff_loss_scalar,
                loss_man_term,
                global_step,
                base_value=self.lambda_man_base,
                ratio_cap=self.ratio_cap_man_vs_diff,
            )
            total_loss = lambda_dyn_eff * loss_dyn_term + lambda_man_eff * loss_man_term
            lambda_eff = float((lambda_dyn_eff + lambda_man_eff).detach().item())
        else:
            effective_prior_raw = loss_dyn_term + self.gamma_man * loss_man_term
            lambda_dyn_eff = zero
            lambda_man_eff = zero
            lambda_scalar = self._lambda_eff(diff_loss_scalar, effective_prior_raw, global_step)
            total_loss = lambda_scalar * effective_prior_raw
            lambda_eff = float(lambda_scalar.detach().item())

        dyn_rel = float(loss_dyn_raw.detach().item() / self.dyn_ref) if self.dyn_ref else 0.0
        man_rel = float(loss_man_raw.detach().item() / self.man_ref) if self.man_ref else 0.0

        return {
            "active": True,
            "loss": total_loss,
            "loss_dyn_raw": loss_dyn_raw,
            "loss_man_raw": loss_man_raw,
            "loss_prior_raw": loss_prior_raw,
            "lambda_eff": lambda_eff,
            "lambda_dyn_eff": float(lambda_dyn_eff.detach().item()),
            "lambda_man_eff": float(lambda_man_eff.detach().item()),
            "active_frac": float(sel_stats["active_frac"]),
            "subbatch_frac": float(sel_stats["subbatch_frac"]),
            "dyn_conf_mean": dyn_conf_mean,
            "dyn_conf_active_frac": dyn_conf_active_frac,
            "dyn_conf_max_mean": dyn_conf_max_mean,
            "dyn_conf_margin_mean": dyn_conf_margin_mean,
            "dyn_conf_entropy_mean": dyn_conf_entropy_mean,
            "dyn_conf_compound_mean": dyn_conf_compound_mean,
            "dyn_unknown_frac": dyn_unknown_frac,
            "man_conf_mean": man_conf_mean,
            "man_conf_active_frac": man_conf_active_frac,
            "dyn_rel": dyn_rel,
            "man_rel": man_rel,
        }
