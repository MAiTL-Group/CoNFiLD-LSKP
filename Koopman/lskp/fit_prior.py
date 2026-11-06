"""Fit a Local Stochastic Koopman Prior (LSKP) from aligned CNF latents."""

import argparse
from pathlib import Path
from typing import Dict

import torch

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

try:
    from .data import (
        LocalStochasticKoopmanPrior,
        build_transition_descriptor,
        compute_lskp_whitened_residual,
        fit_descriptor_stats,
        fit_local_stochastic_koopman_prior,
        normalize_descriptor,
    )
except ImportError:  # pragma: no cover
    from data import (
        LocalStochasticKoopmanPrior,
        build_transition_descriptor,
        compute_lskp_whitened_residual,
        fit_descriptor_stats,
        fit_local_stochastic_koopman_prior,
        normalize_descriptor,
    )

from Koopman.pkcp.data import (
    build_aligned_sequences,
    fit_alignment_stats,
    fit_pca_coarse_stats,
    load_npy,
    project_pca_coarse_observable,
)
from Koopman.pkcp.utils import (
    finish_swanlab_run,
    init_swanlab_run,
    load_yaml,
    save_checkpoint,
    save_json,
    save_yaml,
    set_global_seed,
    swanlab_log,
)


def _format_stage_metrics(metrics: Dict[str, float]) -> Dict[str, str]:
    return {k: f"{float(v):.4g}" for k, v in metrics.items()}


def _stage_progress(total_stages: int, enable_tqdm: bool):
    if enable_tqdm and tqdm is not None:
        return tqdm(total=total_stages, desc="LSKP fit", dynamic_ncols=True)
    return None


def _advance_stage(progress, stage_name: str, metrics: Dict[str, float], run, step: int) -> None:
    if progress is not None:
        progress.set_description(f"LSKP {stage_name}")
        if metrics:
            progress.set_postfix(_format_stage_metrics(metrics))
        progress.update(1)
    else:
        print(f"[LSKP] {stage_name}: {metrics}", flush=True)
    swanlab_log(run, {f"lskp/{k}": v for k, v in metrics.items()}, step=step)


def fit_lskp_prior(cfg: Dict) -> Dict[str, Dict]:
    """Fit and export a local stochastic Koopman prior from aligned CNF latents."""
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(save_dir / "lskp_config.yaml", cfg)

    seed = int(cfg.get("seed", 3407))
    deterministic = bool(cfg.get("deterministic", True))
    enable_tqdm = bool(cfg.get("enable_tqdm", True))
    set_global_seed(seed, deterministic=deterministic)
    run = init_swanlab_run(cfg, default_project="CoNFiLD-LSKP", default_run_name=save_dir.name)
    progress = _stage_progress(total_stages=6, enable_tqdm=enable_tqdm)

    try:
        x_train = load_npy(Path(cfg["train_npy"]))
        x_valid = load_npy(Path(cfg["valid_npy"]))
        _advance_stage(
            progress,
            "load",
            {
                "train_samples": float(x_train.shape[0]),
                "valid_samples": float(x_valid.shape[0]),
                "dim_input": float(x_train.shape[-1]),
            },
            run,
            step=1,
        )

        stats = fit_alignment_stats(
            x_train,
            prior_variant=cfg["prior_variant"],
            repr_mode=cfg["repr_mode"],
            window_size=int(cfg["window_size"]),
            temporal_stride=int(cfg["temporal_stride"]),
        )
        aligned_train = build_aligned_sequences(
            x_train,
            prior_variant=cfg["prior_variant"],
            repr_mode=cfg["repr_mode"],
            window_size=int(cfg["window_size"]),
            temporal_stride=int(cfg["temporal_stride"]),
            stats=stats,
        )
        aligned_valid = build_aligned_sequences(
            x_valid,
            prior_variant=cfg["prior_variant"],
            repr_mode=cfg["repr_mode"],
            window_size=int(cfg["window_size"]),
            temporal_stride=int(cfg["temporal_stride"]),
            stats=stats,
        )
        _advance_stage(
            progress,
            "align",
            {
                "aligned_train_t": float(aligned_train.shape[1]),
                "aligned_valid_t": float(aligned_valid.shape[1]),
            },
            run,
            step=2,
        )

        dim_coarse = int(cfg["dim_coarse"])
        pca_stats = fit_pca_coarse_stats(
            aligned_train,
            energy_ratio=float(cfg.get("coarse_energy_ratio", 0.95)),
            dim_coarse=dim_coarse,
        )
        c_train = project_pca_coarse_observable(aligned_train, pca_stats)
        c_valid = project_pca_coarse_observable(aligned_valid, pca_stats)
        _advance_stage(
            progress,
            "pca",
            {
                "dim_coarse": float(dim_coarse),
                "pca_rank": float(pca_stats.rank),
                "energy_ratio": float(pca_stats.energy_ratio),
            },
            run,
            step=3,
        )

        descriptor_cfg = {
            "include_level": bool(cfg.get("include_level", True)),
            "include_delta": bool(cfg.get("include_delta", True)),
            "include_energy_delta": bool(cfg.get("include_energy_delta", True)),
            "include_prev_delta": bool(cfg.get("include_prev_delta", False)),
            "include_delta2": bool(cfg.get("include_delta2", False)),
        }
        q_train = build_transition_descriptor(c_train, **descriptor_cfg)
        q_valid = build_transition_descriptor(c_valid, **descriptor_cfg)
        descriptor_normalize = bool(cfg.get("descriptor_normalize", False))
        if descriptor_normalize:
            descriptor_mean, descriptor_std = fit_descriptor_stats(q_train)
            q_train_fit = normalize_descriptor(q_train, descriptor_mean, descriptor_std)
        else:
            descriptor_mean, descriptor_std = None, None
            q_train_fit = q_train

        delta_train = c_train[:, 1:, :] - c_train[:, :-1, :]
        burst_proxy_train = torch.sqrt(delta_train.pow(2).sum(dim=-1) + 1.0e-6)
        burst_kmeans_quantile = float(cfg.get("burst_kmeans_quantile", 0.75))
        burst_kmeans_boost = float(cfg.get("burst_kmeans_boost", 1.0))
        burst_proxy_threshold = (
            float(torch.quantile(burst_proxy_train.reshape(-1), burst_kmeans_quantile).item())
            if 0.0 < burst_kmeans_quantile < 1.0
            else None
        )

        prior = fit_local_stochastic_koopman_prior(
            c_train,
            q_train_fit,
            num_regimes=int(cfg.get("num_regimes", 4)),
            ridge=float(cfg.get("lambda_k_ridge", 1.0e-3)),
            tau=float(cfg.get("regime_tau", 5.0)),
            kmeans_iters=int(cfg.get("kmeans_iters", 20)),
            seed=seed,
            enable_tqdm=enable_tqdm,
            burst_scores=burst_proxy_train,
            burst_quantile=burst_kmeans_quantile,
            burst_boost=burst_kmeans_boost,
            burst_regime_count=int(cfg.get("burst_regime_count", 0)),
        )
        prior.descriptor_mean = descriptor_mean
        prior.descriptor_std = descriptor_std
        _advance_stage(
            progress,
            "fit",
            {
                "num_regimes": float(prior.centers.shape[0]),
                "burst_regime_count": float(int(cfg.get("burst_regime_count", 0))),
                "dim_descriptor": float(q_train.shape[-1]),
                "dim_coarse": float(c_train.shape[-1]),
                "sigma2_mean": float(prior.sigma2.mean().item()),
                "burst_proxy_threshold": float(prior.burst_proxy_threshold or 0.0),
            },
            run,
            step=4,
        )

        validation_loss_mode = str(cfg.get("validation_loss_mode", "mse")).lower()
        validation_huber_delta = float(cfg.get("validation_huber_delta", 2.0))
        dyn_whitened_val, regime_conf_val, val_aux = compute_lskp_whitened_residual(
            c_valid,
            q_valid,
            prior,
            loss_mode=validation_loss_mode,
            huber_delta=validation_huber_delta,
            return_aux=True,
        )
        tail_start = max(0, q_valid.shape[1] - max(1, q_valid.shape[1] // 4))
        tail_loss, _, tail_aux = compute_lskp_whitened_residual(
            c_valid[:, tail_start:, :],
            q_valid[:, tail_start:, :],
            prior,
            loss_mode=validation_loss_mode,
            huber_delta=validation_huber_delta,
            return_aux=True,
        )
        lag1_resid = (c_valid[:, 1:, :] - c_valid[:, :-1, :]).pow(2).mean()
        unknown_conf_threshold = float(cfg.get("unknown_conf_threshold", 0.15))
        unknown_rate = float((val_aux["conf_compound_vec"] < unknown_conf_threshold).float().mean().item())
        tail_unknown_rate = float((tail_aux["conf_compound_vec"] < unknown_conf_threshold).float().mean().item())
        _advance_stage(
            progress,
            "validate",
            {
                "dyn_whitened_val": float(dyn_whitened_val.mean().item()),
                "tail_dyn_whitened_val": float(tail_loss.mean().item()),
                "lag1_residual_val": float(lag1_resid.item()),
                "regime_conf_mean": float(regime_conf_val.mean().item()),
                "regime_conf_compound_mean": float(val_aux["conf_compound_vec"].mean().item()),
                "unknown_rate_val": unknown_rate,
                "tail_unknown_rate_val": tail_unknown_rate,
            },
            run,
            step=5,
        )

        artifact = {
            "prior_backend": "lskp",
            "artifact_version": str(cfg.get("artifact_version", "lskp1")).lower(),
            "prior_variant": cfg["prior_variant"],
            "repr_mode": cfg["repr_mode"],
            "temporal_stride": int(cfg["temporal_stride"]),
            "window_size": int(cfg["window_size"]),
            "dim_input": int(cfg["dim_input"]),
            "dim_coarse": int(dim_coarse),
            "alignment_stats": stats.to_dict(),
            "pca_coarse_stats": pca_stats.to_dict(),
            "descriptor_cfg": descriptor_cfg,
            "local_stochastic_prior": prior.to_dict(),
            "best_metric": float(dyn_whitened_val.mean().item()),
            "burst_proxy_threshold": (
                float(prior.burst_proxy_threshold) if prior.burst_proxy_threshold is not None else None
            ),
            "burst_regime_count": int(cfg.get("burst_regime_count", 0)),
            "train_config": cfg,
        }
        save_checkpoint(save_dir / "lskp_prior.pt", artifact)

        ref = {
            "prior_backend": "lskp",
            "artifact_version": str(cfg.get("artifact_version", "lskp1")).lower(),
            "dyn_whitened_val": float(dyn_whitened_val.mean().item()),
            "tail_dyn_whitened_val": float(tail_loss.mean().item()),
            "lag1_residual_val": float(lag1_resid.item()),
            "regime_conf_mean": float(regime_conf_val.mean().item()),
            "regime_conf_compound_mean": float(val_aux["conf_compound_vec"].mean().item()),
            "unknown_rate_val": unknown_rate,
            "tail_unknown_rate_val": tail_unknown_rate,
            "num_regimes": int(prior.centers.shape[0]),
            "burst_regime_count": int(cfg.get("burst_regime_count", 0)),
            "dim_descriptor": int(q_train.shape[-1]),
            "burst_proxy_train_mean": float(burst_proxy_train.mean().item()),
            "burst_proxy_threshold": (
                float(burst_proxy_threshold) if burst_proxy_threshold is not None else None
            ),
        }
        save_json(save_dir / "prior_reference_metrics.json", ref)
        _advance_stage(
            progress,
            "export",
            {
                "saved_artifact": 1.0,
                "saved_reference_metrics": 1.0,
            },
            run,
            step=6,
        )
        return {"artifact": artifact, "reference_metrics": ref}
    finally:
        if progress is not None:
            progress.close()
        finish_swanlab_run(run, step=6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    fit_lskp_prior(load_yaml(args.config))


if __name__ == "__main__":
    main()
