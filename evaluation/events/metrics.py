from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.stats import entropy, kurtosis, ks_2samp, skew, wasserstein_distance

from evaluation.diversity_metrics import (
    cluster_coverage_metrics,
    covariance_geometry,
    local_radius_precision_recall,
    rbf_mmd,
    sample_feature_rows,
)
from evaluation.time_diagnostics import acf_1d, compute_energy_tint_metrics, tint_from_acf

from .common import EPS, EvalParams, SplitSlices, flatten_segments, flatten_frames_for_source, relative_gap, split_slices


def fit_slope(y: np.ndarray) -> float:
    t = np.arange(len(y), dtype=np.float64)
    if len(y) < 2:
        return 0.0
    a, _ = np.polyfit(t, y.astype(np.float64), 1)
    return float(abs(a))


def bootstrap_match(dns_values: np.ndarray, gen_values: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n = min(dns_values.size, gen_values.size)
    dns_idx = rng.choice(dns_values.size, size=n, replace=dns_values.size < n)
    gen_idx = rng.choice(gen_values.size, size=n, replace=gen_values.size < n)
    return dns_values[dns_idx], gen_values[gen_idx]


def js_divergence_from_hist(a: np.ndarray, b: np.ndarray, edges: np.ndarray) -> float:
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    ha = np.clip(ha, EPS, None)
    hb = np.clip(hb, EPS, None)
    ha = ha / ha.sum()
    hb = hb / hb.sum()
    m = 0.5 * (ha + hb)
    return float(0.5 * (entropy(ha, m) + entropy(hb, m)))


def sym_kl_from_hist(a: np.ndarray, b: np.ndarray, edges: np.ndarray) -> float:
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    ha = np.clip(ha, EPS, None)
    hb = np.clip(hb, EPS, None)
    ha = ha / ha.sum()
    hb = hb / hb.sum()
    return float(0.5 * (entropy(ha, hb) + entropy(hb, ha)))


def sample_values(arr: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    flat = np.asarray(arr).reshape(-1)
    if flat.size <= max_count:
        return flat.astype(np.float64)
    idx = rng.choice(flat.size, size=max_count, replace=False)
    return flat[idx].astype(np.float64)


def safe_log_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.log(np.clip(np.asarray(a, dtype=np.float64), EPS, None))
    bb = np.log(np.clip(np.asarray(b, dtype=np.float64), EPS, None))
    return float(np.sqrt(np.mean((aa - bb) ** 2)))


def safe_rel_l1(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.sum(np.abs(aa - bb)) / (np.sum(np.abs(bb)) + EPS))


def fit_spectrum_slope(curve: np.ndarray, edges: np.ndarray) -> float:
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = np.isfinite(curve) & (curve > EPS) & np.isfinite(centers) & (centers > EPS)
    idx = np.where(mask)[0]
    if idx.size < 4:
        return float("nan")
    start = max(1, idx.size // 4)
    end = max(start + 2, (3 * idx.size) // 4)
    sel = idx[start:end]
    if sel.size < 2:
        sel = idx
    x = np.log(centers[sel])
    y = np.log(curve[sel])
    a, _ = np.polyfit(x, y, 1)
    return float(a)


def high_freq_ratio(curve: np.ndarray, edges: np.ndarray, hi_frac: float = 0.25) -> float:
    n = len(curve)
    start = max(0, int(math.floor((1.0 - hi_frac) * n)))
    return float(np.sum(curve[start:]) / (np.sum(curve) + EPS))


def mean_curve(curves: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    return {p: np.mean(v, axis=0) for p, v in curves.items()}


def ess_beta(curves: Dict[int, np.ndarray], target_p: int) -> float:
    s3 = np.clip(curves[3] if 3 in curves else curves[2], EPS, None)
    sp = np.clip(curves[target_p], EPS, None)
    x = np.log(np.asarray(s3, dtype=np.float64))
    y = np.log(np.asarray(sp, dtype=np.float64))
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return float("nan")
    a, _ = np.polyfit(x[mask], y[mask], 1)
    return float(a)


def pdf_metrics(dns_values: np.ndarray, gen_values: np.ndarray, tail_low_q: float, tail_high_q: float, max_mmd_values: int) -> Dict[str, float]:
    low = float(np.quantile(dns_values, tail_low_q))
    high = float(np.quantile(dns_values, tail_high_q))
    all_values = np.concatenate([dns_values, gen_values])
    edges = np.quantile(all_values, np.linspace(0.0, 1.0, 129))
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.linspace(float(all_values.min()), float(all_values.max()) + 1.0e-6, 3)
    mmd2, mmd = rbf_mmd(dns_values[:, None], gen_values[:, None], max_points=max_mmd_values)
    return {
        "dns_mean": float(np.mean(dns_values)),
        "gen_mean": float(np.mean(gen_values)),
        "dns_std": float(np.std(dns_values)),
        "gen_std": float(np.std(gen_values)),
        "dns_skew": float(skew(dns_values)),
        "gen_skew": float(skew(gen_values)),
        "dns_kurt": float(kurtosis(dns_values, fisher=False)),
        "gen_kurt": float(kurtosis(gen_values, fisher=False)),
        "tail_low": float(np.mean(gen_values < low)),
        "tail_high": float(np.mean(gen_values > high)),
        "tail_total": float(np.mean((gen_values < low) | (gen_values > high))),
        "W1": float(wasserstein_distance(dns_values, gen_values)),
        "JS": js_divergence_from_hist(dns_values, gen_values, edges),
        "sym_KL": sym_kl_from_hist(dns_values, gen_values, edges),
        "MMD2": float(mmd2),
        "MMD": float(mmd),
    }


def sample_time_pair_distances(frame_features: np.ndarray, rng: np.random.Generator, delta_max: int, n_pairs: int) -> np.ndarray:
    t = frame_features.shape[0]
    if t < 2:
        return np.zeros(1, dtype=np.float64)
    max_delta = min(int(delta_max), t - 1)
    deltas = rng.integers(1, max_delta + 1, size=n_pairs)
    start = rng.integers(0, t - deltas, size=n_pairs)
    end = start + deltas
    diff = frame_features[end] - frame_features[start]
    return np.linalg.norm(diff, axis=1)


def sample_time_pair_distances_fixed_lag(frame_features: np.ndarray, rng: np.random.Generator, lag: int, n_pairs: int) -> np.ndarray:
    t = frame_features.shape[0]
    lag = int(lag)
    if t <= lag or lag < 1:
        return np.zeros(1, dtype=np.float64)
    start = rng.integers(0, t - lag, size=n_pairs)
    end = start + lag
    diff = frame_features[end] - frame_features[start]
    return np.linalg.norm(diff, axis=1)


def sample_cross_distances(frame_features_st: np.ndarray, rng: np.random.Generator, n_pairs: int) -> np.ndarray:
    s, t, _ = frame_features_st.shape
    if s < 2:
        return np.zeros(1, dtype=np.float64)
    sample_a = rng.integers(0, s, size=n_pairs)
    sample_b = rng.integers(0, s - 1, size=n_pairs)
    sample_b = np.where(sample_b >= sample_a, sample_b + 1, sample_b)
    ti = rng.integers(0, t, size=n_pairs)
    diff = frame_features_st[sample_a, ti] - frame_features_st[sample_b, ti]
    return np.linalg.norm(diff, axis=1)


def _concat_time_distances(frame_features_st: np.ndarray, rng: np.random.Generator, sampler, n_pairs: int, *sampler_args) -> np.ndarray:
    s = frame_features_st.shape[0]
    per_seq = max(1, int(n_pairs) // max(1, s))
    return np.concatenate([sampler(frame_features_st[idx], rng, *sampler_args, per_seq) for idx in range(s)])


def compute_energy_family(energy_gen: np.ndarray, energy_dns: np.ndarray, seg_gen: SplitSlices, seg_dns: SplitSlices, acf_max_lag: int, rng: np.random.Generator) -> Dict[str, object]:
    dns_global = energy_dns.reshape(-1)
    gen_global = energy_gen.reshape(-1)
    dns_boot, gen_boot = bootstrap_match(dns_global, gen_global, rng)
    dns_early, gen_early = bootstrap_match(flatten_segments(energy_dns, seg_dns.early), flatten_segments(energy_gen, seg_gen.early), rng)
    dns_late, gen_late = bootstrap_match(flatten_segments(energy_dns, seg_dns.late), flatten_segments(energy_gen, seg_gen.late), rng)
    out = {
        "energy_gen": energy_gen,
        "energy_dns": energy_dns,
        "dns_mean": float(np.mean(dns_global)),
        "gen_mean": float(np.mean(gen_global)),
        "gen_std": float(np.std(gen_global)),
        "gen_median": float(np.median(gen_global)),
        "gen_p90": float(np.quantile(gen_global, 0.9)),
        "drift_split": float(abs(np.mean(flatten_segments(energy_gen, seg_gen.late)) - np.mean(flatten_segments(energy_gen, seg_gen.early)))),
        "drift_slope": float(np.mean([fit_slope(energy_gen[s]) for s in range(energy_gen.shape[0])])),
        "drift_rel": relative_gap(np.mean(flatten_segments(energy_gen, seg_gen.late)), np.mean(flatten_segments(energy_gen, seg_gen.early))),
        "drift_slope_rel": float(np.mean([fit_slope(energy_gen[s]) / (abs(np.mean(energy_gen[s])) + EPS) for s in range(energy_gen.shape[0])])),
        "tail_p95_p50": float(np.quantile(gen_global, 0.95) / (np.quantile(gen_global, 0.50) + EPS)),
        "kurtosis": float(kurtosis(gen_global, fisher=False)),
        "W1_global": float(wasserstein_distance(dns_boot, gen_boot)),
        "KS_global": float(ks_2samp(dns_boot, gen_boot).statistic),
        "W1_early": float(wasserstein_distance(dns_early, gen_early)),
        "W1_late": float(wasserstein_distance(dns_late, gen_late)),
        "W1_early_late": float(wasserstein_distance(flatten_segments(energy_gen, seg_gen.early), flatten_segments(energy_gen, seg_gen.late))),
    }
    out.update(compute_energy_tint_metrics(energy_dns, energy_gen, seg_dns.late, seg_gen.late, acf_max_lag))
    return out


def compute_time_metrics_detailed(
    phi_dns: np.ndarray,
    phi_gen: np.ndarray,
    energy_dns: np.ndarray,
    energy_gen: np.ndarray,
    seg_dns: SplitSlices,
    seg_gen: SplitSlices,
    rng: np.random.Generator,
    delta_max: int,
    delta_pairs: int,
    lag_values: Iterable[int],
) -> Tuple[Dict[str, object], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    dns_vals = _concat_time_distances(phi_dns, rng, sample_time_pair_distances, delta_pairs, delta_max)
    gen_time_vals = _concat_time_distances(phi_gen, rng, sample_time_pair_distances, delta_pairs, delta_max)
    gen_cross_vals = sample_cross_distances(phi_gen, rng, delta_pairs)

    phi_dns_early = phi_dns[:, seg_dns.early]
    phi_dns_late = phi_dns[:, seg_dns.late]
    phi_gen_early = phi_gen[:, seg_gen.early]
    phi_gen_late = phi_gen[:, seg_gen.late]

    dns_early_vals = _concat_time_distances(phi_dns_early, rng, sample_time_pair_distances, delta_pairs, delta_max)
    gen_early_vals = _concat_time_distances(phi_gen_early, rng, sample_time_pair_distances, delta_pairs, delta_max)
    dns_late_vals = _concat_time_distances(phi_dns_late, rng, sample_time_pair_distances, delta_pairs, delta_max)
    gen_late_vals = _concat_time_distances(phi_gen_late, rng, sample_time_pair_distances, delta_pairs, delta_max)
    gen_tail_cross_vals = sample_cross_distances(phi_gen_late, rng, delta_pairs)

    max_valid_lag = max(0, min(phi_dns.shape[1] - 1, phi_gen.shape[1] - 1, int(delta_max)))
    valid_lags = [int(lag) for lag in lag_values if 1 <= int(lag) <= max_valid_lag]
    lag_w1: Dict[int, float] = {}
    dns_lag_mean: Dict[int, float] = {}
    gen_lag_mean: Dict[int, float] = {}
    dns_lag_dist: Dict[int, np.ndarray] = {}
    for lag in valid_lags:
        dns_lag = _concat_time_distances(phi_dns, rng, sample_time_pair_distances_fixed_lag, delta_pairs, lag)
        gen_lag = _concat_time_distances(phi_gen, rng, sample_time_pair_distances_fixed_lag, delta_pairs, lag)
        dns_lag_dist[lag] = dns_lag
        lag_w1[lag] = float(wasserstein_distance(dns_lag, gen_lag))
        dns_lag_mean[lag] = float(np.mean(dns_lag))
        gen_lag_mean[lag] = float(np.mean(gen_lag))

    metrics = {
        "dns_mean": float(np.mean(dns_vals)),
        "gen_time_mean": float(np.mean(gen_time_vals)),
        "gen_cross_mean": float(np.mean(gen_cross_vals)),
        "w1_time": float(wasserstein_distance(dns_vals, gen_time_vals)),
        "w1_cross": float(wasserstein_distance(dns_vals, gen_cross_vals)),
        "w1_time_early": float(wasserstein_distance(dns_early_vals, gen_early_vals)),
        "w1_time_late": float(wasserstein_distance(dns_late_vals, gen_late_vals)),
        "w1_time_drift": float(wasserstein_distance(dns_late_vals, gen_late_vals) - wasserstein_distance(dns_early_vals, gen_early_vals)),
        "tail_dns_mean": float(np.mean(dns_late_vals)),
        "tail_gen_time_mean": float(np.mean(gen_late_vals)),
        "tail_gen_cross_mean": float(np.mean(gen_tail_cross_vals)),
        "tail_w1_time": float(wasserstein_distance(dns_late_vals, gen_late_vals)),
        "tail_w1_cross": float(wasserstein_distance(dns_late_vals, gen_tail_cross_vals)),
        "lag_values": valid_lags,
        "lag_conditioned_w1_time": {str(lag): lag_w1[lag] for lag in valid_lags},
        "lag_conditioned_dns_mean": {str(lag): dns_lag_mean[lag] for lag in valid_lags},
        "lag_conditioned_gen_time_mean": {str(lag): gen_lag_mean[lag] for lag in valid_lags},
    }
    for lag in valid_lags:
        metrics[f"w1_time_lag_{lag}"] = lag_w1[lag]
        metrics[f"dns_time_lag_mean_{lag}"] = dns_lag_mean[lag]
        metrics[f"gen_time_lag_mean_{lag}"] = gen_lag_mean[lag]

    plot_payload = {
        "dns": dns_vals,
        "gen_time": gen_time_vals,
        "gen_cross": gen_cross_vals,
        "lag_values": np.asarray(valid_lags, dtype=np.int64),
        "w1_time_by_lag": np.asarray([lag_w1[lag] for lag in valid_lags], dtype=np.float64),
        "dns_lag_mean": np.asarray([dns_lag_mean[lag] for lag in valid_lags], dtype=np.float64),
        "gen_time_lag_mean": np.asarray([gen_lag_mean[lag] for lag in valid_lags], dtype=np.float64),
    }

    tint_agg = compute_energy_tint_metrics(energy_dns, energy_gen, seg_dns.late, seg_gen.late, max(1, min(128, energy_dns.shape[1] - 1, energy_gen.shape[1] - 1)))
    dns_tint_ref = float(tint_agg["Tint_dns"])
    dns_tint_late_ref = float(tint_agg["late_tail_Tint_dns"])
    per_sample = {
        "late_tail_Tint_gap": [],
        "tail_w1_time": [],
        "w1_time_lag_1": [],
        "w1_time_lag_16": [],
    }
    for s in range(phi_gen.shape[0]):
        gen_tint = tint_from_acf(acf_1d(energy_gen[s], min(128, energy_gen.shape[1] - 1)))
        gen_tint_late = tint_from_acf(acf_1d(energy_gen[s, seg_gen.late], max(1, min(128, energy_gen[:, seg_gen.late].shape[1] - 1))))
        per_sample["late_tail_Tint_gap"].append(relative_gap(gen_tint_late, dns_tint_late_ref))
        gen_late_dist = sample_time_pair_distances(phi_gen_late[s], rng, delta_max, max(1, delta_pairs // max(1, phi_gen.shape[0])))
        per_sample["tail_w1_time"].append(float(wasserstein_distance(dns_late_vals, gen_late_dist)))
        for lag in [1, 16]:
            if lag in dns_lag_dist:
                gen_lag_dist = sample_time_pair_distances_fixed_lag(phi_gen[s], rng, lag, max(1, delta_pairs // max(1, phi_gen.shape[0])))
                per_sample[f"w1_time_lag_{lag}"].append(float(wasserstein_distance(dns_lag_dist[lag], gen_lag_dist)))
            else:
                per_sample[f"w1_time_lag_{lag}"].append(float("nan"))
    return metrics, plot_payload, {k: np.asarray(v, dtype=np.float64) for k, v in per_sample.items()}


def evaluate_bundle(
    x_gen_eval: np.ndarray,
    x_dns_eval: np.ndarray,
    backend,
    cache,
    params: EvalParams,
    rng: np.random.Generator,
    boundary_idx: Optional[np.ndarray] = None,
    boundary_normals: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    seg_gen = split_slices(x_gen_eval.shape[1], params.early_frac, params.late_frac)
    seg_dns = split_slices(x_dns_eval.shape[1], params.early_frac, params.late_frac)

    energy_gen = backend.energy_series(x_gen_eval)
    energy_dns = backend.energy_series(x_dns_eval)
    energy_metrics = compute_energy_family(energy_gen, energy_dns, seg_gen, seg_dns, params.acf_max_lag, rng)

    flat_gen = x_gen_eval.reshape(-1, *x_gen_eval.shape[2:])
    flat_dns = x_dns_eval.reshape(-1, *x_dns_eval.shape[2:])
    deriv_gen = backend.compute_derivative_frame_stats(flat_gen, cache, batch_size=params.derivative_batch_size, keep_spatial=False)
    deriv_dns = backend.compute_derivative_frame_stats(flat_dns, cache, batch_size=params.derivative_batch_size, keep_spatial=False)
    deriv_gen_frame = {k: deriv_gen[k].reshape(x_gen_eval.shape[0], x_gen_eval.shape[1]) for k in ["enstrophy_frame", "strain2_frame", "grad2_frame", "r_div_frame"]}
    deriv_dns_frame = {k: deriv_dns[k].reshape(x_dns_eval.shape[0], x_dns_eval.shape[1]) for k in ["enstrophy_frame", "strain2_frame", "grad2_frame", "r_div_frame"]}
    derivative_metrics = {
        "enstrophy_dns": float(np.mean(deriv_dns["enstrophy_frame"])),
        "enstrophy_gen": float(np.mean(deriv_gen["enstrophy_frame"])),
        "strain2_dns": float(np.mean(deriv_dns["strain2_frame"])),
        "strain2_gen": float(np.mean(deriv_gen["strain2_frame"])),
        "grad2_dns": float(np.mean(deriv_dns["grad2_frame"])),
        "grad2_gen": float(np.mean(deriv_gen["grad2_frame"])),
        "enstrophy_gap": relative_gap(np.mean(deriv_gen["enstrophy_frame"]), np.mean(deriv_dns["enstrophy_frame"])),
        "strain2_gap": relative_gap(np.mean(deriv_gen["strain2_frame"]), np.mean(deriv_dns["strain2_frame"])),
        "grad2_gap": relative_gap(np.mean(deriv_gen["grad2_frame"]), np.mean(deriv_dns["grad2_frame"])),
        "enstrophy_drift": relative_gap(np.mean(deriv_gen_frame["enstrophy_frame"][:, seg_gen.late]), np.mean(deriv_gen_frame["enstrophy_frame"][:, seg_gen.early])),
        "div_l2_dns": float(deriv_dns["div_l2"]),
        "div_l2_gen": float(deriv_gen["div_l2"]),
        "div_inf_dns": float(deriv_dns["div_inf"]),
        "div_inf_gen": float(deriv_gen["div_inf"]),
        "r_div": relative_gap(np.mean(deriv_gen["r_div_frame"]), np.mean(deriv_dns["r_div_frame"])),
    }
    derivative_metrics.update(backend.compute_boundary_metrics(x_gen_eval, x_dns_eval, boundary_idx, boundary_normals))

    dist_frames_gen, dist_frames_dns = flatten_frames_for_source(
        x_gen_eval, x_dns_eval, min(params.max_structure_frames, params.max_spectrum_frames), backend.field_ndim
    )
    dist_deriv_gen = backend.compute_derivative_frame_stats(dist_frames_gen, cache, batch_size=params.derivative_batch_size, keep_spatial=True)
    dist_deriv_dns = backend.compute_derivative_frame_stats(dist_frames_dns, cache, batch_size=params.derivative_batch_size, keep_spatial=True)
    derivative_metrics["kurt_gap"] = relative_gap(kurtosis(dist_deriv_gen["omega_abs"].reshape(-1), fisher=False), kurtosis(dist_deriv_dns["omega_abs"].reshape(-1), fisher=False))

    dist_pdf_data = {
        "vel_mag": (
            sample_values(np.linalg.norm(dist_frames_dns[..., : min(3, dist_frames_dns.shape[-1])], axis=-1), params.max_distribution_values, rng),
            sample_values(np.linalg.norm(dist_frames_gen[..., : min(3, dist_frames_gen.shape[-1])], axis=-1), params.max_distribution_values, rng),
        ),
        "omega_abs": (
            sample_values(dist_deriv_dns["omega_abs"], params.max_distribution_values, rng),
            sample_values(dist_deriv_gen["omega_abs"], params.max_distribution_values, rng),
        ),
    }
    for c in range(min(2, dist_frames_dns.shape[-1])):
        dist_pdf_data[f"u{c+1}"] = (
            sample_values(dist_frames_dns[..., c], params.max_distribution_values, rng),
            sample_values(dist_frames_gen[..., c], params.max_distribution_values, rng),
        )
    if dist_frames_dns.shape[-1] >= 3:
        dist_pdf_data["aux"] = (
            sample_values(dist_frames_dns[..., 2], params.max_distribution_values, rng),
            sample_values(dist_frames_gen[..., 2], params.max_distribution_values, rng),
        )
    pdf_metrics_all = {name: pdf_metrics(dns_vals, gen_vals, params.tail_low_q, params.tail_high_q, params.max_mmd_values) for name, (dns_vals, gen_vals) in dist_pdf_data.items()}

    spec_gen_global_frames, spec_dns_global_frames = flatten_frames_for_source(x_gen_eval, x_dns_eval, params.max_spectrum_frames, backend.field_ndim)
    spec_gen_early_frames, spec_dns_early_frames = flatten_frames_for_source(
        x_gen_eval[:, seg_gen.early], x_dns_eval[:, seg_dns.early], params.max_spectrum_frames, backend.field_ndim
    )
    spec_gen_late_frames, spec_dns_late_frames = flatten_frames_for_source(
        x_gen_eval[:, seg_gen.late], x_dns_eval[:, seg_dns.late], params.max_spectrum_frames, backend.field_ndim
    )
    spec_dns_global = backend.spectral_energy_curve(spec_dns_global_frames, cache, batch_size=params.spectrum_batch_size)
    spec_gen_global = backend.spectral_energy_curve(spec_gen_global_frames, cache, batch_size=params.spectrum_batch_size)
    spec_dns_early = backend.spectral_energy_curve(spec_dns_early_frames, cache, batch_size=params.spectrum_batch_size)
    spec_gen_early = backend.spectral_energy_curve(spec_gen_early_frames, cache, batch_size=params.spectrum_batch_size)
    spec_dns_late = backend.spectral_energy_curve(spec_dns_late_frames, cache, batch_size=params.spectrum_batch_size)
    spec_gen_late = backend.spectral_energy_curve(spec_gen_late_frames, cache, batch_size=params.spectrum_batch_size)
    slope_dns = fit_spectrum_slope(spec_dns_global["curve_mean"], spec_dns_global["edges"])
    slope_gen = fit_spectrum_slope(spec_gen_global["curve_mean"], spec_gen_global["edges"])
    spectrum_metrics = {
        "Err_logL2": safe_log_l2(spec_gen_global["curve_mean"], spec_dns_global["curve_mean"]),
        "Err_relL1": safe_rel_l1(spec_gen_global["curve_mean"], spec_dns_global["curve_mean"]),
        "slope_dns": slope_dns,
        "slope_gen": slope_gen,
        "slope_diff": float(abs(slope_gen - slope_dns)),
        "r_hi_dns": high_freq_ratio(spec_dns_global["curve_mean"], spec_dns_global["edges"]),
        "r_hi_gen": high_freq_ratio(spec_gen_global["curve_mean"], spec_gen_global["edges"]),
        "r_hi_delta": float(high_freq_ratio(spec_gen_global["curve_mean"], spec_gen_global["edges"]) - high_freq_ratio(spec_dns_global["curve_mean"], spec_dns_global["edges"])),
        "err_early": safe_log_l2(spec_gen_early["curve_mean"], spec_dns_early["curve_mean"]),
        "err_late": safe_log_l2(spec_gen_late["curve_mean"], spec_dns_late["curve_mean"]),
        "drift_err": float(safe_log_l2(spec_gen_late["curve_mean"], spec_dns_late["curve_mean"]) - safe_log_l2(spec_gen_early["curve_mean"], spec_dns_early["curve_mean"])),
        "bin_coverage": float(np.mean(spec_dns_global["counts"] > 0)),
        "bin_min_count": int(np.min(spec_dns_global["counts"])),
        "bin_median_count": float(np.median(spec_dns_global["counts"])),
        "edges": spec_dns_global["edges"],
        "dns_curve": spec_dns_global["curve_mean"],
        "gen_curve": spec_gen_global["curve_mean"],
    }

    struct_gen_global_frames, struct_dns_global_frames = flatten_frames_for_source(x_gen_eval, x_dns_eval, params.max_structure_frames, backend.field_ndim)
    struct_gen_early_frames, struct_dns_early_frames = flatten_frames_for_source(
        x_gen_eval[:, seg_gen.early], x_dns_eval[:, seg_dns.early], params.max_structure_frames, backend.field_ndim
    )
    struct_gen_late_frames, struct_dns_late_frames = flatten_frames_for_source(
        x_gen_eval[:, seg_gen.late], x_dns_eval[:, seg_dns.late], params.max_structure_frames, backend.field_ndim
    )
    curves_gen = backend.structure_curves(struct_gen_global_frames, cache, powers=(2, 3, 4, 6), batch_size=params.derivative_batch_size)
    curves_dns = backend.structure_curves(struct_dns_global_frames, cache, powers=(2, 3, 4, 6), batch_size=params.derivative_batch_size)
    curves_gen_early = backend.structure_curves(struct_gen_early_frames, cache, powers=(6,), batch_size=params.derivative_batch_size)
    curves_dns_early = backend.structure_curves(struct_dns_early_frames, cache, powers=(6,), batch_size=params.derivative_batch_size)
    curves_gen_late = backend.structure_curves(struct_gen_late_frames, cache, powers=(6,), batch_size=params.derivative_batch_size)
    curves_dns_late = backend.structure_curves(struct_dns_late_frames, cache, powers=(6,), batch_size=params.derivative_batch_size)
    mean_gen = mean_curve(curves_gen)
    mean_dns = mean_curve(curves_dns)
    f_gen = mean_gen[4] / np.clip(mean_gen[2] ** 2, EPS, None)
    f_dns = mean_dns[4] / np.clip(mean_dns[2] ** 2, EPS, None)
    h_gen = mean_gen[6] / np.clip(mean_gen[2] ** 3, EPS, None)
    h_dns = mean_dns[6] / np.clip(mean_dns[2] ** 3, EPS, None)
    r_centers = 0.5 * (cache.pair_bin_edges[:-1] + cache.pair_bin_edges[1:])
    structure_metrics = {
        "S2_err_logL2": safe_log_l2(mean_gen[2], mean_dns[2]),
        "S4_err_logL2": safe_log_l2(mean_gen[4], mean_dns[4]),
        "S6_err": safe_log_l2(mean_gen[6], mean_dns[6]),
        "S6_err_relL1": safe_rel_l1(mean_gen[6], mean_dns[6]),
        "F_err": safe_rel_l1(f_gen, f_dns),
        "H_err": safe_rel_l1(h_gen, h_dns),
        "beta_4_dns": ess_beta(mean_dns, 4),
        "beta_4_gen": ess_beta(mean_gen, 4),
        "beta_6_dns": ess_beta(mean_dns, 6),
        "beta_6_gen": ess_beta(mean_gen, 6),
        "beta_4_gap": float(abs(ess_beta(mean_gen, 4) - ess_beta(mean_dns, 4))),
        "beta_6_gap": float(abs(ess_beta(mean_gen, 6) - ess_beta(mean_dns, 6))),
        "S6_drift": float(safe_log_l2(np.mean(curves_gen_late[6], axis=0), np.mean(curves_dns_late[6], axis=0)) - safe_log_l2(np.mean(curves_gen_early[6], axis=0), np.mean(curves_dns_early[6], axis=0))),
        "r_centers": r_centers,
        "F_dns": f_dns,
        "F_gen": f_gen,
        "H_dns": h_dns,
        "H_gen": h_gen,
    }

    spec_feat_dns = backend.spectral_energy_curve(flat_dns, cache, batch_size=params.spectrum_batch_size)["curves"].reshape(x_dns_eval.shape[0], x_dns_eval.shape[1], -1)
    spec_feat_gen = backend.spectral_energy_curve(flat_gen, cache, batch_size=params.spectrum_batch_size)["curves"].reshape(x_gen_eval.shape[0], x_gen_eval.shape[1], -1)
    phi_dns = np.stack([
        backend.time_features(energy_dns[s], spec_feat_dns[s], {k: v[s] for k, v in deriv_dns_frame.items()}, x_dns_eval[s], params.feature_bins)
        for s in range(x_dns_eval.shape[0])
    ], axis=0)
    phi_gen = np.stack([
        backend.time_features(energy_gen[s], spec_feat_gen[s], {k: v[s] for k, v in deriv_gen_frame.items()}, x_gen_eval[s], params.feature_bins)
        for s in range(x_gen_eval.shape[0])
    ], axis=0)
    time_metrics, time_plot_payload, time_core_atoms = compute_time_metrics_detailed(phi_dns, phi_gen, energy_dns, energy_gen, seg_dns, seg_gen, rng, params.delta_max, params.delta_pairs, (1, 2, 4, 8, 16, 32, 64))

    dns_frame_features = sample_feature_rows(phi_dns.reshape(-1, phi_dns.shape[-1]), params.max_feature_frames, rng)
    gen_frame_features = sample_feature_rows(phi_gen.reshape(-1, phi_gen.shape[-1]), params.max_feature_frames, rng)
    pr = local_radius_precision_recall(dns_frame_features, gen_frame_features, k=params.precision_recall_k)
    coverage = cluster_coverage_metrics(dns_frame_features, gen_frame_features, n_clusters=params.cluster_count, rng=rng)
    cov_geo = covariance_geometry(dns_frame_features, gen_frame_features)
    diversity_metrics = {
        "feature_dim": int(dns_frame_features.shape[1]),
        "n_dns_features": int(dns_frame_features.shape[0]),
        "n_gen_features": int(gen_frame_features.shape[0]),
        "precision": pr["precision"],
        "recall": pr["recall"],
        "cluster_hist_l1": coverage["cluster_hist_l1"],
        "uncovered": coverage["uncovered"],
        "trace_dns": cov_geo["trace_dns"],
        "trace_gen": cov_geo["trace_gen"],
        "trace_gap": cov_geo["trace_gap"],
        "logdet_dns": cov_geo["logdet_dns"],
        "logdet_gen": cov_geo["logdet_gen"],
        "logdet_gap": cov_geo["logdet_gap"],
        "cov_fro_gap": cov_geo["cov_fro_gap"],
        "w1_time": time_metrics["w1_time"],
        "w1_cross": time_metrics["w1_cross"],
        "w1_time_drift": time_metrics["w1_time_drift"],
        "dns_mean": time_metrics["dns_mean"],
        "gen_time_mean": time_metrics["gen_time_mean"],
        "gen_cross_mean": time_metrics["gen_cross_mean"],
    }

    spec_per_sample = []
    for s in range(x_gen_eval.shape[0]):
        sample_curve = backend.spectral_energy_curve(x_gen_eval[s], cache, batch_size=params.spectrum_batch_size)["curve_mean"]
        spec_per_sample.append(safe_log_l2(sample_curve, spec_dns_global["curve_mean"]))
    s6_drift_sample = []
    for s in range(x_gen_eval.shape[0]):
        sample_early = backend.structure_curves(
            flatten_frames_for_source(
                x_gen_eval[s : s + 1, seg_gen.early], x_dns_eval[:, seg_dns.early], params.max_structure_frames, backend.field_ndim
            )[0],
            cache,
            powers=(6,),
            batch_size=params.derivative_batch_size,
        )[6]
        sample_late = backend.structure_curves(
            flatten_frames_for_source(
                x_gen_eval[s : s + 1, seg_gen.late], x_dns_eval[:, seg_dns.late], params.max_structure_frames, backend.field_ndim
            )[0],
            cache,
            powers=(6,),
            batch_size=params.derivative_batch_size,
        )[6]
        s6_drift_sample.append(float(safe_log_l2(np.mean(sample_late, axis=0), np.mean(curves_dns_late[6], axis=0)) - safe_log_l2(np.mean(sample_early, axis=0), np.mean(curves_dns_early[6], axis=0))))
    enstrophy_ref = float(np.mean(deriv_dns["enstrophy_frame"]))
    enstrophy_gap_sample = []
    for s in range(x_gen_eval.shape[0]):
        enstrophy_gap_sample.append(relative_gap(np.mean(deriv_gen_frame["enstrophy_frame"][s]), enstrophy_ref))

    core_atoms = {
        "late_tail_Tint_gap": time_core_atoms["late_tail_Tint_gap"],
        "tail_w1_time": time_core_atoms["tail_w1_time"],
        "w1_time_lag_1": time_core_atoms["w1_time_lag_1"],
        "w1_time_lag_16": time_core_atoms["w1_time_lag_16"],
        "S6_drift": np.asarray(s6_drift_sample, dtype=np.float64),
        "enstrophy_gap": np.asarray(enstrophy_gap_sample, dtype=np.float64),
        "Err_logL2": np.asarray(spec_per_sample, dtype=np.float64),
    }

    return {
        "energy": energy_metrics,
        "derivative": derivative_metrics,
        "distribution": pdf_metrics_all,
        "spectrum": spectrum_metrics,
        "structure": structure_metrics,
        "time": time_metrics,
        "diversity": diversity_metrics,
        "core_atoms": core_atoms,
        "time_plot_payload": time_plot_payload,
    }


def build_core_row(summary: Dict[str, object]) -> Dict[str, float]:
    return {
        "late_tail_Tint_gap": float(summary["energy"]["late_tail_Tint_gap"]),
        "tail_w1_time": float(summary["time"]["tail_w1_time"]),
        "w1_time_lag_1": float(summary["time"].get("w1_time_lag_1", float("nan"))),
        "w1_time_lag_16": float(summary["time"].get("w1_time_lag_16", float("nan"))),
        "S6_drift": float(summary["structure"]["S6_drift"]),
        "enstrophy_gap": float(summary["derivative"]["enstrophy_gap"]),
        "Err_logL2": float(summary["spectrum"]["Err_logL2"]),
    }
