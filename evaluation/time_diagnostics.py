from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.stats import wasserstein_distance


EPS = 1.0e-8


def acf_1d(y: np.ndarray, max_lag: int) -> np.ndarray:
    """Return the normalized 1D autocorrelation up to ``max_lag``."""
    y = np.asarray(y, dtype=np.float64)
    y = y - np.mean(y)
    denom = np.dot(y, y) + EPS
    out = np.empty(max_lag + 1, dtype=np.float64)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        if lag >= len(y):
            out[lag] = np.nan
        else:
            out[lag] = np.dot(y[:-lag], y[lag:]) / denom
    return out


def tint_from_acf(acf: np.ndarray, mode: str = "first_zero") -> float:
    """Integrate an autocorrelation curve to an effective time scale."""
    valid = np.asarray(acf, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float("nan")
    if mode == "first_zero":
        idx = np.where(valid <= 0)[0]
        end = idx[0] if idx.size else valid.size
        return float(np.sum(valid[:end]))
    return float(np.sum(valid))


def relative_gap(gen_val: float, dns_val: float) -> float:
    """Relative absolute gap using DNS as the reference scale."""
    return float(abs(gen_val - dns_val) / (abs(dns_val) + EPS))


def sample_time_pair_distances(
    frame_features: np.ndarray,
    rng: np.random.Generator,
    delta_max: int,
    n_pairs: int,
) -> np.ndarray:
    """Sample within-sequence temporal pair distances over mixed lags."""
    t = frame_features.shape[0]
    if t < 2:
        return np.zeros(1, dtype=np.float64)
    max_delta = min(int(delta_max), t - 1)
    deltas = rng.integers(1, max_delta + 1, size=n_pairs)
    start = rng.integers(0, t - deltas, size=n_pairs)
    end = start + deltas
    diff = frame_features[end] - frame_features[start]
    return np.linalg.norm(diff, axis=1)


def sample_time_pair_distances_fixed_lag(
    frame_features: np.ndarray,
    rng: np.random.Generator,
    lag: int,
    n_pairs: int,
) -> np.ndarray:
    """Sample within-sequence temporal pair distances at a fixed lag."""
    t = frame_features.shape[0]
    lag = int(lag)
    if t <= lag or lag < 1:
        return np.zeros(1, dtype=np.float64)
    start = rng.integers(0, t - lag, size=n_pairs)
    end = start + lag
    diff = frame_features[end] - frame_features[start]
    return np.linalg.norm(diff, axis=1)


def sample_cross_distances(
    frame_features_st: np.ndarray,
    rng: np.random.Generator,
    n_pairs: int,
) -> np.ndarray:
    """Sample cross-trajectory distances at matched times."""
    s, t, _ = frame_features_st.shape
    if s < 2:
        return np.zeros(1, dtype=np.float64)
    sample_a = rng.integers(0, s, size=n_pairs)
    sample_b = rng.integers(0, s - 1, size=n_pairs)
    sample_b = np.where(sample_b >= sample_a, sample_b + 1, sample_b)
    ti = rng.integers(0, t, size=n_pairs)
    diff = frame_features_st[sample_a, ti] - frame_features_st[sample_b, ti]
    return np.linalg.norm(diff, axis=1)


def compute_energy_tint_metrics(
    energy_dns: np.ndarray,
    energy_gen: np.ndarray,
    seg_dns_late: slice,
    seg_gen_late: slice,
    acf_max_lag: int,
) -> Dict[str, object]:
    """Compute global and late-tail integral time scales from energy traces."""
    acf_dns = np.mean(np.stack([acf_1d(energy_dns[s], acf_max_lag) for s in range(energy_dns.shape[0])], axis=0), axis=0)
    acf_gen = np.mean(np.stack([acf_1d(energy_gen[s], acf_max_lag) for s in range(energy_gen.shape[0])], axis=0), axis=0)
    late_dns = energy_dns[:, seg_dns_late]
    late_gen = energy_gen[:, seg_gen_late]
    late_max_lag = max(1, min(acf_max_lag, late_dns.shape[1] - 1, late_gen.shape[1] - 1))
    acf_dns_late = np.mean(np.stack([acf_1d(late_dns[s], late_max_lag) for s in range(late_dns.shape[0])], axis=0), axis=0)
    acf_gen_late = np.mean(np.stack([acf_1d(late_gen[s], late_max_lag) for s in range(late_gen.shape[0])], axis=0), axis=0)
    tint_dns = tint_from_acf(acf_dns)
    tint_gen = tint_from_acf(acf_gen)
    late_tint_dns = tint_from_acf(acf_dns_late)
    late_tint_gen = tint_from_acf(acf_gen_late)
    return {
        "Tint_dns": float(tint_dns),
        "Tint_gen": float(tint_gen),
        "Tint_gap": relative_gap(tint_gen, tint_dns),
        "late_tail_Tint_dns": float(late_tint_dns),
        "late_tail_Tint_gen": float(late_tint_gen),
        "late_tail_Tint_gap": relative_gap(late_tint_gen, late_tint_dns),
        "acf_dns": acf_dns,
        "acf_gen": acf_gen,
        "late_tail_acf_dns": acf_dns_late,
        "late_tail_acf_gen": acf_gen_late,
    }


def _concat_time_distances(
    frame_features_st: np.ndarray,
    rng: np.random.Generator,
    sampler,
    n_pairs: int,
    *sampler_args,
) -> np.ndarray:
    s = frame_features_st.shape[0]
    per_seq = max(1, int(n_pairs) // max(1, s))
    return np.concatenate([sampler(frame_features_st[idx], rng, *sampler_args, per_seq) for idx in range(s)])


def compute_time_distance_metrics(
    phi_dns: np.ndarray,
    phi_gen: np.ndarray,
    seg_dns_early: slice,
    seg_dns_late: slice,
    seg_gen_early: slice,
    seg_gen_late: slice,
    rng: np.random.Generator,
    delta_max: int,
    delta_pairs: int,
    lag_values: Iterable[int],
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Compute paired temporal transport diagnostics on full and tail windows."""
    dns_vals = _concat_time_distances(phi_dns, rng, sample_time_pair_distances, delta_pairs, delta_max)
    gen_time_vals = _concat_time_distances(phi_gen, rng, sample_time_pair_distances, delta_pairs, delta_max)
    gen_cross_vals = sample_cross_distances(phi_gen, rng, delta_pairs)

    phi_dns_early = phi_dns[:, seg_dns_early]
    phi_dns_late = phi_dns[:, seg_dns_late]
    phi_gen_early = phi_gen[:, seg_gen_early]
    phi_gen_late = phi_gen[:, seg_gen_late]

    dns_early_vals = _concat_time_distances(phi_dns_early, rng, sample_time_pair_distances, delta_pairs, delta_max)
    gen_early_vals = _concat_time_distances(phi_gen_early, rng, sample_time_pair_distances, delta_pairs, delta_max)
    dns_late_vals = _concat_time_distances(phi_dns_late, rng, sample_time_pair_distances, delta_pairs, delta_max)
    gen_late_vals = _concat_time_distances(phi_gen_late, rng, sample_time_pair_distances, delta_pairs, delta_max)
    gen_tail_cross_vals = sample_cross_distances(phi_gen_late, rng, delta_pairs)

    max_valid_lag = max(0, min(phi_dns.shape[1] - 1, phi_gen.shape[1] - 1, int(delta_max)))
    valid_lags = [int(lag) for lag in lag_values if 1 <= int(lag) <= max_valid_lag]
    lag_w1 = {}
    dns_lag_mean = {}
    gen_lag_mean = {}
    for lag in valid_lags:
        dns_lag = _concat_time_distances(phi_dns, rng, sample_time_pair_distances_fixed_lag, delta_pairs, lag)
        gen_lag = _concat_time_distances(phi_gen, rng, sample_time_pair_distances_fixed_lag, delta_pairs, lag)
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
    return metrics, plot_payload


def compute_temporal_core_atoms(
    phi_dns: np.ndarray,
    phi_gen: np.ndarray,
    energy_dns: np.ndarray,
    energy_gen: np.ndarray,
    seg_dns_late: slice,
    seg_gen_late: slice,
    rng: np.random.Generator,
    delta_max: int,
    delta_pairs: int,
    acf_max_lag: int,
    lag_values: Iterable[int] = (1, 16),
) -> Dict[str, np.ndarray]:
    """Build per-rollout atoms for bootstrap CIs of temporal diagnostics."""
    tint_agg = compute_energy_tint_metrics(energy_dns, energy_gen, seg_dns_late, seg_gen_late, acf_max_lag)
    dns_tint_ref = float(tint_agg["Tint_dns"])
    dns_tint_late_ref = float(tint_agg["late_tail_Tint_dns"])

    phi_dns_late = phi_dns[:, seg_dns_late]
    phi_gen_late = phi_gen[:, seg_gen_late]
    dns_late_vals = _concat_time_distances(phi_dns_late, rng, sample_time_pair_distances, delta_pairs, delta_max)

    max_valid_lag = max(0, min(phi_dns.shape[1] - 1, phi_gen.shape[1] - 1, int(delta_max)))
    valid_lags = [int(lag) for lag in lag_values if 1 <= int(lag) <= max_valid_lag]
    dns_lag_dist = {
        lag: _concat_time_distances(phi_dns, rng, sample_time_pair_distances_fixed_lag, delta_pairs, lag)
        for lag in valid_lags
    }

    per_seq_pairs = max(1, int(delta_pairs) // max(1, phi_gen.shape[0]))
    atoms = {
        "Tint_gap": [],
        "late_tail_Tint_gap": [],
        "tail_w1_time": [],
    }
    for lag in lag_values:
        atoms[f"w1_time_lag_{int(lag)}"] = []

    for s in range(phi_gen.shape[0]):
        gen_tint = tint_from_acf(acf_1d(energy_gen[s], min(acf_max_lag, energy_gen.shape[1] - 1)))
        atoms["Tint_gap"].append(relative_gap(gen_tint, dns_tint_ref))

        late_max_lag = max(1, min(acf_max_lag, energy_gen[:, seg_gen_late].shape[1] - 1))
        gen_tint_late = tint_from_acf(acf_1d(energy_gen[s, seg_gen_late], late_max_lag))
        atoms["late_tail_Tint_gap"].append(relative_gap(gen_tint_late, dns_tint_late_ref))

        gen_late_dist = sample_time_pair_distances(phi_gen_late[s], rng, delta_max, per_seq_pairs)
        atoms["tail_w1_time"].append(float(wasserstein_distance(dns_late_vals, gen_late_dist)))

        for lag in lag_values:
            lag = int(lag)
            key = f"w1_time_lag_{lag}"
            if lag in dns_lag_dist:
                gen_lag_dist = sample_time_pair_distances_fixed_lag(phi_gen[s], rng, lag, per_seq_pairs)
                atoms[key].append(float(wasserstein_distance(dns_lag_dist[lag], gen_lag_dist)))
            else:
                atoms[key].append(float("nan"))

    return {key: np.asarray(value, dtype=np.float64) for key, value in atoms.items()}


def bootstrap_core_atoms(core_atoms: Dict[str, np.ndarray], n_bootstrap: int = 200, seed: int = 42) -> Dict[str, Dict[str, float]]:
    """Bootstrap mean statistics for a dictionary of per-rollout metric atoms."""
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, float]] = {}
    for key, arr in core_atoms.items():
        x = np.asarray(arr, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            out[key] = {"mean": float("nan"), "median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
            continue
        stats = np.empty(int(n_bootstrap), dtype=np.float64)
        for b in range(int(n_bootstrap)):
            idx = rng.integers(0, x.size, size=x.size)
            stats[b] = float(np.mean(x[idx]))
        out[key] = {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "ci_low": float(np.quantile(stats, 0.025)),
            "ci_high": float(np.quantile(stats, 0.975)),
            "n": int(x.size),
        }
    return out
