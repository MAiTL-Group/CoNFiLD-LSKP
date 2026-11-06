from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.stats import wasserstein_distance

EPS = 1.0e-8
TIME_LAG_VALUES = (1, 2, 4, 8, 16, 32, 64)


@dataclass
class SplitSlices:
    early: slice
    middle: slice
    late: slice


@dataclass
class LoadedGenData:
    x_gen: np.ndarray
    z_gen: np.ndarray
    mode: str
    n_runs: int
    pooled_sample_count: int
    run_metas: List[dict]


@dataclass
class PairedPools:
    x_gen_pair: np.ndarray
    x_dns_pair: np.ndarray
    t_ref: int
    stride: int
    policy: str
    n_gen_windows: int
    n_dns_windows: int


@dataclass
class EvalParams:
    early_frac: float
    late_frac: float
    acf_max_lag: int
    delta_max: int
    delta_pairs: int
    feature_bins: int
    max_feature_frames: int
    max_distribution_values: int
    max_mmd_values: int
    max_spectrum_frames: int
    max_structure_frames: int
    precision_recall_k: int
    cluster_count: int
    tail_low_q: float
    tail_high_q: float
    derivative_batch_size: int = 64
    spectrum_batch_size: int = 8


@dataclass
class CaseDefaults:
    dns_fields: str
    coords: Optional[str] = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_builtin(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def relative_gap(gen_val: float, dns_val: float) -> float:
    return float(abs(gen_val - dns_val) / (abs(dns_val) + EPS))


def split_slices(length: int, early_frac: float, late_frac: float) -> SplitSlices:
    early_n = max(1, int(round(length * early_frac)))
    late_n = max(1, int(round(length * late_frac)))
    if early_n + late_n >= length:
        early_n = max(1, length // 4)
        late_n = max(1, length // 4)
    return SplitSlices(early=slice(0, early_n), middle=slice(early_n, length - late_n), late=slice(length - late_n, length))


def resolve_window_stride(t_ref: int, requested: int) -> int:
    if requested and requested > 0:
        return int(requested)
    return max(1, t_ref // 4)


def ensure_sample_axis(x: np.ndarray, field_ndim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim < field_ndim + 1:
        raise ValueError(f"Expected at least [T,...], got {x.shape}")
    if x.ndim == field_ndim + 1:
        return x[None, ...]
    if x.ndim == field_ndim + 2:
        return x
    raise ValueError(f"Unsupported array rank for field_ndim={field_ndim}: {x.shape}")


def build_window_pool(x: np.ndarray, win_len: int, stride: int, field_ndim: int) -> np.ndarray:
    x = ensure_sample_axis(x, field_ndim)
    _, t = x.shape[:2]
    if win_len > t:
        raise ValueError(f"win_len={win_len} exceeds sequence length {t}")
    if win_len == t:
        return x.copy()
    starts = list(range(0, t - win_len + 1, stride))
    last = t - win_len
    if not starts:
        starts = [0]
    elif starts[-1] != last:
        starts.append(last)
    windows = [x[:, st : st + win_len] for st in starts]
    return np.concatenate(windows, axis=0)


def build_sample_windows(x: np.ndarray, win_len: int, stride: int, field_ndim: int) -> Tuple[np.ndarray, List[int]]:
    x = ensure_sample_axis(x, field_ndim)
    if win_len > x.shape[1]:
        raise ValueError(f"win_len={win_len} exceeds sequence length {x.shape[1]}")
    if win_len == x.shape[1]:
        return x.copy(), [0]
    starts = list(range(0, x.shape[1] - win_len + 1, stride))
    last = x.shape[1] - win_len
    if not starts:
        starts = [0]
    elif starts[-1] != last:
        starts.append(last)
    windows = [x[:, st : st + win_len] for st in starts]
    return np.stack(windows, axis=1), starts


def flatten_segments(values: np.ndarray, seg: slice) -> np.ndarray:
    return np.asarray(values[..., seg], dtype=np.float64).reshape(-1)


def maybe_subsample_time_axis(x: np.ndarray, max_frames: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim < 2:
        raise ValueError(f"Unsupported shape for time subsample: {x.shape}")
    if x.ndim == 3:
        t = x.shape[0]
        if t <= max_frames:
            return x
        idx = np.linspace(0, t - 1, max_frames, dtype=int)
        return x[idx]
    s, t = x.shape[:2]
    if t <= max_frames:
        return x
    idx = np.linspace(0, t - 1, max_frames, dtype=int)
    return x[:, idx]


def maybe_subsample_total_frames(x: np.ndarray, max_frames: int, field_ndim: int) -> np.ndarray:
    x = ensure_sample_axis(x, field_ndim)
    flat = x.reshape(-1, *x.shape[2:])
    total = flat.shape[0]
    if total <= max_frames:
        return flat
    idx = np.linspace(0, total - 1, max_frames, dtype=int)
    return flat[idx]


def flatten_frames_for_source(x_gen: np.ndarray, x_dns: np.ndarray, max_frames: int, field_ndim: int) -> Tuple[np.ndarray, np.ndarray]:
    return maybe_subsample_total_frames(x_gen, max_frames, field_ndim), maybe_subsample_total_frames(x_dns, max_frames, field_ndim)


def _window_score_energy_w1(window: np.ndarray, dns_energy_flat: np.ndarray, energy_series_fn) -> float:
    return float(wasserstein_distance(energy_series_fn(window[None, ...]).reshape(-1), dns_energy_flat))


def _window_score_drift(window: np.ndarray, seg: SplitSlices, energy_series_fn) -> float:
    energy = energy_series_fn(window[None, ...])[0]
    early = np.mean(energy[seg.early])
    late = np.mean(energy[seg.late])
    return relative_gap(late, early)


def _select_windows_by_policy(
    x_gen: np.ndarray,
    x_dns: np.ndarray,
    t_ref: int,
    stride: int,
    policy: str,
    worst_window_score: str,
    energy_series_fn,
    early_frac: float,
    late_frac: float,
    field_ndim: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    if policy == "window_pool":
        x_gen_pair = build_window_pool(x_gen, t_ref, stride, field_ndim)
        x_dns_pair = build_window_pool(x_dns, t_ref, stride, field_ndim)
        return x_gen_pair, x_dns_pair, int(x_gen_pair.shape[0]), int(x_dns_pair.shape[0])

    if policy == "head_match":
        x_gen_pair = ensure_sample_axis(x_gen, field_ndim)[:, :t_ref].copy()
        x_dns_pair = build_window_pool(x_dns, t_ref, stride, field_ndim)
        return x_gen_pair, x_dns_pair, int(x_gen_pair.shape[0]), int(x_dns_pair.shape[0])

    if policy == "tail_match":
        x_gen_pair = ensure_sample_axis(x_gen, field_ndim)[:, -t_ref:].copy()
        x_dns_pair = build_window_pool(x_dns, t_ref, stride, field_ndim)
        return x_gen_pair, x_dns_pair, int(x_gen_pair.shape[0]), int(x_dns_pair.shape[0])

    if policy not in {"worst_window", "best_window"}:
        raise ValueError(f"Unsupported paired policy: {policy}")

    dns_energy_flat = energy_series_fn(build_window_pool(x_dns, t_ref, stride, field_ndim)).reshape(-1)
    gen_windows, _ = build_sample_windows(x_gen, t_ref, stride, field_ndim)
    seg = split_slices(t_ref, early_frac, late_frac)
    selected = []
    for s in range(gen_windows.shape[0]):
        sample_windows = gen_windows[s]
        scores = []
        for w in range(sample_windows.shape[0]):
            if worst_window_score == "energy_w1":
                score = _window_score_energy_w1(sample_windows[w], dns_energy_flat, energy_series_fn)
            elif worst_window_score == "drift_rel":
                score = _window_score_drift(sample_windows[w], seg, energy_series_fn)
            else:
                raise ValueError(f"Unsupported worst-window score: {worst_window_score}")
            scores.append(score)
        idx = int(np.argmax(scores) if policy == "worst_window" else np.argmin(scores))
        selected.append(sample_windows[idx])
    x_gen_pair = np.stack(selected, axis=0)
    x_dns_pair = build_window_pool(x_dns, t_ref, stride, field_ndim)
    return x_gen_pair, x_dns_pair, int(x_gen_pair.shape[0]), int(x_dns_pair.shape[0])


def build_paired_pools(
    x_gen: np.ndarray,
    x_dns: np.ndarray,
    stride: int,
    policy: str,
    worst_window_score: str,
    energy_series_fn,
    early_frac: float,
    late_frac: float,
    field_ndim: int,
    pair_length: Optional[int] = None,
) -> PairedPools:
    x_gen = ensure_sample_axis(x_gen, field_ndim)
    x_dns = np.asarray(x_dns, dtype=np.float32)
    t_gen = x_gen.shape[1]
    t_dns = x_dns.shape[0]
    t_ref = min(t_gen, t_dns)
    if pair_length is not None and int(pair_length) > 0:
        t_ref = min(t_ref, int(pair_length))
    stride = resolve_window_stride(t_ref, stride)
    if t_gen <= t_dns and (pair_length is None or t_ref == t_gen):
        x_gen_pair = x_gen[:, :t_ref].copy()
        x_dns_pair = build_window_pool(x_dns, t_ref, stride, field_ndim)
        n_gen_windows = int(x_gen_pair.shape[0])
        n_dns_windows = int(x_dns_pair.shape[0])
        out_policy = "dns_window_pool"
    else:
        x_dns_source = x_dns if (pair_length is None or x_dns.shape[0] <= t_ref) else x_dns[:t_ref]
        x_gen_pair, x_dns_pair, n_gen_windows, n_dns_windows = _select_windows_by_policy(
            x_gen=x_gen,
            x_dns=x_dns_source,
            t_ref=t_ref,
            stride=stride,
            policy=policy,
            worst_window_score=worst_window_score,
            energy_series_fn=energy_series_fn,
            early_frac=early_frac,
            late_frac=late_frac,
            field_ndim=field_ndim,
        )
        out_policy = policy
    return PairedPools(
        x_gen_pair=x_gen_pair,
        x_dns_pair=x_dns_pair,
        t_ref=t_ref,
        stride=stride,
        policy=out_policy,
        n_gen_windows=n_gen_windows,
        n_dns_windows=n_dns_windows,
    )
