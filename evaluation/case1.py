from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


import numpy as np
from scipy.stats import entropy, kurtosis, ks_2samp, skew, wasserstein_distance

from evaluation.case1_ops import (
    Case1OpsCache,
    build_case1_ops_cache,
    load_case1_ops_cache,
    save_case1_ops_cache,
)
from evaluation.diversity_metrics import (
    cluster_coverage_metrics,
    covariance_geometry,
    local_radius_precision_recall,
    rbf_mmd,
    sample_feature_rows,
)
from evaluation.time_diagnostics import (
    bootstrap_core_atoms,
    compute_energy_tint_metrics,
    compute_temporal_core_atoms,
    compute_time_distance_metrics,
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case1 motivation evaluation for generated trajectories.")
    parser.add_argument("--gen-fields", type=str, default=None)
    parser.add_argument("--gen-latents", type=str, default=None)
    parser.add_argument("--gen-manifest", type=str, default=None)
    parser.add_argument(
        "--dns-fields",
        type=str,
        default="data/latents/case1/sn/test_raw_segment.npy",
    )
    parser.add_argument(
        "--coords",
        type=str,
        default="data/raw/case1/coords.npy",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/evaluation/case1_motivation/origin_1280",
    )
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--time-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-frac", type=float, default=0.25)
    parser.add_argument("--late-frac", type=float, default=0.25)
    parser.add_argument("--k-neighbors", type=int, default=12)
    parser.add_argument("--pair-count", type=int, default=20000)
    parser.add_argument("--pair-bins", type=int, default=20)
    parser.add_argument("--spectrum-bins", type=int, default=24)
    parser.add_argument("--max-spectrum-frames", type=int, default=256)
    parser.add_argument("--max-structure-frames", type=int, default=128)
    parser.add_argument("--max-distribution-values", type=int, default=300000)
    parser.add_argument("--max-mmd-values", type=int, default=4096)
    parser.add_argument("--delta-max", type=int, default=64)
    parser.add_argument("--delta-pairs", type=int, default=4096)
    parser.add_argument("--acf-max-lag", type=int, default=128)
    parser.add_argument("--tail-low-q", type=float, default=0.01)
    parser.add_argument("--tail-high-q", type=float, default=0.99)
    parser.add_argument("--protocol-a-pair-length", type=int, default=1280)
    parser.add_argument("--paired-window-stride", type=int, default=0)
    parser.add_argument(
        "--paired-policy",
        type=str,
        default="window_pool",
        choices=["window_pool", "head_match", "tail_match", "worst_window", "best_window"],
    )
    parser.add_argument(
        "--worst-window-score",
        type=str,
        default="energy_w1",
        choices=["energy_w1", "drift_rel"],
    )
    parser.add_argument("--feature-bins", type=int, default=16)
    parser.add_argument("--max-feature-frames", type=int, default=4096)
    parser.add_argument("--precision-recall-k", type=int, default=5)
    parser.add_argument("--cluster-count", type=int, default=10)
    parser.add_argument("--boundary-index", type=str, default=None)
    parser.add_argument("--boundary-normals", type=str, default=None)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_builtin(x):
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


def infer_sample_time_from_latents(gen_latents: np.ndarray, sample_count: Optional[int], time_length: Optional[int]) -> Tuple[int, int]:
    inferred_s, inferred_t = int(gen_latents.shape[0]), int(gen_latents.shape[1])
    if sample_count is not None and sample_count != inferred_s:
        raise ValueError(f"sample_count={sample_count} conflicts with gen_latents.shape[0]={inferred_s}")
    if time_length is not None and time_length != inferred_t:
        raise ValueError(f"time_length={time_length} conflicts with gen_latents.shape[1]={inferred_t}")
    return inferred_s, inferred_t


def load_single_run(
    gen_fields_path: str,
    gen_latents_path: str,
    sample_count: Optional[int],
    time_length: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    gen_fields = np.load(gen_fields_path, mmap_mode="r")
    gen_latents = np.load(gen_latents_path, mmap_mode="r")
    sample_count, time_length = infer_sample_time_from_latents(gen_latents, sample_count, time_length)
    expected = sample_count * time_length
    if gen_fields.shape[0] != expected:
        raise ValueError(f"gen_fields first dim {gen_fields.shape[0]} != sample_count*time_length {expected}")
    x_gen = np.asarray(gen_fields, dtype=np.float32).reshape(sample_count, time_length, gen_fields.shape[1], gen_fields.shape[2])
    z_gen = np.asarray(gen_latents, dtype=np.float32)
    return x_gen, z_gen, sample_count, time_length


def load_manifest_inputs(manifest_path: str) -> LoadedGenData:
    payload = json.loads(open(manifest_path, "r", encoding="utf-8").read())
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError("manifest.runs is empty")

    xs, zs, metas = [], [], []
    ref = None
    for idx, run in enumerate(runs):
        gen_fields = run["gen_fields"]
        gen_latents = run["gen_latents"]
        run_meta = {}
        meta_path = run.get("meta")
        if meta_path and os.path.exists(meta_path):
            run_meta = json.loads(open(meta_path, "r", encoding="utf-8").read())
        x_run, z_run, s_run, t_run = load_single_run(gen_fields, gen_latents, None, None)
        current = {
            "time_length": t_run,
            "nodes": x_run.shape[2],
            "channels": x_run.shape[3],
            "latent_dim": z_run.shape[2],
            "ema_path": run_meta.get("ema_path"),
            "cnf_case_file_path": run_meta.get("cnf_case_file_path"),
        }
        if ref is None:
            ref = current
        else:
            for key in ["time_length", "nodes", "channels", "latent_dim"]:
                if current[key] != ref[key]:
                    raise ValueError(f"manifest run {idx} mismatches on {key}: {current[key]} vs {ref[key]}")
            for key in ["ema_path", "cnf_case_file_path"]:
                if ref[key] is not None and current[key] is not None and current[key] != ref[key]:
                    raise ValueError(f"manifest run {idx} mismatches on {key}: {current[key]} vs {ref[key]}")
        xs.append(x_run)
        zs.append(z_run)
        metas.append({"manifest_entry": run, "run_meta": run_meta, "sample_count": s_run, "time_length": t_run})

    x_gen = np.concatenate(xs, axis=0)
    z_gen = np.concatenate(zs, axis=0)
    return LoadedGenData(
        x_gen=x_gen,
        z_gen=z_gen,
        mode="pool",
        n_runs=len(runs),
        pooled_sample_count=int(x_gen.shape[0]),
        run_metas=metas,
    )


def load_inputs(args: argparse.Namespace) -> Tuple[LoadedGenData, np.ndarray, np.ndarray]:
    if args.gen_manifest:
        loaded = load_manifest_inputs(args.gen_manifest)
    else:
        if not args.gen_fields or not args.gen_latents:
            raise ValueError("single-run mode requires both --gen-fields and --gen-latents")
        x_gen, z_gen, sample_count, time_length = load_single_run(
            args.gen_fields, args.gen_latents, args.sample_count, args.time_length
        )
        loaded = LoadedGenData(
            x_gen=x_gen,
            z_gen=z_gen,
            mode="single",
            n_runs=1,
            pooled_sample_count=int(sample_count),
            run_metas=[
                {
                    "manifest_entry": {
                        "gen_fields": args.gen_fields,
                        "gen_latents": args.gen_latents,
                    },
                    "run_meta": {},
                    "sample_count": sample_count,
                    "time_length": time_length,
                }
            ],
        )
    dns_fields = np.asarray(np.load(args.dns_fields, mmap_mode="r"), dtype=np.float32)
    coords = np.asarray(np.load(args.coords), dtype=np.float32)
    if loaded.x_gen.shape[2] != coords.shape[0] or dns_fields.shape[1] != coords.shape[0]:
        raise ValueError("coords node count is inconsistent with field arrays")
    return loaded, dns_fields, coords


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


def build_window_pool(x: np.ndarray, win_len: int, stride: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[None, ...]
    s, t = x.shape[:2]
    if win_len > t:
        raise ValueError(f"win_len={win_len} exceeds sequence length {t}")
    if win_len == t:
        return x.copy()
    starts = list(range(0, t - win_len + 1, stride))
    last = t - win_len
    if starts[-1] != last:
        starts.append(last)
    windows = [x[:, st : st + win_len] for st in starts]
    return np.concatenate(windows, axis=0)


def build_sample_windows(x: np.ndarray, win_len: int, stride: int) -> Tuple[np.ndarray, List[int]]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected [S,T,N,C], got {x.shape}")
    if win_len > x.shape[1]:
        raise ValueError(f"win_len={win_len} exceeds sequence length {x.shape[1]}")
    if win_len == x.shape[1]:
        return x.copy(), [0]
    starts = list(range(0, x.shape[1] - win_len + 1, stride))
    last = x.shape[1] - win_len
    if starts[-1] != last:
        starts.append(last)
    windows = [x[:, st : st + win_len] for st in starts]
    return np.stack(windows, axis=1), starts


def _window_score_energy_w1(window: np.ndarray, dns_energy_flat: np.ndarray) -> float:
    return float(wasserstein_distance(energy_series(window).reshape(-1), dns_energy_flat))


def _window_score_drift(window: np.ndarray, seg: SplitSlices) -> float:
    energy = energy_series(window[None, ...])[0]
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
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    if policy == "window_pool":
        x_gen_pair = build_window_pool(x_gen, t_ref, stride)
        x_dns_pair = build_window_pool(x_dns, t_ref, stride)
        return x_gen_pair, x_dns_pair, int(x_gen_pair.shape[0]), int(x_dns_pair.shape[0])

    if policy == "head_match":
        return x_gen[:, :t_ref].copy(), build_window_pool(x_dns, t_ref, stride), int(x_gen.shape[0]), int(build_window_pool(x_dns, t_ref, stride).shape[0])

    if policy == "tail_match":
        return x_gen[:, -t_ref:].copy(), build_window_pool(x_dns, t_ref, stride), int(x_gen.shape[0]), int(build_window_pool(x_dns, t_ref, stride).shape[0])

    if policy not in {"worst_window", "best_window"}:
        raise ValueError(f"Unsupported paired policy: {policy}")

    dns_energy_flat = energy_series(build_window_pool(x_dns, t_ref, stride)).reshape(-1)
    gen_windows, _ = build_sample_windows(x_gen, t_ref, stride)
    seg = split_slices(t_ref, 0.25, 0.25)
    selected = []
    for s in range(gen_windows.shape[0]):
        sample_windows = gen_windows[s]
        scores = []
        for w in range(sample_windows.shape[0]):
            if worst_window_score == "energy_w1":
                score = _window_score_energy_w1(sample_windows[w], dns_energy_flat)
            elif worst_window_score == "drift_rel":
                score = _window_score_drift(sample_windows[w], seg)
            else:
                raise ValueError(f"Unsupported worst-window score: {worst_window_score}")
            scores.append(score)
        idx = int(np.argmax(scores) if policy == "worst_window" else np.argmin(scores))
        selected.append(sample_windows[idx])
    x_gen_pair = np.stack(selected, axis=0)
    x_dns_pair = build_window_pool(x_dns, t_ref, stride)
    return x_gen_pair, x_dns_pair, int(x_gen_pair.shape[0]), int(x_dns_pair.shape[0])


def build_paired_pools(x_gen: np.ndarray, x_dns: np.ndarray, stride: int, policy: str, worst_window_score: str) -> PairedPools:
    t_gen = x_gen.shape[1]
    t_dns = x_dns.shape[0]
    t_ref = min(t_gen, t_dns)
    stride = resolve_window_stride(t_ref, stride)
    if t_gen <= t_dns:
        x_gen_pair = x_gen.copy()
        x_dns_pair = build_window_pool(x_dns, t_ref, stride)
        n_gen_windows = int(x_gen_pair.shape[0])
        n_dns_windows = int(x_dns_pair.shape[0])
    else:
        x_gen_pair, x_dns_pair, n_gen_windows, n_dns_windows = _select_windows_by_policy(
            x_gen,
            x_dns,
            t_ref,
            stride,
            policy=policy,
            worst_window_score=worst_window_score,
        )
    return PairedPools(
        x_gen_pair=x_gen_pair,
        x_dns_pair=x_dns_pair,
        t_ref=t_ref,
        stride=stride,
        policy=policy if t_gen > t_dns else "dns_window_pool",
        n_gen_windows=n_gen_windows,
        n_dns_windows=n_dns_windows,
    )


def energy_series(x: np.ndarray) -> np.ndarray:
    vel = x[..., :2]
    return 0.5 * np.mean(np.sum(vel * vel, axis=-1), axis=-1)


def fit_slope(y: np.ndarray) -> float:
    t = np.arange(len(y), dtype=np.float64)
    if len(y) < 2:
        return 0.0
    a, _ = np.polyfit(t, y.astype(np.float64), 1)
    return float(abs(a))


def acf_1d(y: np.ndarray, max_lag: int) -> np.ndarray:
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
    valid = np.asarray(acf, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float("nan")
    if mode == "first_zero":
        idx = np.where(valid <= 0)[0]
        end = idx[0] if idx.size else valid.size
        return float(np.sum(valid[:end]))
    return float(np.sum(valid))


def flatten_segments(values: np.ndarray, seg: slice) -> np.ndarray:
    return np.asarray(values[..., seg], dtype=np.float64).reshape(-1)


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


def apply_gradient_weights(field_bt_n: np.ndarray, neighbor_idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.asarray(field_bt_n, dtype=np.float32)
    reshape_back = x.shape
    x2 = x.reshape(-1, x.shape[-1])
    centers = x2[:, :, None]
    neigh = x2[:, neighbor_idx]
    delta = neigh - centers
    out = np.sum(delta * weights[None, :, :], axis=-1)
    return out.reshape(*reshape_back)


def compute_derivative_frame_stats(x: np.ndarray, cache: Case1OpsCache, batch_size: int = 64, keep_spatial: bool = True) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    out = {
        "enstrophy_frame": [],
        "strain2_frame": [],
        "grad2_frame": [],
        "r_div_frame": [],
        "div_l2_sq_sum": 0.0,
        "div_inf": 0.0,
        "grad2_sum": 0.0,
        "count": 0,
    }
    if keep_spatial:
        out.update({"omega": [], "omega_abs": [], "div": [], "div_abs": [], "p_grad_mag": []})
    for start in range(0, x.shape[0], batch_size):
        batch = x[start : start + batch_size]
        u1 = batch[..., 0]
        u2 = batch[..., 1]
        p = batch[..., 2]
        du1_dx = apply_gradient_weights(u1, cache.neighbor_idx, cache.grad_x_weights)
        du1_dy = apply_gradient_weights(u1, cache.neighbor_idx, cache.grad_y_weights)
        du2_dx = apply_gradient_weights(u2, cache.neighbor_idx, cache.grad_x_weights)
        du2_dy = apply_gradient_weights(u2, cache.neighbor_idx, cache.grad_y_weights)
        dp_dx = apply_gradient_weights(p, cache.neighbor_idx, cache.grad_x_weights)
        dp_dy = apply_gradient_weights(p, cache.neighbor_idx, cache.grad_y_weights)
        omega = du2_dx - du1_dy
        div = du1_dx + du2_dy
        s11 = du1_dx
        s22 = du2_dy
        s12 = 0.5 * (du1_dy + du2_dx)
        strain2_density = s11 * s11 + s22 * s22 + 2.0 * s12 * s12
        grad2_density = du1_dx * du1_dx + du1_dy * du1_dy + du2_dx * du2_dx + du2_dy * du2_dy
        out["enstrophy_frame"].append(np.mean(omega * omega, axis=-1))
        out["strain2_frame"].append(np.mean(strain2_density, axis=-1))
        out["grad2_frame"].append(np.mean(grad2_density, axis=-1))
        out["r_div_frame"].append(np.sqrt(np.mean(div * div, axis=-1)) / (np.sqrt(np.mean(grad2_density, axis=-1)) + EPS))
        out["div_l2_sq_sum"] += float(np.sum(div * div))
        out["div_inf"] = max(out["div_inf"], float(np.max(np.abs(div))))
        out["grad2_sum"] += float(np.sum(grad2_density))
        out["count"] += int(np.prod(div.shape))
        if keep_spatial:
            out["omega"].append(omega)
            out["omega_abs"].append(np.abs(omega))
            out["div"].append(div)
            out["div_abs"].append(np.abs(div))
            out["p_grad_mag"].append(np.sqrt(dp_dx * dp_dx + dp_dy * dp_dy))
    out["enstrophy_frame"] = np.concatenate(out["enstrophy_frame"], axis=0)
    out["strain2_frame"] = np.concatenate(out["strain2_frame"], axis=0)
    out["grad2_frame"] = np.concatenate(out["grad2_frame"], axis=0)
    out["r_div_frame"] = np.concatenate(out["r_div_frame"], axis=0)
    out["div_l2"] = math.sqrt(out["div_l2_sq_sum"] / max(out["count"], 1))
    out["grad2_mean_spatial"] = out["grad2_sum"] / max(out["count"], 1)
    if keep_spatial:
        out["omega"] = np.concatenate(out["omega"], axis=0)
        out["omega_abs"] = np.concatenate(out["omega_abs"], axis=0)
        out["div"] = np.concatenate(out["div"], axis=0)
        out["div_abs"] = np.concatenate(out["div_abs"], axis=0)
        out["p_grad_mag"] = np.concatenate(out["p_grad_mag"], axis=0)
    return out


def maybe_subsample_frames(x: np.ndarray, max_frames: int) -> np.ndarray:
    if x.ndim == 3:
        t = x.shape[0]
        if t <= max_frames:
            return x
        idx = np.linspace(0, t - 1, max_frames, dtype=int)
        return x[idx]
    if x.ndim == 4:
        s, t = x.shape[:2]
        if t <= max_frames:
            return x
        idx = np.linspace(0, t - 1, max_frames, dtype=int)
        return x[:, idx]
    raise ValueError(f"Unsupported shape for frame subsample: {x.shape}")


def spectral_energy_curve(frames: np.ndarray, cache: Case1OpsCache) -> Dict[str, np.ndarray]:
    eigvals = cache.lap_eigvals
    eigvecs = cache.lap_eigvecs
    edges = cache.spectrum_bin_edges
    positive_mask = eigvals > 1.0e-8
    pos_vals = eigvals[positive_mask]
    pos_vecs = eigvecs[:, positive_mask]
    frames = np.asarray(frames, dtype=np.float32)
    u = frames[:, :, :2].astype(np.float32)
    u = u - np.mean(u, axis=1, keepdims=True)
    coeff = np.tensordot(u, pos_vecs, axes=([1], [0]))
    mode_energy = 0.5 * np.sum(coeff * coeff, axis=1)
    bin_ids = np.digitize(pos_vals, edges) - 1
    counts = np.bincount(bin_ids, minlength=len(edges) - 1).astype(np.int64)
    weights = np.zeros((mode_energy.shape[1], len(edges) - 1), dtype=np.float64)
    for b in range(weights.shape[1]):
        if counts[b] > 0:
            weights[bin_ids == b, b] = 1.0 / counts[b]
    curves = mode_energy @ weights
    return {"curve_mean": np.mean(curves, axis=0), "curve_std": np.std(curves, axis=0), "curves": curves, "counts": counts, "edges": edges}


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


def structure_curves(frames: np.ndarray, cache: Case1OpsCache, powers: Iterable[int] = (2, 4, 6)) -> Dict[int, np.ndarray]:
    pair_i = cache.pair_i
    pair_j = cache.pair_j
    edges = cache.pair_bin_edges
    bin_ids = np.digitize(cache.pair_dist, edges) - 1
    vel_i = frames[:, pair_i, :2]
    vel_j = frames[:, pair_j, :2]
    delta = np.linalg.norm(vel_i - vel_j, axis=-1)
    curves = {}
    for p in powers:
        powered = np.power(delta, p, dtype=np.float64)
        curve = np.zeros((frames.shape[0], len(edges) - 1), dtype=np.float64)
        for b in range(curve.shape[1]):
            mask = bin_ids == b
            if np.any(mask):
                curve[:, b] = np.mean(powered[:, mask], axis=1)
        curves[p] = curve
    return curves


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


def pdf_metrics(
    dns_values: np.ndarray,
    gen_values: np.ndarray,
    tail_low_q: float,
    tail_high_q: float,
    max_mmd_values: int,
) -> Dict[str, float]:
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


def time_features(energy_t: np.ndarray, spectral_curves_t: np.ndarray, derivative_stats_t: Dict[str, np.ndarray], pressure_t: np.ndarray, n_bins: int) -> np.ndarray:
    feats = [
        energy_t[:, None],
        spectral_curves_t[:, :n_bins],
        derivative_stats_t["enstrophy_frame"][:, None],
        derivative_stats_t["strain2_frame"][:, None],
        derivative_stats_t["grad2_frame"][:, None],
        derivative_stats_t["r_div_frame"][:, None],
        np.mean(pressure_t, axis=-1)[:, None],
        np.std(pressure_t, axis=-1)[:, None],
    ]
    x = np.concatenate(feats, axis=1).astype(np.float64)
    mu = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + EPS
    return (x - mu) / std


def load_optional_boundary_inputs(args: argparse.Namespace) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    boundary_idx = None
    boundary_normals = None
    if args.boundary_index:
        boundary_idx = np.asarray(np.load(args.boundary_index), dtype=np.int64).reshape(-1)
    if args.boundary_normals:
        boundary_normals = np.asarray(np.load(args.boundary_normals), dtype=np.float32)
    if boundary_normals is not None and boundary_idx is None:
        raise ValueError("--boundary-normals requires --boundary-index")
    if boundary_normals is not None and boundary_normals.shape[0] != boundary_idx.shape[0]:
        raise ValueError("boundary normals size must match boundary index size")
    return boundary_idx, boundary_normals


def compute_boundary_metrics(
    x_gen: np.ndarray,
    x_dns: np.ndarray,
    boundary_idx: Optional[np.ndarray],
    boundary_normals: Optional[np.ndarray],
) -> Dict[str, Optional[float]]:
    if boundary_idx is None:
        return {
            "boundary_u_dns": None,
            "boundary_u_gen": None,
            "boundary_u_gap": None,
            "flux_dns": None,
            "flux_gen": None,
            "flux_gap": None,
        }
    dns_bound = x_dns[..., boundary_idx, :2]
    gen_bound = x_gen[..., boundary_idx, :2]
    dns_speed = np.linalg.norm(dns_bound, axis=-1)
    gen_speed = np.linalg.norm(gen_bound, axis=-1)
    out = {
        "boundary_u_dns": float(np.mean(dns_speed)),
        "boundary_u_gen": float(np.mean(gen_speed)),
        "boundary_u_gap": relative_gap(np.mean(gen_speed), np.mean(dns_speed)),
        "flux_dns": None,
        "flux_gen": None,
        "flux_gap": None,
    }
    if boundary_normals is not None:
        dns_flux = np.mean(np.sum(dns_bound * boundary_normals[None, None, :, :], axis=-1), axis=-1)
        gen_flux = np.mean(np.sum(gen_bound * boundary_normals[None, None, :, :], axis=-1), axis=-1)
        out["flux_dns"] = float(np.mean(np.abs(dns_flux)))
        out["flux_gen"] = float(np.mean(np.abs(gen_flux)))
        out["flux_gap"] = relative_gap(np.mean(np.abs(gen_flux)), np.mean(np.abs(dns_flux)))
    return out


def sample_time_pair_distances(frame_features: np.ndarray, rng: np.random.Generator, delta_max: int, n_pairs: int) -> np.ndarray:
    t = frame_features.shape[0]
    if t < 2:
        return np.zeros(1, dtype=np.float64)
    max_delta = min(delta_max, t - 1)
    deltas = rng.integers(1, max_delta + 1, size=n_pairs)
    start = rng.integers(0, t - deltas, size=n_pairs)
    end = start + deltas
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


def flatten_frames_for_source(x_gen: np.ndarray, x_dns: np.ndarray, max_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    if x_gen.ndim == 4:
        x_gen = maybe_subsample_frames(x_gen, max_frames).reshape(-1, x_gen.shape[-2], x_gen.shape[-1])
    if x_dns.ndim == 4:
        x_dns = maybe_subsample_frames(x_dns, max_frames).reshape(-1, x_dns.shape[-2], x_dns.shape[-1])
    elif x_dns.ndim == 3:
        x_dns = maybe_subsample_frames(x_dns, max_frames)
    return x_gen, x_dns


def relative_gap(gen_val: float, dns_val: float) -> float:
    return float(abs(gen_val - dns_val) / (abs(dns_val) + EPS))






















def write_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(obj), f, ensure_ascii=False, indent=2)


def write_scorecard_csv(path: str, row: Dict[str, float]) -> None:
    headers = list(row.keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        f.write(",".join(str(row[h]) for h in headers) + "\n")


def compute_energy_metrics(x_gen: np.ndarray, x_dns: np.ndarray, seg_gen: SplitSlices, seg_dns: SplitSlices, acf_max_lag: int, rng: np.random.Generator) -> Dict[str, object]:
    energy_gen = energy_series(x_gen)
    energy_dns = energy_series(x_dns)
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
    out.update(
        compute_energy_tint_metrics(
            energy_dns,
            energy_gen,
            seg_dns.late,
            seg_gen.late,
            acf_max_lag,
        )
    )
    return out


def compute_paired_metrics(
    x_gen_pair: np.ndarray,
    x_dns_pair: np.ndarray,
    cache: Case1OpsCache,
    args: argparse.Namespace,
    rng: np.random.Generator,
    boundary_idx: Optional[np.ndarray],
    boundary_normals: Optional[np.ndarray],
) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, np.ndarray]]:
    seg_gen = split_slices(x_gen_pair.shape[1], args.early_frac, args.late_frac)
    seg_dns = split_slices(x_dns_pair.shape[1], args.early_frac, args.late_frac)
    energy_metrics = compute_energy_metrics(x_gen_pair, x_dns_pair, seg_gen, seg_dns, args.acf_max_lag, rng)

    deriv_gen = compute_derivative_frame_stats(x_gen_pair.reshape(-1, x_gen_pair.shape[2], x_gen_pair.shape[3]), cache, keep_spatial=False)
    deriv_dns = compute_derivative_frame_stats(x_dns_pair.reshape(-1, x_dns_pair.shape[2], x_dns_pair.shape[3]), cache, keep_spatial=False)
    deriv_gen_frame = {
        "enstrophy_frame": deriv_gen["enstrophy_frame"].reshape(x_gen_pair.shape[0], x_gen_pair.shape[1]),
        "strain2_frame": deriv_gen["strain2_frame"].reshape(x_gen_pair.shape[0], x_gen_pair.shape[1]),
        "grad2_frame": deriv_gen["grad2_frame"].reshape(x_gen_pair.shape[0], x_gen_pair.shape[1]),
        "r_div_frame": deriv_gen["r_div_frame"].reshape(x_gen_pair.shape[0], x_gen_pair.shape[1]),
    }
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
    derivative_metrics.update(compute_boundary_metrics(x_gen_pair, x_dns_pair, boundary_idx, boundary_normals))

    dist_frames_gen = maybe_subsample_frames(x_gen_pair, min(args.max_structure_frames, args.max_spectrum_frames))
    dist_frames_dns = maybe_subsample_frames(x_dns_pair, min(args.max_structure_frames, args.max_spectrum_frames))
    dist_deriv_gen = compute_derivative_frame_stats(dist_frames_gen.reshape(-1, dist_frames_gen.shape[2], dist_frames_gen.shape[3]), cache, keep_spatial=True)
    dist_deriv_dns = compute_derivative_frame_stats(dist_frames_dns.reshape(-1, dist_frames_dns.shape[2], dist_frames_dns.shape[3]), cache, keep_spatial=True)
    derivative_metrics["kurt_gap"] = relative_gap(kurtosis(dist_deriv_gen["omega_abs"].reshape(-1), fisher=False), kurtosis(dist_deriv_dns["omega_abs"].reshape(-1), fisher=False))

    dist_pdf_data = {
        "u1": (sample_values(dist_frames_dns[..., 0], args.max_distribution_values, rng), sample_values(dist_frames_gen[..., 0], args.max_distribution_values, rng)),
        "u2": (sample_values(dist_frames_dns[..., 1], args.max_distribution_values, rng), sample_values(dist_frames_gen[..., 1], args.max_distribution_values, rng)),
        "p": (sample_values(dist_frames_dns[..., 2], args.max_distribution_values, rng), sample_values(dist_frames_gen[..., 2], args.max_distribution_values, rng)),
        "umag": (
            sample_values(np.linalg.norm(dist_frames_dns[..., :2], axis=-1), args.max_distribution_values, rng),
            sample_values(np.linalg.norm(dist_frames_gen[..., :2], axis=-1), args.max_distribution_values, rng),
        ),
        "omega_abs": (sample_values(dist_deriv_dns["omega_abs"], args.max_distribution_values, rng), sample_values(dist_deriv_gen["omega_abs"], args.max_distribution_values, rng)),
    }
    pdf_metrics_all = {
        name: pdf_metrics(dns_vals, gen_vals, args.tail_low_q, args.tail_high_q, args.max_mmd_values)
        for name, (dns_vals, gen_vals) in dist_pdf_data.items()
    }

    spec_gen_global_frames, spec_dns_global_frames = flatten_frames_for_source(x_gen_pair, x_dns_pair, args.max_spectrum_frames)
    spec_gen_early_frames, spec_dns_early_frames = flatten_frames_for_source(x_gen_pair[:, seg_gen.early], x_dns_pair[:, seg_dns.early], args.max_spectrum_frames)
    spec_gen_late_frames, spec_dns_late_frames = flatten_frames_for_source(x_gen_pair[:, seg_gen.late], x_dns_pair[:, seg_dns.late], args.max_spectrum_frames)
    spec_dns_global = spectral_energy_curve(spec_dns_global_frames, cache)
    spec_gen_global = spectral_energy_curve(spec_gen_global_frames, cache)
    spec_dns_early = spectral_energy_curve(spec_dns_early_frames, cache)
    spec_gen_early = spectral_energy_curve(spec_gen_early_frames, cache)
    spec_dns_late = spectral_energy_curve(spec_dns_late_frames, cache)
    spec_gen_late = spectral_energy_curve(spec_gen_late_frames, cache)
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

    struct_gen_global_frames, struct_dns_global_frames = flatten_frames_for_source(x_gen_pair, x_dns_pair, args.max_structure_frames)
    struct_gen_early_frames, struct_dns_early_frames = flatten_frames_for_source(x_gen_pair[:, seg_gen.early], x_dns_pair[:, seg_dns.early], args.max_structure_frames)
    struct_gen_late_frames, struct_dns_late_frames = flatten_frames_for_source(x_gen_pair[:, seg_gen.late], x_dns_pair[:, seg_dns.late], args.max_structure_frames)
    curves_gen = structure_curves(struct_gen_global_frames, cache, powers=(2, 3, 4, 6))
    curves_dns = structure_curves(struct_dns_global_frames, cache, powers=(2, 3, 4, 6))
    curves_gen_early = structure_curves(struct_gen_early_frames, cache, powers=(6,))
    curves_dns_early = structure_curves(struct_dns_early_frames, cache, powers=(6,))
    curves_gen_late = structure_curves(struct_gen_late_frames, cache, powers=(6,))
    curves_dns_late = structure_curves(struct_dns_late_frames, cache, powers=(6,))
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

    spec_feat_dns = spectral_energy_curve(x_dns_pair.reshape(-1, x_dns_pair.shape[2], x_dns_pair.shape[3]), cache)["curves"]
    spec_feat_gen = spectral_energy_curve(x_gen_pair.reshape(-1, x_gen_pair.shape[2], x_gen_pair.shape[3]), cache)["curves"]
    spec_feat_gen = spec_feat_gen.reshape(x_gen_pair.shape[0], x_gen_pair.shape[1], -1)
    spec_feat_dns = spec_feat_dns.reshape(x_dns_pair.shape[0], x_dns_pair.shape[1], -1)
    deriv_dns_feats = compute_derivative_frame_stats(x_dns_pair.reshape(-1, x_dns_pair.shape[2], x_dns_pair.shape[3]), cache, keep_spatial=False)
    deriv_gen_feats = compute_derivative_frame_stats(x_gen_pair.reshape(-1, x_gen_pair.shape[2], x_gen_pair.shape[3]), cache, keep_spatial=False)
    deriv_dns_feats = {
        "enstrophy_frame": deriv_dns_feats["enstrophy_frame"].reshape(x_dns_pair.shape[0], x_dns_pair.shape[1]),
        "strain2_frame": deriv_dns_feats["strain2_frame"].reshape(x_dns_pair.shape[0], x_dns_pair.shape[1]),
        "grad2_frame": deriv_dns_feats["grad2_frame"].reshape(x_dns_pair.shape[0], x_dns_pair.shape[1]),
        "r_div_frame": deriv_dns_feats["r_div_frame"].reshape(x_dns_pair.shape[0], x_dns_pair.shape[1]),
    }
    deriv_gen_feats = {
        "enstrophy_frame": deriv_gen_feats["enstrophy_frame"].reshape(x_gen_pair.shape[0], x_gen_pair.shape[1]),
        "strain2_frame": deriv_gen_feats["strain2_frame"].reshape(x_gen_pair.shape[0], x_gen_pair.shape[1]),
        "grad2_frame": deriv_gen_feats["grad2_frame"].reshape(x_gen_pair.shape[0], x_gen_pair.shape[1]),
        "r_div_frame": deriv_gen_feats["r_div_frame"].reshape(x_gen_pair.shape[0], x_gen_pair.shape[1]),
    }
    phi_dns = np.stack([time_features(energy_metrics["energy_dns"][s], spec_feat_dns[s], {k: v[s] for k, v in deriv_dns_feats.items()}, x_dns_pair[s, ..., 2], args.feature_bins) for s in range(x_dns_pair.shape[0])], axis=0)
    phi_gen = np.stack([time_features(energy_metrics["energy_gen"][s], spec_feat_gen[s], {k: v[s] for k, v in deriv_gen_feats.items()}, x_gen_pair[s, ..., 2], args.feature_bins) for s in range(x_gen_pair.shape[0])], axis=0)
    time_distance_metrics, time_metrics = compute_time_distance_metrics(
        phi_dns,
        phi_gen,
        seg_dns.early,
        seg_dns.late,
        seg_gen.early,
        seg_gen.late,
        rng,
        args.delta_max,
        args.delta_pairs,
        TIME_LAG_VALUES,
    )
    temporal_core_atoms = compute_temporal_core_atoms(
        phi_dns,
        phi_gen,
        energy_metrics["energy_dns"],
        energy_metrics["energy_gen"],
        seg_dns.late,
        seg_gen.late,
        rng,
        args.delta_max,
        args.delta_pairs,
        args.acf_max_lag,
        (1, 16),
    )

    dns_frame_features = sample_feature_rows(phi_dns.reshape(-1, phi_dns.shape[-1]), args.max_feature_frames, rng)
    gen_frame_features = sample_feature_rows(phi_gen.reshape(-1, phi_gen.shape[-1]), args.max_feature_frames, rng)
    pr = local_radius_precision_recall(dns_frame_features, gen_frame_features, k=args.precision_recall_k)
    coverage = cluster_coverage_metrics(dns_frame_features, gen_frame_features, n_clusters=args.cluster_count, rng=rng)
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
        "w1_time": time_distance_metrics["w1_time"],
        "w1_cross": time_distance_metrics["w1_cross"],
        "w1_time_drift": time_distance_metrics["w1_time_drift"],
        "dns_mean": time_distance_metrics["dns_mean"],
        "gen_time_mean": time_distance_metrics["gen_time_mean"],
        "gen_cross_mean": time_distance_metrics["gen_cross_mean"],
    }
    spec_per_sample = [
        safe_log_l2(spectral_energy_curve(x_gen_pair[s], cache)["curve_mean"], spec_dns_global["curve_mean"])
        for s in range(x_gen_pair.shape[0])
    ]
    s6_drift_sample = []
    for s in range(x_gen_pair.shape[0]):
        sample_early = structure_curves(
            maybe_subsample_frames(x_gen_pair[s : s + 1, seg_gen.early], args.max_structure_frames).reshape(-1, x_gen_pair.shape[2], x_gen_pair.shape[3]),
            cache,
            powers=(6,),
        )[6]
        sample_late = structure_curves(
            maybe_subsample_frames(x_gen_pair[s : s + 1, seg_gen.late], args.max_structure_frames).reshape(-1, x_gen_pair.shape[2], x_gen_pair.shape[3]),
            cache,
            powers=(6,),
        )[6]
        s6_drift_sample.append(
            float(
                safe_log_l2(np.mean(sample_late, axis=0), np.mean(curves_dns_late[6], axis=0))
                - safe_log_l2(np.mean(sample_early, axis=0), np.mean(curves_dns_early[6], axis=0))
            )
        )

    enstrophy_ref = float(np.mean(deriv_dns["enstrophy_frame"]))
    grad2_ref = float(np.mean(deriv_dns["grad2_frame"]))
    core_atoms = {
        **temporal_core_atoms,
        "S6_drift": np.asarray(s6_drift_sample, dtype=np.float64),
        "enstrophy_gap": np.asarray(
            [relative_gap(np.mean(deriv_gen_frame["enstrophy_frame"][s]), enstrophy_ref) for s in range(x_gen_pair.shape[0])],
            dtype=np.float64,
        ),
        "grad2_gap": np.asarray(
            [relative_gap(np.mean(deriv_gen_frame["grad2_frame"][s]), grad2_ref) for s in range(x_gen_pair.shape[0])],
            dtype=np.float64,
        ),
        "Err_logL2": np.asarray(spec_per_sample, dtype=np.float64),
    }
    return energy_metrics, derivative_metrics, pdf_metrics_all, spectrum_metrics, structure_metrics, time_distance_metrics, diversity_metrics, dist_pdf_data, time_metrics, core_atoms


def compute_gen_internal_metrics(x_gen: np.ndarray, cache: Case1OpsCache, args: argparse.Namespace, rng: np.random.Generator) -> Tuple[Dict, Dict, Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
    seg = split_slices(x_gen.shape[1], args.early_frac, args.late_frac)
    energy = energy_series(x_gen)
    energy_metrics = {
        "gen_mean": float(np.mean(energy)),
        "gen_std": float(np.std(energy)),
        "drift_split": float(abs(np.mean(flatten_segments(energy, seg.late)) - np.mean(flatten_segments(energy, seg.early)))),
        "drift_slope": float(np.mean([fit_slope(energy[s]) for s in range(energy.shape[0])])),
        "drift_rel": relative_gap(np.mean(flatten_segments(energy, seg.late)), np.mean(flatten_segments(energy, seg.early))),
        "drift_slope_rel": float(np.mean([fit_slope(energy[s]) / (abs(np.mean(energy[s])) + EPS) for s in range(energy.shape[0])])),
        "energy_gen": energy,
    }

    spec_gen_global_frames = maybe_subsample_frames(x_gen, args.max_spectrum_frames).reshape(-1, x_gen.shape[2], x_gen.shape[3])
    spec_gen_early_frames = maybe_subsample_frames(x_gen[:, seg.early], args.max_spectrum_frames).reshape(-1, x_gen.shape[2], x_gen.shape[3])
    spec_gen_late_frames = maybe_subsample_frames(x_gen[:, seg.late], args.max_spectrum_frames).reshape(-1, x_gen.shape[2], x_gen.shape[3])
    spec_gen_global = spectral_energy_curve(spec_gen_global_frames, cache)
    spec_gen_early = spectral_energy_curve(spec_gen_early_frames, cache)
    spec_gen_late = spectral_energy_curve(spec_gen_late_frames, cache)
    spectrum_metrics = {
        "err_early_late": safe_log_l2(spec_gen_late["curve_mean"], spec_gen_early["curve_mean"]),
        "r_hi_global": high_freq_ratio(spec_gen_global["curve_mean"], spec_gen_global["edges"]),
        "edges": spec_gen_global["edges"],
        "gen_curve": spec_gen_global["curve_mean"],
        "gen_curve_early": spec_gen_early["curve_mean"],
        "gen_curve_late": spec_gen_late["curve_mean"],
    }

    struct_gen_global_frames = maybe_subsample_frames(x_gen, args.max_structure_frames).reshape(-1, x_gen.shape[2], x_gen.shape[3])
    struct_gen_early_frames = maybe_subsample_frames(x_gen[:, seg.early], args.max_structure_frames).reshape(-1, x_gen.shape[2], x_gen.shape[3])
    struct_gen_late_frames = maybe_subsample_frames(x_gen[:, seg.late], args.max_structure_frames).reshape(-1, x_gen.shape[2], x_gen.shape[3])
    curves_gen = structure_curves(struct_gen_global_frames, cache, powers=(2, 3, 4, 6))
    curves_gen_early = structure_curves(struct_gen_early_frames, cache, powers=(6,))
    curves_gen_late = structure_curves(struct_gen_late_frames, cache, powers=(6,))
    mean_gen = mean_curve(curves_gen)
    f_gen = mean_gen[4] / np.clip(mean_gen[2] ** 2, EPS, None)
    h_gen = mean_gen[6] / np.clip(mean_gen[2] ** 3, EPS, None)
    r_centers = 0.5 * (cache.pair_bin_edges[:-1] + cache.pair_bin_edges[1:])
    structure_metrics = {
        "S6_early_late": safe_log_l2(np.mean(curves_gen_late[6], axis=0), np.mean(curves_gen_early[6], axis=0)),
        "r_centers": r_centers,
        "F_gen": f_gen,
        "H_gen": h_gen,
    }

    spec_feat_gen = spectral_energy_curve(x_gen.reshape(-1, x_gen.shape[2], x_gen.shape[3]), cache)["curves"].reshape(x_gen.shape[0], x_gen.shape[1], -1)
    deriv_gen_feats = compute_derivative_frame_stats(x_gen.reshape(-1, x_gen.shape[2], x_gen.shape[3]), cache, keep_spatial=False)
    deriv_gen_feats = {
        "enstrophy_frame": deriv_gen_feats["enstrophy_frame"].reshape(x_gen.shape[0], x_gen.shape[1]),
        "strain2_frame": deriv_gen_feats["strain2_frame"].reshape(x_gen.shape[0], x_gen.shape[1]),
        "grad2_frame": deriv_gen_feats["grad2_frame"].reshape(x_gen.shape[0], x_gen.shape[1]),
        "r_div_frame": deriv_gen_feats["r_div_frame"].reshape(x_gen.shape[0], x_gen.shape[1]),
    }
    phi_gen = np.stack([time_features(energy[s], spec_feat_gen[s], {k: v[s] for k, v in deriv_gen_feats.items()}, x_gen[s, ..., 2], args.feature_bins) for s in range(x_gen.shape[0])], axis=0)
    phi_gen_early = phi_gen[:, seg.early]
    phi_gen_late = phi_gen[:, seg.late]
    early_vals = np.concatenate([sample_time_pair_distances(phi_gen_early[s], rng, args.delta_max, max(1, args.delta_pairs // x_gen.shape[0])) for s in range(x_gen.shape[0])])
    late_vals = np.concatenate([sample_time_pair_distances(phi_gen_late[s], rng, args.delta_max, max(1, args.delta_pairs // x_gen.shape[0])) for s in range(x_gen.shape[0])])
    time_distance_metrics = {
        "gen_time_mean": float(np.mean(np.concatenate([sample_time_pair_distances(phi_gen[s], rng, args.delta_max, max(1, args.delta_pairs // x_gen.shape[0])) for s in range(x_gen.shape[0])]))),
        "gen_time_early": float(np.mean(early_vals)),
        "gen_time_late": float(np.mean(late_vals)),
        "gen_time_drift": float(np.mean(late_vals) - np.mean(early_vals)),
    }
    return energy_metrics, spectrum_metrics, structure_metrics, time_distance_metrics, energy, spec_gen_global["curve_mean"], r_centers


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    figures_dir = os.path.join(args.out_dir, "figures")
    cache_dir = os.path.join(args.out_dir, "cache")
    ensure_dir(figures_dir)
    ensure_dir(cache_dir)

    rng = np.random.default_rng(args.seed)
    loaded, x_dns_raw, coords = load_inputs(args)
    x_gen = loaded.x_gen
    z_gen = loaded.z_gen
    boundary_idx, boundary_normals = load_optional_boundary_inputs(args)

    cache_path = os.path.join(cache_dir, "case1_ops_cache.npz")
    if os.path.exists(cache_path):
        cache = load_case1_ops_cache(cache_dir)
    else:
        cache = build_case1_ops_cache(coords=coords, seed=args.seed, k_neighbors=args.k_neighbors, n_pairs=args.pair_count, pair_bins=args.pair_bins, spectrum_bins=args.spectrum_bins)
        save_case1_ops_cache(cache, cache_dir, meta={"seed": args.seed, "k_neighbors": args.k_neighbors, "pair_count": args.pair_count, "pair_bins": args.pair_bins, "spectrum_bins": args.spectrum_bins})

    paired = build_paired_pools(x_gen, x_dns_raw, args.paired_window_stride, args.paired_policy, args.worst_window_score)
    energy_paired, derivative_paired, pdf_paired, spectrum_paired, structure_paired, time_paired, diversity_paired, dist_pdf_data, time_metrics, core_atoms = compute_paired_metrics(
        paired.x_gen_pair,
        paired.x_dns_pair,
        cache,
        args,
        rng,
        boundary_idx,
        boundary_normals,
    )
    energy_internal, spectrum_internal, structure_internal, time_internal, energy_gen_internal_curve, spec_gen_internal_curve, r_centers_internal = compute_gen_internal_metrics(x_gen, cache, args, rng)

    scorecard = {
        "paired_policy": paired.policy,
        "protocol_tag": "mixed_abc",
        "drift_rel": energy_paired["drift_rel"],
        "drift_slope_rel": energy_paired["drift_slope_rel"],
        "W1_global": energy_paired["W1_global"],
        "Tint_gap": energy_paired["Tint_gap"],
        "late_tail_Tint_gap": energy_paired["late_tail_Tint_gap"],
        "Err_logL2": spectrum_paired["Err_logL2"],
        "slope_diff": spectrum_paired["slope_diff"],
        "r_hi_delta": spectrum_paired["r_hi_delta"],
        "drift_err": spectrum_paired["drift_err"],
        "S6_err": structure_paired["S6_err"],
        "F_err": structure_paired["F_err"],
        "H_err": structure_paired["H_err"],
        "S6_drift": structure_paired["S6_drift"],
        "enstrophy_gap": derivative_paired["enstrophy_gap"],
        "strain2_gap": derivative_paired["strain2_gap"],
        "grad2_gap": derivative_paired["grad2_gap"],
        "kurt_gap": derivative_paired["kurt_gap"],
        "precision": diversity_paired["precision"],
        "recall": diversity_paired["recall"],
        "uncovered": diversity_paired["uncovered"],
        "w1_time": time_paired["w1_time"],
        "w1_cross": time_paired["w1_cross"],
        "tail_w1_time": time_paired["tail_w1_time"],
        "tail_w1_cross": time_paired["tail_w1_cross"],
        "w1_time_drift": time_paired["w1_time_drift"],
        "r_div": derivative_paired["r_div"],
        "boundary_u_gap": derivative_paired["boundary_u_gap"],
        "flux_gap": derivative_paired["flux_gap"],
        "enstrophy_drift": derivative_paired["enstrophy_drift"],
    }
    for lag in TIME_LAG_VALUES:
        scorecard[f"w1_time_lag_{lag}"] = time_paired.get(f"w1_time_lag_{lag}", np.nan)

    protocol_summary = {
        "protocol_a": {
            "description": "same-length fair comparison",
            "paired_length": int(paired.t_ref),
            "paired_policy": paired.policy,
            "metrics": ["drift_rel", "drift_slope_rel", "W1_global", "KS_global"],
        },
        "protocol_b": {
            "description": "distributional comparison with matched sampling",
            "metrics": [
                "pdf.W1",
                "pdf.JS",
                "pdf.sym_KL",
                "pdf.MMD",
                "precision",
                "recall",
                "uncovered",
                "cluster_hist_l1",
                "cov_fro_gap",
            ],
        },
        "protocol_c": {
            "description": "long-horizon and degradation diagnostics",
            "metrics": [
                "Tint_gap",
                "late_tail_Tint_gap",
                "drift_err",
                "S6_drift",
                "tail_w1_time",
                "w1_time_drift",
                "gen_internal.energy.drift_rel",
                "gen_internal.time_distance.gen_time_drift",
            ],
        },
        "family_defaults": {
            "energy": "C+A",
            "distribution": "B+A",
            "spectrum": "C+B",
            "structure": "C+B",
            "derivative": "B+C",
            "diversity": "B",
            "physics_plausibility": "B+C",
        },
    }

    meta = {
        "mode": loaded.mode,
        "n_runs": loaded.n_runs,
        "pooled_sample_count": loaded.pooled_sample_count,
        "gen_fields": None if args.gen_manifest else args.gen_fields,
        "gen_latents": None if args.gen_manifest else args.gen_latents,
        "gen_manifest": args.gen_manifest,
        "dns_fields": args.dns_fields,
        "coords": args.coords,
        "gen_shape": list(x_gen.shape),
        "z_gen_shape": list(z_gen.shape),
        "dns_shape": list(x_dns_raw.shape),
        "coords_shape": list(coords.shape),
        "seed": args.seed,
        "early_frac": args.early_frac,
        "late_frac": args.late_frac,
        "k_neighbors": args.k_neighbors,
        "pair_count": args.pair_count,
        "pair_bins": args.pair_bins,
        "spectrum_bins": args.spectrum_bins,
        "max_spectrum_frames": args.max_spectrum_frames,
        "max_structure_frames": args.max_structure_frames,
        "max_distribution_values": args.max_distribution_values,
        "max_mmd_values": args.max_mmd_values,
        "delta_max": args.delta_max,
        "delta_pairs": args.delta_pairs,
        "acf_max_lag": args.acf_max_lag,
        "gap_definition": "relative",
        "protocol_a_pair_length": args.protocol_a_pair_length,
        "paired_policy": paired.policy,
        "requested_paired_policy": args.paired_policy,
        "worst_window_score": args.worst_window_score,
        "dns_length": int(x_dns_raw.shape[0]),
        "gen_length": int(x_gen.shape[1]),
        "paired_length": int(paired.t_ref),
        "length_policy": paired.policy,
        "paired_window_stride": int(paired.stride),
        "n_paired_windows_gen": int(paired.n_gen_windows),
        "n_paired_windows_dns": int(paired.n_dns_windows),
        "feature_bins": int(args.feature_bins),
        "max_feature_frames": int(args.max_feature_frames),
        "precision_recall_k": int(args.precision_recall_k),
        "cluster_count": int(args.cluster_count),
        "boundary_metrics_enabled": bool(boundary_idx is not None),
        "boundary_flux_enabled": bool(boundary_normals is not None),
        "protocols": protocol_summary,
        "pooling_scope": "same_model_same_length_cross_seed",
        "run_metas": loaded.run_metas,
    }

    np.savez_compressed(
        os.path.join(args.out_dir, "energy_curves.npz"),
        energy_gen_paired=energy_paired["energy_gen"],
        energy_dns_paired=energy_paired["energy_dns"],
        acf_dns=energy_paired["acf_dns"],
        acf_gen=energy_paired["acf_gen"],
        energy_gen_internal=energy_gen_internal_curve,
    )

    write_json(os.path.join(args.out_dir, "meta.json"), meta)
    write_json(
        os.path.join(args.out_dir, "metrics_raw.json"),
        {
            "protocols": protocol_summary,
            "paired": {
                "energy": energy_paired,
                "distribution": pdf_paired,
                "spectrum": spectrum_paired,
                "structure": structure_paired,
                "derivative": derivative_paired,
                "time_distance": time_paired,
                "diversity": diversity_paired,
            },
            "gen_internal": {
                "energy": energy_internal,
                "spectrum": spectrum_internal,
                "structure": structure_internal,
                "time_distance": time_internal,
            },
        },
    )
    write_json(os.path.join(args.out_dir, "pdf_stats.json"), pdf_paired)
    write_json(os.path.join(args.out_dir, "spectrum_metrics.json"), spectrum_paired)
    write_json(os.path.join(args.out_dir, "structure_metrics.json"), structure_paired)
    write_json(os.path.join(args.out_dir, "derivative_metrics.json"), derivative_paired)
    write_json(os.path.join(args.out_dir, "time_distance_metrics.json"), time_paired)
    write_json(os.path.join(args.out_dir, "diversity_metrics.json"), diversity_paired)
    write_json(os.path.join(args.out_dir, "core_atoms.json"), core_atoms)
    write_json(os.path.join(args.out_dir, "bootstrap_core.json"), bootstrap_core_atoms(core_atoms, n_bootstrap=200, seed=args.seed))
    write_json(os.path.join(args.out_dir, "protocol_summary.json"), protocol_summary)
    write_json(os.path.join(args.out_dir, "scorecard.json"), scorecard)
    write_scorecard_csv(os.path.join(args.out_dir, "scorecard.csv"), scorecard)

    pass
    pass
    pass
    pass
    pass
    pass
    pass

    if x_gen.shape[1] > paired.t_ref:
        pass
        pass
        pass

    print(f"[done] motivation evaluation written to {args.out_dir}")


if __name__ == "__main__":
    main()
