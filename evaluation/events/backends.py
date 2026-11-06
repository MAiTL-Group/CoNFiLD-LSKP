from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from evaluation.case1_ops import Case1OpsCache, build_case1_ops_cache, load_case1_ops_cache, save_case1_ops_cache
from evaluation.case2_grid_ops import Case2GridOpsCache, build_case2_grid_ops_cache, load_case2_grid_ops_cache, save_case2_grid_ops_cache
from evaluation.case3_ops import Case3OpsCache, build_case3_ops_cache, load_case3_ops_cache, save_case3_ops_cache

from .common import EPS, ensure_sample_axis, maybe_subsample_time_axis, relative_gap


@dataclass
class BackendCache:
    payload: Any
    meta: Dict[str, Any]


class BaseBackend:
    case: str
    field_ndim: int

    def prepare_cache(self, cache_dir: Optional[str], **kwargs) -> BackendCache:
        raise NotImplementedError

    def energy_series(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def compute_derivative_frame_stats(self, x: np.ndarray, cache: Any, batch_size: int = 64, keep_spatial: bool = True) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def spectral_energy_curve(self, frames: np.ndarray, cache: Any, batch_size: int = 8) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def structure_curves(self, frames: np.ndarray, cache: Any, powers: Iterable[int] = (2, 3, 4, 6), batch_size: int = 16) -> Dict[int, np.ndarray]:
        raise NotImplementedError

    def time_features(self, energy_t: np.ndarray, spectral_curves_t: np.ndarray, derivative_stats_t: Dict[str, np.ndarray], x_t: np.ndarray, n_bins: int) -> np.ndarray:
        raise NotImplementedError

    def compute_boundary_metrics(self, x_gen: np.ndarray, x_dns: np.ndarray, boundary_idx: Optional[np.ndarray], boundary_normals: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
        raise NotImplementedError

    def flatten_total_frames(self, x: np.ndarray, max_frames: int) -> np.ndarray:
        x_s = ensure_sample_axis(x, self.field_ndim)
        if x_s.shape[0] * x_s.shape[1] <= max_frames:
            return x_s.reshape(-1, *x_s.shape[2:])
        return maybe_subsample_time_axis(x_s, max_frames).reshape(-1, *x_s.shape[2:])


def apply_gradient_weights(field_bt_n: np.ndarray, neighbor_idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.asarray(field_bt_n, dtype=np.float32)
    reshape_back = x.shape
    x2 = x.reshape(-1, x.shape[-1])
    centers = x2[:, :, None]
    neigh = x2[:, neighbor_idx]
    delta = neigh - centers
    out = np.sum(delta * weights[None, :, :], axis=-1)
    return out.reshape(*reshape_back)


class PointCloud2DBackend(BaseBackend):
    def __init__(self, case: str):
        self.case = case
        self.field_ndim = 2
        if case == "case1":
            self._build = build_case1_ops_cache
            self._load = load_case1_ops_cache
            self._save = save_case1_ops_cache
            self._cache_name = "case1_ops_meta.json"
        elif case == "case3":
            self._build = build_case3_ops_cache
            self._load = load_case3_ops_cache
            self._save = save_case3_ops_cache
            self._cache_name = "case3_ops_meta.json"
        else:
            raise ValueError(f"Unsupported point-cloud case: {case}")

    def prepare_cache(self, cache_dir: Optional[str], **kwargs) -> BackendCache:
        coords = kwargs["coords"]
        if cache_dir and os.path.exists(cache_dir):
            cache = self._load(cache_dir)
            meta_path = os.path.join(cache_dir, self._cache_name)
            meta = json.loads(open(meta_path, "r", encoding="utf-8").read()) if os.path.exists(meta_path) else {}
            return BackendCache(payload=cache, meta=meta)
        cache = self._build(
            coords,
            seed=kwargs.get("seed", 42),
            k_neighbors=kwargs.get("k_neighbors", 12),
            n_pairs=kwargs.get("pair_count", 20000),
            pair_bins=kwargs.get("pair_bins", 20),
            spectrum_bins=kwargs.get("spectrum_bins", 24),
        )
        meta = {
            "case": self.case,
            "seed": kwargs.get("seed", 42),
            "k_neighbors": kwargs.get("k_neighbors", 12),
            "pair_count": kwargs.get("pair_count", 20000),
            "pair_bins": kwargs.get("pair_bins", 20),
            "spectrum_bins": kwargs.get("spectrum_bins", 24),
        }
        if cache_dir:
            self._save(cache, cache_dir, meta)
        return BackendCache(payload=cache, meta=meta)

    def energy_series(self, x: np.ndarray) -> np.ndarray:
        vel = np.asarray(x, dtype=np.float32)[..., :2]
        return 0.5 * np.mean(np.sum(vel * vel, axis=-1), axis=-1)

    def get_aux_scalar_field(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] >= 3:
            return x[..., 2]
        return np.linalg.norm(x[..., :2], axis=-1).astype(np.float32)

    def compute_derivative_frame_stats(self, x: np.ndarray, cache: Case1OpsCache, batch_size: int = 64, keep_spatial: bool = True) -> Dict[str, np.ndarray]:
        x = np.asarray(x, dtype=np.float32)
        has_scalar = x.shape[-1] >= 3
        out: Dict[str, Any] = {
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
            aux_key = "p_grad_mag" if has_scalar else "aux_grad_mag"
            out.update({"omega": [], "omega_abs": [], "div": [], "div_abs": [], aux_key: []})
        for start in range(0, x.shape[0], batch_size):
            batch = x[start : start + batch_size]
            u1 = batch[..., 0]
            u2 = batch[..., 1]
            du1_dx = apply_gradient_weights(u1, cache.neighbor_idx, cache.grad_x_weights)
            du1_dy = apply_gradient_weights(u1, cache.neighbor_idx, cache.grad_y_weights)
            du2_dx = apply_gradient_weights(u2, cache.neighbor_idx, cache.grad_x_weights)
            du2_dy = apply_gradient_weights(u2, cache.neighbor_idx, cache.grad_y_weights)
            if has_scalar:
                aux = batch[..., 2]
                daux_dx = apply_gradient_weights(aux, cache.neighbor_idx, cache.grad_x_weights)
                daux_dy = apply_gradient_weights(aux, cache.neighbor_idx, cache.grad_y_weights)
            else:
                daux_dx = np.zeros_like(du1_dx)
                daux_dy = np.zeros_like(du1_dy)
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
                aux_key = "p_grad_mag" if has_scalar else "aux_grad_mag"
                out[aux_key].append(np.sqrt(daux_dx * daux_dx + daux_dy * daux_dy))
        out["enstrophy_frame"] = np.concatenate(out["enstrophy_frame"], axis=0)
        out["strain2_frame"] = np.concatenate(out["strain2_frame"], axis=0)
        out["grad2_frame"] = np.concatenate(out["grad2_frame"], axis=0)
        out["r_div_frame"] = np.concatenate(out["r_div_frame"], axis=0)
        out["div_l2"] = math.sqrt(out["div_l2_sq_sum"] / max(out["count"], 1))
        out["grad2_mean_spatial"] = out["grad2_sum"] / max(out["count"], 1)
        if keep_spatial:
            for key in [k for k in out.keys() if isinstance(out[k], list)]:
                out[key] = np.concatenate(out[key], axis=0)
        return out  # type: ignore[return-value]

    def spectral_energy_curve(self, frames: np.ndarray, cache: Case1OpsCache, batch_size: int = 8) -> Dict[str, np.ndarray]:
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

    def structure_curves(self, frames: np.ndarray, cache: Case1OpsCache, powers: Iterable[int] = (2, 3, 4, 6), batch_size: int = 16) -> Dict[int, np.ndarray]:
        pair_i = cache.pair_i
        pair_j = cache.pair_j
        edges = cache.pair_bin_edges
        bin_ids = np.digitize(cache.pair_dist, edges) - 1
        vel_i = frames[:, pair_i, :2]
        vel_j = frames[:, pair_j, :2]
        delta = np.linalg.norm(vel_i - vel_j, axis=-1)
        curves: Dict[int, np.ndarray] = {}
        for p in powers:
            powered = np.power(delta, p, dtype=np.float64)
            curve = np.zeros((frames.shape[0], len(edges) - 1), dtype=np.float64)
            for b in range(curve.shape[1]):
                mask = bin_ids == b
                if np.any(mask):
                    curve[:, b] = np.mean(powered[:, mask], axis=1)
            curves[p] = curve
        return curves

    def time_features(self, energy_t: np.ndarray, spectral_curves_t: np.ndarray, derivative_stats_t: Dict[str, np.ndarray], x_t: np.ndarray, n_bins: int) -> np.ndarray:
        aux_scalar_t = self.get_aux_scalar_field(x_t)
        feats = [
            energy_t[:, None],
            spectral_curves_t[:, :n_bins],
            derivative_stats_t["enstrophy_frame"][:, None],
            derivative_stats_t["strain2_frame"][:, None],
            derivative_stats_t["grad2_frame"][:, None],
            derivative_stats_t["r_div_frame"][:, None],
            np.mean(aux_scalar_t, axis=-1)[:, None],
            np.std(aux_scalar_t, axis=-1)[:, None],
        ]
        arr = np.concatenate(feats, axis=1).astype(np.float64)
        mu = np.mean(arr, axis=0, keepdims=True)
        std = np.std(arr, axis=0, keepdims=True) + EPS
        return (arr - mu) / std

    def compute_boundary_metrics(self, x_gen: np.ndarray, x_dns: np.ndarray, boundary_idx: Optional[np.ndarray], boundary_normals: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
        if boundary_idx is None:
            return {"boundary_u_dns": None, "boundary_u_gen": None, "boundary_u_gap": None, "flux_dns": None, "flux_gen": None, "flux_gap": None}
        dns_bound = x_dns[..., boundary_idx, :2]
        gen_bound = x_gen[..., boundary_idx, :2]
        dns_speed = np.linalg.norm(dns_bound, axis=-1)
        gen_speed = np.linalg.norm(gen_bound, axis=-1)
        out: Dict[str, Optional[float]] = {
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


class GridCase2Backend(BaseBackend):
    case = "case2"
    field_ndim = 3

    def prepare_cache(self, cache_dir: Optional[str], **kwargs) -> BackendCache:
        if cache_dir and os.path.exists(cache_dir):
            cache = load_case2_grid_ops_cache(cache_dir)
            meta_path = os.path.join(cache_dir, "case2_grid_ops_meta.json")
            meta = json.loads(open(meta_path, "r", encoding="utf-8").read()) if os.path.exists(meta_path) else {}
            return BackendCache(payload=cache, meta=meta)
        cache = build_case2_grid_ops_cache(
            height=kwargs.get("grid_height", 400),
            width=kwargs.get("grid_width", 100),
            seed=kwargs.get("seed", 42),
            n_pairs=kwargs.get("pair_count", 20000),
            pair_bins=kwargs.get("pair_bins", 20),
            spectrum_bins=kwargs.get("spectrum_bins", 24),
        )
        meta = {
            "case": self.case,
            "grid_height": kwargs.get("grid_height", 400),
            "grid_width": kwargs.get("grid_width", 100),
            "seed": kwargs.get("seed", 42),
            "pair_count": kwargs.get("pair_count", 20000),
            "pair_bins": kwargs.get("pair_bins", 20),
            "spectrum_bins": kwargs.get("spectrum_bins", 24),
        }
        if cache_dir:
            save_case2_grid_ops_cache(cache, cache_dir, meta)
        return BackendCache(payload=cache, meta=meta)

    def energy_series(self, x: np.ndarray) -> np.ndarray:
        vel = np.asarray(x, dtype=np.float32)[..., :3]
        return 0.5 * np.mean(np.sum(vel * vel, axis=-1), axis=(-2, -1))

    def flatten_spatial(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32).reshape(*x.shape[:-3], -1, x.shape[-1])

    def compute_derivative_frame_stats(self, x: np.ndarray, cache: Case2GridOpsCache, batch_size: int = 8, keep_spatial: bool = True) -> Dict[str, np.ndarray]:
        x = np.asarray(x, dtype=np.float32)
        out: Dict[str, Any] = {
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
        dy = float(cache.dy)
        dz = float(cache.dz)
        for start in range(0, x.shape[0], batch_size):
            batch = x[start : start + batch_size]
            u = batch[..., 0]
            v = batch[..., 1]
            w = batch[..., 2]
            p = batch[..., 3]
            du_dy, du_dz = np.gradient(u, dy, dz, axis=(1, 2), edge_order=1)
            dv_dy, dv_dz = np.gradient(v, dy, dz, axis=(1, 2), edge_order=1)
            dw_dy, dw_dz = np.gradient(w, dy, dz, axis=(1, 2), edge_order=1)
            dp_dy, dp_dz = np.gradient(p, dy, dz, axis=(1, 2), edge_order=1)
            div = dv_dy + dw_dz
            omega_x = dw_dy - dv_dz
            omega_y = du_dz
            omega_z = -du_dy
            omega_sq = omega_x * omega_x + omega_y * omega_y + omega_z * omega_z
            s_yy = dv_dy
            s_zz = dw_dz
            s_yz = 0.5 * (dv_dz + dw_dy)
            s_xy = 0.5 * du_dy
            s_xz = 0.5 * du_dz
            strain2_density = s_yy * s_yy + s_zz * s_zz + 2.0 * s_yz * s_yz + 2.0 * s_xy * s_xy + 2.0 * s_xz * s_xz
            grad2_density = du_dy * du_dy + du_dz * du_dz + dv_dy * dv_dy + dv_dz * dv_dz + dw_dy * dw_dy + dw_dz * dw_dz
            out["enstrophy_frame"].append(np.mean(omega_sq, axis=(-2, -1)))
            out["strain2_frame"].append(np.mean(strain2_density, axis=(-2, -1)))
            out["grad2_frame"].append(np.mean(grad2_density, axis=(-2, -1)))
            out["r_div_frame"].append(np.sqrt(np.mean(div * div, axis=(-2, -1))) / (np.sqrt(np.mean(grad2_density, axis=(-2, -1))) + EPS))
            out["div_l2_sq_sum"] += float(np.sum(div * div))
            out["div_inf"] = max(out["div_inf"], float(np.max(np.abs(div))))
            out["grad2_sum"] += float(np.sum(grad2_density))
            out["count"] += int(np.prod(div.shape))
            if keep_spatial:
                out["omega"].append(np.sqrt(np.maximum(omega_sq, 0.0)))
                out["omega_abs"].append(np.sqrt(np.maximum(omega_sq, 0.0)))
                out["div"].append(div)
                out["div_abs"].append(np.abs(div))
                out["p_grad_mag"].append(np.sqrt(dp_dy * dp_dy + dp_dz * dp_dz))
        out["enstrophy_frame"] = np.concatenate(out["enstrophy_frame"], axis=0)
        out["strain2_frame"] = np.concatenate(out["strain2_frame"], axis=0)
        out["grad2_frame"] = np.concatenate(out["grad2_frame"], axis=0)
        out["r_div_frame"] = np.concatenate(out["r_div_frame"], axis=0)
        out["div_l2"] = math.sqrt(out["div_l2_sq_sum"] / max(out["count"], 1))
        out["grad2_mean_spatial"] = out["grad2_sum"] / max(out["count"], 1)
        if keep_spatial:
            for key in [k for k in out.keys() if isinstance(out[k], list)]:
                out[key] = np.concatenate(out[key], axis=0)
        return out  # type: ignore[return-value]

    def spectral_energy_curve(self, frames: np.ndarray, cache: Case2GridOpsCache, batch_size: int = 8) -> Dict[str, np.ndarray]:
        frames = np.asarray(frames, dtype=np.float32)
        curves = []
        pos_mask = cache.spectrum_pos_mask
        weights = cache.spectrum_shell_weights.astype(np.float64)
        for start in range(0, frames.shape[0], batch_size):
            batch = frames[start : start + batch_size]
            vel = batch[..., :3].astype(np.float32)
            vel = vel - np.mean(vel, axis=(1, 2), keepdims=True)
            vel = np.moveaxis(vel, -1, 1)
            coeff = np.fft.rfftn(vel, axes=(-2, -1))
            mode_energy = 0.5 * np.sum(np.abs(coeff) ** 2, axis=1)
            flat = mode_energy.reshape(batch.shape[0], -1)[:, pos_mask]
            curves.append(flat @ weights)
        curves = np.concatenate(curves, axis=0)
        return {"curve_mean": np.mean(curves, axis=0), "curve_std": np.std(curves, axis=0), "curves": curves, "counts": cache.spectrum_counts, "edges": cache.spectrum_bin_edges}

    def structure_curves(self, frames: np.ndarray, cache: Case2GridOpsCache, powers: Iterable[int] = (2, 3, 4, 6), batch_size: int = 16) -> Dict[int, np.ndarray]:
        pair_i = cache.pair_i
        pair_j = cache.pair_j
        edges = cache.pair_bin_edges
        bin_ids = np.digitize(cache.pair_dist, edges) - 1
        curves = {p: [] for p in powers}
        for start in range(0, frames.shape[0], batch_size):
            batch = np.asarray(frames[start : start + batch_size], dtype=np.float32)
            flat = batch.reshape(batch.shape[0], -1, batch.shape[-1])
            vel_i = flat[:, pair_i, :3]
            vel_j = flat[:, pair_j, :3]
            delta = np.linalg.norm(vel_i - vel_j, axis=-1)
            for p in powers:
                powered = np.power(delta, p, dtype=np.float64)
                curve = np.zeros((batch.shape[0], len(edges) - 1), dtype=np.float64)
                for b in range(curve.shape[1]):
                    mask = bin_ids == b
                    if np.any(mask):
                        curve[:, b] = np.mean(powered[:, mask], axis=1)
                curves[p].append(curve)
        return {p: np.concatenate(v, axis=0) for p, v in curves.items()}

    def time_features(self, energy_t: np.ndarray, spectral_curves_t: np.ndarray, derivative_stats_t: Dict[str, np.ndarray], x_t: np.ndarray, n_bins: int) -> np.ndarray:
        pressure_t = np.asarray(x_t, dtype=np.float32)[..., 3]
        feats = [
            energy_t[:, None],
            spectral_curves_t[:, :n_bins],
            derivative_stats_t["enstrophy_frame"][:, None],
            derivative_stats_t["strain2_frame"][:, None],
            derivative_stats_t["grad2_frame"][:, None],
            derivative_stats_t["r_div_frame"][:, None],
            np.mean(pressure_t, axis=(-2, -1))[:, None],
            np.std(pressure_t, axis=(-2, -1))[:, None],
        ]
        arr = np.concatenate(feats, axis=1).astype(np.float64)
        mu = np.mean(arr, axis=0, keepdims=True)
        std = np.std(arr, axis=0, keepdims=True) + EPS
        return (arr - mu) / std

    def compute_boundary_metrics(self, x_gen: np.ndarray, x_dns: np.ndarray, boundary_idx: Optional[np.ndarray], boundary_normals: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
        if boundary_idx is None:
            return {"boundary_u_dns": None, "boundary_u_gen": None, "boundary_u_gap": None, "flux_dns": None, "flux_gen": None, "flux_gap": None}
        dns_flat = self.flatten_spatial(x_dns)
        gen_flat = self.flatten_spatial(x_gen)
        dns_bound = dns_flat[..., boundary_idx, 1:3]
        gen_bound = gen_flat[..., boundary_idx, 1:3]
        dns_speed = np.linalg.norm(dns_bound, axis=-1)
        gen_speed = np.linalg.norm(gen_bound, axis=-1)
        out: Dict[str, Optional[float]] = {
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


BACKENDS = {
    "case1": PointCloud2DBackend("case1"),
    "case2": GridCase2Backend(),
    "case3": PointCloud2DBackend("case3"),
}


def get_backend(case: str) -> BaseBackend:
    if case not in BACKENDS:
        raise ValueError(f"Unsupported case: {case}")
    return BACKENDS[case]
