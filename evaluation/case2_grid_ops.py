from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class Case2GridOpsCache:
    grid_shape: np.ndarray
    coords: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_dist: np.ndarray
    pair_bin_edges: np.ndarray
    spectrum_bin_edges: np.ndarray
    spectrum_pos_mask: np.ndarray
    spectrum_shell_ids: np.ndarray
    spectrum_shell_weights: np.ndarray
    spectrum_counts: np.ndarray


def generate_case2_coords(height: int = 400, width: int = 100) -> Tuple[np.ndarray, float, float]:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    z = np.linspace(0.0, 1.0, width, dtype=np.float32)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    coords = np.stack([yy, zz], axis=-1).reshape(-1, 2)
    dy = float(y[1] - y[0]) if height > 1 else 1.0
    dz = float(z[1] - z[0]) if width > 1 else 1.0
    return coords.astype(np.float32), dy, dz


def _sample_pairs(
    coords: np.ndarray,
    rng: np.random.Generator,
    n_pairs: int = 20000,
    n_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = coords.shape[0]
    pair_i = rng.integers(0, n, size=n_pairs, endpoint=False)
    pair_j = rng.integers(0, n - 1, size=n_pairs, endpoint=False)
    pair_j = np.where(pair_j >= pair_i, pair_j + 1, pair_j)
    diff = coords[pair_i] - coords[pair_j]
    dist = np.linalg.norm(diff, axis=1)
    positive = dist[dist > 0]
    if positive.size == 0:
        raise ValueError("All sampled pair distances are zero.")
    bin_edges = np.quantile(positive, np.linspace(0.0, 1.0, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if bin_edges.size < 3:
        dmin = float(positive.min())
        dmax = float(positive.max()) + 1.0e-6
        bin_edges = np.linspace(dmin, dmax, 3)
    return (
        pair_i.astype(np.int32),
        pair_j.astype(np.int32),
        dist.astype(np.float32),
        bin_edges.astype(np.float32),
    )


def _fft_shell_cache(
    height: int,
    width: int,
    dy: float,
    dz: float,
    spectrum_bins: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ky = np.fft.fftfreq(height, d=dy).astype(np.float32)
    kz = np.fft.rfftfreq(width, d=dz).astype(np.float32)
    ky_grid, kz_grid = np.meshgrid(ky, kz, indexing="ij")
    radius = np.sqrt(ky_grid * ky_grid + kz_grid * kz_grid).reshape(-1)
    pos_mask = radius > 1.0e-8
    positive = radius[pos_mask]
    if positive.size == 0:
        raise ValueError("No positive FFT shell radii found.")
    bin_edges = np.geomspace(float(positive.min()), float(positive.max()) + 1.0e-8, spectrum_bins + 1)
    shell_ids = np.digitize(positive, bin_edges) - 1
    shell_ids = np.clip(shell_ids, 0, len(bin_edges) - 2)
    counts = np.bincount(shell_ids, minlength=len(bin_edges) - 1).astype(np.int64)
    weights = np.zeros((positive.size, len(bin_edges) - 1), dtype=np.float32)
    for b in range(weights.shape[1]):
        if counts[b] > 0:
            weights[shell_ids == b, b] = 1.0 / float(counts[b])
    return (
        bin_edges.astype(np.float32),
        pos_mask.astype(bool),
        shell_ids.astype(np.int32),
        weights.astype(np.float32),
        counts.astype(np.int64),
    )


def build_case2_grid_ops_cache(
    height: int = 400,
    width: int = 100,
    seed: int = 42,
    n_pairs: int = 20000,
    pair_bins: int = 20,
    spectrum_bins: int = 24,
) -> Case2GridOpsCache:
    coords, dy, dz = generate_case2_coords(height=height, width=width)
    rng = np.random.default_rng(seed)
    pair_i, pair_j, pair_dist, pair_bin_edges = _sample_pairs(coords, rng=rng, n_pairs=n_pairs, n_bins=pair_bins)
    spectrum_bin_edges, pos_mask, shell_ids, shell_weights, counts = _fft_shell_cache(
        height=height,
        width=width,
        dy=dy,
        dz=dz,
        spectrum_bins=spectrum_bins,
    )
    return Case2GridOpsCache(
        grid_shape=np.asarray([height, width], dtype=np.int32),
        coords=coords,
        dy=np.asarray(dy, dtype=np.float32),
        dz=np.asarray(dz, dtype=np.float32),
        pair_i=pair_i,
        pair_j=pair_j,
        pair_dist=pair_dist,
        pair_bin_edges=pair_bin_edges,
        spectrum_bin_edges=spectrum_bin_edges,
        spectrum_pos_mask=pos_mask,
        spectrum_shell_ids=shell_ids,
        spectrum_shell_weights=shell_weights,
        spectrum_counts=counts,
    )


def save_case2_grid_ops_cache(cache: Case2GridOpsCache, out_dir: str, meta: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "case2_grid_ops_cache.npz"),
        grid_shape=cache.grid_shape,
        coords=cache.coords,
        dy=cache.dy,
        dz=cache.dz,
        pair_i=cache.pair_i,
        pair_j=cache.pair_j,
        pair_dist=cache.pair_dist,
        pair_bin_edges=cache.pair_bin_edges,
        spectrum_bin_edges=cache.spectrum_bin_edges,
        spectrum_pos_mask=cache.spectrum_pos_mask,
        spectrum_shell_ids=cache.spectrum_shell_ids,
        spectrum_shell_weights=cache.spectrum_shell_weights,
        spectrum_counts=cache.spectrum_counts,
    )
    with open(os.path.join(out_dir, "case2_grid_ops_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_case2_grid_ops_cache(out_dir: str) -> Case2GridOpsCache:
    path = os.path.join(out_dir, "case2_grid_ops_cache.npz")
    data = np.load(path)
    return Case2GridOpsCache(
        grid_shape=data["grid_shape"],
        coords=data["coords"],
        dy=data["dy"],
        dz=data["dz"],
        pair_i=data["pair_i"],
        pair_j=data["pair_j"],
        pair_dist=data["pair_dist"],
        pair_bin_edges=data["pair_bin_edges"],
        spectrum_bin_edges=data["spectrum_bin_edges"],
        spectrum_pos_mask=data["spectrum_pos_mask"],
        spectrum_shell_ids=data["spectrum_shell_ids"],
        spectrum_shell_weights=data["spectrum_shell_weights"],
        spectrum_counts=data["spectrum_counts"],
    )
