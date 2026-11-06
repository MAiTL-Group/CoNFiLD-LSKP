from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Case3OpsCache:
    coords: np.ndarray
    neighbor_idx: np.ndarray
    grad_x_weights: np.ndarray
    grad_y_weights: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_dist: np.ndarray
    pair_bin_edges: np.ndarray
    lap_eigvals: np.ndarray
    lap_eigvecs: np.ndarray
    spectrum_bin_edges: np.ndarray


def _solve_local_gradient_weights(
    coords: np.ndarray, neighbor_idx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute local weighted least-squares gradient coefficients.

    For each node i and scalar field f, estimate gradient by
    grad f_i = B_i @ (f(neigh) - f_i), where B_i is shape [2, k].
    """

    n, k = neighbor_idx.shape
    wx = np.zeros((n, k), dtype=np.float32)
    wy = np.zeros((n, k), dtype=np.float32)
    eye2 = np.eye(2, dtype=np.float64)

    for i in range(n):
        neigh = neighbor_idx[i]
        dx = coords[neigh] - coords[i : i + 1]
        dist2 = np.sum(dx * dx, axis=1)
        sigma2 = max(np.median(dist2), 1.0e-12)
        w = np.exp(-dist2 / sigma2)
        w = np.clip(w, 1.0e-8, None)
        a = dx.astype(np.float64)
        wa = a * w[:, None]
        ata = a.T @ wa
        try:
            inv = np.linalg.inv(ata + 1.0e-8 * eye2)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(ata + 1.0e-6 * eye2)
        b = inv @ (a.T * w[None, :])
        wx[i] = b[0].astype(np.float32)
        wy[i] = b[1].astype(np.float32)
    return wx, wy


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
    return pair_i.astype(np.int32), pair_j.astype(np.int32), dist.astype(np.float32), bin_edges.astype(np.float32)


def _graph_spectrum(
    coords: np.ndarray,
    neighbor_idx: np.ndarray,
    spectrum_bins: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, k = neighbor_idx.shape
    w = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        neigh = neighbor_idx[i]
        dx = coords[neigh] - coords[i : i + 1]
        dist2 = np.sum(dx * dx, axis=1)
        sigma2 = max(np.median(dist2), 1.0e-12)
        wij = np.exp(-dist2 / sigma2)
        w[i, neigh] = np.maximum(w[i, neigh], wij)
    w = np.maximum(w, w.T)
    deg = np.sum(w, axis=1)
    inv_sqrt_deg = 1.0 / np.sqrt(np.clip(deg, 1.0e-12, None))
    l = np.eye(n, dtype=np.float64) - (inv_sqrt_deg[:, None] * w * inv_sqrt_deg[None, :])
    eigvals, eigvecs = np.linalg.eigh(l)
    eigvals = eigvals.astype(np.float32)
    eigvecs = eigvecs.astype(np.float32)
    positive = eigvals[eigvals > 1.0e-8]
    if positive.size == 0:
        raise ValueError("No positive graph Laplacian eigenvalues found.")
    bin_edges = np.geomspace(float(positive.min()), float(positive.max()) + 1.0e-8, spectrum_bins + 1)
    return eigvals, eigvecs, bin_edges.astype(np.float32)


def build_case3_ops_cache(
    coords: np.ndarray,
    seed: int = 42,
    k_neighbors: int = 12,
    n_pairs: int = 20000,
    pair_bins: int = 20,
    spectrum_bins: int = 24,
) -> Case3OpsCache:
    tree = cKDTree(coords)
    _, neigh = tree.query(coords, k=k_neighbors + 1)
    neigh = neigh[:, 1:]
    grad_x_weights, grad_y_weights = _solve_local_gradient_weights(coords, neigh)
    rng = np.random.default_rng(seed)
    pair_i, pair_j, pair_dist, pair_bin_edges = _sample_pairs(
        coords, rng=rng, n_pairs=n_pairs, n_bins=pair_bins
    )
    eigvals, eigvecs, spectrum_bin_edges = _graph_spectrum(
        coords, neighbor_idx=neigh, spectrum_bins=spectrum_bins
    )
    return Case3OpsCache(
        coords=coords.astype(np.float32),
        neighbor_idx=neigh.astype(np.int32),
        grad_x_weights=grad_x_weights,
        grad_y_weights=grad_y_weights,
        pair_i=pair_i,
        pair_j=pair_j,
        pair_dist=pair_dist,
        pair_bin_edges=pair_bin_edges,
        lap_eigvals=eigvals,
        lap_eigvecs=eigvecs,
        spectrum_bin_edges=spectrum_bin_edges,
    )


def save_case3_ops_cache(cache: Case3OpsCache, out_dir: str, meta: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "case3_ops_cache.npz"),
        coords=cache.coords,
        neighbor_idx=cache.neighbor_idx,
        grad_x_weights=cache.grad_x_weights,
        grad_y_weights=cache.grad_y_weights,
        pair_i=cache.pair_i,
        pair_j=cache.pair_j,
        pair_dist=cache.pair_dist,
        pair_bin_edges=cache.pair_bin_edges,
        lap_eigvals=cache.lap_eigvals,
        lap_eigvecs=cache.lap_eigvecs,
        spectrum_bin_edges=cache.spectrum_bin_edges,
    )
    with open(os.path.join(out_dir, "case3_ops_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_case3_ops_cache(out_dir: str) -> Case3OpsCache:
    path = os.path.join(out_dir, "case3_ops_cache.npz")
    data = np.load(path)
    return Case3OpsCache(
        coords=data["coords"],
        neighbor_idx=data["neighbor_idx"],
        grad_x_weights=data["grad_x_weights"],
        grad_y_weights=data["grad_y_weights"],
        pair_i=data["pair_i"],
        pair_j=data["pair_j"],
        pair_dist=data["pair_dist"],
        pair_bin_edges=data["pair_bin_edges"],
        lap_eigvals=data["lap_eigvals"],
        lap_eigvecs=data["lap_eigvecs"],
        spectrum_bin_edges=data["spectrum_bin_edges"],
    )
