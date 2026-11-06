from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

EPS = 1.0e-8


def sample_feature_rows(features: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.shape[0] <= max_rows:
        return x
    idx = rng.choice(x.shape[0], size=max_rows, replace=False)
    return x[idx]


def _subsample_rows(x: np.ndarray, max_points: int) -> np.ndarray:
    if x.shape[0] <= max_points:
        return x
    idx = np.linspace(0, x.shape[0] - 1, max_points, dtype=int)
    return x[idx]


def rbf_mmd(a: np.ndarray, b: np.ndarray, max_points: int = 4096) -> Tuple[float, float]:
    x = np.asarray(a, dtype=np.float64).reshape(-1, a.shape[-1] if np.asarray(a).ndim > 1 else 1)
    y = np.asarray(b, dtype=np.float64).reshape(-1, b.shape[-1] if np.asarray(b).ndim > 1 else 1)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan"), float("nan")
    if max_points is not None and max_points > 0:
        x = _subsample_rows(x, int(max_points))
        y = _subsample_rows(y, int(max_points))
    z = np.concatenate([x, y], axis=0)
    z2 = np.sum(z * z, axis=1, keepdims=True)
    dist2 = np.clip(z2 + z2.T - 2.0 * (z @ z.T), 0.0, None)
    tri = dist2[np.triu_indices(dist2.shape[0], k=1)]
    sigma2 = float(np.median(tri[tri > 0])) if np.any(tri > 0) else 1.0
    gamma = 1.0 / max(2.0 * sigma2, EPS)

    xx2 = np.sum(x * x, axis=1, keepdims=True)
    yy2 = np.sum(y * y, axis=1, keepdims=True)
    k_xx = np.exp(-gamma * np.clip(xx2 + xx2.T - 2.0 * (x @ x.T), 0.0, None))
    k_yy = np.exp(-gamma * np.clip(yy2 + yy2.T - 2.0 * (y @ y.T), 0.0, None))
    k_xy = np.exp(-gamma * np.clip(xx2 + yy2.T - 2.0 * (x @ y.T), 0.0, None))

    n = x.shape[0]
    m = y.shape[0]
    mmd2 = (
        (np.sum(k_xx) - np.trace(k_xx)) / max(n * (n - 1), 1)
        + (np.sum(k_yy) - np.trace(k_yy)) / max(m * (m - 1), 1)
        - 2.0 * np.mean(k_xy)
    )
    return float(max(mmd2, 0.0)), float(math.sqrt(max(mmd2, 0.0)))


def _local_radius(features: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.shape[0] <= 1:
        return np.full(x.shape[0], 0.0, dtype=np.float64)
    kk = min(max(2, k + 1), x.shape[0])
    tree = cKDTree(x)
    dists, _ = tree.query(x, k=kk)
    if dists.ndim == 1:
        dists = dists[:, None]
    return np.asarray(dists[:, -1], dtype=np.float64)


def local_radius_precision_recall(dns_features: np.ndarray, gen_features: np.ndarray, k: int = 5) -> Dict[str, float]:
    dns = np.asarray(dns_features, dtype=np.float64)
    gen = np.asarray(gen_features, dtype=np.float64)
    if dns.shape[0] == 0 or gen.shape[0] == 0:
        return {"precision": float("nan"), "recall": float("nan")}

    dns_radius = _local_radius(dns, k)
    gen_radius = _local_radius(gen, k)
    dns_tree = cKDTree(dns)
    gen_tree = cKDTree(gen)

    d_gen_to_dns, idx_gen_to_dns = dns_tree.query(gen, k=1)
    precision = np.mean(d_gen_to_dns <= dns_radius[idx_gen_to_dns] + EPS)

    d_dns_to_gen, idx_dns_to_gen = gen_tree.query(dns, k=1)
    recall = np.mean(d_dns_to_gen <= gen_radius[idx_dns_to_gen] + EPS)
    return {"precision": float(precision), "recall": float(recall)}


def cluster_coverage_metrics(dns_features: np.ndarray, gen_features: np.ndarray, n_clusters: int, rng: np.random.Generator) -> Dict[str, float]:
    dns = np.asarray(dns_features, dtype=np.float64)
    gen = np.asarray(gen_features, dtype=np.float64)
    if dns.shape[0] == 0 or gen.shape[0] == 0:
        return {"cluster_hist_l1": float("nan"), "uncovered": float("nan")}
    n_clusters = int(max(2, min(n_clusters, dns.shape[0] + gen.shape[0])))
    data = np.concatenate([dns, gen], axis=0)
    # Deterministic init under provided RNG.
    init_idx = rng.choice(data.shape[0], size=n_clusters, replace=False)
    centroids, labels = kmeans2(data, data[init_idx], minit="matrix", iter=30)
    dns_labels = labels[: dns.shape[0]]
    gen_labels = labels[dns.shape[0] :]
    dns_hist = np.bincount(dns_labels, minlength=n_clusters).astype(np.float64)
    gen_hist = np.bincount(gen_labels, minlength=n_clusters).astype(np.float64)
    dns_hist = dns_hist / max(np.sum(dns_hist), 1.0)
    gen_hist = gen_hist / max(np.sum(gen_hist), 1.0)
    uncovered = np.mean((dns_hist > 0.0) & (gen_hist <= 0.0))
    return {
        "cluster_hist_l1": float(np.sum(np.abs(dns_hist - gen_hist))),
        "uncovered": float(uncovered),
        "n_clusters": int(n_clusters),
    }


def covariance_geometry(dns_features: np.ndarray, gen_features: np.ndarray, jitter: float = 1.0e-6) -> Dict[str, float]:
    dns = np.asarray(dns_features, dtype=np.float64)
    gen = np.asarray(gen_features, dtype=np.float64)
    if dns.shape[0] < 2 or gen.shape[0] < 2:
        return {
            "trace_dns": float("nan"),
            "trace_gen": float("nan"),
            "trace_gap": float("nan"),
            "logdet_dns": float("nan"),
            "logdet_gen": float("nan"),
            "logdet_gap": float("nan"),
            "cov_fro_gap": float("nan"),
        }
    cov_dns = np.cov(dns, rowvar=False)
    cov_gen = np.cov(gen, rowvar=False)
    eye_dns = np.eye(cov_dns.shape[0], dtype=np.float64) * jitter
    eye_gen = np.eye(cov_gen.shape[0], dtype=np.float64) * jitter
    sign_dns, logdet_dns = np.linalg.slogdet(cov_dns + eye_dns)
    sign_gen, logdet_gen = np.linalg.slogdet(cov_gen + eye_gen)
    if sign_dns <= 0:
        logdet_dns = float("nan")
    if sign_gen <= 0:
        logdet_gen = float("nan")
    trace_dns = float(np.trace(cov_dns))
    trace_gen = float(np.trace(cov_gen))
    return {
        "trace_dns": trace_dns,
        "trace_gen": trace_gen,
        "trace_gap": float(abs(trace_gen - trace_dns) / (abs(trace_dns) + EPS)),
        "logdet_dns": float(logdet_dns),
        "logdet_gen": float(logdet_gen),
        "logdet_gap": float(abs(logdet_gen - logdet_dns) / (abs(logdet_dns) + EPS)) if np.isfinite(logdet_dns) and np.isfinite(logdet_gen) else float("nan"),
        "cov_fro_gap": float(np.linalg.norm(cov_gen - cov_dns, ord="fro")),
    }
