"""Numerical kernels preserved from the research implementation."""
from __future__ import annotations
import argparse
import json
import math
import re
import statistics as stats
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from .tube import align_and_project, as_tensor, generated_paths, iter_npy_chunks, score_chunk
CONFIDENCE_THRESHOLD = 0.15
BURST_CONFIDENCE_THRESHOLD = 0.25
SCALE_EPS = 1e-6


def burst_threshold(artifact: Dict[str, object]) -> float:
    value = artifact.get("burst_proxy_threshold")
    if value is None:
        value = artifact["local_stochastic_prior"].get("burst_proxy_threshold")
    if value is None:
        raise ValueError("Case1 LSKP prior has no stored burst-proxy threshold")
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def score_with_burst(
    c_seq: torch.Tensor,
    artifact: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rms, confidence, _ = score_chunk(c_seq, artifact)
    delta = c_seq[:, 1:, :] - c_seq[:, :-1, :]
    n_sequences, n_transitions = delta.shape[:2]
    rms = rms.reshape(n_sequences, n_transitions)
    confidence = confidence.reshape(n_sequences, n_transitions)
    burst = torch.sqrt(delta.pow(2).sum(dim=-1) + SCALE_EPS).numpy()
    if not (rms.shape == confidence.shape == burst.shape):
        raise ValueError(
            f"transition arrays are misaligned: rms={rms.shape}, "
            f"confidence={confidence.shape}, burst={burst.shape}"
        )
    return rms, confidence, burst


def collect_chunks(
    chunks: Iterable[torch.Tensor],
    artifact: Dict[str, object],
    *,
    full_sequence: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rms_parts: List[np.ndarray] = []
    confidence_parts: List[np.ndarray] = []
    burst_parts: List[np.ndarray] = []
    for x in chunks:
        c_seq = align_and_project(x, artifact, full_sequence=full_sequence)
        rms, confidence, burst = score_with_burst(c_seq, artifact)
        rms_parts.append(rms)
        confidence_parts.append(confidence)
        burst_parts.append(burst)
    if not rms_parts:
        raise ValueError("no transitions found for audit split")
    return (
        np.concatenate(rms_parts, axis=0),
        np.concatenate(confidence_parts, axis=0),
        np.concatenate(burst_parts, axis=0),
    )


def _aggregate_clusters(
    rms: np.ndarray,
    confidence: np.ndarray,
    burst: np.ndarray,
    stored_burst_threshold: float,
    *,
    rows_per_cluster: int,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Aggregate transition statistics without splitting stored sequence windows."""
    if rms.ndim != 2:
        raise ValueError(f"expected [sequence, transition] scores, got {rms.shape}")
    low_conf = confidence < CONFIDENCE_THRESHOLD
    burst_relaxed = (burst >= stored_burst_threshold) & (
        confidence < BURST_CONFIDENCE_THRESHOLD
    )
    gate_affected = low_conf | burst_relaxed
    joint = burst_relaxed
    unaffected = ~gate_affected
    if not np.any(joint) or not np.any(unaffected):
        raise ValueError("bootstrap comparison requires both comparison subgroups")

    tail_threshold = float(np.quantile(rms[unaffected], 0.95))
    n_rows, n_transitions = rms.shape
    row_cluster = np.arange(n_rows, dtype=np.int64) // rows_per_cluster
    n_clusters = int(row_cluster[-1]) + 1
    cluster_ids = np.repeat(row_cluster, n_transitions)

    def totals(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        flat_mask = mask.reshape(-1)
        ids = cluster_ids[flat_mask]
        values = rms.reshape(-1)[flat_mask]
        count = np.bincount(ids, minlength=n_clusters).astype(np.float64)
        score_sum = np.bincount(ids, weights=values, minlength=n_clusters)
        exceed_sum = np.bincount(
            ids,
            weights=(values > tail_threshold).astype(np.float64),
            minlength=n_clusters,
        )
        return count, score_sum, exceed_sum

    joint_count, joint_sum, joint_exceed = totals(joint)
    unaffected_count, unaffected_sum, unaffected_exceed = totals(unaffected)
    return {
        "joint_count": joint_count,
        "joint_sum": joint_sum,
        "joint_exceed": joint_exceed,
        "unaffected_count": unaffected_count,
        "unaffected_sum": unaffected_sum,
        "unaffected_exceed": unaffected_exceed,
    }, tail_threshold


def _contrast_from_totals(
    totals: Dict[str, np.ndarray], indices: np.ndarray
) -> Dict[str, float]:
    joint_count = float(totals["joint_count"][indices].sum())
    unaffected_count = float(totals["unaffected_count"][indices].sum())
    if joint_count <= 0 or unaffected_count <= 0:
        raise ValueError("bootstrap draw omitted one comparison subgroup")
    joint_mean = float(totals["joint_sum"][indices].sum() / joint_count)
    unaffected_mean = float(totals["unaffected_sum"][indices].sum() / unaffected_count)
    joint_tail_rate = float(totals["joint_exceed"][indices].sum() / joint_count)
    unaffected_tail_rate = float(
        totals["unaffected_exceed"][indices].sum() / unaffected_count
    )
    return {
        "joint_mean": joint_mean,
        "unaffected_mean": unaffected_mean,
        "mean_difference": joint_mean - unaffected_mean,
        "mean_relative_difference": joint_mean / unaffected_mean - 1.0,
        "joint_tail_exceedance": joint_tail_rate,
        "unaffected_tail_exceedance": unaffected_tail_rate,
        "tail_exceedance_difference": joint_tail_rate - unaffected_tail_rate,
    }


def cluster_bootstrap(
    split: str,
    rms: np.ndarray,
    confidence: np.ndarray,
    burst: np.ndarray,
    stored_burst_threshold: float,
    *,
    rows_per_cluster: int,
    resamples: int,
    seed: int,
    resampling_unit: str,
) -> Dict[str, object]:
    totals, tail_threshold = _aggregate_clusters(
        rms,
        confidence,
        burst,
        stored_burst_threshold,
        rows_per_cluster=rows_per_cluster,
    )
    n_clusters = len(totals["joint_count"])
    point = _contrast_from_totals(totals, np.arange(n_clusters))
    rng = np.random.default_rng(seed)
    samples: Dict[str, List[float]] = {key: [] for key in point}
    valid_draws = 0
    for _ in range(resamples):
        indices = rng.integers(0, n_clusters, size=n_clusters)
        try:
            draw = _contrast_from_totals(totals, indices)
        except ValueError:
            continue
        valid_draws += 1
        for key, value in draw.items():
            samples[key].append(value)
    if valid_draws < max(100, int(0.9 * resamples)):
        raise RuntimeError(
            f"too few valid bootstrap draws for {split}/{resampling_unit}: "
            f"{valid_draws}/{resamples}"
        )

    intervals = {
        key: {
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
        }
        for key, values in samples.items()
    }
    return {
        "split": split,
        "resampling_unit": resampling_unit,
        "rows_per_cluster": rows_per_cluster,
        "n_source_rows": int(rms.shape[0]),
        "n_clusters": n_clusters,
        "clusters_with_joint_transitions": int(
            np.count_nonzero(totals["joint_count"])
        ),
        "joint_transition_count": int(totals["joint_count"].sum()),
        "unaffected_transition_count": int(totals["unaffected_count"].sum()),
        "tail_threshold_from_unaffected_q95": tail_threshold,
        "requested_resamples": resamples,
        "valid_resamples": valid_draws,
        "seed": seed,
        "point": point,
        "intervals": intervals,
    }


def generated_chunks(root: str) -> Iterable[torch.Tensor]:
    for path in generated_paths(root):
        yield torch.from_numpy(np.load(path).astype(np.float32, copy=True))


def summarize_group(
    split: str,
    group: str,
    mask: np.ndarray,
    rms: np.ndarray,
    confidence: np.ndarray,
    burst: np.ndarray,
) -> Dict[str, object]:
    values = rms[mask]
    count = int(mask.sum())
    if count == 0:
        return {
            "split": split,
            "group": group,
            "n": 0,
            "fraction": 0.0,
            "confidence_mean": None,
            "burst_proxy_mean": None,
            "tube_rms_mean": None,
            "tube_rms_median": None,
            "tube_rms_q90": None,
            "tube_rms_q95": None,
            "tube_rms_q99": None,
        }
    return {
        "split": split,
        "group": group,
        "n": count,
        "fraction": float(count / rms.size),
        "confidence_mean": float(confidence[mask].mean()),
        "burst_proxy_mean": float(burst[mask].mean()),
        "tube_rms_mean": float(values.mean()),
        "tube_rms_median": float(np.median(values)),
        "tube_rms_q90": float(np.quantile(values, 0.90)),
        "tube_rms_q95": float(np.quantile(values, 0.95)),
        "tube_rms_q99": float(np.quantile(values, 0.99)),
    }


def summarize_split(
    split: str,
    rms: np.ndarray,
    confidence: np.ndarray,
    burst: np.ndarray,
    stored_burst_threshold: float,
) -> List[Dict[str, object]]:
    low_conf = confidence < CONFIDENCE_THRESHOLD
    burst_mask = burst >= stored_burst_threshold
    burst_relaxed = burst_mask & (confidence < BURST_CONFIDENCE_THRESHOLD)
    gate_affected = low_conf | burst_relaxed
    groups = {
        "all": np.ones(rms.shape, dtype=bool),
        "confidence_below_0.15": low_conf,
        "confidence_at_least_0.15": ~low_conf,
        "burst": burst_mask,
        "non_burst": ~burst_mask,
        "burst_and_confidence_below_0.25": burst_relaxed,
        "gate_affected_union": gate_affected,
        "gate_unaffected": ~gate_affected,
    }
    return [
        summarize_group(split, name, mask, rms, confidence, burst)
        for name, mask in groups.items()
    ]
