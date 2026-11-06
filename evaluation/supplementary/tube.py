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



def as_tensor(value: object) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"expected tensor, got {type(value)!r}")
    return value.detach().cpu().float()


def build_transition_descriptor(c_seq: torch.Tensor, descriptor_cfg: Dict[str, bool], eps: float = 1.0e-6) -> torch.Tensor:
    c_now = c_seq[:, :-1, :]
    c_next = c_seq[:, 1:, :]
    delta = c_next - c_now
    zero_delta = torch.zeros_like(delta[:, :1, :])
    delta_prev = torch.cat([zero_delta, delta[:, :-1, :]], dim=1)
    delta2 = delta - delta_prev
    feats: List[torch.Tensor] = []
    if descriptor_cfg.get("include_level", True):
        feats.append(c_now)
    if descriptor_cfg.get("include_delta", True):
        feats.append(delta)
    if descriptor_cfg.get("include_prev_delta", False):
        feats.append(delta_prev)
    if descriptor_cfg.get("include_delta2", False):
        feats.append(delta2)
    if descriptor_cfg.get("include_energy_delta", True):
        e_now = c_now.pow(2).sum(dim=-1, keepdim=True)
        e_next = c_next.pow(2).sum(dim=-1, keepdim=True)
        feats.append(e_next - e_now)
        feats.append(torch.sqrt(delta.pow(2).sum(dim=-1, keepdim=True) + float(eps)))
    if not feats:
        raise ValueError("descriptor config produced no features")
    return torch.cat(feats, dim=-1)


def select_window(x: torch.Tensor, artifact: Dict[str, object]) -> torch.Tensor:
    window_size = int(artifact["window_size"])
    stride = int(artifact["temporal_stride"])
    needed = 1 + (window_size - 1) * stride
    idx = torch.arange(0, needed, stride, dtype=torch.long)
    return x.index_select(1, idx)


def align_and_project(x_raw: torch.Tensor, artifact: Dict[str, object], *, full_sequence: bool) -> torch.Tensor:
    stats = artifact["alignment_stats"]
    pca = artifact["pca_coarse_stats"]
    x_seq = x_raw if full_sequence else select_window(x_raw, artifact)
    data_mean = as_tensor(stats["data_mean"])
    data_std = as_tensor(stats["data_std"])
    aligned = (x_seq - data_mean) / data_std
    repr_mode = str(artifact["repr_mode"])
    if repr_mode == "delta_from_first":
        aligned = aligned - aligned[:, :1, :]
    elif repr_mode != "absolute":
        raise ValueError(f"unsupported repr_mode={repr_mode}")
    if stats.get("repr_mean") is not None and stats.get("repr_std") is not None:
        aligned = (aligned - as_tensor(stats["repr_mean"])) / as_tensor(stats["repr_std"])
    mean = as_tensor(pca["mean"])
    basis = as_tensor(pca["basis"])
    flat = aligned.reshape(-1, aligned.shape[-1])
    coeff = (flat - mean) @ basis
    return coeff.reshape(aligned.shape[0], aligned.shape[1], basis.shape[1])


def confidence_from_alpha(alpha: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    num_regimes = int(alpha.shape[-1])
    top_vals = alpha.topk(min(2, num_regimes), dim=-1).values
    conf_max = top_vals[..., 0]
    conf_margin = top_vals[..., 0] - top_vals[..., 1] if num_regimes > 1 else top_vals[..., 0]
    entropy = -(alpha * torch.log(alpha.clamp(min=eps))).sum(dim=-1)
    entropy_norm = entropy / float(math.log(num_regimes)) if num_regimes > 1 else torch.zeros_like(entropy)
    return (conf_max * conf_margin * (1.0 - entropy_norm)).clamp(min=0.0, max=1.0)


def score_chunk(c_seq: torch.Tensor, artifact: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior = artifact["local_stochastic_prior"]
    centers = as_tensor(prior["centers"])
    k_stack = as_tensor(prior["koopman_k"])
    sigma2 = as_tensor(prior["sigma2"])
    tau = float(prior["tau"])
    q = build_transition_descriptor(c_seq, dict(artifact["descriptor_cfg"]))
    if prior.get("descriptor_mean") is not None and prior.get("descriptor_std") is not None:
        q = (q - as_tensor(prior["descriptor_mean"])) / as_tensor(prior["descriptor_std"]).clamp(min=1.0e-6)
    q_flat = q.reshape(-1, q.shape[-1])
    dist2 = (q_flat[:, None, :] - centers[None, :, :]).pow(2).sum(dim=-1)
    alpha = torch.softmax(-tau * dist2, dim=-1).reshape(q.shape[0], q.shape[1], -1)
    z0 = c_seq[:, :-1, :]
    z1 = c_seq[:, 1:, :]
    z_pred_r = torch.einsum("rdi,bti->btrd", k_stack, z0)
    z_pred = (alpha.unsqueeze(-1) * z_pred_r).sum(dim=2)
    sigma2_mix = (alpha.unsqueeze(-1) * sigma2.unsqueeze(0).unsqueeze(0)).sum(dim=2)
    resid_white = (z1 - z_pred) / torch.sqrt(sigma2_mix + 1.0e-6)
    rms = torch.sqrt(resid_white.pow(2).mean(dim=-1)).reshape(-1)
    conf = confidence_from_alpha(alpha).reshape(-1)
    labels = alpha.argmax(dim=-1).reshape(-1)
    return rms.numpy(), conf.numpy(), labels.numpy()


def iter_npy_chunks(path: str, chunk_size: int) -> Iterable[torch.Tensor]:
    data = np.load(path, mmap_mode="r")
    for start in range(0, data.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), data.shape[0])
        yield torch.from_numpy(np.array(data[start:stop], copy=True)).float()


def generated_paths(root: str) -> List[Path]:
    base = Path(root)
    return sorted(base.glob("seed_*/gen_latents.npy"))


def summarize_arrays(case: str, split: str, rms: np.ndarray, conf: np.ndarray, labels: np.ndarray, num_regimes: int) -> Dict[str, object]:
    if rms.size == 0:
        raise ValueError(f"empty calibration array for {case} {split}")
    counts = np.bincount(labels.astype(np.int64), minlength=num_regimes)
    if np.std(conf) > 1.0e-12 and np.std(rms) > 1.0e-12:
        conf_error_corr = float(np.corrcoef(conf, rms)[0, 1])
    else:
        conf_error_corr = float("nan")
    return {
        "case": case,
        "split": split,
        "n_transitions": int(rms.size),
        "rms_mean": float(np.mean(rms)),
        "rms_median": float(np.median(rms)),
        "rms_q90": float(np.quantile(rms, 0.90)),
        "rms_q95": float(np.quantile(rms, 0.95)),
        "rms_q99": float(np.quantile(rms, 0.99)),
        "coverage_1sigma": float(np.mean(rms <= 1.0)),
        "coverage_2sigma": float(np.mean(rms <= 2.0)),
        "coverage_3sigma": float(np.mean(rms <= 3.0)),
        "confidence_mean": float(np.mean(conf)),
        "confidence_error_corr": conf_error_corr,
        "active_regimes": int(np.count_nonzero(counts)),
        "regime_counts": [int(x) for x in counts.tolist()],
    }


def audit_case(case: str, spec: Dict[str, str], chunk_size: int) -> List[Dict[str, object]]:
    artifact = torch.load(spec["prior"], map_location="cpu")
    num_regimes = int(as_tensor(artifact["local_stochastic_prior"]["centers"]).shape[0])
    rows: List[Dict[str, object]] = []
    for split in ("train", "valid", "test"):
        rms_parts: List[np.ndarray] = []
        conf_parts: List[np.ndarray] = []
        label_parts: List[np.ndarray] = []
        for x in iter_npy_chunks(spec[split], chunk_size):
            c = align_and_project(x, artifact, full_sequence=False)
            rms, conf, labels = score_chunk(c, artifact)
            rms_parts.append(rms)
            conf_parts.append(conf)
            label_parts.append(labels)
        rows.append(
            summarize_arrays(
                case,
                split,
                np.concatenate(rms_parts),
                np.concatenate(conf_parts),
                np.concatenate(label_parts),
                num_regimes,
            )
        )
    gen_rms: List[np.ndarray] = []
    gen_conf: List[np.ndarray] = []
    gen_labels: List[np.ndarray] = []
    for path in generated_paths(spec["generated"]):
        x = torch.from_numpy(np.load(path).astype(np.float32, copy=True))
        c = align_and_project(x, artifact, full_sequence=True)
        rms, conf, labels = score_chunk(c, artifact)
        gen_rms.append(rms)
        gen_conf.append(conf)
        gen_labels.append(labels)
    if gen_rms:
        rows.append(
            summarize_arrays(
                case,
                "generated_lskp",
                np.concatenate(gen_rms),
                np.concatenate(gen_conf),
                np.concatenate(gen_labels),
                num_regimes,
            )
        )
    return rows
