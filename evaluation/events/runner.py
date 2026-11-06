from __future__ import annotations

import json
import os
from argparse import Namespace
from typing import Dict, List, Optional

import numpy as np

from .backends import get_backend
from .bootstrap import bootstrap_core_atoms
from .common import EvalParams, ensure_dir, to_builtin, build_paired_pools, build_window_pool
from .io import load_coords, load_dns_fields, load_inputs, load_optional_boundary_inputs
from .metrics import build_core_row, evaluate_bundle
from .reporting import save_core_table, save_json


def _params_from_args(args: Namespace) -> EvalParams:
    return EvalParams(
        early_frac=args.early_frac,
        late_frac=args.late_frac,
        acf_max_lag=args.acf_max_lag,
        delta_max=args.delta_max,
        delta_pairs=args.delta_pairs,
        feature_bins=args.feature_bins,
        max_feature_frames=args.max_feature_frames,
        max_distribution_values=args.max_distribution_values,
        max_mmd_values=args.max_mmd_values,
        max_spectrum_frames=args.max_spectrum_frames,
        max_structure_frames=args.max_structure_frames,
        precision_recall_k=args.precision_recall_k,
        cluster_count=args.cluster_count,
        tail_low_q=args.tail_low_q,
        tail_high_q=args.tail_high_q,
        derivative_batch_size=args.derivative_batch_size,
        spectrum_batch_size=args.spectrum_batch_size,
    )


def _split_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _split_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _evaluate_protocol_a(args: Namespace, x_gen: np.ndarray, x_dns: np.ndarray, backend, cache, params: EvalParams, rng: np.random.Generator, boundary_idx, boundary_normals, entries: List[Dict]) -> None:
    policies = _split_list(args.protocol_a_policies)
    pair_len = int(args.protocol_a_pair_length) if args.protocol_a_pair_length > 0 else None
    base_dir = os.path.join(args.out_dir, "protocol_a")
    for policy in policies:
        paired = build_paired_pools(
            x_gen,
            x_dns,
            stride=args.paired_window_stride,
            policy=policy,
            worst_window_score=args.worst_window_score,
            energy_series_fn=backend.energy_series,
            early_frac=args.early_frac,
            late_frac=args.late_frac,
            field_ndim=backend.field_ndim,
            pair_length=pair_len,
        )
        summary = evaluate_bundle(paired.x_gen_pair, paired.x_dns_pair, backend, cache, params, rng, boundary_idx, boundary_normals)
        core = build_core_row(summary)
        out = {
            "protocol": "A",
            "label": args.label,
            "policy": policy,
            "pair_length": paired.t_ref,
            "paired_meta": {k: getattr(paired, k) for k in paired.__dataclass_fields__.keys() if k not in {"x_gen_pair", "x_dns_pair"}},
            "core": core,
            "summary": summary,
        }
        out_dir = os.path.join(base_dir, policy)
        save_json(os.path.join(out_dir, "summary.json"), out)
        entries.append({"protocol": "A", "label": args.label, "policy": policy, "horizon": paired.t_ref, "core": core})


def _evaluate_protocol_b(args: Namespace, x_gen: np.ndarray, x_dns: np.ndarray, backend, cache, params: EvalParams, rng: np.random.Generator, boundary_idx, boundary_normals, entries: List[Dict]) -> None:
    x_dns_full = x_dns[None, ...]
    summary = evaluate_bundle(x_gen, x_dns_full, backend, cache, params, rng, boundary_idx, boundary_normals)
    core = build_core_row(summary)
    out = {
        "protocol": "B",
        "label": args.label,
        "policy": "distributional_full",
        "pair_length": None,
        "core": core,
        "summary": summary,
    }
    out_dir = os.path.join(args.out_dir, "protocol_b")
    save_json(os.path.join(out_dir, "summary.json"), out)
    entries.append({"protocol": "B", "label": args.label, "policy": "distributional_full", "horizon": None, "core": core})


def _event_window_length(args: Namespace, t: int) -> int:
    return max(8, min(int(args.event_window_length), int(t)))


def _event_window_stride(args: Namespace, event_len: int) -> int:
    return max(1, min(int(args.event_window_stride), event_len))


def _build_event_subwindows(x: np.ndarray, event_len: int, event_stride: int, field_ndim: int) -> np.ndarray:
    return build_window_pool(x, event_len, event_stride, field_ndim)


def _compute_event_scores(x_windows: np.ndarray, backend, cache, params: EvalParams, score_name: str) -> np.ndarray:
    flat = x_windows.reshape(-1, *x_windows.shape[2:])
    deriv = backend.compute_derivative_frame_stats(flat, cache, batch_size=params.derivative_batch_size, keep_spatial=False)
    n_win, t = x_windows.shape[:2]
    if score_name == "dns_window_p95_enstrophy":
        values = deriv["enstrophy_frame"].reshape(n_win, t)
        return np.quantile(values, 0.95, axis=1).astype(np.float64)
    if score_name == "dns_window_mean_enstrophy":
        values = deriv["enstrophy_frame"].reshape(n_win, t)
        return np.mean(values, axis=1).astype(np.float64)
    if score_name == "dns_window_max_grad2":
        values = deriv["grad2_frame"].reshape(n_win, t)
        return np.max(values, axis=1).astype(np.float64)
    raise ValueError(f"Unsupported event score: {score_name}")


def _assign_event_groups(score: np.ndarray, q_low: float, q_high: float) -> Dict[str, np.ndarray]:
    score = np.asarray(score, dtype=np.float64)
    idx_all = np.arange(score.shape[0], dtype=np.int64)
    return {
        "calm": idx_all[score <= q_low],
        "typical": idx_all[(score > q_low) & (score < q_high)],
        "burst": idx_all[score >= q_high],
    }


def _event_group_payload(name: str, gen_score: np.ndarray, dns_score: np.ndarray, gen_idx: np.ndarray, dns_idx: np.ndarray) -> Dict[str, object]:
    gen_vals = np.asarray(gen_score[gen_idx], dtype=np.float64) if gen_idx.size > 0 else np.zeros(0, dtype=np.float64)
    dns_vals = np.asarray(dns_score[dns_idx], dtype=np.float64) if dns_idx.size > 0 else np.zeros(0, dtype=np.float64)
    total_gen = max(1, int(gen_score.shape[0]))
    total_dns = max(1, int(dns_score.shape[0]))
    payload: Dict[str, object] = {
        "name": name,
        "n_gen": int(gen_idx.size),
        "n_dns": int(dns_idx.size),
        "gen_frac": float(gen_idx.size / total_gen),
        "dns_frac": float(dns_idx.size / total_dns),
        "occupancy_gap": float(abs((gen_idx.size / total_gen) - (dns_idx.size / total_dns))),
    }
    if gen_vals.size > 0:
        payload["gen_score_mean"] = float(np.mean(gen_vals))
        payload["gen_score_p95"] = float(np.quantile(gen_vals, 0.95))
    if dns_vals.size > 0:
        payload["dns_score_mean"] = float(np.mean(dns_vals))
        payload["dns_score_p95"] = float(np.quantile(dns_vals, 0.95))
    return payload


def _evaluate_protocol_e(
    args: Namespace,
    summary: Dict[str, object],
    x_gen_pair: np.ndarray,
    x_dns_pair: np.ndarray,
    backend,
    cache,
    params: EvalParams,
    rng: np.random.Generator,
    boundary_idx,
    boundary_normals,
    out_dir: str,
) -> None:
    del summary
    event_len = _event_window_length(args, min(x_gen_pair.shape[1], x_dns_pair.shape[1]))
    event_stride = _event_window_stride(args, event_len)
    x_gen_event = _build_event_subwindows(x_gen_pair, event_len, event_stride, backend.field_ndim)
    x_dns_event = _build_event_subwindows(x_dns_pair, event_len, event_stride, backend.field_ndim)

    dns_score_all = _compute_event_scores(x_dns_event, backend, cache, params, args.event_score)
    gen_score_all = _compute_event_scores(x_gen_event, backend, cache, params, args.event_score)
    dns_valid = np.isfinite(dns_score_all)
    gen_valid = np.isfinite(gen_score_all)
    x_dns_event = x_dns_event[dns_valid]
    x_gen_event = x_gen_event[gen_valid]
    dns_score = dns_score_all[dns_valid]
    gen_score = gen_score_all[gen_valid]
    if dns_score.size == 0 or gen_score.size == 0:
        payload = {
            "status": "insufficient_samples",
            "reason": "no_finite_event_scores",
            "event_score": args.event_score,
            "event_threshold_source": args.event_threshold_source,
        }
        save_json(os.path.join(out_dir, "event_split.json"), payload)
        save_json(os.path.join(out_dir, "event_conditioned.json"), payload)
        return

    if args.event_threshold_source != "dns_reference":
        raise ValueError(f"Unsupported event threshold source: {args.event_threshold_source}")
    q_low = float(np.quantile(dns_score, args.event_calm_q))
    q_high = float(np.quantile(dns_score, args.event_burst_q))
    dns_groups = _assign_event_groups(dns_score, q_low, q_high)
    gen_groups = _assign_event_groups(gen_score, q_low, q_high)

    split_payload = {
        "status": "ok",
        "event_score": args.event_score,
        "event_threshold_source": args.event_threshold_source,
        "event_window_length": int(event_len),
        "event_window_stride": int(event_stride),
        "thresholds": {
            "calm_q": float(args.event_calm_q),
            "burst_q": float(args.event_burst_q),
            "score_q_low": q_low,
            "score_q_high": q_high,
        },
        "total_windows": {
            "gen": int(x_gen_event.shape[0]),
            "dns": int(x_dns_event.shape[0]),
        },
        "groups": {},
    }
    for name in ["calm", "typical", "burst"]:
        split_payload["groups"][name] = _event_group_payload(name, gen_score, dns_score, gen_groups[name], dns_groups[name])
    save_json(os.path.join(out_dir, "event_split.json"), split_payload)

    metrics_payload: Dict[str, object] = {
        "status": "ok",
        "event_score": args.event_score,
        "event_threshold_source": args.event_threshold_source,
        "event_window_length": int(event_len),
        "event_window_stride": int(event_stride),
        "thresholds": split_payload["thresholds"],
        "groups": {},
    }
    min_group = max(1, int(args.event_min_group_samples))
    for name in ["calm", "typical", "burst"]:
        gen_idx = gen_groups[name]
        dns_idx = dns_groups[name]
        group_payload = dict(split_payload["groups"][name])
        if gen_idx.size < min_group or dns_idx.size < min_group:
            group_payload["status"] = "insufficient_samples"
            group_payload["min_group_samples"] = int(min_group)
            metrics_payload["groups"][name] = group_payload
            continue
        sub_summary = evaluate_bundle(
            x_gen_event[gen_idx],
            x_dns_event[dns_idx],
            backend,
            cache,
            params,
            rng,
            boundary_idx,
            boundary_normals,
        )
        group_payload["status"] = "ok"
        group_payload["core"] = build_core_row(sub_summary)
        if args.bootstrap_samples > 0:
            group_payload["bootstrap_core"] = bootstrap_core_atoms(
                sub_summary["core_atoms"],
                n_bootstrap=args.bootstrap_samples,
                seed=args.seed,
            )
        metrics_payload["groups"][name] = group_payload
    save_json(os.path.join(out_dir, "event_conditioned.json"), metrics_payload)


def _evaluate_protocol_c_plus(args: Namespace, x_gen: np.ndarray, x_dns: np.ndarray, backend, cache, params: EvalParams, rng: np.random.Generator, boundary_idx, boundary_normals, entries: List[Dict]) -> None:
    horizons = _split_int_list(args.horizons)
    policies = _split_list(args.protocol_c_policies)
    base_dir = os.path.join(args.out_dir, "protocol_c_plus")
    t_gen = x_gen.shape[1]
    t_dns = x_dns.shape[0]
    for horizon in horizons:
        if horizon > min(t_gen, t_dns):
            continue
        for policy in policies:
            paired = build_paired_pools(
                x_gen,
                x_dns,
                stride=args.paired_window_stride,
                policy=policy,
                worst_window_score=args.worst_window_score,
                energy_series_fn=backend.energy_series,
                early_frac=args.early_frac,
                late_frac=args.late_frac,
                field_ndim=backend.field_ndim,
                pair_length=horizon,
            )
            summary = evaluate_bundle(paired.x_gen_pair, paired.x_dns_pair, backend, cache, params, rng, boundary_idx, boundary_normals)
            core = build_core_row(summary)
            out = {
                "protocol": "C+",
                "label": args.label,
                "policy": policy,
                "horizon": int(horizon),
                "paired_meta": {k: getattr(paired, k) for k in paired.__dataclass_fields__.keys() if k not in {"x_gen_pair", "x_dns_pair"}},
                "core": core,
                "summary": summary,
            }
            out_dir = os.path.join(base_dir, f"horizon_{int(horizon):05d}", policy)
            save_json(os.path.join(out_dir, "summary.json"), out)
            if args.bootstrap_samples > 0:
                boot = bootstrap_core_atoms(summary["core_atoms"], n_bootstrap=args.bootstrap_samples, seed=args.seed)
                save_json(os.path.join(out_dir, "bootstrap_core.json"), boot)
            if args.enable_protocol_e:
                _evaluate_protocol_e(args, summary, paired.x_gen_pair, paired.x_dns_pair, backend, cache, params, rng, boundary_idx, boundary_normals, out_dir)
            entries.append({"protocol": "C+", "label": args.label, "policy": policy, "horizon": int(horizon), "core": core})


def run_motivation_v2(args: Namespace) -> Dict[str, object]:
    ensure_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)
    params = _params_from_args(args)
    loaded = load_inputs(args.gen_fields, args.gen_latents, args.gen_manifest, args.sample_count, args.time_length)
    x_dns = load_dns_fields(args.case, args.dns_fields)
    coords = load_coords(args.case, args.coords)
    boundary_idx, boundary_normals = load_optional_boundary_inputs(args.boundary_index, args.boundary_normals)

    backend = get_backend(args.case)
    cache_bundle = backend.prepare_cache(
        args.cache_dir,
        coords=coords,
        seed=args.seed,
        k_neighbors=args.k_neighbors,
        pair_count=args.pair_count,
        pair_bins=args.pair_bins,
        spectrum_bins=args.spectrum_bins,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
    )
    cache = cache_bundle.payload

    run_config = {
        "case": args.case,
        "label": args.label,
        "seed": args.seed,
        "protocols": _split_list(args.protocols),
        "horizons": _split_int_list(args.horizons),
        "protocol_a_policies": _split_list(args.protocol_a_policies),
        "protocol_c_policies": _split_list(args.protocol_c_policies),
        "bootstrap_samples": int(args.bootstrap_samples),
        "enable_protocol_e": bool(args.enable_protocol_e),
        "event_calm_q": float(args.event_calm_q),
        "event_burst_q": float(args.event_burst_q),
        "event_window_length": int(args.event_window_length),
        "event_window_stride": int(args.event_window_stride),
        "event_score": args.event_score,
        "event_threshold_source": args.event_threshold_source,
        "event_min_group_samples": int(args.event_min_group_samples),
        "loaded_mode": loaded.mode,
        "pooled_sample_count": loaded.pooled_sample_count,
        "cache_meta": cache_bundle.meta,
    }
    save_json(os.path.join(args.out_dir, "run_config.json"), run_config)

    entries: List[Dict] = []
    protocols = set(_split_list(args.protocols))
    if "A" in protocols:
        _evaluate_protocol_a(args, loaded.x_gen, x_dns, backend, cache, params, rng, boundary_idx, boundary_normals, entries)
    if "B" in protocols:
        _evaluate_protocol_b(args, loaded.x_gen, x_dns, backend, cache, params, rng, boundary_idx, boundary_normals, entries)
    if "C+" in protocols:
        _evaluate_protocol_c_plus(args, loaded.x_gen, x_dns, backend, cache, params, rng, boundary_idx, boundary_normals, entries)

    save_core_table(args.out_dir, entries)
    result = {"status": "ok", "out_dir": args.out_dir, "entries": entries}
    save_json(os.path.join(args.out_dir, "result_manifest.json"), result)
    return result
