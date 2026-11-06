from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np

from .common import CaseDefaults, LoadedGenData

CASE_DEFAULTS = {
    "case1": CaseDefaults(
        dns_fields="data/latents/case1/sn/test_raw_segment.npy",
        coords="data/raw/case1/coords.npy",
    ),
    "case2": CaseDefaults(
        dns_fields="data/latents/case2/sn/test_raw_segment.npy",
        coords=None,
    ),
    "case3": CaseDefaults(
        dns_fields="data/latents/case3/sn/test_raw_segment.npy",
        coords="data/raw/case3/coords.npy",
    ),
}


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
    x_gen = np.asarray(gen_fields, dtype=np.float32).reshape(sample_count, time_length, *gen_fields.shape[1:])
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
            "field_shape": tuple(x_run.shape[2:]),
            "latent_dim": z_run.shape[2],
            "ema_path": run_meta.get("ema_path"),
            "cnf_case_file_path": run_meta.get("cnf_case_file_path"),
        }
        if ref is None:
            ref = current
        else:
            for key in ["time_length", "field_shape", "latent_dim"]:
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


def load_inputs(
    gen_fields: Optional[str],
    gen_latents: Optional[str],
    gen_manifest: Optional[str],
    sample_count: Optional[int],
    time_length: Optional[int],
) -> LoadedGenData:
    if gen_manifest:
        return load_manifest_inputs(gen_manifest)
    if not gen_fields or not gen_latents:
        raise ValueError("single-run mode requires both --gen-fields and --gen-latents")
    x_gen, z_gen, s_run, t_run = load_single_run(gen_fields, gen_latents, sample_count, time_length)
    return LoadedGenData(
        x_gen=x_gen,
        z_gen=z_gen,
        mode="single",
        n_runs=1,
        pooled_sample_count=int(s_run),
        run_metas=[
            {
                "manifest_entry": {"gen_fields": gen_fields, "gen_latents": gen_latents},
                "run_meta": {},
                "sample_count": s_run,
                "time_length": t_run,
            }
        ],
    )


def load_dns_fields(case: str, override_path: Optional[str] = None) -> np.ndarray:
    path = override_path or CASE_DEFAULTS[case].dns_fields
    return np.asarray(np.load(path), dtype=np.float32)


def load_coords(case: str, override_path: Optional[str] = None) -> Optional[np.ndarray]:
    path = override_path or CASE_DEFAULTS[case].coords
    if path is None:
        return None
    return np.asarray(np.load(path), dtype=np.float32)


def load_optional_boundary_inputs(boundary_index: Optional[str], boundary_normals: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    boundary_idx = None
    boundary_normals_arr = None
    if boundary_index:
        boundary_idx = np.asarray(np.load(boundary_index), dtype=np.int64).reshape(-1)
    if boundary_normals:
        boundary_normals_arr = np.asarray(np.load(boundary_normals), dtype=np.float32)
    if boundary_normals_arr is not None and boundary_idx is None:
        raise ValueError("boundary normals require boundary index")
    if boundary_normals_arr is not None and boundary_normals_arr.shape[0] != boundary_idx.shape[0]:
        raise ValueError("boundary normals size must match boundary index size")
    return boundary_idx, boundary_normals_arr
