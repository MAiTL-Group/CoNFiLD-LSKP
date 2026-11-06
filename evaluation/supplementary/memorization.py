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

from .tube import align_and_project


def evenly_sample_indices(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, max_n, dtype=np.int64))


def load_generated_windows(root: str, window: int) -> np.ndarray:
    windows: List[np.ndarray] = []
    for path in sorted(Path(root).glob("seed_*/gen_latents.npy")):
        arr = np.load(path).astype(np.float32, copy=False)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"expected generated latent shape [B,T,D], got {arr.shape} at {path}")
        batch, total_t, dim = arr.shape
        usable = (total_t // window) * window
        if usable <= 0:
            continue
        sliced = arr[:, :usable, :].reshape(batch, usable // window, window, dim)
        windows.append(sliced.reshape(-1, window, dim))
    if not windows:
        raise FileNotFoundError(f"no generated latent windows found under {root}")
    return np.concatenate(windows, axis=0).astype(np.float32, copy=False)


def project_windows(x: np.ndarray, artifact: Dict[str, object], *, chunk_size: int = 256) -> np.ndarray:
    parts: List[np.ndarray] = []
    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        with torch.no_grad():
            c = align_and_project(torch.from_numpy(np.array(x[start:stop], copy=True)).float(), artifact, full_sequence=False)
        parts.append(c.numpy().astype(np.float32, copy=False))
    return np.concatenate(parts, axis=0)


def iter_train_chunks(path: str, chunk_size: int) -> Iterable[np.ndarray]:
    arr = np.load(path, mmap_mode="r")
    for start in range(0, arr.shape[0], chunk_size):
        stop = min(start + chunk_size, arr.shape[0])
        yield np.array(arr[start:stop], dtype=np.float32, copy=True)


def nearest_rms(query: np.ndarray, train_path_or_array, *, train_chunk: int) -> np.ndarray:
    q = query.reshape(query.shape[0], -1).astype(np.float32, copy=False)
    best = np.full(q.shape[0], np.inf, dtype=np.float64)
    if isinstance(train_path_or_array, (str, Path)):
        chunks = iter_train_chunks(str(train_path_or_array), train_chunk)
    else:
        arr = np.asarray(train_path_or_array, dtype=np.float32)
        chunks = (arr[start : start + train_chunk] for start in range(0, arr.shape[0], train_chunk))
    for chunk in chunks:
        t = chunk.reshape(chunk.shape[0], -1).astype(np.float32, copy=False)
        # Exact chunked squared Euclidean distance, reported as per-coordinate RMS.
        q_norm = np.sum(q * q, axis=1, dtype=np.float64)[:, None]
        t_norm = np.sum(t * t, axis=1, dtype=np.float64)[None, :]
        dist2 = q_norm + t_norm - 2.0 * (q.astype(np.float64) @ t.astype(np.float64).T)
        dist2 = np.maximum(dist2, 0.0)
        best = np.minimum(best, np.min(dist2, axis=1))
    return np.sqrt(best / q.shape[1])


def summarize_dist(case: str, space: str, source: str, dist: np.ndarray, duplicate_threshold: float) -> Dict[str, object]:
    return {
        "case": case,
        "space": space,
        "source": source,
        "n_query": int(dist.size),
        "min_rms": float(np.min(dist)),
        "q05_rms": float(np.quantile(dist, 0.05)),
        "median_rms": float(np.median(dist)),
        "q95_rms": float(np.quantile(dist, 0.95)),
        "max_rms": float(np.max(dist)),
        "duplicates_below_threshold": int(np.sum(dist <= duplicate_threshold)),
        "duplicate_threshold": float(duplicate_threshold),
    }


def audit_case(case: str, spec: Dict[str, object], args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    train_path = str(spec["train"])
    test_path = str(spec["test"])
    window = int(spec["window"])
    artifact = torch.load(str(spec["prior"]), map_location="cpu")

    train = np.load(train_path, mmap_mode="r")
    test = np.load(test_path, mmap_mode="r")
    gen = load_generated_windows(str(spec["generated"]), window)
    test_idx = evenly_sample_indices(test.shape[0], int(args.test_sample))
    test_sample = np.array(test[test_idx], dtype=np.float32, copy=True)

    rows: List[Dict[str, object]] = []
    rows.append(summarize_dist(case, "latent", "generated_to_train", nearest_rms(gen, train_path, train_chunk=args.train_chunk), args.duplicate_threshold))
    rows.append(summarize_dist(case, "latent", "test_to_train", nearest_rms(test_sample, train_path, train_chunk=args.train_chunk), args.duplicate_threshold))

    # Coarse observable NN uses the same PCA/alignment map as the fitted LSKP prior.
    train_c_parts: List[np.ndarray] = []
    for chunk in iter_train_chunks(train_path, max(1, args.train_chunk)):
        train_c_parts.append(project_windows(chunk, artifact, chunk_size=max(1, args.train_chunk)))
    train_c = np.concatenate(train_c_parts, axis=0)
    gen_c = project_windows(gen, artifact, chunk_size=256)
    test_c = project_windows(test_sample, artifact, chunk_size=256)
    rows.append(summarize_dist(case, "coarse_observable", "generated_to_train", nearest_rms(gen_c, train_c, train_chunk=max(512, args.train_chunk)), args.duplicate_threshold))
    rows.append(summarize_dist(case, "coarse_observable", "test_to_train", nearest_rms(test_c, train_c, train_chunk=max(512, args.train_chunk)), args.duplicate_threshold))

    by_key = {(r["space"], r["source"]): r for r in rows}
    ratios = {}
    for space in ["latent", "coarse_observable"]:
        gen_med = float(by_key[(space, "generated_to_train")]["median_rms"])
        test_med = float(by_key[(space, "test_to_train")]["median_rms"])
        ratios[space] = gen_med / test_med if test_med > 0 else float("nan")
    meta = {
        "case": case,
        "train_windows": int(train.shape[0]),
        "test_windows_total": int(test.shape[0]),
        "test_windows_sampled": int(test_sample.shape[0]),
        "generated_windows": int(gen.shape[0]),
        "generated_root": str(spec["generated"]),
        "median_ratio_generated_over_test": ratios,
    }
    return rows, meta
