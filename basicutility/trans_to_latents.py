"""Extract optimized CNF snapshot codes and create temporal data partitions.

Use scripts/run.py extract-latents for the fixed per-case paper partitions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def make_overlapping_windows(latents: np.ndarray, T: int, s: int) -> np.ndarray:
    """
    Window [t_total, L] latents along time with length T and stride s.
    Return an array shaped [N, T, L].
    """
    t_total, _ = latents.shape
    if T > t_total:
        raise ValueError(f"Window length exceeds sequence length: window_length={T}, t_total={t_total}")
    if s < 1:
        raise ValueError(f"Stride must be a positive integer: stride={s}")
    N = (t_total - T) // s + 1
    idx = np.arange(T)[None, :] + s * np.arange(N)[:, None]  # [N, T]
    return latents[idx]


def make_overlapping_windows_with_starts(
    seq: np.ndarray, T: int, s: int, base_start: int = 0
) -> tuple:
    """
    Window a [t_total, ...] sequence and return global start indices.
    Returns:
      windows: [N, T, ...]
      starts:  [N], global start indices.
    """
    t_total = seq.shape[0]
    if T > t_total:
        raise ValueError(f"Window length exceeds sequence length: window_length={T}, t_total={t_total}")
    if s < 1:
        raise ValueError(f"Stride must be a positive integer: stride={s}")
    n_win = (t_total - T) // s + 1
    local_starts = s * np.arange(n_win, dtype=np.int64)  # [N]
    idx = local_starts[:, None] + np.arange(T, dtype=np.int64)[None, :]  # [N, T]
    windows = seq[idx]
    starts = local_starts + int(base_start)
    return windows, starts


def save_np(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def make_time_first_splits(
    latents: np.ndarray,
    window_length: int,
    stride: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    delay_l: int,
    purge_gap: int,
    time_split_index: int,
    time_split_index_2: int,
):
    """
    Split train/valid/test along time before windowing each partition,
    preventing overlapping windows from leaking across partitions.

    Returns:
    - train_data: [N_train, T, L]
    - valid_data: [N_valid, T, L]
    - test_data: [N_test, T, L]
    - meta: Split metadata dictionary.
    """
    t_total = latents.shape[0]
    T = window_length

    if train_ratio <= 0 or valid_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio/valid_ratio/test_ratio must all be > 0")
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + valid_ratio + test_ratio must equal 1; got {ratio_sum:.6f}"
        )

    if time_split_index is None:
        split_idx_1 = int(t_total * train_ratio)
    else:
        split_idx_1 = int(time_split_index)

    if time_split_index_2 is None:
        if time_split_index is None:
            split_idx_2 = int(t_total * (train_ratio + valid_ratio))
        else:
            remain = t_total - split_idx_1
            valid_share = valid_ratio / (valid_ratio + test_ratio)
            split_idx_2 = split_idx_1 + int(remain * valid_share)
    else:
        split_idx_2 = int(time_split_index_2)

    if not (0 < split_idx_1 < split_idx_2 < t_total):
        raise ValueError(
            f"Invalid time split: split1={split_idx_1}, split2={split_idx_2}, t_total={t_total}; "
            "require 0 < split1 < split2 < t_total"
        )

    # Infer the boundary gap from window length and delay_L if unspecified.
    gap = int(purge_gap)
    if gap < 0:
        gap = T + max(int(delay_l), 0) - 1
    if gap < 0:
        gap = 0

    train_start, train_end = 0, split_idx_1
    valid_start, valid_end = split_idx_1 + gap, split_idx_2
    test_start, test_end = split_idx_2 + gap, t_total

    train_len = train_end - train_start
    valid_len = valid_end - valid_start
    test_len = test_end - test_start
    if train_len < T:
        raise ValueError(
            f"Training segment too short: train_len={train_len}, window_length={T}, "
            f"increase train_ratio or decrease window_length"
        )
    if valid_len < T:
        raise ValueError(
            f"Validation segment too short: valid_len={valid_len}, window_length={T}, "
            f"decrease purge_gap, train_ratio or window_length"
        )
    if test_len < T:
        raise ValueError(
            f"Test segment too short: test_len={test_len}, window_length={T}, "
            f"decrease purge_gap, adjust split ratios or decrease window_length"
        )

    train_lat = latents[train_start:train_end]
    valid_lat = latents[valid_start:valid_end]
    test_lat = latents[test_start:test_end]
    train_data = make_overlapping_windows(train_lat, T=T, s=stride)
    valid_data = make_overlapping_windows(valid_lat, T=T, s=stride)
    test_data = make_overlapping_windows(test_lat, T=T, s=stride)

    meta = {
        "split_mode": "time_first",
        "window_length": int(T),
        "stride": int(stride),
        "delay_l": int(delay_l),
        "purge_gap": int(gap),
        "time_split_index_1": int(split_idx_1),
        "time_split_index_2": int(split_idx_2),
        "train_time_range": [int(train_start), int(train_end)],
        "valid_time_range": [int(valid_start), int(valid_end)],
        "test_time_range": [int(test_start), int(test_end)],
        "train_num_windows": int(train_data.shape[0]),
        "valid_num_windows": int(valid_data.shape[0]),
        "test_num_windows": int(test_data.shape[0]),
    }
    return train_data, valid_data, test_data, meta


def make_time_train_valid_split(
    latents: np.ndarray,
    window_length: int,
    train_stride: int,
    valid_stride: int,
    delay_l: int,
    purge_gap: int,
    time_split_index: int,
):
    """
    Split train/valid along time before windowing each partition.

    Returns:
    - train_data: [N_train, T, L]
    - valid_data: [N_valid, T, L]
    - meta: Split metadata dictionary.
    """
    t_total = latents.shape[0]
    T = window_length

    if time_split_index is None:
        raise ValueError("time_train_valid requires an explicit --time-split-index")
    split_idx_1 = int(time_split_index)
    if not (0 < split_idx_1 < t_total):
        raise ValueError(
            f"Invalid time split: split1={split_idx_1}, t_total={t_total}; require 0 < split1 < t_total"
        )

    gap = int(purge_gap)
    if gap < 0:
        gap = T + max(int(delay_l), 0) - 1
    if gap < 0:
        gap = 0

    train_start, train_end = 0, split_idx_1
    valid_start, valid_end = split_idx_1 + gap, t_total

    train_len = train_end - train_start
    valid_len = valid_end - valid_start
    if train_len < T:
        raise ValueError(
            f"Training segment too short: train_len={train_len}, window_length={T}, "
            f"increase split1 or decrease window_length"
        )
    if valid_len < T:
        raise ValueError(
            f"Validation segment too short: valid_len={valid_len}, window_length={T}, "
            f"decrease purge_gap, split1 or window_length"
        )

    train_lat = latents[train_start:train_end]
    valid_lat = latents[valid_start:valid_end]
    train_data = make_overlapping_windows(train_lat, T=T, s=train_stride)
    valid_data = make_overlapping_windows(valid_lat, T=T, s=valid_stride)

    meta = {
        "split_mode": "time_train_valid",
        "window_length": int(T),
        "stride": None,
        "train_stride": int(train_stride),
        "valid_stride": int(valid_stride),
        "delay_l": int(delay_l),
        "purge_gap": int(gap),
        "time_split_index_1": int(split_idx_1),
        "train_time_range": [int(train_start), int(train_end)],
        "valid_time_range": [int(valid_start), int(valid_end)],
        "train_num_windows": int(train_data.shape[0]),
        "valid_num_windows": int(valid_data.shape[0]),
    }
    return train_data, valid_data, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CNF snapshot codes and optionally save sliding windows")
    parser.add_argument("--ckpt", type=str, required=True, help="CNF checkpoint path")
    parser.add_argument("--out-root", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--raw-data-npy",
        type=str,
        default=None,
        help="Optional raw-field NPY with time as the first axis, for exporting test fields",
    )
    parser.add_argument(
        "--raw-time-offset",
        type=int,
        default=None,
        help=(
            "Optional raw-to-latent time offset: raw_idx = latent_idx + offset. "
            "None infers the offset: raw_t==latent_t -> 0; raw_t==latent_t+1 -> 1."
        ),
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=None,
        help="Window length (e.g. time_length); omit to save only latents.npy without windowing",
    )
    parser.add_argument("--stride", type=int, default=1, help="Window stride, default 1")
    parser.add_argument(
        "--train-stride",
        type=int,
        default=None,
        help="Training-window stride in time_train_valid mode; defaults to --stride",
    )
    parser.add_argument(
        "--valid-stride",
        type=int,
        default=None,
        help="Validation-window stride in time_train_valid mode; defaults to --stride",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training fraction along time, default 0.8",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Validation fraction along time, default 0.1",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test fraction along time, default 0.1",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle windows: within each partition for time_first, before splitting for window_random",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling, default 42",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="time_first",
        choices=["time_first", "time_train_valid", "window_random"],
        help=(
            "Split mode: time_first (recommended; split time before windowing) / "
            "time_train_valid (train/valid only, for Koopman fitting) / "
            "window_random (overlapping windows may leak across partitions)"
        ),
    )
    parser.add_argument(
        "--delay-l",
        type=int,
        default=0,
        help="Koopman delay_L for automatic purge_gap calculation in time_first mode",
    )
    parser.add_argument(
        "--purge-gap",
        type=int,
        default=-1,
        help="Boundary gap for time_first (shared by train-valid and valid-test); <0 uses window_length + delay_l - 1",
    )
    parser.add_argument(
        "--time-split-index",
        type=int,
        default=None,
        help="First split index for time_first/time_train_valid; overrides train_ratio",
    )
    parser.add_argument(
        "--time-split-index-2",
        type=int,
        default=None,
        help="Second split index for time_first; overrides valid_ratio/test_ratio",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    latents = ckpt["hidden_states"]["latents"]  # [t_total, L]
    latents_np = latents.detach().cpu().numpy()
    t_total = latents_np.shape[0]
    print(f"Loaded latent shape: {latents_np.shape}")

    raw_data_np = None
    raw_offset = 0
    if args.raw_data_npy is not None:
        raw_data_np = np.load(args.raw_data_npy)
        raw_t = int(raw_data_np.shape[0])
        if args.raw_time_offset is None:
            if raw_t == t_total:
                raw_offset = 0
            elif raw_t == t_total + 1:
                # Case1's load_elbow_flow(path)[1:] drops the first raw frame.
                raw_offset = 1
                print(
                    "[Info] Aligning raw/latent time axes: raw_t = latent_t + 1; "
                    "using raw_time_offset=1, so raw[1:] aligns with latent[0:]."
                )
            else:
                raise ValueError(
                    "Raw-field and latent time lengths differ; cannot infer the offset: "
                    f"raw_t={raw_t}, latent_t={t_total}. "
                    "Specify --raw-time-offset explicitly."
                )
        else:
            raw_offset = int(args.raw_time_offset)

        raw_start = raw_offset
        raw_end = raw_offset + t_total
        if raw_start < 0 or raw_end > raw_t:
            raise ValueError(
                f"Invalid raw_time_offset: offset={raw_offset}, raw_t={raw_t}, latent_t={t_total}; "
                f"require 0 <= offset and offset+latent_t <= raw_t (raw range [{raw_start}, {raw_end}))"
            )
        raw_data_np = raw_data_np[raw_start:raw_end]
        print(f"Loaded raw-field shape: {raw_data_np.shape}")

    out_root = Path(args.out_root)
    latents_path = out_root / "latents.npy"
    save_np(latents_path, latents_np)
    print(f"Saved full latents to {latents_path}")

    if args.window_length is not None:
        test_data = None
        test_window_start_idx = None
        if args.split_mode == "time_first":
            train_data, valid_data, test_data, split_meta = make_time_first_splits(
                latents=latents_np,
                window_length=args.window_length,
                stride=args.stride,
                train_ratio=args.train_ratio,
                valid_ratio=args.valid_ratio,
                test_ratio=args.test_ratio,
                delay_l=args.delay_l,
                purge_gap=args.purge_gap,
                time_split_index=args.time_split_index,
                time_split_index_2=args.time_split_index_2,
            )
            # Global start indices aligned with test_data.
            _, test_window_start_idx = make_overlapping_windows_with_starts(
                latents_np[
                    split_meta["test_time_range"][0] : split_meta["test_time_range"][1]
                ],
                T=args.window_length,
                s=args.stride,
                base_start=split_meta["test_time_range"][0],
            )
            if args.shuffle:
                train_data = train_data[rng.permutation(train_data.shape[0])]
                valid_data = valid_data[rng.permutation(valid_data.shape[0])]
                perm_test = rng.permutation(test_data.shape[0])
                test_data = test_data[perm_test]
                test_window_start_idx = test_window_start_idx[perm_test]
            split_meta["shuffle"] = bool(args.shuffle)
            split_meta["seed"] = int(args.seed)
            with open(out_root / "split_meta.json", "w", encoding="utf-8") as f:
                json.dump(split_meta, f, ensure_ascii=False, indent=2)
            print(
                f"time_first split complete: train={train_data.shape}, valid={valid_data.shape}, test={test_data.shape}, "
                f"purge_gap={split_meta['purge_gap']}, split1={split_meta['time_split_index_1']}, "
                f"split2={split_meta['time_split_index_2']}"
            )
        elif args.split_mode == "time_train_valid":
            train_stride = int(args.train_stride) if args.train_stride is not None else int(args.stride)
            valid_stride = int(args.valid_stride) if args.valid_stride is not None else int(args.stride)
            train_data, valid_data, split_meta = make_time_train_valid_split(
                latents=latents_np,
                window_length=args.window_length,
                train_stride=train_stride,
                valid_stride=valid_stride,
                delay_l=args.delay_l,
                purge_gap=args.purge_gap,
                time_split_index=args.time_split_index,
            )
            if args.shuffle:
                train_data = train_data[rng.permutation(train_data.shape[0])]
                valid_data = valid_data[rng.permutation(valid_data.shape[0])]
            split_meta["shuffle"] = bool(args.shuffle)
            split_meta["seed"] = int(args.seed)
            with open(out_root / "split_meta.json", "w", encoding="utf-8") as f:
                json.dump(split_meta, f, ensure_ascii=False, indent=2)
            print(
                f"time_train_valid split complete: train={train_data.shape}, valid={valid_data.shape}, "
                f"train_stride={split_meta['train_stride']}, valid_stride={split_meta['valid_stride']}, "
                f"purge_gap={split_meta['purge_gap']}, split1={split_meta['time_split_index_1']}"
            )
        else:
            # Windowing before random splitting can leak overlapping frames
            # across train/valid/test partitions, especially at stride=1.
            data, start_idx = make_overlapping_windows_with_starts(
                latents_np, T=args.window_length, s=args.stride, base_start=0
            )
            if args.shuffle:
                perm = rng.permutation(data.shape[0])
                data = data[perm]
                start_idx = start_idx[perm]
            N = data.shape[0]
            ratio_sum = args.train_ratio + args.valid_ratio + args.test_ratio
            if abs(ratio_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"window_random train/valid/test fractions must sum to 1; got {ratio_sum:.6f}"
                )
            n_train = int(N * args.train_ratio)
            n_valid = int(N * args.valid_ratio)
            train_data = data[:n_train]
            valid_data = data[n_train : n_train + n_valid]
            test_data = data[n_train + n_valid :]
            test_window_start_idx = start_idx[n_train + n_valid :]
            split_meta = {
                "split_mode": "window_random",
                "window_length": int(args.window_length),
                "stride": int(args.stride),
                "train_ratio": float(args.train_ratio),
                "valid_ratio": float(args.valid_ratio),
                "test_ratio": float(args.test_ratio),
                "shuffle": bool(args.shuffle),
                "seed": int(args.seed),
                "leakage_risk": True,
                "train_num_windows": int(train_data.shape[0]),
                "valid_num_windows": int(valid_data.shape[0]),
                "test_num_windows": int(test_data.shape[0]),
            }
            with open(out_root / "split_meta.json", "w", encoding="utf-8") as f:
                json.dump(split_meta, f, ensure_ascii=False, indent=2)
            print(
                "[Warning] window_random may leak overlapping windows across partitions; "
                "use --split-mode time_first for the paper's temporal split."
            )

        save_np(out_root / "train_data.npy", train_data)
        save_np(out_root / "valid_data.npy", valid_data)
        if test_data is not None:
            save_np(out_root / "test_data.npy", test_data)
            save_np(
                out_root / "test_window_start_idx.npy",
                test_window_start_idx.astype(np.int64),
            )

        # Optionally export raw test fields in the same order as test_data.
        if raw_data_np is not None and test_window_start_idx is not None:
            idx = test_window_start_idx[:, None] + np.arange(args.window_length, dtype=np.int64)[None, :]
            test_raw_windows = raw_data_np[idx]
            save_np(out_root / "test_raw_windows.npy", test_raw_windows)
            save_np(out_root / "test_raw_window_start_idx.npy", (test_window_start_idx + raw_offset).astype(np.int64))
            if args.split_mode == "time_first":
                test_start, test_end = split_meta["test_time_range"]
                test_raw_segment = raw_data_np[test_start:test_end]
                save_np(out_root / "test_raw_segment.npy", test_raw_segment)
            split_meta["raw_data_npy"] = str(args.raw_data_npy)
            split_meta["raw_time_offset"] = int(raw_offset)
            split_meta["test_window_start_idx_file"] = "test_window_start_idx.npy"
            split_meta["test_raw_window_start_idx_file"] = "test_raw_window_start_idx.npy"
            split_meta["test_raw_windows_file"] = "test_raw_windows.npy"
            if args.split_mode == "time_first":
                split_meta["test_raw_segment_file"] = "test_raw_segment.npy"
            with open(out_root / "split_meta.json", "w", encoding="utf-8") as f:
                json.dump(split_meta, f, ensure_ascii=False, indent=2)
            print(
                f"Saved raw test fields: windows {test_raw_windows.shape}"
                + (
                    f", segment {test_raw_segment.shape}"
                    if args.split_mode == "time_first"
                    else ""
                )
            )
        elif raw_data_np is not None:
            print("[Info] time_train_valid has no test-field windows; skipping raw test export.")

        if test_data is not None:
            print(
                f"train/valid/test window shapes: "
                f"train {train_data.shape}, valid {valid_data.shape}, test {test_data.shape}"
            )
        else:
            print(f"train/valid window shapes: train {train_data.shape}, valid {valid_data.shape}")
        print(f"Saved windowed data to {out_root}")
    else:
        print("No window-length specified; skipping windowing")


if __name__ == "__main__":
    main()
