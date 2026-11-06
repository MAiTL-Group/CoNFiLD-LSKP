#!/usr/bin/env python3
"""Select a saved CNF checkpoint from local reconstruction-loss logs."""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as exc:  # pragma: no cover - runtime dependency guard
    EventAccumulator = None


CKPT_RE = re.compile(r"^checkpoint_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select best CNF checkpoint from TensorBoard or SwanLab metrics.")
    p.add_argument("--logdir", type=str, required=True, help="CNF save_path (contains checkpoints + tfevents).")
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "tensorboard", "swanlab"],
        help="Metric backend. auto prefers TensorBoard, then SwanLab.",
    )
    p.add_argument(
        "--swan-backup",
        type=str,
        default=None,
        help="Path to SwanLab backup.swanlab. Required for backend=swanlab if not under logdir.",
    )
    p.add_argument(
        "--metric-prefix",
        type=str,
        default="test_error_",
        help="TensorBoard scalar tag prefix. Default: test_error_",
    )
    p.add_argument(
        "--swan-metric-prefix",
        type=str,
        default="val/test_error_",
        help="SwanLab scalar key prefix. Default: val/test_error_",
    )
    p.add_argument(
        "--cap",
        type=float,
        default=2.0,
        help="Hard cap for channel error. Any channel > cap is filtered out.",
    )
    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.10,
        help="Ignore early steps below this ratio of max step. Default: 0.10",
    )
    p.add_argument(
        "--min-step",
        type=int,
        default=None,
        help="Absolute minimal step to consider. Overrides warmup-ratio when provided.",
    )
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=0.2,
        help="EMA alpha over raw mean score. Default: 0.2",
    )
    p.add_argument(
        "--roll-window",
        type=int,
        default=8,
        help="Window size for rolling std tie-break. Default: 8",
    )
    p.add_argument(
        "--tie-delta",
        type=float,
        default=1.0e-4,
        help="EMA tie threshold. Candidates within min_ema + tie_delta enter tie-break.",
    )
    p.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Output report path. Default: <logdir>/selection_report.json",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of top candidates in report. Default: 10",
    )
    return p.parse_args()


def find_checkpoints(logdir: Path) -> Dict[int, Path]:
    ckpts: Dict[int, Path] = {}
    for fp in logdir.iterdir():
        if not fp.is_file():
            continue
        m = CKPT_RE.match(fp.name)
        if m is None:
            continue
        step = int(m.group(1))
        ckpts[step] = fp
    return ckpts


def find_event_files(logdir: Path) -> List[Path]:
    return sorted(
        [p for p in logdir.iterdir() if p.is_file() and p.name.startswith("events.out.tfevents")]
    )


def load_metrics_from_events(event_files: List[Path], metric_prefix: str) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """
    Returns:
      step_to_channel[channel_id] = (value, wall_time)
    For overlapping event files, retain the record with the later wall_time.
    """
    step_to_channel: Dict[int, Dict[int, Tuple[float, float]]] = {}
    tag_re = re.compile(rf"^{re.escape(metric_prefix)}(\d+)$")

    if EventAccumulator is None:
        raise RuntimeError(
            "tensorboard is required for backend=tensorboard. "
            "Install with `pip install tensorboard`."
        )

    for ev in event_files:
        acc = EventAccumulator(str(ev), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            m = tag_re.match(tag)
            if m is None:
                continue
            cid = int(m.group(1))
            for item in acc.Scalars(tag):
                step = int(item.step)
                val = float(item.value)
                wt = float(item.wall_time)
                if step not in step_to_channel:
                    step_to_channel[step] = {}
                old = step_to_channel[step].get(cid, None)
                if old is None or wt >= old[1]:
                    step_to_channel[step][cid] = (val, wt)
    return step_to_channel


def resolve_swan_backup(logdir: Path, swan_backup_arg: Optional[str]) -> Path:
    if swan_backup_arg:
        p = Path(swan_backup_arg)
    else:
        p = logdir / "backup.swanlab"
    if not p.exists():
        raise FileNotFoundError(
            f"SwanLab backup not found: {p}. "
            "Please pass --swan-backup swanlog/run-xxxx/backup.swanlab"
        )
    return p


def load_metrics_from_swan_backup(
    backup_path: Path, metric_prefix: str
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """
    Returns:
      step_to_channel[channel_id] = (value, order_index)
    When records overlap, later records in read order replace earlier ones.
    """
    step_to_channel: Dict[int, Dict[int, Tuple[float, float]]] = {}
    key_re = re.compile(rf"^{re.escape(metric_prefix)}(\d+)$")

    with backup_path.open("rb") as f:
        for order, raw in enumerate(f):
            s = raw.decode("utf-8", errors="ignore")
            j = s.find('{"model_type"')
            if j < 0:
                continue
            rec = s[j:].strip()
            try:
                obj = json.loads(rec)
            except Exception:
                continue
            if obj.get("model_type") != "Scalar":
                continue
            data = obj.get("data", {})
            key = data.get("key", "")
            m = key_re.match(key)
            if m is None:
                continue
            cid = int(m.group(1))
            step = int(data.get("step", 0))
            metric = data.get("metric", {})
            val = metric.get("data", None)
            if val is None:
                continue
            val = float(val)
            order_float = float(order)
            if step not in step_to_channel:
                step_to_channel[step] = {}
            old = step_to_channel[step].get(cid, None)
            if old is None or order_float >= old[1]:
                step_to_channel[step][cid] = (val, order_float)
    return step_to_channel


def rolling_std(values: List[float], window: int) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    if window <= 1:
        return [0.0 for _ in values]
    for i in range(len(arr)):
        l = max(0, i - window + 1)
        seg = arr[l : i + 1]
        out[i] = float(np.std(seg))
    return out.tolist()


def main() -> None:
    args = parse_args()
    logdir = Path(args.logdir)
    if not logdir.exists():
        raise FileNotFoundError(f"logdir not found: {logdir}")

    ckpts = find_checkpoints(logdir)
    if not ckpts:
        raise RuntimeError(f"No checkpoint_*.pt found in {logdir}")

    backend = args.backend
    event_files: List[Path] = []
    swan_backup: Optional[Path] = None

    if backend in ("auto", "tensorboard"):
        event_files = find_event_files(logdir)
        if backend == "tensorboard" and not event_files:
            raise RuntimeError(f"No TensorBoard event files found in {logdir}")

    if backend == "auto":
        if event_files:
            backend = "tensorboard"
        else:
            backend = "swanlab"

    if backend == "tensorboard":
        raw = load_metrics_from_events(event_files, metric_prefix=args.metric_prefix)
        if not raw:
            raise RuntimeError(
                f"No scalar tags matched prefix '{args.metric_prefix}' in {logdir}"
            )
    elif backend == "swanlab":
        swan_backup = resolve_swan_backup(logdir, args.swan_backup)
        raw = load_metrics_from_swan_backup(
            swan_backup, metric_prefix=args.swan_metric_prefix
        )
        if not raw:
            raise RuntimeError(
                f"No SwanLab scalar keys matched prefix '{args.swan_metric_prefix}' in {swan_backup}"
            )
    else:
        raise RuntimeError(f"Unknown backend: {backend}")

    all_channels = sorted({cid for per_step in raw.values() for cid in per_step.keys()})
    if not all_channels:
        raise RuntimeError("No channel metrics found.")

    # Evaluate only steps with a saved checkpoint.
    common_steps = sorted(set(raw.keys()) & set(ckpts.keys()))
    if not common_steps:
        raise RuntimeError("No overlap between metric steps and checkpoint steps.")

    step_scores: List[Dict] = []
    for step in common_steps:
        channel_map = raw[step]
        # Require all channels at this step.
        if any(cid not in channel_map for cid in all_channels):
            continue
        vals = [float(channel_map[cid][0]) for cid in all_channels]
        if not all(math.isfinite(v) for v in vals):
            continue
        if any(v > args.cap for v in vals):
            continue
        raw_mean = float(np.mean(vals))
        step_scores.append(
            {
                "step": int(step),
                "checkpoint": str(ckpts[step]),
                "channel_values": {str(cid): vals[i] for i, cid in enumerate(all_channels)},
                "raw_mean": raw_mean,
            }
        )

    if not step_scores:
        raise RuntimeError("All candidates filtered out (check cap/channels/event tags).")

    # warmup cutoff
    max_step = max(item["step"] for item in step_scores)
    if args.min_step is not None:
        cutoff = int(args.min_step)
    else:
        cutoff = int(max_step * float(args.warmup_ratio))
    step_scores = [x for x in step_scores if x["step"] >= cutoff]
    if not step_scores:
        raise RuntimeError("No candidates left after warmup/min-step filtering.")

    # Compute EMA and rolling standard deviation in ascending step order.
    step_scores.sort(key=lambda x: x["step"])
    alpha = float(args.ema_alpha)
    ema_vals: List[float] = []
    cur = None
    raw_means = [x["raw_mean"] for x in step_scores]
    for v in raw_means:
        if cur is None:
            cur = v
        else:
            cur = alpha * v + (1.0 - alpha) * cur
        ema_vals.append(float(cur))
    var_vals = rolling_std(raw_means, int(args.roll_window))

    for i in range(len(step_scores)):
        step_scores[i]["ema"] = ema_vals[i]
        step_scores[i]["roll_std"] = var_vals[i]

    # Minimize EMA; break ties by rolling std, raw mean, then later step.
    min_ema = min(x["ema"] for x in step_scores)
    tie = [x for x in step_scores if x["ema"] <= min_ema + float(args.tie_delta)]
    tie.sort(
        key=lambda x: (
            float("inf") if not math.isfinite(x["roll_std"]) else x["roll_std"],
            x["raw_mean"],
            -x["step"],
        )
    )
    best = tie[0]

    # Display and save the selected result.
    ranked = sorted(step_scores, key=lambda x: (x["ema"], x["roll_std"], x["raw_mean"], -x["step"]))
    topk = ranked[: max(1, int(args.topk))]

    report = {
        "logdir": str(logdir),
        "backend": backend,
        "metric_prefix": args.metric_prefix,
        "swan_metric_prefix": args.swan_metric_prefix,
        "swan_backup": None if swan_backup is None else str(swan_backup),
        "channels": all_channels,
        "num_event_files": len(event_files),
        "num_checkpoints": len(ckpts),
        "num_candidates_after_filter": len(step_scores),
        "filter": {
            "cap": float(args.cap),
            "warmup_ratio": float(args.warmup_ratio),
            "min_step": None if args.min_step is None else int(args.min_step),
            "ema_alpha": float(args.ema_alpha),
            "roll_window": int(args.roll_window),
            "tie_delta": float(args.tie_delta),
        },
        "best": best,
        "topk": topk,
    }

    out_json = Path(args.out_json) if args.out_json else (logdir / "selection_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"[CNF] logdir: {logdir}")
    print(f"[CNF] backend: {backend}")
    if swan_backup is not None:
        print(f"[CNF] swan_backup: {swan_backup}")
    print(f"[CNF] channels: {all_channels}")
    print(f"[CNF] candidates(after filter): {len(step_scores)}")
    print(f"[CNF] BEST step={best['step']}, ema={best['ema']:.6f}, raw_mean={best['raw_mean']:.6f}")
    print(f"[CNF] BEST checkpoint: {best['checkpoint']}")
    print(f"[CNF] report: {out_json}")
    print("-" * 80)
    for i, item in enumerate(topk, 1):
        print(
            f"#{i:02d} step={item['step']:>6d}  ema={item['ema']:.6f}  "
            f"raw={item['raw_mean']:.6f}  std={item['roll_std']:.6f}  ckpt={Path(item['checkpoint']).name}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
