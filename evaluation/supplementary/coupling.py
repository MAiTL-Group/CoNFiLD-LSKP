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

import yaml
TABLE_RE = re.compile(r"^\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*$")

def parse_float(text: str) -> Optional[float]:
    text = text.strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_log(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    blocks: List[Dict[str, float]] = []
    cur: Dict[str, float] = {}
    for raw in path.read_text(errors="ignore").splitlines():
        m = TABLE_RE.match(raw)
        if m:
            key, val = m.group(1).strip(), m.group(2).strip()
            f = parse_float(val)
            if f is not None:
                cur[key] = f
            continue
        if raw.startswith("---") and cur:
            blocks.append(cur)
            cur = {}
    if cur:
        blocks.append(cur)
    # Keep only blocks with a step, and sort in logged order. Do not de-duplicate:
    # a validation block and the following training block can share adjacent steps.
    return [b for b in blocks if "step" in b]


def median(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return None
    return float(stats.median(xs))


def mean(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def pct(vals: Iterable[Optional[float]], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in vals if v is not None and math.isfinite(float(v)))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    idx = (len(xs) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)


def load_recipe(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_best(run_dir: Path) -> Dict:
    p = run_dir / "best_report.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def summarize_run(run: Dict, root: Path) -> Dict:
    recipe = load_recipe(root / run["recipe"])
    koop = recipe.get("koopman", {}) if isinstance(recipe, dict) else {}
    run_dir = root / run["run_dir"]
    blocks = parse_log(run_dir / "log.txt")
    best = load_best(run_dir)

    warmup = int(koop.get("warmup_steps", 0) or 0)
    target_step = int(run.get("target_step", 0) or 0)
    lambda_base = float(koop.get("lambda_base", 0.0) or 0.0)
    ratio_cap = float(koop.get("ratio_cap_vs_diff", 0.0) or 0.0)
    interval = int(koop.get("koop_interval", 1) or 1)
    subbatch = int(koop.get("koop_subbatch", 0) or 0)

    latest_step = int(max([b.get("step", 0) for b in blocks], default=0))
    post = [b for b in blocks if b.get("step", 0) >= warmup]
    active = [b for b in post if b.get("koopman/loss_prior_raw", 0.0) > 0.0]
    active_sorted = sorted(active, key=lambda b: b.get("step", 0))
    tail_n = max(1, min(30, len(active_sorted) // 5 or len(active_sorted))) if active_sorted else 0
    tail = active_sorted[-tail_n:] if tail_n else []

    def col(blocks_: List[Dict[str, float]], key: str) -> List[Optional[float]]:
        return [b.get(key) for b in blocks_]

    # Logged prior contribution is approximate because logs expose loss_prior_raw,
    # while lambda may be computed from an effective weighted prior term. It is still
    # a useful scale proxy across runs.
    prior_contrib = []
    prior_to_train = []
    cap_fraction = []
    for b in active_sorted:
        lam = b.get("koopman/lambda_eff")
        raw = b.get("koopman/loss_prior_raw")
        train = b.get("train/mse") or b.get("train/loss")
        if lam is not None and raw is not None:
            pc = lam * raw
            prior_contrib.append(pc)
            if train and train > 0:
                prior_to_train.append(pc / train)
        if lam is not None and lambda_base > 0:
            cap_fraction.append(lam / lambda_base)

    tail_prior_contrib = []
    tail_prior_to_train = []
    tail_cap_fraction = []
    for b in tail:
        lam = b.get("koopman/lambda_eff")
        raw = b.get("koopman/loss_prior_raw")
        train = b.get("train/mse") or b.get("train/loss")
        if lam is not None and raw is not None:
            pc = lam * raw
            tail_prior_contrib.append(pc)
            if train and train > 0:
                tail_prior_to_train.append(pc / train)
        if lam is not None and lambda_base > 0:
            tail_cap_fraction.append(lam / lambda_base)

    cap_limited_tail = None
    med_tail_cap = median(tail_cap_fraction)
    if med_tail_cap is not None:
        cap_limited_tail = med_tail_cap < 0.75

    strength_label = "unknown"
    med_tail_ratio = median(tail_prior_to_train)
    med_tail_active = median(col(tail, "koopman/active_frac"))
    med_tail_sub = median(col(tail, "koopman/subbatch_frac"))
    if med_tail_ratio is not None:
        if med_tail_ratio < 0.01:
            strength_label = "very_weak_logged_scale"
        elif med_tail_ratio < 0.025:
            strength_label = "conservative"
        else:
            strength_label = "near_configured_cap"
    if run["role"] == "old_multiregime_unstable":
        strength_label = "pathological_prior_scale"

    return {
        "run_id": run["run_id"],
        "case": run["case"],
        "role": run["role"],
        "status": run["status"],
        "recipe": run["recipe"],
        "run_dir": run["run_dir"],
        "target_step": target_step,
        "latest_step": latest_step,
        "progress_frac": latest_step / target_step if target_step else None,
        "best_step": best.get("step"),
        "best_metric_raw": best.get("metric_raw"),
        "best_metric_ema": best.get("metric_ema"),
        "warmup_steps": warmup,
        "koop_interval": interval,
        "interval_duty": 1.0 / interval if interval > 0 else None,
        "koop_subbatch": subbatch,
        "sigma_max": koop.get("sigma_max"),
        "lambda_base": lambda_base,
        "ratio_cap_vs_diff": ratio_cap,
        "selective_conf_threshold": koop.get("selective_conf_threshold"),
        "selective_weight_scale": koop.get("selective_weight_scale"),
        "selective_burst_conf_threshold": koop.get("selective_burst_conf_threshold"),
        "selective_burst_scale": koop.get("selective_burst_scale"),
        "n_log_blocks": len(blocks),
        "n_post_warmup_blocks": len(post),
        "n_active_blocks": len(active_sorted),
        "median_lambda_eff_all": median(col(active_sorted, "koopman/lambda_eff")),
        "median_lambda_eff_tail": median(col(tail, "koopman/lambda_eff")),
        "median_cap_fraction_all": median(cap_fraction),
        "median_cap_fraction_tail": median(tail_cap_fraction),
        "cap_limited_tail": cap_limited_tail,
        "median_loss_prior_raw_all": median(col(active_sorted, "koopman/loss_prior_raw")),
        "median_loss_prior_raw_tail": median(col(tail, "koopman/loss_prior_raw")),
        "p90_loss_prior_raw_all": pct(col(active_sorted, "koopman/loss_prior_raw"), 0.90),
        "median_dyn_rel_all": median(col(active_sorted, "koopman/dyn_rel")),
        "median_dyn_rel_tail": median(col(tail, "koopman/dyn_rel")),
        "p90_dyn_rel_all": pct(col(active_sorted, "koopman/dyn_rel"), 0.90),
        "median_active_frac_tail": med_tail_active,
        "median_dyn_conf_active_frac_tail": median(col(tail, "koopman/dyn_conf_active_frac")),
        "median_subbatch_frac_tail": med_tail_sub,
        "median_train_mse_tail": median(col(tail, "train/mse")),
        "median_prior_contrib_all": median(prior_contrib),
        "median_prior_contrib_tail": median(tail_prior_contrib),
        "median_prior_to_train_mse_all": median(prior_to_train),
        "median_prior_to_train_mse_tail": med_tail_ratio,
        "strength_label": strength_label,
    }
