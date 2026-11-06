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

TABLE_ROW = re.compile(r"^\|\s*(.*?)\s*\|\s*([^|]+?)\s*\|$")
BEST_UPDATE = re.compile(r"^\[best\] updated step=(\d+), raw=([^,]+), ema=([^,]+), file=(.+)$")
BEST_RESUME = re.compile(r"^\[best/resume\] restored previous best from .*: step=(\d+), raw=([^,]+), ema=(.+)$")


def _number(value):
    if value is None:
        return None
    try:
        result = float(value.strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_log(path):
    """Return logged validation rows and accepted best-update events."""
    if not path.is_file():
        return [], []

    rows = []
    updates = []
    resumes = []
    block = {}

    def flush():
        if "step" in block and "valid_diff_loss" in block:
            step = _number(block["step"])
            loss = _number(block["valid_diff_loss"])
            if step is not None and loss is not None:
                rows.append(
                    {
                        "step": int(round(step)),
                        "valid_diff_loss": loss,
                        "best_ema_shown": _number(block.get("best/ema")),
                        "best_raw_shown": _number(block.get("best/raw")),
                        "best_step_shown": _number(block.get("best/step")),
                    }
                )
        block.clear()

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line and set(line) == {"-"}:
                flush()
                continue
            match = TABLE_ROW.match(line)
            if match:
                block[match.group(1).strip()] = match.group(2).strip()
                continue
            match = BEST_UPDATE.match(line)
            if match:
                updates.append(
                    {
                        "step": int(match.group(1)),
                        "metric_raw": _number(match.group(2)),
                        "metric_ema": _number(match.group(3)),
                        "filename": match.group(4).strip(),
                    }
                )
                continue
            match = BEST_RESUME.match(line)
            if match:
                resumes.append(
                    {
                        "step": int(match.group(1)),
                        "metric_raw": _number(match.group(2)),
                        "metric_ema": _number(match.group(3)),
                    }
                )
    flush()
    return rows, updates, resumes


def load_best_report(path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
