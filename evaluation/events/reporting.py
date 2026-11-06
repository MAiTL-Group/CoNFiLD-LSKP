from __future__ import annotations

import csv
import json
import os
from typing import Dict, Iterable, List, Optional

from .common import ensure_dir, to_builtin

CORE_KEYS = [
    "late_tail_Tint_gap",
    "tail_w1_time",
    "w1_time_lag_1",
    "w1_time_lag_16",
    "S6_drift",
    "enstrophy_gap",
    "Err_logL2",
]


def save_json(path: str, payload: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, ensure_ascii=False, indent=2)


def build_core_table(entries: Iterable[Dict]) -> List[Dict]:
    rows = []
    for entry in entries:
        row = {
            "protocol": entry["protocol"],
            "label": entry["label"],
            "policy": entry.get("policy"),
            "horizon": entry.get("horizon"),
        }
        for key in CORE_KEYS:
            row[key] = entry["core"][key]
        rows.append(row)
    return rows


def save_core_table(out_dir: str, rows: List[Dict]) -> None:
    save_json(os.path.join(out_dir, "core_table.json"), {"rows": rows, "core_keys": CORE_KEYS})
    if not rows:
        return
    csv_path = os.path.join(out_dir, "core_table.csv")
    ensure_dir(os.path.dirname(csv_path))
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_core_table(result_dir: str) -> List[Dict]:
    path = os.path.join(result_dir, "core_table.json")
    payload = json.loads(open(path, "r", encoding="utf-8").read())
    return payload.get("rows", [])


def select_core_row(rows: List[Dict], protocol: str, horizon: Optional[int] = None, policy: Optional[str] = None, label: Optional[str] = None) -> Dict:
    matches = []
    for row in rows:
        if row["protocol"] != protocol:
            continue
        if horizon is not None and int(row.get("horizon") or -1) != int(horizon):
            continue
        if policy is not None and row.get("policy") != policy:
            continue
        if label is not None and row.get("label") != label:
            continue
        matches.append(row)
    if not matches:
        raise ValueError(f"No row found for protocol={protocol}, horizon={horizon}, policy={policy}, label={label}")
    if len(matches) > 1:
        raise ValueError(f"Multiple rows found for protocol={protocol}, horizon={horizon}, policy={policy}, label={label}")
    return matches[0]


def compare_core_tables(result_dirs: List[str], labels: List[str], protocol: str, horizon: Optional[int], policy: Optional[str], out_path: str) -> Dict:
    rows = []
    for result_dir, label in zip(result_dirs, labels):
        table = load_core_table(result_dir)
        row = select_core_row(table, protocol=protocol, horizon=horizon, policy=policy)
        row = dict(row)
        row["result_dir"] = result_dir
        row["label"] = label
        rows.append(row)
    payload = {
        "protocol": protocol,
        "horizon": horizon,
        "policy": policy,
        "core_keys": CORE_KEYS,
        "rows": rows,
    }
    save_json(out_path, payload)
    return payload
