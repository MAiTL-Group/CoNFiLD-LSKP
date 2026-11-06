"""Aggregate freshly computed metrics without changing selection or atomization."""

import json
import math
import statistics
from pathlib import Path


METRICS = ["w1_time_lag_1", "w1_time_lag_16", "Tint_gap", "Err_logL2"]


def mean_std(values):
    """Report sample SD across independent training runs, not evaluation windows."""
    values = list(values)
    return {"mean": statistics.mean(values), "std": statistics.stdev(values),
            "min": min(values), "max": max(values), "n_training_runs": len(values)}


def summarize_metrics(records, task, metrics=None):
    """Read all declared cells; require complete seed or common-step designs.

    Each record contains model and a fresh bootstrap_core.json path. Training-seed
    records additionally contain train_seed; common-step records contain step.
    Window intervals are copied per cell and never pooled as training replicates.
    """
    metrics = metrics or METRICS
    if not records:
        raise ValueError("Metric manifest has no runs")
    axis = {"training-seeds": "train_seed", "common-steps": "step"}.get(task)
    expected = None
    if task == "training-seeds":
        expected = {(m, s) for m in ("sn", "lskp") for s in (42, 1042, 2042)}
    elif task == "common-steps":
        expected = {(m, s) for m in ("origin", "sn", "lskp") for s in (50000, 100000, 155000)}
    keys = [(r["model"], int(r[axis])) if axis else (r["model"], r.get("label", "")) for r in records]
    if len(keys) != len(set(keys)) or (expected is not None and set(keys) != expected):
        raise ValueError("Duplicate or incomplete comparison cells for " + task)
    if expected is not None and len({str(Path(r["metrics"]).resolve()) for r in records}) != len(records):
        raise ValueError("Each locked comparison cell requires its own evaluation file")
    rows, values = [], {}
    for record, key in zip(records, keys):
        payload = json.loads(Path(record["metrics"]).read_text())
        values[key] = {}
        for metric in metrics:
            entry = payload[metric]
            value = float(entry["mean"])
            if not math.isfinite(value) or int(entry["n"]) < 1:
                raise ValueError("Non-finite or empty metric cell: {} {}".format(key, metric))
            if expected is not None and int(entry["n"]) != 57:
                raise ValueError("Locked Case3 protocol requires 57 evaluation windows per cell")
            values[key][metric] = value
            row = {"model": record["model"], "metric": metric, "mean": value,
                   "ci_low": entry["ci_low"], "ci_high": entry["ci_high"],
                   "evaluation_atoms": int(entry["n"]), "metrics_source": str(record["metrics"])}
            if axis:
                row[axis] = int(record[axis])
            else:
                row["label"] = record.get("label", "")
            rows.append(row)
    result = {"rows": rows, "comparison": task,
              "window_intervals": "per-cell window bootstrap; not training-run intervals"}
    if task == "training-seeds":
        result["groups"] = {model: {metric: mean_std(values[model, seed][metric]
                                             for seed in (42, 1042, 2042)) for metric in metrics}
                            for model in ("sn", "lskp")}
        result["paired_differences"] = {}
        for metric in metrics:
            differences = [values["lskp", s][metric] - values["sn", s][metric] for s in (42, 1042, 2042)]
            result["paired_differences"][metric] = dict(mean_std(differences),
                                                       lskp_lower_count=sum(x < 0 for x in differences))
        result["scope"] = "diffusion-training seeds with fixed chart/prior; not end-to-end refits"
    elif task == "common-steps":
        result["lskp_lower_than_both"] = {
            str(step): sum(values["lskp", step][metric] < min(values["sn", step][metric],
                          values["origin", step][metric]) for metric in metrics)
            for step in (50000, 100000, 155000)}
        result["scope"] = "all fixed steps; not matched validation cadence or a new checkpoint selector"
    return result
