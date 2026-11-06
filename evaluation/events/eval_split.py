from __future__ import annotations

import json
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional

from .cli_defaults import DEFAULT_RUN_ARGS, LIST_ARG_KEYS, MODEL_ARG_KEYS, PATH_ARG_KEYS
from .reporting import compare_core_tables, save_json
from .runner import run_motivation_v2


SPLIT_RESERVED_KEYS = {"name", "registry", "preset", "models", "protocol_m"}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(value: Optional[str], base_dir: str) -> Optional[str]:
    if value is None:
        return None
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def _normalize_cli_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key in LIST_ARG_KEYS:
        if isinstance(value, list):
            return ",".join(str(x) for x in value)
        return str(value)
    return value


def _resolve_config_paths(config: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    out = dict(config)
    for key in PATH_ARG_KEYS:
        if key in out:
            out[key] = _resolve_path(out[key], base_dir)
    return out


def load_eval_split(split_path: str) -> Dict[str, Any]:
    split_path = os.path.abspath(split_path)
    base_dir = os.path.dirname(split_path)
    split = _load_json(split_path)

    merged: Dict[str, Any] = {}
    registry_path = split.get("registry")
    preset_name = split.get("preset")
    if registry_path and preset_name:
        registry = _load_json(_resolve_path(registry_path, base_dir))
        presets = registry.get("presets", {})
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' not found in {registry_path}")
        merged.update(presets[preset_name])

    for key, value in split.items():
        if key in SPLIT_RESERVED_KEYS:
            continue
        merged[key] = value

    merged = _resolve_config_paths(merged, base_dir)

    models = []
    for model in split.get("models", []):
        resolved = dict(model)
        for key in PATH_ARG_KEYS.intersection(model.keys()):
            resolved[key] = _resolve_path(model[key], base_dir)
        models.append(resolved)

    protocol_m = []
    for job in split.get("protocol_m", []):
        protocol_m.append(dict(job))

    resolved_split = {
        "name": split.get("name", os.path.splitext(os.path.basename(split_path))[0]),
        "split_path": split_path,
        "base_dir": base_dir,
        "config": merged,
        "models": models,
        "protocol_m": protocol_m,
        "registry": _resolve_path(registry_path, base_dir) if registry_path else None,
        "preset": preset_name,
    }
    return resolved_split


def _build_namespace(common_cfg: Dict[str, Any], model_cfg: Dict[str, Any], out_dir: str) -> Namespace:
    args = dict(DEFAULT_RUN_ARGS)
    args.update(common_cfg)
    args.update({k: v for k, v in model_cfg.items() if k in MODEL_ARG_KEYS})
    args["out_dir"] = out_dir

    for key, value in list(args.items()):
        args[key] = _normalize_cli_value(key, value)

    if not args.get("case"):
        raise ValueError("eval-split requires 'case'")
    if not args.get("label"):
        raise ValueError("eval-split model requires 'label'")
    if not args.get("out_dir"):
        raise ValueError("eval-split requires resolvable 'out_dir'")
    if not args.get("gen_manifest") and not (args.get("gen_fields") and args.get("gen_latents")):
        raise ValueError("eval-split model requires gen_manifest or both gen_fields/gen_latents")
    return Namespace(**args)


def run_eval_split(split_path: str, out_root_override: Optional[str] = None) -> Dict[str, Any]:
    split = load_eval_split(split_path)
    common_cfg = dict(split["config"])
    split_name = split["name"]
    out_root = out_root_override or common_cfg.get("out_root")
    if not out_root:
        raise ValueError("eval-split requires out_root in split or --eval-split-out-root")
    out_root = os.path.abspath(out_root)
    os.makedirs(out_root, exist_ok=True)

    payload = {
        "name": split_name,
        "split_path": split["split_path"],
        "registry": split.get("registry"),
        "preset": split.get("preset"),
        "out_root": out_root,
        "config": common_cfg,
        "models": split["models"],
        "protocol_m": split["protocol_m"],
    }
    save_json(os.path.join(out_root, "eval_split_resolved.json"), payload)

    result_dirs: List[str] = []
    labels: List[str] = []
    model_runs: List[Dict[str, Any]] = []
    for model in split["models"]:
        label = model["label"]
        model_out_dir = os.path.join(out_root, label)
        ns = _build_namespace(common_cfg, model, model_out_dir)
        result = run_motivation_v2(ns)
        result_dirs.append(model_out_dir)
        labels.append(label)
        model_runs.append({
            "label": label,
            "out_dir": model_out_dir,
            "result_manifest": os.path.join(model_out_dir, "result_manifest.json"),
            "status": result.get("status"),
            "entry_count": len(result.get("entries", [])),
        })

    compare_jobs = []
    protocol_m_dir = os.path.join(out_root, "protocol_m")
    os.makedirs(protocol_m_dir, exist_ok=True)
    for job in split["protocol_m"]:
        protocol = job["protocol"]
        horizon = job.get("horizon")
        policies = job.get("policies") or [job.get("policy")]
        for policy in policies:
            if policy is None:
                continue
            out_name = job.get("out_name")
            if out_name:
                out_path = os.path.join(protocol_m_dir, f"{out_name}_{policy}.json")
            else:
                h_token = "none" if horizon is None else f"h{int(horizon):05d}"
                out_path = os.path.join(protocol_m_dir, f"{protocol}_{h_token}_{policy}.json")
            compare_payload = compare_core_tables(
                result_dirs=result_dirs,
                labels=labels,
                protocol=protocol,
                horizon=horizon,
                policy=policy,
                out_path=out_path,
            )
            compare_jobs.append({
                "protocol": protocol,
                "horizon": horizon,
                "policy": policy,
                "out_path": out_path,
                "row_count": len(compare_payload.get("rows", [])),
            })

    result = {
        "status": "ok",
        "name": split_name,
        "out_root": out_root,
        "model_runs": model_runs,
        "protocol_m_outputs": compare_jobs,
    }
    save_json(os.path.join(out_root, "eval_split_result.json"), result)
    return result
