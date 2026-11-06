"""Resolve portable recipes without depending on the caller's working directory."""

import sys
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = {
    "data_path", "coor_path", "save_path", "save_dir", "log_path",
    "train_data_path", "valid_data_path", "train_npy", "valid_npy",
    "checkpoint_path", "resume_checkpoint", "ema_path", "min_val", "max_val",
    "cnf_config", "cnf_checkpoint", "cnf_normalizer", "coords", "raw_data_npy",
    "reference_metrics_path",
}


def activate_imports():
    """Add only this checkout and its vendored runtime roots to Python's path."""
    for suffix in ("UnconditionalDiffusionTraining_and_Generation", "ConditionalNeuralField", ""):
        value = str(ROOT / suffix)
        if value not in sys.path:
            sys.path.insert(0, value)


def resolve_path(value, root=ROOT):
    """Expand a path; relative recipe paths are relative to the release root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(root) / path


def resolve_config(cfg, root=ROOT):
    """Return a mapping with known path fields expanded for the local checkout."""
    def resolve(obj):
        if isinstance(obj, dict):
            return {key: str(resolve_path(value, root))
                    if key in PATH_KEYS and isinstance(value, str) and value
                    else resolve(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [resolve(value) for value in obj]
        return obj

    return resolve(cfg)


def _read_mapping(path, root):
    with resolve_path(path, root).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Recipe must contain a YAML mapping")
    return cfg


def load_config(path, root=ROOT):
    """Load a YAML mapping and expand paths, without applying run variants."""
    return resolve_config(_read_mapping(path, root), root)


def apply_overrides(config, overrides):
    """Copy a recipe and apply explicit dotted-key YAML values without file writes.

    Existing root keys and the optional generic_regularizer section are allowed.
    Nested fields may be added to existing mappings for documented runtime knobs.
    A scalar cannot be traversed; misspelled root keys fail instead of being ignored.
    """
    cfg = deepcopy(config)
    for expression in overrides or []:
        if "=" not in expression:
            raise ValueError("Override requires dotted.key=value: " + expression)
        key, raw = expression.split("=", 1)
        parts = key.split(".")
        if not all(parts) or parts[0] not in set(cfg) | {"generic_regularizer", "train_seed"}:
            raise ValueError("Unknown or empty override key: " + key)
        current = cfg
        for part in parts[:-1]:
            if part not in current and part == "generic_regularizer" and current is cfg:
                current[part] = {}
            if not isinstance(current.get(part), dict):
                raise ValueError("Override parent must be an existing mapping: " + key)
            current = current[part]
        current[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_recipe(path, stage, model=None, chart=None, training_seed=None, root=ROOT, case=None,
                overrides=None):
    """Apply an optional named variant before expanding paths; never write files."""
    cfg = _read_mapping(path, root)
    if model is not None or chart is not None or training_seed is not None:
        from .variants import apply_variant
        name = case or cfg.get("case") or Path(path).stem
        cfg = apply_variant(cfg, stage, name, model=model, chart=chart, training_seed=training_seed)
    return resolve_config(apply_overrides(cfg, overrides), root)


def require_files(paths):
    """Fail before computation if any explicitly required file is missing."""
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))
