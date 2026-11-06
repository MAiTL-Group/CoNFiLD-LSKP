"""Original reverse diffusion plus explicit, DNS-independent CNF decoding."""

import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import activate_imports, load_config, load_recipe, require_files


def load_decoder(cfg, device):
    """Load a named CNF checkpoint, its normalizers, and its query coordinates.

    No raw DNS field is loaded, no latent code is fitted, and no 'latest'
    checkpoint is selected. Returned tensors and network live on ``device``.
    """
    activate_imports()
    from ConditionalNeuralField.cnf import nf_networks
    from ConditionalNeuralField.cnf.utils.normalize import Normalizer_ts

    require_files([cfg["cnf_config"], cfg["cnf_checkpoint"], cfg["cnf_normalizer"]])
    if cfg.get("model") in ("origin", "sn", "lskp"):
        chart = "origin" if cfg["model"] == "origin" else "sn"
        cnf = load_recipe(cfg["cnf_config"], "cnf", chart=chart, case=cfg.get("case"))
    else:
        cnf = load_config(cfg["cnf_config"])
    spec = cnf["NF"]
    kwargs = dict(spec.get("kwargs", {}))
    if not kwargs:
        kwargs = {key: spec[key] for key in
                  ("out_features", "num_hidden_layers", "hidden_features")}
        kwargs.update(in_coord_features=cnf["dims"], in_latent_features=cnf["hidden_size"],
                      use_sn=spec.get("use_sn", False), sn_coeff=spec.get("sn_coeff", 1.0),
                      sn_power_iter=spec.get("sn_power_iter", 1))
    model = getattr(nf_networks, spec["name"])(**kwargs)
    checkpoint = torch.load(cfg["cnf_checkpoint"], map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    params = torch.load(cfg["cnf_normalizer"], map_location="cpu")
    x_norm, y_norm = Normalizer_ts(**cnf["normalizer"]), Normalizer_ts(**cnf["normalizer"])
    x_norm.params, y_norm.params = params["x_normalizer_params"], params["y_normalizer_params"]
    if cfg.get("coords"):
        require_files([cfg["coords"]])
        coords = np.load(cfg["coords"], allow_pickle=False)
    else:
        shape = cfg.get("grid_shape")
        if not shape or len(shape) != cnf["dims"] or any(int(n) <= 0 for n in shape):
            raise ValueError("Supply coordinates or a positive grid_shape matching CNF dimensions")
        coords = np.stack(np.meshgrid(*[np.linspace(0, 1, n) for n in shape], indexing="ij"), axis=-1)
    if coords.shape[-1] != cnf["dims"] or coords.ndim not in (2, cnf["dims"] + 1):
        raise ValueError("Query coordinate shape disagrees with the CNF configuration")
    coords = torch.as_tensor(coords, dtype=torch.float32, device=device)
    return model, x_norm, y_norm, coords, int(cnf["hidden_size"])


def decode_fields(latents, cfg, device="cpu", destination=None):
    """Decode [sample,time,latent] codes, optionally streaming fields to an NPY.

    Fields use the original flattened sample/time layout. Default batch size one
    retains the original decoder call order, including SN power iterations.
    """
    model, x_norm, y_norm, coords, dim = load_decoder(cfg, device)
    latents = torch.as_tensor(latents, dtype=torch.float32, device=device)
    if latents.ndim != 3 or latents.shape[-1] != dim:
        raise ValueError("Expected [sample,time,{}] latent codes".format(dim))
    batch_size = int(cfg.get("decode_batch_size", 1))
    if batch_size < 1:
        raise ValueError("decode_batch_size must be positive")
    count, time = latents.shape[:2]
    result = None
    query = x_norm.normalize(coords)
    with torch.no_grad():
        for sample in range(count):
            for start in range(0, time, batch_size):
                block = latents[sample, start:start + batch_size]
                block = block[:, None, None] if coords.ndim > 2 else block[:, None]
                fields = y_norm.denormalize(model(query, block)).cpu().numpy()
                if result is None:
                    shape = (count * time,) + tuple(fields.shape[1:])
                    if destination is None:
                        result = np.empty(shape, dtype=fields.dtype)
                    else:
                        result = np.lib.format.open_memmap(str(destination), mode="w+",
                                                          dtype=fields.dtype, shape=shape)
                offset = sample * time + start
                result[offset:offset + len(fields)] = fields
    if result is None:
        raise ValueError("Cannot decode an empty latent array")
    if isinstance(result, np.memmap):
        result.flush()
    return result


def sample(cfg, device="cuda"):
    """Run unchanged reverse diffusion and decode; write fields, codes, and metadata."""
    activate_imports()
    from src.script_util import create_gaussian_diffusion, create_model

    required = [cfg[key] for key in ("ema_path", "min_val", "max_val", "cnf_config",
                                    "cnf_checkpoint", "cnf_normalizer")]
    if cfg.get("coords"):
        required.append(cfg["coords"])
    require_files(required)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a small smoke test")
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    keys = ("image_size", "num_channels", "num_res_blocks", "num_heads", "num_head_channels",
            "attention_resolutions", "channel_mult")
    model = create_model(**{key: cfg[key] for key in keys if key in cfg})
    model.load_state_dict(torch.load(cfg["ema_path"], map_location="cpu"), strict=True)
    model.to(device).eval()
    diffusion = create_gaussian_diffusion(steps=int(cfg["steps"]), noise_schedule=cfg["noise_schedule"])
    shape = (int(cfg["test_batch_size"]), 1, int(cfg["time_length"]), int(cfg["latent_length"]))
    if any(size <= 0 for size in shape):
        raise ValueError("Sampling dimensions must be positive")
    output = Path(cfg["save_dir"]) / "seed_{:04d}".format(seed)
    output.mkdir(parents=True, exist_ok=True)
    names = ("gen_fields.npy", "gen_latents.npy", "meta.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Refusing to overwrite an existing sample: {}".format(output))
    minimum, maximum = np.load(cfg["min_val"], allow_pickle=False), np.load(cfg["max_val"], allow_pickle=False)
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all() or not (maximum > minimum).all():
        raise ValueError("Invalid latent training min/max statistics")
    with torch.no_grad():
        latents = diffusion.p_sample_loop(model, shape, device=torch.device(device))[:, 0]
        lo = torch.as_tensor(minimum, device=device)
        hi = torch.as_tensor(maximum, device=device)
        latents = (latents + 1) * (hi - lo) / 2.0 + lo
    array = latents.cpu().numpy()
    np.save(output / "gen_latents.npy", array)
    fields = decode_fields(array, cfg, device=device, destination=output / "gen_fields.npy")
    meta = {key: cfg[key] for key in cfg if key != "save_dir"}
    meta.update(seed=seed, test_batch_size=shape[0], time_length=shape[2], latent_length=shape[3],
                cnf_case_file_path=cfg["cnf_config"], field_shape=list(fields.shape),
                save_path=str(output / "gen_fields.npy"), save_latents=str(output / "gen_latents.npy"))
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return output
