# CoNFiLD-LSKP modification notice (2026):
# Added local prior integration, configuration, seeding and logging.
# Derived from CoNFiLD; see third_party_licenses/README.md for attribution.
#Imports
import sys
import os
import json
import random
import torch
import torch.distributed as dist
import numpy as np
from src.script_util import create_model, create_gaussian_diffusion
from src.train_util import TrainLoop
from src.koopman_regularizer_v6 import validate_path1_v6_recipe
from src.runtime_config import configure_torch_runtime
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from src.dist_util import setup_dist, dev
from src.logger import configure, log, set_swanlab_run
try:  # optional experiment logger
    import swanlab  # type: ignore
except ImportError:
    swanlab = None
from basicutility import ReadInput as ri

# Optional split Koopman autoencoder backend.
from Koopman.path1.koopman_ae_split import KoopmanAESplit as KoopmanAEModel

inp = ri.basic_input(sys.argv[1])

## Setup
train_seed = int(getattr(inp, "train_seed", 42))
random.seed(train_seed)
np.random.seed(train_seed)
torch.manual_seed(train_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(train_seed)
ORIGINAL_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")

use_mpi = bool(getattr(inp, "use_mpi", True))
setup_dist(use_mpi=use_mpi)
rank = dist.get_rank() if dist.is_initialized() else 0
configure(dir=inp.log_path, format_strs=["stdout","log","tensorboard_new"])
runtime_cfg = getattr(inp, "runtime", {}) if hasattr(inp, "runtime") else {}
runtime_state = configure_torch_runtime(runtime_cfg)
log(f"training seed: {train_seed}")
log(f"distributed launch: {'MPI' if use_mpi else 'single process'}")
log(f"torch runtime: {runtime_state}")
# optional SwanLab experiment logging
enable_swanlab = getattr(inp, "enable_swanlab", False)
swanlab_run = None
if enable_swanlab and rank == 0:
    if swanlab is None:
        log("swanlab is not installed, ignore enable_swanlab flag")
    else:
        def _sanitize_cfg(obj):
            if isinstance(obj, dict):
                return {k: _sanitize_cfg(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize_cfg(v) for v in obj]
            if isinstance(obj, np.generic):
                return obj.item()
            return obj

        swanlab_kwargs = {
            "project": getattr(inp, "swanlab_project", "CoNFiLD-CNF-origin"),
            "experiment_name": getattr(inp, "swanlab_run_name", None) or inp.log_path,
            "tags": getattr(inp, "swanlab_tags", None),
            "description": getattr(inp, "swanlab_notes", None),
            "resume": getattr(inp, "swanlab_resume", False),
            "id": getattr(inp, "swanlab_resume_id", None),
        }
        swanlab_kwargs = {k: v for k, v in swanlab_kwargs.items() if v is not None}
        try:
            swanlab_run = swanlab.init(**swanlab_kwargs)
        except TypeError:
            # older API may use name instead of experiment_name
            if "experiment_name" in swanlab_kwargs:
                swanlab_kwargs["name"] = swanlab_kwargs.pop("experiment_name")
            swanlab_run = swanlab.init(**swanlab_kwargs)
        if swanlab_run is not None:
            cfg_dict = ri.read_yaml(inp.yamlpath)
            if cfg_dict:
                cfg_dict = _sanitize_cfg(cfg_dict)
                cfg_api = getattr(swanlab_run, "config", None)
                if cfg_api is not None:
                    update_fn = getattr(cfg_api, "update", None)
                    if callable(update_fn):
                        update_fn(cfg_dict)
                    else:
                        try:
                            cfg_api.update(cfg_dict)
                        except Exception:
                            pass
                else:
                    try:
                        swanlab_run.config = cfg_dict
                    except Exception:
                        pass
elif enable_swanlab:
    log("swanlab logging is enabled but will only run on rank 0; skipping on this rank")
set_swanlab_run(swanlab_run if rank == 0 else None)

## HyperParams (Change according to the case)
batch_size = inp.batch_size
test_batch_size = inp.test_batch_size
valid_t_seed = getattr(inp, "valid_t_seed", 42)
validation_rng_cfg = (
    getattr(inp, "validation_rng", {}) if hasattr(inp, "validation_rng") else {}
)
profile_memory_path = getattr(inp, "profile_memory_path", "")
profile_memory_reset_before_loop = bool(getattr(inp, "profile_memory_reset_before_loop", True))
disable_checkpoint_saving = bool(getattr(inp, "disable_checkpoint_saving", False))

image_size= inp.image_size
num_channels= inp.num_channels
num_res_blocks= inp.num_res_blocks
num_heads=inp.num_heads
num_head_channels= inp.num_head_channels
attention_resolutions= inp.attention_resolutions
channel_mult = inp.channel_mult

steps= inp.steps
noise_schedule= inp.noise_schedule

microbatch= inp.microbatch
lr = inp.lr
ema_rate= inp.ema_rate
log_interval= inp.log_interval
save_interval= inp.save_interval
lr_anneal_steps= inp.lr_anneal_steps
max_train_steps = getattr(inp, "max_train_steps", 0)
valid_interval = getattr(inp, "valid_interval", 1)
resume_checkpoint = getattr(inp, "resume_checkpoint", "")
resume_optimizer = getattr(inp, "resume_optimizer", True)
resume_ema = getattr(inp, "resume_ema", True)
resume_step_override = getattr(inp, "resume_step_override", None)
training_schedule_cfg = getattr(inp, "training_schedule", {}) if hasattr(inp, "training_schedule") else {}
selection_cfg = getattr(inp, "selection", {}) if hasattr(inp, "selection") else {}
checkpoint_policy_cfg = (
    getattr(inp, "checkpoint_policy", {}) if hasattr(inp, "checkpoint_policy") else {}
)
training_trace_cfg = (
    getattr(inp, "training_trace", {}) if hasattr(inp, "training_trace") else {}
)
schedule_mode = str((training_schedule_cfg or {}).get("mode", "")).lower()
path_schedule_enabled = schedule_mode in {"path1_v2", "path1_v3", "path1_v4", "path1_v5"}
if path_schedule_enabled:
    total_steps_override = int((training_schedule_cfg or {}).get("total_steps", 0) or 0)
    if total_steps_override <= 0:
        raise ValueError(f"{schedule_mode or 'path1'} recipe requires training_schedule.total_steps > 0")
    max_train_steps = total_steps_override
    best_ckpt_cfg = {}
else:
    best_ckpt_cfg = getattr(inp, "best_ckpt", {}) if hasattr(inp, "best_ckpt") else {}

## Data Preprocessing
train_data = np.load(inp.train_data_path)
valid_data = np.load(inp.valid_data_path)
data_dim_x = int(train_data.shape[-1])

max_val, min_val = np.max(train_data, keepdims=True), np.min(train_data, keepdims=True)
# persist min/max for inference denormalization
if getattr(inp, "max_val", None) and getattr(inp, "min_val", None):
    max_dir = os.path.dirname(inp.max_val)
    min_dir = os.path.dirname(inp.min_val)
    if max_dir:
        os.makedirs(max_dir, exist_ok=True)
    if min_dir:
        os.makedirs(min_dir, exist_ok=True)
    np.save(inp.max_val, max_val)
    np.save(inp.min_val, min_val)
    log(f"saved max/min to {inp.max_val} and {inp.min_val}")

norm_train_data = -1 + (train_data - min_val)*2. / (max_val - min_val)
norm_valid_data = -1 + (valid_data - min_val)*2. / (max_val - min_val)

norm_train_data = torch.from_numpy(norm_train_data[:, None, ...])
norm_valid_data = torch.from_numpy(norm_valid_data[:, None, ...])

log("creating data loader...")

world_size = dist.get_world_size() if dist.is_initialized() else 1
# path-lite distilled skeleton mode needs global train indices for reliable-mask lookup.
koopman_cfg = getattr(inp, "koopman", {}) if hasattr(inp, "koopman") else {}
generic_regularizer_cfg = (
    getattr(inp, "generic_regularizer", {}) if hasattr(inp, "generic_regularizer") else {}
)
koopman_mode = str(koopman_cfg.get("mode", "")).lower()
distilled_mode = bool(koopman_cfg.get("enabled", False)) and koopman_mode == "distilled_skeleton"
path1_v6_mode = bool(koopman_cfg.get("enabled", False)) and koopman_mode == "path1_v6_calibrated"
if distilled_mode:
    train_ds = TensorDataset(
        norm_train_data,
        torch.arange(norm_train_data.shape[0], dtype=torch.long),
    )
else:
    train_ds = TensorDataset(norm_train_data)
valid_ds = TensorDataset(
    norm_valid_data,
    torch.arange(norm_valid_data.shape[0], dtype=torch.long),
)
train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None

dl_train = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=train_sampler is None,
    sampler=train_sampler,
)
valid_loader_generator = None
if bool((validation_rng_cfg or {}).get("enabled", False)):
    if "loader_seed" not in validation_rng_cfg:
        raise ValueError("validation_rng.loader_seed is required when enabled")
    valid_loader_generator = torch.Generator(device="cpu")
    valid_loader_generator.manual_seed(int(validation_rng_cfg["loader_seed"]))

dl_valid = DataLoader(
    valid_ds,
    batch_size=test_batch_size,
    shuffle=False,
    generator=valid_loader_generator,
)

def dl_iter(dl, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dl:
            yield batch
        epoch += 1

## Unet Model
log("creating model and diffusion...")

unet_model = create_model(image_size=image_size,
                          num_channels= num_channels,
                          num_res_blocks= num_res_blocks,
                          num_heads=num_heads,
                          num_head_channels=num_head_channels,
                          attention_resolutions=attention_resolutions,
                          channel_mult=channel_mult
                        )

unet_model.to(dev())

## Gaussian Diffusion
diff_model = create_gaussian_diffusion(steps=steps,
                                       noise_schedule=noise_schedule
                                    )

## Optional KoopmanAE.
koopman_enabled = koopman_cfg.get("enabled", False)
koopman_model = None
koopman_prior_artifact = None
koopman_norm = None
if koopman_enabled and (not distilled_mode):
    ckpt_path = koopman_cfg.get("checkpoint_path", None)
    if ckpt_path is None:
        raise ValueError("koopman.enabled=True requires checkpoint_path")
    prior_backend = str(koopman_cfg.get("prior_backend", "split_ae")).lower()
    if path1_v6_mode and prior_backend in {"pkcp", "hokp", "lskp"}:
        log(f"loading {prior_backend.upper()} prior artifact from {ckpt_path}")
        artifact = torch.load(ckpt_path, map_location=dev())
        artifact_backend = str(artifact.get("prior_backend", "")).lower()
        if artifact_backend != prior_backend:
            raise ValueError(f"expected {prior_backend} prior artifact, got prior_backend={artifact_backend}")
        ckpt_dim_x = int(artifact.get("dim_input", 0) or 0)
        if ckpt_dim_x != data_dim_x:
            raise ValueError(
                f"{prior_backend.upper()} prior artifact dimension does not match diffusion latents: "
                f"artifact dim_input={ckpt_dim_x}, train_data dim_x={data_dim_x}, "
                f"checkpoint_path={ckpt_path}, train_data_path={inp.train_data_path}"
            )
        ckpt_repr_mode = str(artifact.get("repr_mode", "absolute")).lower()
        recipe_repr_mode = koopman_cfg.get("repr_mode", None)
        if recipe_repr_mode is None or str(recipe_repr_mode).lower() == "auto":
            koopman_cfg = dict(koopman_cfg)
            koopman_cfg["repr_mode"] = ckpt_repr_mode
            log(f"{prior_backend.upper()} repr_mode inherited from artifact: {ckpt_repr_mode}")
        elif str(recipe_repr_mode).lower() != ckpt_repr_mode:
            raise ValueError(
                f"{prior_backend.upper()} repr_mode mismatch: recipe={str(recipe_repr_mode).lower()}, artifact={ckpt_repr_mode}"
            )
        alignment_stats = artifact.get("alignment_stats", {}) or {}
        koopman_norm = {
            "mean": alignment_stats.get("data_mean", None),
            "std": alignment_stats.get("data_std", None),
            "repr_mean": alignment_stats.get("repr_mean", None),
            "repr_std": alignment_stats.get("repr_std", None),
            "min": min_val,
            "max": max_val,
        }
        koopman_prior_artifact = artifact
        validated_v6 = validate_path1_v6_recipe(
            koopman_cfg=koopman_cfg,
            time_length=int(getattr(inp, "time_length", 0) or 0),
            checkpoint_repr_mode=ckpt_repr_mode,
            checkpoint_delay_l=0,
        )
        log(
            f"[path1_v6/{prior_backend}] validated recipe: "
            f"prior_variant={validated_v6['prior_variant']}, "
            f"repr_mode={validated_v6['repr_mode']}, "
            f"window_size={validated_v6['window_size']}, "
            f"temporal_stride={validated_v6['temporal_stride']}, "
            f"time_length={validated_v6['time_length']}"
        )
        log(f"{prior_backend.upper()} prior artifact loaded.")
    else:
        log(f"loading KoopmanAE checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=dev())
        ckpt_dim_x = ckpt.get("dim_x", None)
        if ckpt_dim_x is None:
            raise ValueError("Koopman checkpoint is missing dim_x; cannot validate latent dimensions")
        ckpt_dim_x = int(ckpt_dim_x)
        if ckpt_dim_x != data_dim_x:
            raise ValueError(
                "Koopman checkpoint dimension does not match diffusion latents: "
                f"ckpt dim_x={ckpt_dim_x}, train_data dim_x={data_dim_x}, "
                f"checkpoint_path={ckpt_path}, train_data_path={inp.train_data_path}"
            )
        args = ckpt.get("args", {})
        mcfg = ckpt.get("model_config", {}) or {}
        tcfg = ckpt.get("train_config", {}) or {}
        ckpt_repr_mode = str(tcfg.get("repr_mode", "absolute")).lower()
        ckpt_repr_mean = ckpt.get("repr_mean", None)
        ckpt_repr_std = ckpt.get("repr_std", None)
        recipe_repr_mode = koopman_cfg.get("repr_mode", None)
        if recipe_repr_mode is None or str(recipe_repr_mode).lower() == "auto":
            if ckpt_repr_mode != "absolute":
                koopman_cfg = dict(koopman_cfg)
                koopman_cfg["repr_mode"] = ckpt_repr_mode
                log(f"Koopman repr_mode inherited from checkpoint: {ckpt_repr_mode}")
            else:
                koopman_cfg = dict(koopman_cfg)
                koopman_cfg["repr_mode"] = "absolute"
        else:
            recipe_repr_mode = str(recipe_repr_mode).lower()
            if recipe_repr_mode != ckpt_repr_mode:
                raise ValueError(
                    f"Koopman repr_mode mismatch: recipe={recipe_repr_mode}, checkpoint={ckpt_repr_mode}"
                )
        if ckpt_repr_mode != "absolute" and (ckpt_repr_mean is None or ckpt_repr_std is None):
            raise ValueError(
                "Koopman checkpoint uses a non-absolute repr_mode but lacks repr_mean/std; "
                f"retrain or use a checkpoint with representation normalization. checkpoint_path={ckpt_path}"
            )
        dim_z = mcfg.get("dim_z") or args.get("dim_z")
        dim_z_dyn = mcfg.get("dim_z_dyn")
        if dim_z is None or dim_z_dyn is None:
            raise ValueError(
                "Koopman checkpoint is not compatible with split model: "
                "missing model_config.dim_z or model_config.dim_z_dyn"
            )
        koopman_model = KoopmanAEModel(
            dim_x=ckpt_dim_x,
            dim_z=dim_z,
            dim_z_dyn=dim_z_dyn,
            hidden_dims_enc=mcfg.get("hidden_dims_enc") or args.get("hidden_enc"),
            hidden_dims_dec=mcfg.get("hidden_dims_dec") or args.get("hidden_dec"),
            alpha=mcfg.get("alpha", args.get("alpha", 0.99)),
            sn_coeff=mcfg.get("sn_coeff", args.get("sn_coeff", 1.0)),
            sn_power_iter=mcfg.get("sn_power_iter", args.get("sn_power_iter", 1)),
            orthogonal_init=mcfg.get("orthogonal_init", True),
            delay_L=mcfg.get("delay_L", 0),
        ).to(dev())
        koopman_model.load_state_dict(ckpt["model_state_dict"])
        koopman_model.eval()
        koopman_model.requires_grad_(False)
        if hasattr(koopman_model, "set_buffer_update"):
            koopman_model.set_buffer_update(False)
            log("KoopmanAE buffers frozen (u/K_u will not update).")
        koopman_norm = {
            "mean": ckpt.get("data_mean", None),
            "std": ckpt.get("data_std", None),
            "repr_mean": ckpt_repr_mean,
            "repr_std": ckpt_repr_std,
            "min": min_val,
            "max": max_val,
        }
        if path1_v6_mode:
            validated_v6 = validate_path1_v6_recipe(
                koopman_cfg=koopman_cfg,
                time_length=int(getattr(inp, "time_length", 0) or 0),
                checkpoint_repr_mode=ckpt_repr_mode,
                checkpoint_delay_l=int(mcfg.get("delay_L", 0) or 0),
            )
            log(
                "[path1_v6] validated recipe: "
                f"prior_variant={validated_v6['prior_variant']}, "
                f"repr_mode={validated_v6['repr_mode']}, "
                f"window_size={validated_v6['window_size']}, "
                f"temporal_stride={validated_v6['temporal_stride']}, "
                f"time_length={validated_v6['time_length']}"
            )
        log("KoopmanAE loaded and frozen.")
elif koopman_enabled and distilled_mode:
    artifact_path = koopman_cfg.get("artifact_path", None)
    if not artifact_path:
        raise ValueError("koopman.mode=distilled_skeleton requires artifact_path")
    log(f"using distilled Koopman surrogate artifact: {artifact_path}")

## Training Loop
log("training...")

train_uncond_model = TrainLoop(
                                model=unet_model,
                                diffusion=diff_model,
                                train_data = dl_iter(dl_train, train_sampler),
                                valid_data=dl_valid,
                                batch_size= batch_size,
                                microbatch= microbatch,
                                lr = lr,
                                ema_rate=ema_rate,
                                log_interval=log_interval,
                                save_interval=save_interval,
                                lr_anneal_steps=lr_anneal_steps,
                                max_train_steps=max_train_steps,
                                valid_interval=valid_interval,
                                resume_checkpoint=resume_checkpoint,
                                resume_optimizer=resume_optimizer,
                                resume_ema=resume_ema,
                                resume_step_override=resume_step_override,
                                best_ckpt_cfg=best_ckpt_cfg,
                                training_schedule_cfg=training_schedule_cfg,
                                selection_cfg=selection_cfg,
                                koopman_model=koopman_model,
                                koopman_prior_artifact=koopman_prior_artifact,
                                koopman_cfg=koopman_cfg,
                                koopman_norm=koopman_norm,
                                generic_regularizer_cfg=generic_regularizer_cfg,
                                train_returns_indices=distilled_mode,
                                valid_t_seed=valid_t_seed,
                                validation_rng_cfg=validation_rng_cfg,
                                disable_checkpoint_saving=disable_checkpoint_saving,
                                checkpoint_policy_cfg=checkpoint_policy_cfg,
                                training_trace_cfg=training_trace_cfg)

profile_memory_baseline = {}
if profile_memory_path:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        profile_memory_baseline = {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
        }
        if profile_memory_reset_before_loop:
            torch.cuda.reset_peak_memory_stats()
    else:
        profile_memory_baseline = {"status": "cuda_unavailable"}

train_uncond_model.run_loop()

if profile_memory_path and rank == 0:
    profile_dir = os.path.dirname(profile_memory_path)
    if profile_dir:
        os.makedirs(profile_dir, exist_ok=True)
    summary = {
        "yaml_path": inp.yamlpath,
        "log_path": inp.log_path,
        "original_cuda_visible_devices": ORIGINAL_CUDA_VISIBLE_DEVICES,
        "effective_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "rank": int(rank),
        "world_size": int(dist.get_world_size()) if dist.is_initialized() else 1,
        "max_train_steps": int(max_train_steps),
        "completed_local_steps": int(getattr(train_uncond_model, "step", 0)),
        "resume_step": int(getattr(train_uncond_model, "resume_step", 0)),
        "profile_memory_reset_before_loop": profile_memory_reset_before_loop,
        "baseline_before_loop": profile_memory_baseline,
    }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        summary.update(
            {
                "cuda_available": True,
                "torch_device_index": int(device),
                "device_name": props.name,
                "device_total_memory_bytes": int(props.total_memory),
                "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "current_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    else:
        summary.update({"cuda_available": False})
    with open(profile_memory_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    log(f"profile memory summary written to {profile_memory_path}")

# finish swanlab run if enabled
if swanlab_run is not None:
    finish_fn = getattr(swanlab_run, "finish", None)
    if callable(finish_fn):
        finish_fn()
set_swanlab_run(None)
