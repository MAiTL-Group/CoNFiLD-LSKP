# CoNFiLD-LSKP modification notice (2026):
# Added prior regularization, validation selection and training support.
# Derived from CoNFiLD; see third_party_licenses/README.md for attribution.
import copy
import functools
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer, master_params_to_model_params
from .koopman_regularizer import LegacyKoopmanRegularizer, ModernKoopmanRegularizer
from .koopman_regularizer_v6 import Path1V6CalibratedRegularizer
from .koopman_surrogate import DistilledSkeletonRegularizer
from .generic_transition_regularizer import GenericTransitionRegularizer
from .checkpoint_policy import CheckpointPolicy
from .multi_objective_balance import create_lambda_balancer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        train_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        valid_data=None,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        final_lr=0.,
        max_train_steps=0,
        valid_interval=1,
        resume_optimizer=True,
        resume_ema=True,
        resume_step_override=None,
        best_ckpt_cfg=None,
        training_schedule_cfg=None,
        selection_cfg=None,
        koopman_model=None,
        koopman_prior_artifact=None,
        koopman_cfg=None,
        koopman_norm=None,
        generic_regularizer_cfg=None,
        train_returns_indices=False,
        valid_t_seed=42,
        validation_rng_cfg=None,
        disable_checkpoint_saving=False,
        checkpoint_policy_cfg=None,
        training_trace_cfg=None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.train_data = train_data
        self.valid_data = valid_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.final_lr = final_lr
        self.max_train_steps = max_train_steps
        self.valid_interval = max(1, int(valid_interval))
        self.resume_optimizer = bool(resume_optimizer)
        self.resume_ema = bool(resume_ema)
        self.resume_step_override = resume_step_override
        self.training_schedule_cfg = training_schedule_cfg or {}
        self.selection_cfg = selection_cfg or {}
        self.koopman_cfg = koopman_cfg or {}
        self.koopman_norm = koopman_norm or {}
        self.koopman_prior_artifact = koopman_prior_artifact or {}
        self.generic_regularizer_cfg = generic_regularizer_cfg or {}
        self.train_returns_indices = bool(train_returns_indices)
        self.koopman_mode = str(self.koopman_cfg.get("mode", "")).lower()
        self.distilled_skeleton_enabled = bool(
            self.koopman_cfg.get("enabled", False) and self.koopman_mode == "distilled_skeleton"
        )
        self.path1_v6_enabled = bool(
            self.koopman_cfg.get("enabled", False) and self.koopman_mode == "path1_v6_calibrated"
        )
        self.path_schedule_mode = str(
            self.training_schedule_cfg.get("mode", "")
        ).lower()
        self.path1_v2_enabled = self.path_schedule_mode == "path1_v2"
        self.path1_v3_enabled = self.path_schedule_mode == "path1_v3"
        self.path1_v4_enabled = self.path_schedule_mode == "path1_v4"
        self.path1_v5_enabled = self.path_schedule_mode == "path1_v5"
        self.path_schedule_enabled = (
            self.path1_v2_enabled
            or self.path1_v3_enabled
            or self.path1_v4_enabled
            or self.path1_v5_enabled
        )
        self.total_train_steps = int(
            self.training_schedule_cfg.get(
                "total_steps",
                max_train_steps if max_train_steps > 0 else lr_anneal_steps,
            )
            or 0
        )
        if self.path_schedule_enabled and self.total_train_steps <= 0:
            raise ValueError(f"{self.path_schedule_mode} requires training_schedule.total_steps > 0")
        if self.path_schedule_enabled:
            self.max_train_steps = self.total_train_steps
        if self.distilled_skeleton_enabled and self.path_schedule_enabled:
            raise ValueError("distilled_skeleton mode cannot be combined with path1_v2/v3/v4/v5 schedules")
        if self.distilled_skeleton_enabled and (not self.train_returns_indices):
            raise ValueError("distilled_skeleton mode requires train_returns_indices=True")
        if self.path1_v6_enabled and self.path_schedule_enabled:
            raise ValueError("path1_v6_calibrated cannot be combined with path1_v2/v3/v4/v5 schedules")
        self.best_ckpt_cfg = best_ckpt_cfg or {}
        self.valid_t_seed = int(valid_t_seed)
        self.validation_rng_cfg = dict(validation_rng_cfg or {})
        self.validation_rng_enabled = bool(self.validation_rng_cfg.get("enabled", False))
        self.valid_noise_seed = int(self.validation_rng_cfg.get("noise_seed", 0))
        self.valid_loader_seed = int(self.validation_rng_cfg.get("loader_seed", 0))
        if self.validation_rng_enabled:
            missing = {
                key for key in ("noise_seed", "loader_seed")
                if key not in self.validation_rng_cfg
            }
            if missing:
                raise ValueError(
                    "validation_rng requires explicit seeds: " + ", ".join(sorted(missing))
                )
            logger.log(
                "[validation] isolated fixed RNG enabled: "
                f"noise_seed={self.valid_noise_seed}, loader_seed={self.valid_loader_seed}"
            )
        self.disable_checkpoint_saving = bool(disable_checkpoint_saving)
        self.checkpoint_policy = CheckpointPolicy.from_config(checkpoint_policy_cfg)
        self.training_trace_cfg = training_trace_cfg or {}
        self.training_trace_enabled = bool(self.training_trace_cfg.get("enabled", False))
        self.training_trace_filename = str(
            self.training_trace_cfg.get("filename", "training_trace.jsonl")
        )
        if self.checkpoint_policy.enabled:
            logger.log(
                "[checkpoint] sparse policy enabled: "
                f"ema_milestones={list(self.checkpoint_policy.ema_milestones)}, "
                f"skip_step_zero={self.checkpoint_policy.skip_step_zero}, "
                f"lightweight_best={self.checkpoint_policy.lightweight_best}, "
                f"terminal_full={self.checkpoint_policy.terminal_full}"
            )
        self.best_ckpt_enabled = (not self.path_schedule_enabled) and bool(
            self.best_ckpt_cfg.get("enabled", False)
        )
        self.best_metric_name = str(self.best_ckpt_cfg.get("metric", "valid_diff_loss"))
        self.best_ema_decay = float(self.best_ckpt_cfg.get("ema_decay", 0.9))
        self.best_warmup_valid_count = max(0, int(self.best_ckpt_cfg.get("warmup_valid_count", 0)))
        self.best_min_delta = float(self.best_ckpt_cfg.get("min_delta", 0.0))
        self.best_filename = str(self.best_ckpt_cfg.get("filename", "checkpoint_best.pt"))
        self.best_trace_filename = str(
            self.best_ckpt_cfg.get("trace_filename", "") or ""
        ).strip()
        if self.best_metric_name != "valid_diff_loss":
            logger.log(
                f"[best] unsupported metric={self.best_metric_name}, fallback to valid_diff_loss"
            )
            self.best_metric_name = "valid_diff_loss"
        if not (0.0 <= self.best_ema_decay < 1.0):
            raise ValueError(f"best_ckpt.ema_decay must be in [0,1), got {self.best_ema_decay}")
        self.best_metric = float("inf")
        self.best_metric_ema = None
        self.best_step = -1
        self.valid_count = 0
        if self.best_ckpt_enabled:
            logger.log(
                f"[best] enabled metric={self.best_metric_name}, ema_decay={self.best_ema_decay}, "
                f"warmup_valid_count={self.best_warmup_valid_count}, min_delta={self.best_min_delta}, "
                f"filename={self.best_filename}"
            )
            self._maybe_restore_best_state_from_report()
        self.selection_metric_name = "valid_diff_loss"
        self.selection_ema_alpha = float(self.selection_cfg.get("ema_alpha", 0.2))
        self.selection_min_delta = float(self.selection_cfg.get("min_delta", 0.0))
        self.selection_patience_valids = max(
            1, int(self.selection_cfg.get("patience_valids", 80))
        )
        self.selection_metric_ema = None
        self.selection_valid_count = 0
        self.selection_filename = None
        self.selection_report_filename = None
        self.selection_best_metric = float("inf")
        self.selection_best_step = -1
        self.selection_best_raw = None
        self.selection_best_support = {}
        self.selection_last_improve_valid = 0
        self.primary_filename = None
        self.primary_report_filename = None
        self.primary_best_metric = float("inf")
        self.primary_best_step = -1
        self.primary_best_raw = None
        self.primary_best_phase = None
        self.primary_last_improve_valid = 0
        self.structured_filename = None
        self.structured_report_filename = None
        self.structured_best_metric = float("inf")
        self.structured_best_step = -1
        self.structured_best_raw = None
        self.structured_best_phase = None
        self.structured_best_support = {}
        self.structured_last_improve_valid = 0
        self.warmup_filename = None
        self.warmup_report_filename = None
        self.warmup_best_metric = float("inf")
        self.warmup_best_step = -1
        self.warmup_best_raw = None
        self.warmup_last_improve_valid = 0
        self.final_filename = None
        self.final_report_filename = None
        self.final_best_metric = float("inf")
        self.final_best_step = -1
        self.final_best_raw = None
        self.final_best_phase = None
        self.final_best_source = None
        self.final_last_improve_valid = 0
        self.selection_min_n_eff = 0.0
        self.selection_min_weight_sum = 0.0
        self.selection_max_overflow_p95 = 1.0
        self.selection_grad_ratio_min = 0.0
        self.selection_grad_ratio_max = float("inf")
        self.structured_max_overflow_p95_valid = 1.0
        self.structured_min_n_eff_train = 0.0
        self.structured_min_weight_sum_train = 0.0
        self.structured_grad_ratio_min_train = 0.0
        self.structured_grad_ratio_max_train = float("inf")
        self.path1_v3_dyn_start_step = 0
        self.path1_v3_dyn1_full_step = 0
        self.path1_v3_dyn_multi_start_step = 0
        self.path1_v3_dyn_multi_full_step = 0
        self.path1_v3_cons_start_step = 0
        self.path1_v3_cons_full_step = 0
        self._path1_v3_state = {}
        self.path1_v4_warmup_min_step = 0
        self.path1_v4_warmup_force_step = 0
        self.path1_v4_warmup_plateau_valids = 4
        self.path1_v4_dyn1_ramp_start_step = 0
        self.path1_v4_dyn1_full_step = 0
        self.path1_v4_dyn_multi_earliest_step = 0
        self.path1_v4_dyn_multi_deadline_step = 0
        self.path1_v4_dyn_multi_full_step = 0
        self.path1_v4_cons_earliest_step = 0
        self.path1_v4_cons_deadline_step = 0
        self.path1_v4_cons_full_step = 0
        self.path1_v4_dyn1_observe_valids = 1
        self.path1_v4_dyn_observe_valids = 1
        self.path1_v4_reject_patience_valids = 1
        self.path1_v4_max_degrade_vs_warmup_best = 0.0
        self.path1_v4_max_degrade_vs_prev_phase_best = 0.0
        self.path1_v4_reject_degrade_vs_warmup = 0.0
        self.path1_v4_reject_degrade_vs_prev_phase = 0.0
        self.path1_v4_overflow_enter = 1.0
        self.path1_v4_overflow_exit = 1.0
        self.path1_v4_overflow_valid_ema_alpha = 0.20
        self.path1_v4_structured_min_n_eff_train = 0.0
        self.path1_v4_structured_min_weight_sum_train = 0.0
        self.path1_v4_structured_grad_ratio_min_train = 0.0
        self.path1_v4_structured_grad_ratio_max_train = float("inf")
        self.path1_v4_target_ratio_dyn1_full = 0.0
        self.path1_v4_target_ratio_dyn_full = 0.0
        self.path1_v4_target_ratio_cons_full = 0.0
        self.path1_v4_dyn_multi_full_share = 0.0
        self.path1_v4_cons_total_full_share = 0.0
        self.path1_v4_controller_phase = "warmup"
        self.path1_v4_phase_accepted = {}
        self.path1_v4_phase_rejected = {}
        self.path1_v4_phase_rejection_reason = {}
        self.path1_v4_phase_start_step = {}
        self.path1_v4_phase_start_valid = {}
        self.path1_v4_phase_best_metric = {}
        self.path1_v4_phase_best_step = {}
        self.path1_v4_phase_last_improve_valid = {}
        self.path1_v4_phase_entry_reason = {}
        self.path1_v4_structured_state = False
        self.path1_v4_overflow_valid_ema = None
        self.path1_v4_last_structured_summary = {}
        self.path1_v4_dyn_rejected = False
        self.path1_v4_cons_rejected = False
        self._path1_v4_state = {}
        self.path1_v5_warmup_min_step = 0
        self.path1_v5_warmup_force_step = 0
        self.path1_v5_warmup_plateau_valids = 1
        self.path1_v5_dyn1_start_step = 0
        self.path1_v5_dyn1_full_step = 0
        self.path1_v5_dyn1_observe_valids = 1
        self.path1_v5_dyn_probe_start_step = 0
        self.path1_v5_dyn_probe_full_step = 0
        self.path1_v5_dyn_probe_observe_valids = 1
        self.path1_v5_dyn_start_step = 0
        self.path1_v5_dyn_full_step = 0
        self.path1_v5_dyn_observe_valids = 1
        self.path1_v5_cons_probe_start_step = 0
        self.path1_v5_cons_probe_full_step = 0
        self.path1_v5_cons_probe_observe_valids = 1
        self.path1_v5_cons_start_step = 0
        self.path1_v5_cons_full_step = 0
        self.path1_v5_cons_observe_valids = 1
        self.path1_v5_finalize_patience_valids = 1
        self.path1_v5_reject_patience_valids = 1
        self.path1_v5_final_min_delta = 0.0
        self.path1_v5_max_degrade_vs_prev_phase_best = 0.0
        self.path1_v5_reject_degrade_vs_prev_phase = 0.0
        self.path1_v5_structured_hold_valids = 1
        self.path1_v5_overflow_enter = 1.0
        self.path1_v5_overflow_exit = 1.0
        self.path1_v5_overflow_valid_ema_alpha = 0.20
        self.path1_v5_structured_min_n_eff_train = 0.0
        self.path1_v5_structured_min_weight_sum_train = 0.0
        self.path1_v5_structured_grad_ratio_min_train = 0.0
        self.path1_v5_structured_grad_ratio_max_train = float("inf")
        self.path1_v5_target_ratio_dyn1_full = 0.0
        self.path1_v5_target_ratio_dyn_probe_full = 0.0
        self.path1_v5_target_ratio_dyn_full = 0.0
        self.path1_v5_target_ratio_cons_probe_full = 0.0
        self.path1_v5_target_ratio_cons_full = 0.0
        self.path1_v5_dyn_multi_probe_share = 0.0
        self.path1_v5_dyn_multi_full_share = 0.0
        self.path1_v5_cons_total_probe_share = 0.0
        self.path1_v5_cons_total_full_share = 0.0
        self.path1_v5_controller_phase = "warmup"
        self.path1_v5_previous_phase = None
        self.path1_v5_phase_accepted = {}
        self.path1_v5_phase_rejected = {}
        self.path1_v5_phase_rejection_reason = {}
        self.path1_v5_phase_promotion_blocked_reason = {}
        self.path1_v5_phase_start_step = {}
        self.path1_v5_phase_start_valid = {}
        self.path1_v5_phase_best_metric = {}
        self.path1_v5_phase_best_step = {}
        self.path1_v5_phase_best_raw = {}
        self.path1_v5_phase_last_improve_valid = {}
        self.path1_v5_phase_entry_reason = {}
        self.path1_v5_phase_metric_ema = {}
        self.path1_v5_structured_state = False
        self.path1_v5_structured_true_streak = 0
        self.path1_v5_overflow_valid_ema = None
        self.path1_v5_last_structured_summary = {}
        self.path1_v5_finalize_reason = None
        self.path1_v5_rollback_source_checkpoint = None
        self.path1_v5_rollback_source_phase = None
        self.path1_v5_pending_rollback = None
        self._path1_v5_state = {}
        self.dyn1_filename = None
        self.dyn1_report_filename = None
        self.dyn1_best_metric = float("inf")
        self.dyn1_best_step = -1
        self.dyn1_best_raw = None
        self.dyn1_last_improve_valid = 0
        self.dyn_filename = None
        self.dyn_report_filename = None
        self.dyn_best_metric = float("inf")
        self.dyn_best_step = -1
        self.dyn_best_raw = None
        self.dyn_last_improve_valid = 0
        self.cons_filename = None
        self.cons_report_filename = None
        self.cons_best_metric = float("inf")
        self.cons_best_step = -1
        self.cons_best_raw = None
        self.cons_last_improve_valid = 0
        self._support_window_stats = defaultdict(list)
        self._latest_support_summary = {}
        self._early_stop_triggered = False
        self._early_stop_reason = None
        self._early_stop_enabled = True
        self._early_stop_start_step = 0

        if self.path1_v2_enabled:
            self.selection_filename = str(
                self.selection_cfg.get("filename", "checkpoint_best_window.pt")
            )
            self.selection_report_filename = str(
                self.selection_cfg.get("report_filename", "best_window_report.json")
            )
            support_guard = self.selection_cfg.get("support_guard", {}) or {}
            self.selection_min_n_eff = float(support_guard.get("min_N_eff", 0.0))
            self.selection_min_weight_sum = float(
                support_guard.get("min_weight_sum", 0.0)
            )
            self.selection_max_overflow_p95 = float(
                support_guard.get("max_overflow_p95", 1.0)
            )
            self.selection_grad_ratio_min = float(
                support_guard.get("grad_ratio_min", 0.0)
            )
            self.selection_grad_ratio_max = float(
                support_guard.get("grad_ratio_max", float("inf"))
            )
            early_stop_cfg = self.training_schedule_cfg.get("early_stop", {}) or {}
            self._early_stop_enabled = bool(early_stop_cfg.get("enabled", True))
            self._early_stop_start_step = self._resolve_schedule_point(
                explicit_step=early_stop_cfg.get("start_step", None),
                frac=early_stop_cfg.get(
                    "start_frac",
                    self.training_schedule_cfg.get("early_stop_start_frac", 0.30),
                ),
                default=0,
            )
            logger.log(
                "[path1_v2] enabled: "
                f"total_steps={self.total_train_steps}, "
                f"selection_ema_alpha={self.selection_ema_alpha}, "
                f"best_filename={self.selection_filename}, "
                f"early_stop_enabled={self._early_stop_enabled}, "
                f"early_stop_start_step={self._early_stop_start_step}, "
                f"patience_valids={self.selection_patience_valids}"
            )
        elif self.path1_v3_enabled:
            self.primary_filename = str(
                self.selection_cfg.get("primary_filename", "checkpoint_best_bc.pt")
            )
            self.primary_report_filename = str(
                self.selection_cfg.get("primary_report_filename", "best_bc_report.json")
            )
            self.structured_filename = str(
                self.selection_cfg.get("structured_filename", "checkpoint_best_structured.pt")
            )
            self.structured_report_filename = str(
                self.selection_cfg.get("structured_report_filename", "best_structured_report.json")
            )
            structured_guard = self.selection_cfg.get("structured_guard", {}) or {}
            self.structured_max_overflow_p95_valid = float(
                structured_guard.get("max_overflow_p95_valid", 1.0)
            )
            self.structured_min_n_eff_train = float(
                structured_guard.get("min_N_eff_train", 0.0)
            )
            self.structured_min_weight_sum_train = float(
                structured_guard.get("min_weight_sum_train", 0.0)
            )
            self.structured_grad_ratio_min_train = float(
                structured_guard.get("grad_ratio_min_train", 0.0)
            )
            self.structured_grad_ratio_max_train = float(
                structured_guard.get("grad_ratio_max_train", float("inf"))
            )
            early_stop_cfg = self.training_schedule_cfg.get("early_stop", {}) or {}
            self._early_stop_enabled = bool(early_stop_cfg.get("enabled", True))
            self.path1_v3_dyn_start_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("dyn_start_step", None),
                frac=None,
                default=0,
            )
            self.path1_v3_dyn1_full_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("dyn1_full_step", None),
                frac=None,
                default=self.path1_v3_dyn_start_step,
            )
            self.path1_v3_dyn_multi_start_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("dyn_multi_start_step", None),
                frac=None,
                default=self.path1_v3_dyn1_full_step,
            )
            self.path1_v3_dyn_multi_full_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("dyn_multi_full_step", None),
                frac=None,
                default=self.path1_v3_dyn_multi_start_step,
            )
            self.path1_v3_cons_start_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("cons_start_step", None),
                frac=None,
                default=self.path1_v3_dyn_multi_full_step,
            )
            self.path1_v3_cons_full_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("cons_full_step", None),
                frac=None,
                default=self.path1_v3_cons_start_step,
            )
            self._early_stop_start_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("early_stop_start_step", None),
                frac=None,
                default=self.path1_v3_cons_start_step,
            )
            logger.log(
                "[path1_v3] enabled: "
                f"total_steps={self.total_train_steps}, "
                f"selection_ema_alpha={self.selection_ema_alpha}, "
                f"primary_filename={self.primary_filename}, "
                f"structured_filename={self.structured_filename}, "
                f"early_stop_enabled={self._early_stop_enabled}, "
                f"dyn_start_step={self.path1_v3_dyn_start_step}, "
                f"dyn1_full_step={self.path1_v3_dyn1_full_step}, "
                f"dyn_multi_start_step={self.path1_v3_dyn_multi_start_step}, "
                f"dyn_multi_full_step={self.path1_v3_dyn_multi_full_step}, "
                f"cons_start_step={self.path1_v3_cons_start_step}, "
                f"cons_full_step={self.path1_v3_cons_full_step}, "
                f"early_stop_start_step={self._early_stop_start_step}, "
                f"patience_valids={self.selection_patience_valids}"
            )
        elif self.path1_v4_enabled:
            self.warmup_filename = str(
                self.selection_cfg.get("warmup_filename", "checkpoint_best_warmup.pt")
            )
            self.warmup_report_filename = str(
                self.selection_cfg.get("warmup_report_filename", "best_warmup_report.json")
            )
            self.primary_filename = str(
                self.selection_cfg.get("primary_filename", "checkpoint_best_bc.pt")
            )
            self.primary_report_filename = str(
                self.selection_cfg.get("primary_report_filename", "best_bc_report.json")
            )
            self.structured_filename = str(
                self.selection_cfg.get("structured_filename", "checkpoint_best_structured.pt")
            )
            self.structured_report_filename = str(
                self.selection_cfg.get("structured_report_filename", "best_structured_report.json")
            )
            self.final_filename = str(
                self.selection_cfg.get("final_filename", "checkpoint_best_final.pt")
            )
            self.final_report_filename = str(
                self.selection_cfg.get("final_report_filename", "best_final_report.json")
            )
            structured_guard = self.selection_cfg.get("structured_guard", {}) or {}
            self.path1_v4_overflow_enter = float(structured_guard.get("overflow_enter", 1.0))
            self.path1_v4_overflow_exit = float(
                structured_guard.get("overflow_exit", self.path1_v4_overflow_enter)
            )
            self.path1_v4_overflow_valid_ema_alpha = float(
                structured_guard.get("overflow_valid_ema_alpha", 0.20)
            )
            self.path1_v4_structured_min_n_eff_train = float(
                structured_guard.get("min_N_eff_train", 0.0)
            )
            self.path1_v4_structured_min_weight_sum_train = float(
                structured_guard.get("min_weight_sum_train", 0.0)
            )
            self.path1_v4_structured_grad_ratio_min_train = float(
                structured_guard.get("grad_ratio_min_train", 0.0)
            )
            self.path1_v4_structured_grad_ratio_max_train = float(
                structured_guard.get("grad_ratio_max_train", float("inf"))
            )
            self._early_stop_enabled = bool(
                (self.training_schedule_cfg.get("early_stop", {}) or {}).get("enabled", True)
            )
            self.path1_v4_warmup_min_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("warmup_min_step", None),
                frac=None,
                default=0,
            )
            self.path1_v4_warmup_force_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("warmup_force_step", None),
                frac=None,
                default=self.path1_v4_warmup_min_step,
            )
            self.path1_v4_warmup_plateau_valids = max(
                1, int(self.training_schedule_cfg.get("warmup_plateau_valids", 4))
            )
            self.path1_v4_dyn1_ramp_start_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("dyn1_ramp_start_step", None),
                frac=None,
                default=self.path1_v4_warmup_force_step,
            )
            self.path1_v4_dyn1_full_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("dyn1_full_step", None),
                frac=None,
                default=self.path1_v4_dyn1_ramp_start_step,
            )
            self.path1_v4_dyn_multi_earliest_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("dyn_multi_earliest_step", None),
                frac=None,
                default=self.path1_v4_dyn1_full_step,
            )
            self.path1_v4_dyn_multi_deadline_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("dyn_multi_deadline_step", None),
                frac=None,
                default=self.path1_v4_dyn_multi_earliest_step,
            )
            self.path1_v4_dyn_multi_full_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("dyn_multi_full_step", None),
                frac=None,
                default=self.path1_v4_dyn_multi_deadline_step,
            )
            self.path1_v4_cons_earliest_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("cons_earliest_step", None),
                frac=None,
                default=self.path1_v4_dyn_multi_full_step,
            )
            self.path1_v4_cons_deadline_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("cons_deadline_step", None),
                frac=None,
                default=self.path1_v4_cons_earliest_step,
            )
            self.path1_v4_cons_full_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("cons_full_step", None),
                frac=None,
                default=self.path1_v4_cons_deadline_step,
            )
            self._early_stop_start_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("early_stop_start_step", None),
                frac=None,
                default=self.path1_v4_cons_earliest_step,
            )
            self.path1_v4_dyn1_observe_valids = max(
                1, int(self.training_schedule_cfg.get("dyn1_observe_valids", 1))
            )
            self.path1_v4_dyn_observe_valids = max(
                1, int(self.training_schedule_cfg.get("dyn_observe_valids", 1))
            )
            self.path1_v4_reject_patience_valids = max(
                1, int(self.training_schedule_cfg.get("reject_patience_valids", 1))
            )
            self.path1_v4_max_degrade_vs_warmup_best = float(
                self.training_schedule_cfg.get("max_degrade_vs_warmup_best", 0.0)
            )
            self.path1_v4_max_degrade_vs_prev_phase_best = float(
                self.training_schedule_cfg.get("max_degrade_vs_prev_phase_best", 0.0)
            )
            self.path1_v4_reject_degrade_vs_warmup = float(
                self.training_schedule_cfg.get("reject_degrade_vs_warmup", 0.0)
            )
            self.path1_v4_reject_degrade_vs_prev_phase = float(
                self.training_schedule_cfg.get("reject_degrade_vs_prev_phase", 0.0)
            )
            balance_cfg = self.koopman_cfg.get("balance", {}) or {}
            ratio_cfg = balance_cfg.get("target_ratio_schedule", {}) or {}
            self.path1_v4_target_ratio_dyn1_full = float(
                ratio_cfg.get("dyn1_full", balance_cfg.get("target_ratio", 0.0))
            )
            self.path1_v4_target_ratio_dyn_full = float(
                ratio_cfg.get("dyn_full", self.path1_v4_target_ratio_dyn1_full)
            )
            self.path1_v4_target_ratio_cons_full = float(
                ratio_cfg.get("cons_full", self.path1_v4_target_ratio_dyn_full)
            )
            dyn_cfg = self.koopman_cfg.get("dyn", {}) or {}
            cons_cfg = self.koopman_cfg.get("consistency", {}) or {}
            self.path1_v4_dyn_multi_full_share = float(
                dyn_cfg.get("multi_full_share", dyn_cfg.get("dyn_multi_full_share", 0.0))
            )
            self.path1_v4_cons_total_full_share = float(
                cons_cfg.get("total_full_share", cons_cfg.get("cons_total_full_share", 0.0))
            )
            self.path1_v4_phase_accepted = {"warmup": True, "dyn1": False, "dyn": False, "cons": False}
            self.path1_v4_phase_rejected = {"dyn1": False, "dyn": False, "cons": False}
            self.path1_v4_phase_rejection_reason = {}
            self.path1_v4_phase_start_step = {"warmup": 0}
            self.path1_v4_phase_start_valid = {"warmup": 0}
            self.path1_v4_phase_best_metric = {}
            self.path1_v4_phase_best_step = {}
            self.path1_v4_phase_last_improve_valid = {}
            self.path1_v4_phase_entry_reason = {"warmup": "initial"}
            self.path1_v4_controller_phase = "warmup"
            logger.log(
                "[path1_v4] enabled: "
                f"total_steps={self.total_train_steps}, "
                f"warmup_min_step={self.path1_v4_warmup_min_step}, "
                f"warmup_force_step={self.path1_v4_warmup_force_step}, "
                f"dyn1_full_step={self.path1_v4_dyn1_full_step}, "
                f"dyn_multi_earliest_step={self.path1_v4_dyn_multi_earliest_step}, "
                f"dyn_multi_deadline_step={self.path1_v4_dyn_multi_deadline_step}, "
                f"cons_earliest_step={self.path1_v4_cons_earliest_step}, "
                f"cons_deadline_step={self.path1_v4_cons_deadline_step}, "
                f"early_stop_start_step={self._early_stop_start_step}, "
                f"overflow_enter={self.path1_v4_overflow_enter}, "
                f"overflow_exit={self.path1_v4_overflow_exit}"
            )
        elif self.path1_v5_enabled:
            self.warmup_filename = str(
                self.selection_cfg.get("warmup_filename", "checkpoint_best_warmup.pt")
            )
            self.warmup_report_filename = str(
                self.selection_cfg.get("warmup_report_filename", "best_warmup_report.json")
            )
            self.dyn1_filename = str(
                self.selection_cfg.get("dyn1_filename", "checkpoint_best_dyn1.pt")
            )
            self.dyn1_report_filename = str(
                self.selection_cfg.get("dyn1_report_filename", "best_dyn1_report.json")
            )
            self.dyn_filename = str(
                self.selection_cfg.get("dyn_filename", "checkpoint_best_dyn.pt")
            )
            self.dyn_report_filename = str(
                self.selection_cfg.get("dyn_report_filename", "best_dyn_report.json")
            )
            self.cons_filename = str(
                self.selection_cfg.get("cons_filename", "checkpoint_best_cons.pt")
            )
            self.cons_report_filename = str(
                self.selection_cfg.get("cons_report_filename", "best_cons_report.json")
            )
            self.structured_filename = str(
                self.selection_cfg.get("structured_filename", "checkpoint_best_structured.pt")
            )
            self.structured_report_filename = str(
                self.selection_cfg.get("structured_report_filename", "best_structured_report.json")
            )
            self.final_filename = str(
                self.selection_cfg.get("final_filename", "checkpoint_best_final.pt")
            )
            self.final_report_filename = str(
                self.selection_cfg.get("final_report_filename", "best_final_report.json")
            )
            structured_guard = self.selection_cfg.get("structured_guard", {}) or {}
            self.path1_v5_overflow_enter = float(structured_guard.get("overflow_enter", 1.0))
            self.path1_v5_overflow_exit = float(
                structured_guard.get("overflow_exit", self.path1_v5_overflow_enter)
            )
            self.path1_v5_overflow_valid_ema_alpha = float(
                structured_guard.get("overflow_valid_ema_alpha", 0.20)
            )
            self.path1_v5_structured_min_n_eff_train = float(
                structured_guard.get("min_N_eff_train", 0.0)
            )
            self.path1_v5_structured_min_weight_sum_train = float(
                structured_guard.get("min_weight_sum_train", 0.0)
            )
            self.path1_v5_structured_grad_ratio_min_train = float(
                structured_guard.get("grad_ratio_min_train", 0.0)
            )
            self.path1_v5_structured_grad_ratio_max_train = float(
                structured_guard.get("grad_ratio_max_train", float("inf"))
            )
            self._early_stop_enabled = bool(
                (self.training_schedule_cfg.get("early_stop", {}) or {}).get("enabled", True)
            )
            self.path1_v5_warmup_min_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("warmup_min_step", None),
                frac=None,
                default=0,
            )
            self.path1_v5_warmup_force_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("warmup_force_step", None),
                frac=None,
                default=self.path1_v5_warmup_min_step,
            )
            self.path1_v5_warmup_plateau_valids = max(
                1, int(self.training_schedule_cfg.get("warmup_plateau_valids", 1))
            )
            self.path1_v5_dyn1_start_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("dyn1_start_step", None),
                frac=None,
                default=self.path1_v5_warmup_force_step,
            )
            self.path1_v5_dyn1_full_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("dyn1_full_step", None),
                frac=None,
                default=self.path1_v5_dyn1_start_step,
            )
            self.path1_v5_dyn1_observe_valids = max(
                1, int(self.training_schedule_cfg.get("dyn1_observe_valids", 1))
            )
            self.path1_v5_dyn_probe_start_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("dyn_probe_start_step", None),
                frac=None,
                default=self.path1_v5_dyn1_full_step,
            )
            self.path1_v5_dyn_probe_full_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("dyn_probe_full_step", None),
                frac=None,
                default=self.path1_v5_dyn_probe_start_step,
            )
            self.path1_v5_dyn_probe_observe_valids = max(
                1, int(self.training_schedule_cfg.get("dyn_probe_observe_valids", 1))
            )
            self.path1_v5_dyn_start_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("dyn_start_step", None),
                frac=None,
                default=self.path1_v5_dyn_probe_full_step,
            )
            self.path1_v5_dyn_full_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("dyn_full_step", None),
                frac=None,
                default=self.path1_v5_dyn_start_step,
            )
            self.path1_v5_dyn_observe_valids = max(
                1, int(self.training_schedule_cfg.get("dyn_observe_valids", 1))
            )
            self.path1_v5_cons_probe_start_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("cons_probe_start_step", None),
                frac=None,
                default=self.path1_v5_dyn_full_step,
            )
            self.path1_v5_cons_probe_full_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("cons_probe_full_step", None),
                frac=None,
                default=self.path1_v5_cons_probe_start_step,
            )
            self.path1_v5_cons_probe_observe_valids = max(
                1, int(self.training_schedule_cfg.get("cons_probe_observe_valids", 1))
            )
            self.path1_v5_cons_start_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("cons_start_step", None),
                frac=None,
                default=self.path1_v5_cons_probe_full_step,
            )
            self.path1_v5_cons_full_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("cons_full_step", None),
                frac=None,
                default=self.path1_v5_cons_start_step,
            )
            self.path1_v5_cons_observe_valids = max(
                1, int(self.training_schedule_cfg.get("cons_observe_valids", 1))
            )
            self.path1_v5_finalize_patience_valids = max(
                1, int(self.training_schedule_cfg.get("finalize_patience_valids", 1))
            )
            self.path1_v5_reject_patience_valids = max(
                1, int(self.training_schedule_cfg.get("reject_patience_valids", 1))
            )
            self.path1_v5_final_min_delta = float(
                self.training_schedule_cfg.get("final_min_delta", self.selection_min_delta)
            )
            self.path1_v5_max_degrade_vs_prev_phase_best = float(
                self.training_schedule_cfg.get("max_degrade_vs_prev_phase_best", 0.0)
            )
            self.path1_v5_reject_degrade_vs_prev_phase = float(
                self.training_schedule_cfg.get("reject_degrade_vs_prev_phase", 0.0)
            )
            self.path1_v5_structured_hold_valids = max(
                1, int(self.training_schedule_cfg.get("structured_hold_valids", 1))
            )
            self._early_stop_start_step = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("early_stop_start_step", None),
                frac=None,
                default=self.path1_v5_cons_start_step,
            )
            balance_cfg = self.koopman_cfg.get("balance", {}) or {}
            ratio_cfg = balance_cfg.get("target_ratio_schedule", {}) or {}
            base_ratio = float(balance_cfg.get("target_ratio", 0.0))
            self.path1_v5_target_ratio_dyn1_full = float(
                ratio_cfg.get("dyn1_full", base_ratio)
            )
            self.path1_v5_target_ratio_dyn_probe_full = float(
                ratio_cfg.get("dyn_probe_full", self.path1_v5_target_ratio_dyn1_full)
            )
            self.path1_v5_target_ratio_dyn_full = float(
                ratio_cfg.get("dyn_full", self.path1_v5_target_ratio_dyn_probe_full)
            )
            self.path1_v5_target_ratio_cons_probe_full = float(
                ratio_cfg.get("cons_probe_full", self.path1_v5_target_ratio_dyn_full)
            )
            self.path1_v5_target_ratio_cons_full = float(
                ratio_cfg.get("cons_full", self.path1_v5_target_ratio_cons_probe_full)
            )
            dyn_cfg = self.koopman_cfg.get("dyn", {}) or {}
            cons_cfg = self.koopman_cfg.get("consistency", {}) or {}
            self.path1_v5_dyn_multi_probe_share = float(
                dyn_cfg.get("multi_probe_share", 0.0)
            )
            self.path1_v5_dyn_multi_full_share = float(
                dyn_cfg.get("multi_full_share", dyn_cfg.get("dyn_multi_full_share", 0.0))
            )
            self.path1_v5_cons_total_probe_share = float(
                cons_cfg.get("total_probe_share", 0.0)
            )
            self.path1_v5_cons_total_full_share = float(
                cons_cfg.get("total_full_share", cons_cfg.get("cons_total_full_share", 0.0))
            )
            self.path1_v5_phase_accepted = {
                "warmup": True,
                "dyn1": False,
                "dyn_probe": False,
                "dyn": False,
                "cons_probe": False,
                "cons": False,
                "finalize": False,
            }
            self.path1_v5_phase_rejected = {
                "dyn1": False,
                "dyn_probe": False,
                "dyn": False,
                "cons_probe": False,
                "cons": False,
            }
            self.path1_v5_phase_rejection_reason = {}
            self.path1_v5_phase_promotion_blocked_reason = {}
            self.path1_v5_phase_start_step = {"warmup": 0}
            self.path1_v5_phase_start_valid = {"warmup": 0}
            self.path1_v5_phase_best_metric = {}
            self.path1_v5_phase_best_step = {}
            self.path1_v5_phase_best_raw = {}
            self.path1_v5_phase_last_improve_valid = {}
            self.path1_v5_phase_entry_reason = {"warmup": "initial"}
            self.path1_v5_phase_metric_ema = {}
            self.path1_v5_controller_phase = "warmup"
            self.path1_v5_previous_phase = None
            logger.log(
                "[path1_v5] enabled: "
                f"total_steps={self.total_train_steps}, "
                f"warmup_force_step={self.path1_v5_warmup_force_step}, "
                f"dyn1_full_step={self.path1_v5_dyn1_full_step}, "
                f"dyn_probe_start_step={self.path1_v5_dyn_probe_start_step}, "
                f"dyn_start_step={self.path1_v5_dyn_start_step}, "
                f"cons_probe_start_step={self.path1_v5_cons_probe_start_step}, "
                f"cons_start_step={self.path1_v5_cons_start_step}, "
                f"overflow_enter={self.path1_v5_overflow_enter}, "
                f"overflow_exit={self.path1_v5_overflow_exit}"
            )
        self.koopman_model = koopman_model
        self.koopman_surrogate = None
        self.koopman_v6_regularizer = None
        self.valid_t_all = None
        if self.valid_data is not None:
            dataset = getattr(self.valid_data, "dataset", None)
            if dataset is not None:
                self.valid_t_all = build_fixed_uniform_timesteps(
                    num_samples=len(dataset),
                    num_timesteps=self.diffusion.num_timesteps,
                    seed=self.valid_t_seed,
                )
        # Select the configured Koopman regularizer; default to legacy.
        regularizer_cfg = self.koopman_cfg.get("regularizer", {}) or {}
        self.koopman_regularizer_mode = str(
            regularizer_cfg.get("mode", self.koopman_cfg.get("regularizer_mode", "legacy"))
        ).lower()
        self.koopman_regularizer = None
        self.lambda_balancer = None
        self.generic_regularizer = None
        if bool(self.generic_regularizer_cfg.get("enabled", False)):
            if bool(self.koopman_cfg.get("enabled", False)):
                raise ValueError(
                    "generic_regularizer.enabled=True cannot be combined with "
                    "koopman.enabled=True; run it as a separate baseline."
                )
            self.generic_regularizer = GenericTransitionRegularizer(
                diffusion=self.diffusion,
                cfg=self.generic_regularizer_cfg,
                device=dist_util.dev(),
            )
            logger.log(
                "[generic_regularizer] enabled: "
                f"mode={self.generic_regularizer.mode}, "
                f"warmup_steps={self.generic_regularizer.warmup_steps}, "
                f"interval={self.generic_regularizer.interval}, "
                f"subbatch={self.generic_regularizer.subbatch}, "
                f"window_size={self.generic_regularizer.window_size}, "
                f"sigma_max={self.generic_regularizer.sigma_max}, "
                f"lambda_base={self.generic_regularizer.lambda_base}, "
                f"ratio_cap_vs_diff={self.generic_regularizer.ratio_cap_vs_diff}"
            )
        if self.distilled_skeleton_enabled:
            artifact_path = self.koopman_cfg.get("artifact_path", None)
            if not artifact_path:
                raise ValueError("distilled_skeleton mode requires koopman.artifact_path")
            self.koopman_surrogate = DistilledSkeletonRegularizer(
                artifact_path=artifact_path,
                koopman_cfg=self.koopman_cfg,
                diffusion=self.diffusion,
                device=dist_util.dev(),
            )
            logger.log(
                f"[pathlite] distilled_skeleton enabled: artifact={artifact_path}, "
                f"repr_mode={self.koopman_surrogate.repr_mode}, interval={self.koopman_surrogate.interval}, "
                f"lambda_base={self.koopman_surrogate.lambda_base}, "
                f"ratio_cap_vs_diff={self.koopman_surrogate.ratio_cap_vs_diff}"
            )
        elif self.path1_v6_enabled:
            prior_backend = str(self.koopman_cfg.get("prior_backend", "split_ae")).lower()
            if prior_backend == "split_ae" and self.koopman_model is None:
                raise ValueError("path1_v6_calibrated split_ae backend requires a frozen Koopman model")
            if prior_backend in {"pkcp", "hokp", "lskp"} and not self.koopman_prior_artifact:
                raise ValueError(f"path1_v6_calibrated {prior_backend} backend requires a prior artifact")
            self.koopman_v6_regularizer = Path1V6CalibratedRegularizer(
                diffusion=self.diffusion,
                koopman_model=self.koopman_model,
                prior_artifact=self.koopman_prior_artifact,
                koopman_cfg=self.koopman_cfg,
                koopman_norm=self.koopman_norm,
                device=dist_util.dev(),
            )
            logger.log(
                "[path1_v6] enabled: "
                f"prior_backend={prior_backend}, "
                f"prior_variant={self.koopman_v6_regularizer.prior_variant}, "
                f"repr_mode={self.koopman_v6_regularizer.repr_mode}, "
                f"window_size={self.koopman_v6_regularizer.window_size}, "
                f"temporal_stride={self.koopman_v6_regularizer.temporal_stride}, "
                f"warmup_steps={self.koopman_v6_regularizer.warmup_steps}, "
                f"interval={self.koopman_v6_regularizer.interval}, "
                f"koop_subbatch={self.koopman_v6_regularizer.koop_subbatch}, "
                f"lambda_base={self.koopman_v6_regularizer.lambda_base}, "
                f"ratio_cap_vs_diff={self.koopman_v6_regularizer.ratio_cap_vs_diff}, "
                f"gamma_man={self.koopman_v6_regularizer.gamma_man}"
            )
        elif self.koopman_model is not None:
            if self.koopman_regularizer_mode == "modern":
                self.koopman_regularizer = ModernKoopmanRegularizer(
                    diffusion=self.diffusion,
                    koopman_model=self.koopman_model,
                    koopman_cfg=self.koopman_cfg,
                    koopman_norm=self.koopman_norm,
                )
            else:
                self.koopman_regularizer = LegacyKoopmanRegularizer(
                    diffusion=self.diffusion,
                    koopman_model=self.koopman_model,
                    koopman_cfg=self.koopman_cfg,
                    koopman_norm=self.koopman_norm,
                )
            # Configure task-level weighting, fixed by default.
            balance_cfg = self.koopman_cfg.get("balance", {}) or {}
            lambda_base = float(self.koopman_cfg.get("lambda_k", 0.0))
            self.lambda_balancer = create_lambda_balancer(balance_cfg, base_lambda=lambda_base)
            # Log the effective configuration at startup.
            bal_method = str(balance_cfg.get("method", "fixed")).lower()
            logger.log(
                f"[koopman] regularizer_mode={self.koopman_regularizer_mode}, "
                f"lambda_k={lambda_base}, balance_method={bal_method}"
            )
        # PCGrad requires single-process fp32 training for gradient consistency.
        self.pcgrad_enabled = bool(
            self.lambda_balancer is not None
            and getattr(self.lambda_balancer, "uses_pcgrad", False)
        )
        self.pcgrad_supported = (not self.use_fp16) and (dist.get_world_size() == 1)
        if self.pcgrad_enabled and (not self.pcgrad_supported):
            logger.log(
                "[pcgrad] requires use_fp16=False and world_size=1; "
                "falling back to weighted-sum backpropagation."
            )
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
        if self.resume_step and self.resume_ema:
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _resolve_schedule_point(self, explicit_step=None, frac=None, default=0):
        if explicit_step is not None:
            return max(0, int(explicit_step))
        if frac is None:
            return max(0, int(default))
        return max(0, int(round(float(frac) * float(self.total_train_steps))))

    @staticmethod
    def _cosine_ramp(step, start_step, full_step):
        if full_step <= start_step:
            return 1.0 if step >= start_step else 0.0
        if step <= start_step:
            return 0.0
        if step >= full_step:
            return 1.0
        ratio = (float(step) - float(start_step)) / float(full_step - start_step)
        return 0.5 * (1.0 - math.cos(math.pi * ratio))

    @staticmethod
    def _median(values):
        if not values:
            return None
        ordered = sorted(float(v) for v in values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    @staticmethod
    def _percentile(values, q):
        if not values:
            return None
        ordered = sorted(float(v) for v in values)
        if len(ordered) == 1:
            return ordered[0]
        q = min(1.0, max(0.0, float(q)))
        pos = q * (len(ordered) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return ordered[lo]
        frac = pos - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    def _record_support_stat(self, key, value):
        if value is None:
            return
        if not math.isfinite(float(value)):
            return
        self._support_window_stats[str(key)].append(float(value))

    def _consume_support_summary(self):
        if not self._support_window_stats:
            return {}
        summary = self._summarize_support_values(self._support_window_stats)
        self._support_window_stats = defaultdict(list)
        self._latest_support_summary = summary
        return summary

    @classmethod
    def _summarize_support_values(cls, support_values):
        summary = {}
        for key, values in (support_values or {}).items():
            if not values:
                continue
            summary[f"{key}_median"] = cls._median(values)
            summary[f"{key}_p95"] = cls._percentile(values, 0.95)
        return summary

    @staticmethod
    def _merge_support_summaries(valid_summary, train_summary):
        merged = {}
        valid_summary = valid_summary or {}
        train_summary = train_summary or {}
        for key in (
            "n_eff_median",
            "n_eff_p95",
            "weight_sum_median",
            "weight_sum_p95",
            "overflow_rate_median",
            "overflow_rate_p95",
        ):
            if key in valid_summary:
                merged[key] = valid_summary[key]
        for key in ("effective_grad_ratio_median", "effective_grad_ratio_p95"):
            if key in train_summary:
                merged[key] = train_summary[key]
        return merged

    @staticmethod
    def _merge_path1_v3_structured_support(valid_summary, train_summary):
        valid_summary = valid_summary or {}
        train_summary = train_summary or {}
        merged = {}
        if "overflow_rate_p95" in valid_summary:
            merged["overflow_rate_p95_valid"] = valid_summary["overflow_rate_p95"]
        if "effective_grad_ratio_median" in train_summary:
            merged["effective_grad_ratio_median_train"] = train_summary["effective_grad_ratio_median"]
        if "n_eff_median" in train_summary:
            merged["n_eff_median_train"] = train_summary["n_eff_median"]
        if "weight_sum_median" in train_summary:
            merged["weight_sum_median_train"] = train_summary["weight_sum_median"]
        return merged

    def _support_guard_ok(self, support_summary):
        if not self.koopman_cfg.get("enabled", False):
            return True
        if not support_summary:
            return False
        n_eff = support_summary.get("n_eff_median", None)
        weight_sum = support_summary.get("weight_sum_median", None)
        overflow_p95 = support_summary.get("overflow_rate_p95", None)
        grad_ratio = support_summary.get("effective_grad_ratio_median", None)
        if n_eff is None or n_eff < self.selection_min_n_eff:
            return False
        if weight_sum is None or weight_sum < self.selection_min_weight_sum:
            return False
        if overflow_p95 is None or overflow_p95 > self.selection_max_overflow_p95:
            return False
        if grad_ratio is None:
            return False
        if grad_ratio < self.selection_grad_ratio_min:
            return False
        if grad_ratio > self.selection_grad_ratio_max:
            return False
        return True

    def _path1_v3_structured_guard_ok(self, merged_summary):
        if not self.koopman_cfg.get("enabled", False):
            return True
        if not merged_summary:
            return False
        overflow_valid = merged_summary.get("overflow_rate_p95_valid", None)
        grad_ratio_train = merged_summary.get("effective_grad_ratio_median_train", None)
        n_eff_train = merged_summary.get("n_eff_median_train", None)
        weight_sum_train = merged_summary.get("weight_sum_median_train", None)
        if overflow_valid is None or overflow_valid > self.structured_max_overflow_p95_valid:
            return False
        if grad_ratio_train is None:
            return False
        if grad_ratio_train < self.structured_grad_ratio_min_train:
            return False
        if grad_ratio_train > self.structured_grad_ratio_max_train:
            return False
        if n_eff_train is None or n_eff_train < self.structured_min_n_eff_train:
            return False
        if weight_sum_train is None or weight_sum_train < self.structured_min_weight_sum_train:
            return False
        return True

    @staticmethod
    def _merge_path1_v4_structured_support(valid_summary, train_summary):
        valid_summary = valid_summary or {}
        train_summary = train_summary or {}
        merged = {}
        if "overflow_rate_p95" in valid_summary:
            merged["overflow_rate_p95_valid"] = valid_summary["overflow_rate_p95"]
        if "effective_grad_ratio_median" in train_summary:
            merged["effective_grad_ratio_median_train"] = train_summary["effective_grad_ratio_median"]
        if "n_eff_median" in train_summary:
            merged["n_eff_median_train"] = train_summary["n_eff_median"]
        if "weight_sum_median" in train_summary:
            merged["weight_sum_median_train"] = train_summary["weight_sum_median"]
        return merged

    def _path1_v4_train_guard_ok(self, merged_summary):
        if not self.koopman_cfg.get("enabled", False):
            return True
        if not merged_summary:
            return False
        grad_ratio = merged_summary.get("effective_grad_ratio_median_train", None)
        n_eff_train = merged_summary.get("n_eff_median_train", None)
        weight_sum_train = merged_summary.get("weight_sum_median_train", None)
        if grad_ratio is None:
            return False
        if grad_ratio < self.path1_v4_structured_grad_ratio_min_train:
            return False
        if grad_ratio > self.path1_v4_structured_grad_ratio_max_train:
            return False
        if n_eff_train is None or n_eff_train < self.path1_v4_structured_min_n_eff_train:
            return False
        if (
            weight_sum_train is None
            or weight_sum_train < self.path1_v4_structured_min_weight_sum_train
        ):
            return False
        return True

    def _path1_v4_update_structured_state(self, merged_summary):
        merged_summary = dict(merged_summary or {})
        if not self.koopman_cfg.get("enabled", False):
            merged_summary["overflow_rate_p95_valid_ema"] = 0.0
            merged_summary["train_guard_ok"] = True
            merged_summary["valid_guard_ok"] = True
            merged_summary["structured_state"] = True
            self.path1_v4_structured_state = True
            self.path1_v4_last_structured_summary = dict(merged_summary)
            return merged_summary

        overflow_valid = merged_summary.get("overflow_rate_p95_valid", None)
        if overflow_valid is not None and math.isfinite(float(overflow_valid)):
            overflow_valid = float(overflow_valid)
            if self.path1_v4_overflow_valid_ema is None:
                self.path1_v4_overflow_valid_ema = overflow_valid
            else:
                alpha = float(self.path1_v4_overflow_valid_ema_alpha)
                self.path1_v4_overflow_valid_ema = (
                    (1.0 - alpha) * float(self.path1_v4_overflow_valid_ema)
                    + alpha * overflow_valid
                )
        overflow_ema = self.path1_v4_overflow_valid_ema
        train_ok = self._path1_v4_train_guard_ok(merged_summary)
        if overflow_ema is None:
            valid_ok = False
        elif self.path1_v4_structured_state:
            valid_ok = float(overflow_ema) <= float(self.path1_v4_overflow_exit)
        else:
            valid_ok = float(overflow_ema) <= float(self.path1_v4_overflow_enter)
        structured_state = bool(train_ok and valid_ok)
        self.path1_v4_structured_state = structured_state
        merged_summary["overflow_rate_p95_valid_ema"] = (
            None if overflow_ema is None else float(overflow_ema)
        )
        merged_summary["train_guard_ok"] = bool(train_ok)
        merged_summary["valid_guard_ok"] = bool(valid_ok)
        merged_summary["structured_state"] = bool(structured_state)
        self.path1_v4_last_structured_summary = dict(merged_summary)
        return merged_summary

    @staticmethod
    def _path1_v4_phase_group(phase_name):
        if phase_name == "tail_warmup":
            return "warmup"
        if phase_name == "tail_dyn1":
            return "dyn1"
        if phase_name == "tail_dyn":
            return "dyn"
        return phase_name

    def _path1_v4_set_phase(
        self,
        phase_name,
        global_step,
        *,
        reason,
        accepted_phase=None,
        rejected_phase=None,
    ):
        phase_name = str(phase_name)
        prev = self.path1_v4_controller_phase
        if accepted_phase is not None:
            self.path1_v4_phase_accepted[str(accepted_phase)] = True
        if rejected_phase is not None:
            rejected_phase = str(rejected_phase)
            self.path1_v4_phase_rejected[rejected_phase] = True
            self.path1_v4_phase_rejection_reason[rejected_phase] = str(reason)
        if prev != phase_name:
            self.path1_v4_controller_phase = phase_name
            self.path1_v4_phase_start_step[phase_name] = int(global_step)
            self.path1_v4_phase_start_valid[phase_name] = int(self.selection_valid_count)
            self.path1_v4_phase_entry_reason[phase_name] = str(reason)
            logger.log(
                f"[path1_v4] phase transition {prev} -> {phase_name} "
                f"at step={int(global_step)} reason={reason}"
            )

    def _path1_v4_record_phase_metric(self, phase_name, metric_ema, global_step):
        phase_group = self._path1_v4_phase_group(phase_name)
        if phase_group not in {"dyn1", "dyn", "cons"}:
            return
        metric_ema = float(metric_ema)
        best = self.path1_v4_phase_best_metric.get(phase_group, float("inf"))
        if metric_ema < (best - self.selection_min_delta):
            self.path1_v4_phase_best_metric[phase_group] = metric_ema
            self.path1_v4_phase_best_step[phase_group] = int(global_step)
            self.path1_v4_phase_last_improve_valid[phase_group] = int(self.selection_valid_count)

    def _path1_v4_promote_warmup_if_ready(self, global_step):
        if self.path1_v4_controller_phase != "warmup":
            return
        plateau = (
            global_step >= self.path1_v4_warmup_min_step
            and self.warmup_best_step >= 0
            and (self.selection_valid_count - self.warmup_last_improve_valid)
            >= self.path1_v4_warmup_plateau_valids
        )
        if global_step >= self.path1_v4_warmup_force_step or plateau:
            reason = "force_step" if global_step >= self.path1_v4_warmup_force_step else "plateau"
            self._path1_v4_set_phase(
                "dyn1",
                global_step,
                reason=reason,
                accepted_phase="dyn1",
            )

    def _path1_v4_best_metric(self, key):
        if key == "warmup":
            return self.warmup_best_metric
        if key == "best_bc":
            return self.primary_best_metric
        if key == "best_structured":
            return self.structured_best_metric
        return self.path1_v4_phase_best_metric.get(key, float("inf"))

    def _path1_v4_handle_phase_control(self, global_step, metric_ema, structured_summary):
        phase = self.path1_v4_controller_phase
        structured_state = bool((structured_summary or {}).get("structured_state", False))
        current_metric = float(metric_ema)
        if phase == "warmup":
            self._path1_v4_promote_warmup_if_ready(global_step)
            return

        if phase == "dyn1":
            dyn1_last_improve = self.path1_v4_phase_last_improve_valid.get("dyn1", 0)
            dyn1_best = self.path1_v4_phase_best_metric.get("dyn1", float("inf"))
            if (
                global_step >= self.path1_v4_dyn1_full_step
                and self.warmup_best_step >= 0
                and dyn1_best < float("inf")
                and (self.selection_valid_count - dyn1_last_improve)
                >= self.path1_v4_reject_patience_valids
                and current_metric
                > float(self.warmup_best_metric) * (1.0 + self.path1_v4_reject_degrade_vs_warmup)
            ):
                self._path1_v4_set_phase(
                    "tail_warmup",
                    global_step,
                    reason="dyn1_reject",
                    rejected_phase="dyn1",
                )
                self.path1_v4_dyn_rejected = True
                self.path1_v4_cons_rejected = True
                return

            dyn1_valids = self.selection_valid_count - self.path1_v4_phase_start_valid.get("dyn1", 0)
            dyn1_ok = (
                global_step >= self.path1_v4_dyn_multi_earliest_step
                and dyn1_valids >= self.path1_v4_dyn1_observe_valids
                and structured_state
                and self.warmup_best_step >= 0
                and dyn1_best < float("inf")
                and dyn1_best
                <= float(self.warmup_best_metric)
                * (1.0 + self.path1_v4_max_degrade_vs_warmup_best)
            )
            if dyn1_ok:
                self._path1_v4_set_phase(
                    "dyn",
                    global_step,
                    reason="dyn_admitted",
                    accepted_phase="dyn",
                )
                return
            if global_step >= self.path1_v4_dyn_multi_deadline_step:
                self._path1_v4_set_phase(
                    "tail_dyn1",
                    global_step,
                    reason="dyn_deadline_missed",
                    rejected_phase="dyn",
                )
                self.path1_v4_dyn_rejected = True
                self.path1_v4_cons_rejected = True
                return

        if phase == "dyn":
            dyn_last_improve = self.path1_v4_phase_last_improve_valid.get("dyn", 0)
            dyn_best = self.path1_v4_phase_best_metric.get("dyn", float("inf"))
            prev_best = self.path1_v4_phase_best_metric.get("dyn1", float("inf"))
            if (
                global_step >= self.path1_v4_dyn_multi_full_step
                and prev_best < float("inf")
                and dyn_best < float("inf")
                and (self.selection_valid_count - dyn_last_improve)
                >= self.path1_v4_reject_patience_valids
                and current_metric
                > float(prev_best) * (1.0 + self.path1_v4_reject_degrade_vs_prev_phase)
            ):
                self._path1_v4_set_phase(
                    "tail_dyn1",
                    global_step,
                    reason="dyn_reject",
                    rejected_phase="dyn",
                )
                self.path1_v4_dyn_rejected = True
                self.path1_v4_cons_rejected = True
                return

            dyn_valids = self.selection_valid_count - self.path1_v4_phase_start_valid.get("dyn", 0)
            dyn_ok = (
                global_step >= self.path1_v4_cons_earliest_step
                and dyn_valids >= self.path1_v4_dyn_observe_valids
                and structured_state
                and prev_best < float("inf")
                and dyn_best < float("inf")
                and dyn_best <= float(prev_best) * (1.0 + self.path1_v4_max_degrade_vs_prev_phase_best)
            )
            if dyn_ok:
                self._path1_v4_set_phase(
                    "cons",
                    global_step,
                    reason="cons_admitted",
                    accepted_phase="cons",
                )
                return
            if global_step >= self.path1_v4_cons_deadline_step:
                self._path1_v4_set_phase(
                    "tail_dyn",
                    global_step,
                    reason="cons_deadline_missed",
                    rejected_phase="cons",
                )
                self.path1_v4_cons_rejected = True
                return

        if phase == "cons":
            cons_last_improve = self.path1_v4_phase_last_improve_valid.get("cons", 0)
            cons_best = self.path1_v4_phase_best_metric.get("cons", float("inf"))
            prev_best = self.path1_v4_phase_best_metric.get("dyn", float("inf"))
            if (
                global_step >= self.path1_v4_cons_full_step
                and prev_best < float("inf")
                and cons_best < float("inf")
                and (self.selection_valid_count - cons_last_improve)
                >= self.path1_v4_reject_patience_valids
                and current_metric
                > float(prev_best) * (1.0 + self.path1_v4_reject_degrade_vs_prev_phase)
            ):
                self._path1_v4_set_phase(
                    "tail_dyn",
                    global_step,
                    reason="cons_reject",
                    rejected_phase="cons",
                )
                self.path1_v4_cons_rejected = True

    def _path1_v4_phase_state(self, global_step):
        if not self.path1_v4_enabled:
            return None
        koopman_enabled = self.koopman_model is not None and self.koopman_cfg.get("enabled", False)
        if not koopman_enabled:
            return None

        phase_name = self.path1_v4_controller_phase
        cons_cfg = self.koopman_cfg.get("consistency", {}) or {}
        cons1_target = max(
            0.0, float(cons_cfg.get("lambda_cons1_target", cons_cfg.get("lambda_cons1", 0.0)))
        )
        cons_multi_target = max(
            0.0, float(cons_cfg.get("lambda_cons_multi_target", cons_cfg.get("lambda_cons_multi", 0.0)))
        )
        cons_weight_sum = cons1_target + cons_multi_target
        cons1_frac = cons1_target / cons_weight_sum if cons_weight_sum > 0.0 else 0.0
        cons_multi_frac = cons_multi_target / cons_weight_sum if cons_weight_sum > 0.0 else 0.0

        dyn1_phase = self._cosine_ramp(
            global_step, self.path1_v4_dyn1_ramp_start_step, self.path1_v4_dyn1_full_step
        )
        dyn_multi_phase = self._cosine_ramp(
            global_step, self.path1_v4_dyn_multi_earliest_step, self.path1_v4_dyn_multi_full_step
        )
        cons_phase = self._cosine_ramp(
            global_step, self.path1_v4_cons_earliest_step, self.path1_v4_cons_full_step
        )

        target_ratio = 0.0
        share_dyn1 = 0.0
        share_dyn_multi = 0.0
        share_cons_total = 0.0

        if phase_name == "warmup" or phase_name == "tail_warmup":
            target_ratio = 0.0
        elif phase_name == "dyn1":
            target_ratio = self.path1_v4_target_ratio_dyn1_full * dyn1_phase
            share_dyn1 = 1.0 if dyn1_phase > 0.0 else 0.0
        elif phase_name == "dyn":
            target_ratio = (
                self.path1_v4_target_ratio_dyn1_full
                + (self.path1_v4_target_ratio_dyn_full - self.path1_v4_target_ratio_dyn1_full)
                * dyn_multi_phase
            )
            share_dyn_multi = self.path1_v4_dyn_multi_full_share * dyn_multi_phase
            share_dyn_multi = min(max(share_dyn_multi, 0.0), 0.95)
            share_dyn1 = max(0.0, 1.0 - share_dyn_multi)
        elif phase_name == "cons":
            target_ratio = (
                self.path1_v4_target_ratio_dyn_full
                + (self.path1_v4_target_ratio_cons_full - self.path1_v4_target_ratio_dyn_full)
                * cons_phase
            )
            share_dyn_multi = min(max(self.path1_v4_dyn_multi_full_share, 0.0), 0.95)
            share_cons_total = self.path1_v4_cons_total_full_share * cons_phase
            share_cons_total = min(max(share_cons_total, 0.0), 0.95)
            share_dyn1 = max(0.0, 1.0 - share_dyn_multi - share_cons_total)
        elif phase_name == "tail_dyn1":
            target_ratio = self.path1_v4_target_ratio_dyn1_full
            share_dyn1 = 1.0
        elif phase_name == "tail_dyn":
            target_ratio = self.path1_v4_target_ratio_dyn_full
            share_dyn_multi = min(max(self.path1_v4_dyn_multi_full_share, 0.0), 0.95)
            share_dyn1 = max(0.0, 1.0 - share_dyn_multi)

        share_cons1 = share_cons_total * cons1_frac
        share_cons_multi = share_cons_total * cons_multi_frac
        return {
            "phase": phase_name,
            "dyn1_phase": float(dyn1_phase),
            "dyn_multi_phase": float(dyn_multi_phase),
            "cons_phase": float(cons_phase),
            "target_ratio": float(target_ratio),
            "share_dyn1": float(share_dyn1),
            "share_dyn_multi": float(share_dyn_multi),
            "share_cons1": float(share_cons1),
            "share_cons_multi": float(share_cons_multi),
            "structured_state": bool(self.path1_v4_structured_state),
        }

    def _apply_path1_v4_schedule(self, global_step):
        if (not self.path1_v4_enabled) or self.koopman_regularizer is None:
            return None
        phase_state = self._path1_v4_phase_state(global_step)
        if phase_state is None:
            self.koopman_regularizer.clear_lambda_overrides()
            return None
        self.koopman_regularizer.set_lambda_overrides(
            lambda_k1=1.0 if phase_state["share_dyn1"] > 0.0 else 0.0,
            lambda_k_multi=1.0 if phase_state["share_dyn_multi"] > 0.0 else 0.0,
            lambda_cons1=1.0 if phase_state["share_cons1"] > 0.0 else 0.0,
            lambda_cons_multi=1.0 if phase_state["share_cons_multi"] > 0.0 else 0.0,
        )
        return phase_state

    @staticmethod
    def _reduce_weighted_scalar(weight_koop, per_sample, eps=1e-8):
        weight_sum = weight_koop.sum()
        return (weight_koop * per_sample).sum() / (weight_sum + eps)

    @staticmethod
    def _cap_consistency_contrib(
        dyn1_loss,
        dyn_multi_loss,
        cons1_loss,
        cons_multi_loss,
        lambda_k1,
        lambda_k_multi,
        lambda_cons1,
        lambda_cons_multi,
        cap_ratio,
        eps=1e-12,
    ):
        if th.is_tensor(dyn1_loss):
            ref = dyn1_loss
        elif th.is_tensor(dyn_multi_loss):
            ref = dyn_multi_loss
        elif th.is_tensor(cons1_loss):
            ref = cons1_loss
        elif th.is_tensor(cons_multi_loss):
            ref = cons_multi_loss
        else:
            ref = th.tensor(float(dyn1_loss), dtype=th.float32)

        def _to_tensor(value):
            if th.is_tensor(value):
                return value.to(device=ref.device, dtype=ref.dtype)
            return th.tensor(float(value), device=ref.device, dtype=ref.dtype)

        dyn1_loss = _to_tensor(dyn1_loss)
        dyn_multi_loss = _to_tensor(dyn_multi_loss)
        cons1_loss = _to_tensor(cons1_loss)
        cons_multi_loss = _to_tensor(cons_multi_loss)
        lambda_k1 = _to_tensor(lambda_k1)
        lambda_k_multi = _to_tensor(lambda_k_multi)
        lambda_cons1 = _to_tensor(lambda_cons1)
        lambda_cons_multi = _to_tensor(lambda_cons_multi)
        cap_ratio = _to_tensor(cap_ratio)

        dyn_contrib = lambda_k1 * dyn1_loss + lambda_k_multi * dyn_multi_loss
        cons_contrib_raw = lambda_cons1 * cons1_loss + lambda_cons_multi * cons_multi_loss

        if float(dyn_contrib.detach().item()) <= eps or float(cap_ratio.detach().item()) <= 0.0:
            cons_cap_scale = _to_tensor(0.0)
        elif float(cons_contrib_raw.detach().item()) > float((cap_ratio * dyn_contrib).detach().item()) and float(cons_contrib_raw.detach().item()) > eps:
            cons_cap_scale = (cap_ratio * dyn_contrib) / (cons_contrib_raw + _to_tensor(eps))
        else:
            cons_cap_scale = _to_tensor(1.0)

        lambda_cons1_capped = lambda_cons1 * cons_cap_scale
        lambda_cons_multi_capped = lambda_cons_multi * cons_cap_scale
        cons_contrib_capped = lambda_cons1_capped * cons1_loss + lambda_cons_multi_capped * cons_multi_loss
        koop_loss_capped = dyn_contrib + cons_contrib_capped
        return {
            "dyn_contrib": dyn_contrib,
            "cons_contrib_raw": cons_contrib_raw,
            "cons_contrib_capped": cons_contrib_capped,
            "koop_loss_capped": koop_loss_capped,
            "cons_cap_scale": cons_cap_scale,
            "lambda_cons1_capped": lambda_cons1_capped,
            "lambda_cons_multi_capped": lambda_cons_multi_capped,
        }

    def _should_trigger_path1_v2_early_stop(self, global_step):
        if not self.path1_v2_enabled:
            return False
        if not self._early_stop_enabled:
            return False
        if global_step < self._early_stop_start_step:
            return False
        if self.selection_best_step < 0:
            return False
        return (self.selection_valid_count - self.selection_last_improve_valid) >= self.selection_patience_valids

    def _should_trigger_path1_v3_early_stop(self, global_step):
        if not self.path1_v3_enabled:
            return False
        if not self._early_stop_enabled:
            return False
        if global_step < self._early_stop_start_step:
            return False
        if global_step < self.path1_v3_cons_start_step:
            return False
        if self.primary_best_step < 0:
            return False
        return (self.selection_valid_count - self.primary_last_improve_valid) >= self.selection_patience_valids

    def _should_trigger_path1_v4_early_stop(self, global_step):
        if not self.path1_v4_enabled:
            return False
        if not self._early_stop_enabled:
            return False
        if self.final_best_step < 0:
            return False
        phase_name = self.path1_v4_controller_phase
        if phase_name.startswith("tail_"):
            return (
                self.selection_valid_count - self.final_last_improve_valid
            ) >= self.selection_patience_valids
        if global_step < self._early_stop_start_step:
            return False
        if phase_name != "cons":
            return False
        return (
            self.selection_valid_count - self.final_last_improve_valid
        ) >= self.selection_patience_valids

    def _path1_v2_phase_lambdas(self, global_step):
        if not self.path1_v2_enabled:
            return None
        if self.koopman_model is None or (not self.koopman_cfg.get("enabled", False)):
            return None

        phase_cfg = self.koopman_cfg.get("phase_schedule", {}) or {}
        dyn_cfg = self.koopman_cfg.get("dyn", {}) or {}
        cons_cfg = self.koopman_cfg.get("consistency", {}) or {}

        dyn_start = self._resolve_schedule_point(
            explicit_step=phase_cfg.get("dyn_start_step", None),
            frac=phase_cfg.get(
                "dyn_start_frac",
                self.training_schedule_cfg.get("warmup_frac", 0.03),
            ),
            default=0,
        )
        dyn_full = self._resolve_schedule_point(
            explicit_step=phase_cfg.get("dyn_full_step", None),
            frac=phase_cfg.get(
                "dyn_full_frac",
                self.training_schedule_cfg.get("dyn_full_frac", 0.12),
            ),
            default=dyn_start,
        )
        cons_start = self._resolve_schedule_point(
            explicit_step=phase_cfg.get("cons_start_step", None),
            frac=phase_cfg.get(
                "cons_start_frac",
                self.training_schedule_cfg.get("cons_start_frac", 0.12),
            ),
            default=dyn_full,
        )
        cons_full = self._resolve_schedule_point(
            explicit_step=phase_cfg.get("cons_full_step", None),
            frac=phase_cfg.get(
                "cons_full_frac",
                self.training_schedule_cfg.get("cons_full_frac", 0.25),
            ),
            default=cons_start,
        )

        dyn_scale = self._cosine_ramp(global_step, dyn_start, dyn_full)
        cons_scale = self._cosine_ramp(global_step, cons_start, cons_full)

        lambda_k1_target = float(dyn_cfg.get("lambda_k1_target", dyn_cfg.get("lambda_k1", 0.0)))
        lambda_k_multi_target = float(
            dyn_cfg.get("lambda_k_multi_target", dyn_cfg.get("lambda_k_multi", 0.0))
        )
        lambda_cons1_target = float(
            cons_cfg.get("lambda_cons1_target", cons_cfg.get("lambda_cons1", 0.0))
        )
        lambda_cons_multi_target = float(
            cons_cfg.get("lambda_cons_multi_target", cons_cfg.get("lambda_cons_multi", 0.0))
        )
        lambda_k1_eff = lambda_k1_target * dyn_scale
        lambda_k_multi_eff = lambda_k_multi_target * dyn_scale
        lambda_cons1_eff = lambda_cons1_target * cons_scale
        lambda_cons_multi_eff = lambda_cons_multi_target * cons_scale

        return {
            "dyn_scale": float(dyn_scale),
            "cons_scale": float(cons_scale),
            "lambda_k1": float(lambda_k1_eff),
            "lambda_k_multi": float(lambda_k_multi_eff),
            "lambda_cons1": float(lambda_cons1_eff),
            "lambda_cons_multi": float(lambda_cons_multi_eff),
        }

    def _path1_v3_phase_state(self, global_step):
        if not self.path1_v3_enabled:
            return None
        koopman_enabled = self.koopman_model is not None and self.koopman_cfg.get("enabled", False)
        if not koopman_enabled:
            return None

        dyn_cfg = self.koopman_cfg.get("dyn", {}) or {}
        cons_cfg = self.koopman_cfg.get("consistency", {}) or {}
        balance_cfg = self.koopman_cfg.get("balance", {}) or {}
        target_ratio_cfg = balance_cfg.get("target_ratio_schedule", {}) or {}

        dyn1_phase = self._cosine_ramp(global_step, self.path1_v3_dyn_start_step, self.path1_v3_dyn1_full_step)
        dyn_multi_phase = self._cosine_ramp(global_step, self.path1_v3_dyn_multi_start_step, self.path1_v3_dyn_multi_full_step)
        cons_phase = self._cosine_ramp(global_step, self.path1_v3_cons_start_step, self.path1_v3_cons_full_step)

        branch_weights = {
            "dyn1": max(0.0, float(dyn_cfg.get("lambda_k1_target", dyn_cfg.get("lambda_k1", 0.0)))) * dyn1_phase,
            "dyn_multi": max(0.0, float(dyn_cfg.get("lambda_k_multi_target", dyn_cfg.get("lambda_k_multi", 0.0)))) * dyn_multi_phase,
            "cons1": max(0.0, float(cons_cfg.get("lambda_cons1_target", cons_cfg.get("lambda_cons1", 0.0)))) * cons_phase,
            "cons_multi": max(0.0, float(cons_cfg.get("lambda_cons_multi_target", cons_cfg.get("lambda_cons_multi", 0.0)))) * cons_phase,
        }
        total_branch_weight = sum(branch_weights.values())
        if total_branch_weight > 0.0:
            shares = {
                "share_dyn1": branch_weights["dyn1"] / total_branch_weight,
                "share_dyn_multi": branch_weights["dyn_multi"] / total_branch_weight,
                "share_cons1": branch_weights["cons1"] / total_branch_weight,
                "share_cons_multi": branch_weights["cons_multi"] / total_branch_weight,
            }
        else:
            shares = {
                "share_dyn1": 0.0,
                "share_dyn_multi": 0.0,
                "share_cons1": 0.0,
                "share_cons_multi": 0.0,
            }

        target_dyn1 = float(target_ratio_cfg.get("dyn1_full", balance_cfg.get("target_ratio", 0.0)))
        target_dyn_multi = float(target_ratio_cfg.get("dyn_multi_full", target_dyn1))
        target_cons = float(target_ratio_cfg.get("cons_full", target_dyn_multi))
        if global_step < self.path1_v3_dyn_start_step:
            target_ratio = 0.0
            phase_name = "warmup"
        elif global_step < self.path1_v3_dyn1_full_step:
            target_ratio = target_dyn1 * dyn1_phase
            phase_name = "dyn1"
        elif global_step < self.path1_v3_dyn_multi_start_step:
            target_ratio = target_dyn1
            phase_name = "dyn1_hold"
        elif global_step < self.path1_v3_dyn_multi_full_step:
            target_ratio = target_dyn1 + (target_dyn_multi - target_dyn1) * dyn_multi_phase
            phase_name = "dyn"
        elif global_step < self.path1_v3_cons_start_step:
            target_ratio = target_dyn_multi
            phase_name = "dyn_hold"
        elif global_step < self.path1_v3_cons_full_step:
            target_ratio = target_dyn_multi + (target_cons - target_dyn_multi) * cons_phase
            phase_name = "dyn_cons"
        else:
            target_ratio = target_cons
            phase_name = "dyn_cons_hold"

        return {
            "phase": phase_name,
            "dyn1_phase": float(dyn1_phase),
            "dyn_multi_phase": float(dyn_multi_phase),
            "cons_phase": float(cons_phase),
            "target_ratio": float(target_ratio),
            **shares,
        }

    def _apply_path1_v2_schedule(self, global_step):
        if (not self.path1_v2_enabled) or self.koopman_regularizer is None:
            return None
        lambdas = self._path1_v2_phase_lambdas(global_step)
        if lambdas is None:
            self.koopman_regularizer.clear_lambda_overrides()
            return None
        self.koopman_regularizer.set_lambda_overrides(
            lambda_k1=lambdas["lambda_k1"],
            lambda_k_multi=lambdas["lambda_k_multi"],
            lambda_cons1=lambdas["lambda_cons1"],
            lambda_cons_multi=lambdas["lambda_cons_multi"],
        )
        return lambdas

    def _apply_path1_v3_schedule(self, global_step):
        if (not self.path1_v3_enabled) or self.koopman_regularizer is None:
            return None
        phase_state = self._path1_v3_phase_state(global_step)
        if phase_state is None:
            self.koopman_regularizer.clear_lambda_overrides()
            return None
        self.koopman_regularizer.set_lambda_overrides(
            lambda_k1=1.0 if phase_state["share_dyn1"] > 0.0 else 0.0,
            lambda_k_multi=1.0 if phase_state["share_dyn_multi"] > 0.0 else 0.0,
            lambda_cons1=1.0 if phase_state["share_cons1"] > 0.0 else 0.0,
            lambda_cons_multi=1.0 if phase_state["share_cons_multi"] > 0.0 else 0.0,
        )
        return phase_state

    def _path1_v3_reduce_branch_losses(self, weight_koop, loss_info):
        required = (
            "loss_dyn1_per_sample",
            "loss_dyn_multi_per_sample",
            "loss_cons1_per_sample",
            "loss_cons_multi_per_sample",
        )
        if any(loss_info.get(key, None) is None for key in required):
            return None
        return {
            "dyn1_loss": self._reduce_weighted_scalar(weight_koop, loss_info["loss_dyn1_per_sample"]),
            "dyn_multi_loss": self._reduce_weighted_scalar(weight_koop, loss_info["loss_dyn_multi_per_sample"]),
            "cons1_loss": self._reduce_weighted_scalar(weight_koop, loss_info["loss_cons1_per_sample"]),
            "cons_multi_loss": self._reduce_weighted_scalar(weight_koop, loss_info["loss_cons_multi_per_sample"]),
        }

    def _compose_path1_v3_koop_loss(self, *, loss_info, weight_koop, phase_state):
        branch_losses = self._path1_v3_reduce_branch_losses(weight_koop, loss_info)
        if branch_losses is None:
            return None
        share_dyn1 = float(phase_state.get("share_dyn1", 0.0))
        share_dyn_multi = float(phase_state.get("share_dyn_multi", 0.0))
        share_cons1 = float(phase_state.get("share_cons1", 0.0))
        share_cons_multi = float(phase_state.get("share_cons_multi", 0.0))
        cons_cfg = self.koopman_cfg.get("consistency", {}) or {}
        cap_ratio = float(cons_cfg.get("cap_ratio_vs_dyn", cons_cfg.get("cap_ratio", 1.0)))
        contrib_info = self._cap_consistency_contrib(
            dyn1_loss=branch_losses["dyn1_loss"],
            dyn_multi_loss=branch_losses["dyn_multi_loss"],
            cons1_loss=branch_losses["cons1_loss"],
            cons_multi_loss=branch_losses["cons_multi_loss"],
            lambda_k1=share_dyn1,
            lambda_k_multi=share_dyn_multi,
            lambda_cons1=share_cons1,
            lambda_cons_multi=share_cons_multi,
            cap_ratio=cap_ratio,
        )
        return {
            "branch_losses": branch_losses,
            "contrib_info": contrib_info,
            "koop_loss": contrib_info["koop_loss_capped"],
            "share_dyn1": share_dyn1,
            "share_dyn_multi": share_dyn_multi,
            "share_cons1": share_cons1,
            "share_cons_multi": share_cons_multi,
            "share_cons1_capped": float(contrib_info["lambda_cons1_capped"].detach().item()),
            "share_cons_multi_capped": float(contrib_info["lambda_cons_multi_capped"].detach().item()),
        }

    def _compose_path1_v4_koop_loss(self, *, loss_info, weight_koop, phase_state):
        branch_losses = self._path1_v3_reduce_branch_losses(weight_koop, loss_info)
        if branch_losses is None:
            return None
        share_dyn1 = float(phase_state.get("share_dyn1", 0.0))
        share_dyn_multi = float(phase_state.get("share_dyn_multi", 0.0))
        share_cons1 = float(phase_state.get("share_cons1", 0.0))
        share_cons_multi = float(phase_state.get("share_cons_multi", 0.0))
        cons_cfg = self.koopman_cfg.get("consistency", {}) or {}
        cap_ratio = float(cons_cfg.get("cap_ratio_vs_dyn", cons_cfg.get("cap_ratio", 1.0)))
        contrib_info = self._cap_consistency_contrib(
            dyn1_loss=branch_losses["dyn1_loss"],
            dyn_multi_loss=branch_losses["dyn_multi_loss"],
            cons1_loss=branch_losses["cons1_loss"],
            cons_multi_loss=branch_losses["cons_multi_loss"],
            lambda_k1=share_dyn1,
            lambda_k_multi=share_dyn_multi,
            lambda_cons1=share_cons1,
            lambda_cons_multi=share_cons_multi,
            cap_ratio=cap_ratio,
        )
        return {
            "branch_losses": branch_losses,
            "contrib_info": contrib_info,
            "koop_loss": contrib_info["koop_loss_capped"],
            "share_dyn1": share_dyn1,
            "share_dyn_multi": share_dyn_multi,
            "share_cons1": share_cons1,
            "share_cons_multi": share_cons_multi,
            "share_cons1_capped": float(contrib_info["lambda_cons1_capped"].detach().item()),
            "share_cons_multi_capped": float(contrib_info["lambda_cons_multi_capped"].detach().item()),
        }

    def _maybe_restore_best_state_from_report(self):
        if not self.resume_checkpoint:
            return
        try:
            _, _, report_filename = self._best_artifact_names(self.best_filename)
            report_path = bf.join(get_blob_logdir(), report_filename)
            if not bf.exists(report_path):
                return
            with bf.BlobFile(report_path, "rb") as f:
                report = json.loads(f.read().decode("utf-8"))
            metric_ema = report.get("metric_ema", report.get("best_metric_ema", None))
            metric_raw = report.get("metric_raw", report.get("best_metric_raw", None))
            step = report.get("step", report.get("best_step", None))
            if metric_ema is None or step is None:
                return
            self.best_metric = float(metric_ema)
            self.best_metric_ema = float(metric_ema)
            self.best_step = int(step)
            self.valid_count = max(self.valid_count, self.best_warmup_valid_count + 1)
            logger.log(
                f"[best/resume] restored previous best from {report_path}: "
                f"step={self.best_step}, raw={metric_raw}, ema={self.best_metric:.6f}"
            )
        except Exception as exc:
            logger.log(f"[best/resume] failed to restore previous best state: {exc}")

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            state_or_payload = dist_util.load_state_dict(
                resume_checkpoint, map_location=dist_util.dev()
            )
            model_state = state_or_payload
            if isinstance(state_or_payload, dict) and "model_state_dict" in state_or_payload:
                model_state = state_or_payload["model_state_dict"]
                if self.resume_step == 0 and self.resume_step_override is None:
                    payload_step = state_or_payload.get("step", None)
                    if payload_step is not None:
                        try:
                            self.resume_step = int(payload_step)
                        except Exception:
                            pass
            if self.resume_step_override is not None:
                self.resume_step = int(self.resume_step_override)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(model_state)

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if not bf.exists(opt_checkpoint):
            # Restore the optimizer and additional state from opt_best.pt.
            basename = bf.basename(main_checkpoint)
            opt_candidates = []
            if basename.startswith("model_best") or basename.startswith("checkpoint_best"):
                suffix = "best"
                if basename.startswith("model_"):
                    suffix = basename[len("model_") : -3]
                elif basename.startswith("checkpoint_"):
                    suffix = basename[len("checkpoint_") : -3]
                opt_candidates.append(bf.join(bf.dirname(main_checkpoint), f"opt_{suffix}.pt"))
            opt_candidates.append(bf.join(bf.dirname(main_checkpoint), "opt_best.pt"))
            for candidate in opt_candidates:
                if bf.exists(candidate):
                    opt_checkpoint = candidate
                    break
        if bf.exists(opt_checkpoint):
            logger.log(f"loading training state from checkpoint: {opt_checkpoint}")
            payload = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            if isinstance(payload, dict) and "optimizer_state_dict" in payload:
                opt_state = payload.get("optimizer_state_dict", None)
                reg_state = payload.get("koopman_regularizer_state", None)
                bal_state = payload.get("lambda_balancer_state", None)
                path1_v4_state = payload.get("path1_v4_state", None)
                path1_v5_state = payload.get("path1_v5_state", None)
            else:
                # Also accept opt*.pt containing optimizer.state_dict() directly.
                opt_state = payload
                reg_state = None
                bal_state = None
                path1_v4_state = None
                path1_v5_state = None

            if self.resume_optimizer and (opt_state is not None):
                self.opt.load_state_dict(opt_state)
                logger.log("[resume] optimizer state loaded.")
            elif not self.resume_optimizer:
                logger.log("[resume] resume_optimizer=False; skipping optimizer state restoration.")

            if (
                self.koopman_regularizer is not None
                and reg_state is not None
                and hasattr(self.koopman_regularizer, "load_state_dict")
            ):
                self.koopman_regularizer.load_state_dict(reg_state)
                logger.log("[resume] koopman_regularizer state loaded.")

            if (
                self.lambda_balancer is not None
                and bal_state is not None
                and hasattr(self.lambda_balancer, "load_state_dict")
            ):
                self.lambda_balancer.load_state_dict(bal_state)
                logger.log("[resume] lambda_balancer state loaded.")
            if self.path1_v4_enabled and path1_v4_state is not None:
                self._load_path1_v4_state_dict(path1_v4_state)
                logger.log("[resume] path1_v4 controller state loaded.")
            if self.path1_v5_enabled and path1_v5_state is not None:
                self._load_path1_v5_state_dict(path1_v5_state)
                logger.log("[resume] path1_v5 controller state loaded.")

    def run_loop(self):
        def should_continue():
            if self._early_stop_triggered:
                return False
            if self.max_train_steps and self.max_train_steps > 0:
                return self.step + self.resume_step < self.max_train_steps
            return (not self.lr_anneal_steps) or (
                self.step + self.resume_step < self.lr_anneal_steps
            )

        while should_continue():
            train_batch, train_indices = self._unpack_train_batch(next(self.train_data))
            val_info = self.run_step(train_batch, None, None, train_indices=train_indices) #self.run_step(batch, cond)
            if val_info is not None:
                self._maybe_update_best(val_info)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            global_step = self.step + self.resume_step
            if self.checkpoint_policy.enabled:
                if (
                    not self.disable_checkpoint_saving
                    and self.checkpoint_policy.should_save_ema(global_step)
                ):
                    self.save_ema_only()
                    if self.checkpoint_policy.should_snapshot_best(global_step):
                        self.snapshot_lightweight_best(global_step)
            elif self.step % self.save_interval == 0:
                if not self.disable_checkpoint_saving:
                    self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        if self.disable_checkpoint_saving:
            return
        if self.checkpoint_policy.enabled:
            global_step = self.step + self.resume_step
            if (
                not self.disable_checkpoint_saving
                and self.checkpoint_policy.should_snapshot_best(global_step)
            ):
                self.snapshot_lightweight_best(global_step)
            if self.checkpoint_policy.terminal_full:
                self.save()
            elif self.checkpoint_policy.should_save_ema(global_step):
                self.save_ema_only()
            return
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def _unpack_train_batch(self, full_batch):
        if isinstance(full_batch, (list, tuple)):
            if len(full_batch) == 2:
                return full_batch[0], full_batch[1]
            if len(full_batch) == 1:
                return full_batch[0], None
            raise ValueError(f"unexpected train batch structure: len={len(full_batch)}")
        return full_batch, None

    def run_step(self, train_batch, valid_batch, cond, train_indices=None):
        do_valid = (self.step + self.resume_step) % self.valid_interval == 0
        valid_info = None
        if cond:
            self.forward_backward(train_batch, cond[0], batch_indices=train_indices)
            if do_valid:
                valid_info = self.forward_valid(valid_batch, cond[1])
        else:
            self.forward_backward(train_batch, batch_indices=train_indices)
            if do_valid:
                valid_info = self.forward_valid(valid_batch)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return valid_info

    def _koopman_loss(self, x0_hat, window_size):
        """
        x0_hat: [B, 1, T, D] or [B, T, D] depending on data shape
        window_size: Window length L+1.
        Returns: A dict containing loss_per_sample, num_per_sample and den_per_sample.
        """
        # Return zero losses when no regularizer is initialized.
        if self.koopman_regularizer is None:
            if x0_hat.dim() == 4 and x0_hat.shape[1] == 1:
                bsz = x0_hat.shape[0]
            else:
                bsz = x0_hat.shape[0]
            zeros = th.zeros(bsz, device=x0_hat.device, dtype=x0_hat.dtype)
            return {
                "loss_per_sample": zeros,
                "num_per_sample": zeros,
                "den_per_sample": zeros,
                "stats": {},
            }
        return self.koopman_regularizer.per_sample_loss(x0_hat, window_size)

    def _backward_pcgrad(self, base_loss, koop_term, params):
        """
        PCGrad backward pass.
        Compute diffusion and koop_term gradients separately, project conflicts
        in the Koopman gradients, then combine them in param.grad with
        microbatch accumulation.
        """
        if not params:
            return {"method": "pcgrad", "cos": 0.0, "conflict": 0.0, "proj_coef": 0.0}

        grads_diff = list(
            th.autograd.grad(
                base_loss,
                params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )
        )
        grads_koop = list(
            th.autograd.grad(
                koop_term,
                params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )
        )
        proj_koop, pc_stats = self.lambda_balancer.project_koop_grads(grads_diff, grads_koop)

        for p, gd, gk in zip(params, grads_diff, proj_koop):
            g_total = None
            if gd is not None:
                g_total = gd.detach()
            if gk is not None:
                gk_det = gk.detach()
                g_total = gk_det if g_total is None else (g_total + gk_det)
            if g_total is None:
                continue
            if p.grad is None:
                p.grad = g_total
            else:
                p.grad.add_(g_total)
        return pc_stats

    def forward_backward(self, batch, cond={}, batch_indices=None):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_indices = None
            if batch_indices is not None:
                micro_indices = batch_indices[i : i + self.microbatch]
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
             }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            trace_noise = None
            trace_payload = None
            if self.training_trace_enabled:
                cpu_rng_before = th.get_rng_state()
                cuda_rng_before = (
                    th.cuda.get_rng_state(dist_util.dev())
                    if th.cuda.is_available()
                    else None
                )
                trace_noise = th.randn_like(micro)
                trace_payload = {
                    "step": int(self.step + self.resume_step),
                    "micro_offset": int(i),
                    "batch_sha256": self._trace_tensor_sha256(micro),
                    "timestep_sha256": self._trace_tensor_sha256(t),
                    "weights_sha256": self._trace_tensor_sha256(weights),
                    "noise_sha256": self._trace_tensor_sha256(trace_noise),
                    "cpu_rng_before_noise_sha256": self._trace_tensor_sha256(cpu_rng_before),
                    "cpu_rng_after_noise_sha256": self._trace_tensor_sha256(th.get_rng_state()),
                    "cuda_rng_before_noise_sha256": (
                        self._trace_tensor_sha256(cuda_rng_before)
                        if cuda_rng_before is not None
                        else None
                    ),
                    "cuda_rng_after_noise_sha256": (
                        self._trace_tensor_sha256(th.cuda.get_rng_state(dist_util.dev()))
                        if th.cuda.is_available()
                        else None
                    ),
                }
            current_step = self.step + self.resume_step
            schedule_lambdas = self._apply_path1_v2_schedule(current_step) if self.path1_v2_enabled else None
            phase_state_v3 = self._apply_path1_v3_schedule(current_step) if self.path1_v3_enabled else None
            phase_state_v4 = self._apply_path1_v4_schedule(current_step) if self.path1_v4_enabled else None
            phase_state_v5 = self._apply_path1_v5_schedule(current_step) if self.path1_v5_enabled else None
            if self.path1_v3_enabled:
                self._path1_v3_state = {
                    "phase": (phase_state_v3 or {}).get("phase", "unknown"),
                    "phase_state": dict(phase_state_v3 or {}),
                    "lambda_koop_eff": 0.0,
                }
            if self.path1_v4_enabled:
                self._path1_v4_state = {
                    "phase": (phase_state_v4 or {}).get("phase", "unknown"),
                    "phase_state": dict(phase_state_v4 or {}),
                    "lambda_koop_eff": 0.0,
                }
            if self.path1_v5_enabled:
                self._path1_v5_state = {
                    "phase": (phase_state_v5 or {}).get("phase", "unknown"),
                    "phase_state": dict(phase_state_v5 or {}),
                    "lambda_koop_eff": 0.0,
                }

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                noise=trace_noise,
                valid=False
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            if trace_payload is not None:
                self._append_training_trace(trace_payload)

            # Remove auxiliary terms before passing losses to log_loss_dict.
            x0_hat_aux = losses.pop("pred_xstart", None)
            eps_aux = losses.pop("eps", None)

            base_loss = (losses["loss"] * weights).mean()
            loss = base_loss
            manual_backward_done = False
            pcgrad_stats = {}

            # path-lite distilled skeleton prior
            if self.distilled_skeleton_enabled and self.koopman_surrogate is not None:
                x0_hat = x0_hat_aux
                if x0_hat is None and eps_aux is not None:
                    x0_hat = self.diffusion._predict_xstart_from_eps(micro, t, eps_aux)
                if x0_hat is not None and micro_indices is not None:
                    skel_info = self.koopman_surrogate.compute_train_loss(
                        x0_hat=x0_hat,
                        t=t,
                        batch_indices=micro_indices,
                        global_step=current_step,
                        diff_loss_scalar=base_loss,
                    )
                    loss = loss + skel_info["loss"]
                    logger.logkv_mean("koopman/skel_loss_raw", float(skel_info["loss_raw"].detach().item()))
                    logger.logkv_mean("koopman/skel_loss", float(skel_info["loss"].detach().item()))
                    logger.logkv_mean("koopman/lambda_eff", float(skel_info["lambda_eff"]))
                    logger.logkv_mean("koopman/active_frac", float(skel_info["active_frac"]))
                    logger.logkv_mean("koopman/reliable_frac", float(skel_info["reliable_frac"]))
                    logger.logkv_mean("koopman/noise_gate_mean", float(skel_info["noise_gate_mean"]))
                    logger.logkv_mean("koopman/resid_abs_median", float(skel_info["resid_abs_median"]))

            # Configured noise-gated prior regularization.
            elif self.path1_v6_enabled and self.koopman_v6_regularizer is not None:
                x0_hat = x0_hat_aux
                if x0_hat is None and eps_aux is not None:
                    x0_hat = self.diffusion._predict_xstart_from_eps(micro, t, eps_aux)
                if x0_hat is not None:
                    v6_info = self.koopman_v6_regularizer.compute_train_loss(
                        x0_hat=x0_hat,
                        t=t,
                        global_step=current_step,
                        diff_loss_scalar=base_loss,
                    )
                    loss = loss + v6_info["loss"]
                    logger.logkv_mean(
                        "koopman/loss_dyn_raw",
                        float(v6_info["loss_dyn_raw"].detach().item()),
                    )
                    logger.logkv_mean(
                        "koopman/loss_man_raw",
                        float(v6_info["loss_man_raw"].detach().item()),
                    )
                    logger.logkv_mean(
                        "koopman/loss_prior_raw",
                        float(v6_info["loss_prior_raw"].detach().item()),
                    )
                    logger.logkv_mean("koopman/lambda_eff", float(v6_info["lambda_eff"]))
                    logger.logkv_mean("koopman/lambda_dyn_eff", float(v6_info.get("lambda_dyn_eff", 0.0)))
                    logger.logkv_mean("koopman/lambda_man_eff", float(v6_info.get("lambda_man_eff", 0.0)))
                    logger.logkv_mean("koopman/active_frac", float(v6_info["active_frac"]))
                    logger.logkv_mean("koopman/subbatch_frac", float(v6_info["subbatch_frac"]))
                    logger.logkv_mean("koopman/dyn_conf_mean", float(v6_info.get("dyn_conf_mean", 0.0)))
                    logger.logkv_mean(
                        "koopman/dyn_conf_active_frac",
                        float(v6_info.get("dyn_conf_active_frac", 0.0)),
                    )
                    logger.logkv_mean(
                        "koopman/dyn_conf_max_mean",
                        float(v6_info.get("dyn_conf_max_mean", 0.0)),
                    )
                    logger.logkv_mean(
                        "koopman/dyn_conf_margin_mean",
                        float(v6_info.get("dyn_conf_margin_mean", 0.0)),
                    )
                    logger.logkv_mean(
                        "koopman/dyn_conf_entropy_mean",
                        float(v6_info.get("dyn_conf_entropy_mean", 0.0)),
                    )
                    logger.logkv_mean(
                        "koopman/dyn_conf_compound_mean",
                        float(v6_info.get("dyn_conf_compound_mean", 0.0)),
                    )
                    logger.logkv_mean(
                        "koopman/dyn_unknown_frac",
                        float(v6_info.get("dyn_unknown_frac", 0.0)),
                    )
                    logger.logkv_mean("koopman/man_conf_mean", float(v6_info.get("man_conf_mean", 0.0)))
                    logger.logkv_mean(
                        "koopman/man_conf_active_frac",
                        float(v6_info.get("man_conf_active_frac", 0.0)),
                    )
                    logger.logkv_mean("koopman/dyn_rel", float(v6_info["dyn_rel"]))
                    logger.logkv_mean("koopman/man_rel", float(v6_info["man_rel"]))

            elif self.generic_regularizer is not None:
                x0_hat = x0_hat_aux
                if x0_hat is None and eps_aux is not None:
                    x0_hat = self.diffusion._predict_xstart_from_eps(micro, t, eps_aux)
                if x0_hat is not None:
                    gen_info = self.generic_regularizer.compute_train_loss(
                        x0_hat=x0_hat,
                        t=t,
                        global_step=current_step,
                        diff_loss_scalar=base_loss,
                    )
                    loss = loss + gen_info["loss"]
                    logger.logkv_mean(
                        "generic_reg/loss_raw",
                        float(gen_info["loss_raw"].detach().item()),
                    )
                    logger.logkv_mean(
                        "generic_reg/loss",
                        float(gen_info["loss"].detach().item()),
                    )
                    logger.logkv_mean("generic_reg/lambda_eff", float(gen_info["lambda_eff"]))
                    logger.logkv_mean("generic_reg/active_frac", float(gen_info["active_frac"]))
                    logger.logkv_mean("generic_reg/subbatch_frac", float(gen_info["subbatch_frac"]))
                    logger.logkv_mean("generic_reg/noise_gate_mean", float(gen_info["noise_gate_mean"]))

            # Koopman regularization.
            elif self.koopman_model is not None and self.koopman_cfg.get("enabled", False):
                lambda_k = float(self.koopman_cfg.get("lambda_k", 0.0))
                if lambda_k > 0:
                    koop_interval = int(self.koopman_cfg.get("interval", 1))
                    if koop_interval > 1 and (self.step + self.resume_step) % koop_interval != 0:
                        log_loss_dict(
                            self.diffusion, t, {k: v * weights for k, v in losses.items()}, prefix="train"
                        )
                        self.mp_trainer.backward(loss)
                        continue
                    if koop_interval > 1:
                        lambda_k = lambda_k * koop_interval
                    window_size = int(self.koopman_cfg.get("window_size", 16))
                    # Optionally include sampling weights in the Koopman term.
                    weight_info = self.koopman_regularizer.sample_weights(
                        t, schedule_weights=weights
                    )
                    weight_koop = weight_info["weight"]
                    beta = weight_info["beta"]
                    alpha_bar = weight_info["alpha_bar"]
                    # Prefer pred_xstart, the predicted-clean latent estimate.
                    x0_hat = x0_hat_aux
                    if x0_hat is None:
                        eps = eps_aux
                        if eps is not None:
                            x0_hat = self.diffusion._predict_xstart_from_eps(micro, t, eps)
                    if x0_hat is not None:
                        loss_info = self._koopman_loss(x0_hat, window_size)
                        loss_per_sample = loss_info["loss_per_sample"]
                        num_per_sample = loss_info["num_per_sample"]
                        den_per_sample = loss_info["den_per_sample"]
                        reduce_info = self.koopman_regularizer.reduce_weighted(
                            weight_koop,
                            loss_per_sample,
                            num_per_sample,
                            den_per_sample,
                        )
                        koop_loss = reduce_info["koop_loss"]
                        koop_num = reduce_info["koop_num"]
                        koop_den = reduce_info["koop_den"]
                        contrib_info = None
                        path_schedule_comp = None
                        effective_schedule_lambdas = dict(schedule_lambdas) if schedule_lambdas is not None else None
                        if (
                            self.path1_v2_enabled
                            and self.koopman_regularizer_mode == "modern"
                            and schedule_lambdas is not None
                        ):
                            loss_dyn1_per_sample = loss_info.get("loss_dyn1_per_sample", None)
                            loss_dyn_multi_per_sample = loss_info.get("loss_dyn_multi_per_sample", None)
                            loss_cons1_per_sample = loss_info.get("loss_cons1_per_sample", None)
                            loss_cons_multi_per_sample = loss_info.get("loss_cons_multi_per_sample", None)
                            if (
                                loss_dyn1_per_sample is not None
                                and loss_dyn_multi_per_sample is not None
                                and loss_cons1_per_sample is not None
                                and loss_cons_multi_per_sample is not None
                            ):
                                dyn1_loss = self._reduce_weighted_scalar(weight_koop, loss_dyn1_per_sample)
                                dyn_multi_loss = self._reduce_weighted_scalar(weight_koop, loss_dyn_multi_per_sample)
                                cons1_loss = self._reduce_weighted_scalar(weight_koop, loss_cons1_per_sample)
                                cons_multi_loss = self._reduce_weighted_scalar(weight_koop, loss_cons_multi_per_sample)
                                cons_cfg = self.koopman_cfg.get("consistency", {}) or {}
                                cap_ratio = float(
                                    cons_cfg.get("cap_ratio_vs_dyn", cons_cfg.get("cap_ratio", 1.0))
                                )
                                contrib_info = self._cap_consistency_contrib(
                                    dyn1_loss=dyn1_loss,
                                    dyn_multi_loss=dyn_multi_loss,
                                    cons1_loss=cons1_loss,
                                    cons_multi_loss=cons_multi_loss,
                                    lambda_k1=schedule_lambdas["lambda_k1"],
                                    lambda_k_multi=schedule_lambdas["lambda_k_multi"],
                                    lambda_cons1=schedule_lambdas["lambda_cons1"],
                                    lambda_cons_multi=schedule_lambdas["lambda_cons_multi"],
                                    cap_ratio=cap_ratio,
                                )
                                koop_loss = contrib_info["koop_loss_capped"]
                                effective_schedule_lambdas["lambda_cons1"] = float(
                                    contrib_info["lambda_cons1_capped"].detach().item()
                                )
                                effective_schedule_lambdas["lambda_cons_multi"] = float(
                                    contrib_info["lambda_cons_multi_capped"].detach().item()
                                )
                        elif (
                            self.path1_v3_enabled
                            and self.koopman_regularizer_mode == "modern"
                            and phase_state_v3 is not None
                        ):
                            path_schedule_comp = self._compose_path1_v3_koop_loss(
                                loss_info=loss_info,
                                weight_koop=weight_koop,
                                phase_state=phase_state_v3,
                            )
                            if path_schedule_comp is not None:
                                contrib_info = path_schedule_comp["contrib_info"]
                                koop_loss = path_schedule_comp["koop_loss"]
                        elif (
                            self.path1_v4_enabled
                            and self.koopman_regularizer_mode == "modern"
                            and phase_state_v4 is not None
                        ):
                            path_schedule_comp = self._compose_path1_v4_koop_loss(
                                loss_info=loss_info,
                                weight_koop=weight_koop,
                                phase_state=phase_state_v4,
                            )
                            if path_schedule_comp is not None:
                                contrib_info = path_schedule_comp["contrib_info"]
                                koop_loss = path_schedule_comp["koop_loss"]
                        elif (
                            self.path1_v5_enabled
                            and self.koopman_regularizer_mode == "modern"
                            and phase_state_v5 is not None
                        ):
                            path_schedule_comp = self._compose_path1_v5_koop_loss(
                                loss_info=loss_info,
                                weight_koop=weight_koop,
                                phase_state=phase_state_v5,
                            )
                            if path_schedule_comp is not None:
                                contrib_info = path_schedule_comp["contrib_info"]
                                koop_loss = path_schedule_comp["koop_loss"]
                        # The balancer returns lambda_eff; fixed mode uses lambda_k.
                        path_schedule_active = False
                        if phase_state_v3 is not None:
                            path_schedule_active = (
                                phase_state_v3.get("share_dyn1", 0.0)
                                + phase_state_v3.get("share_dyn_multi", 0.0)
                                + phase_state_v3.get("share_cons1", 0.0)
                                + phase_state_v3.get("share_cons_multi", 0.0)
                            ) > 0.0
                        if phase_state_v4 is not None:
                            path_schedule_active = (
                                phase_state_v4.get("share_dyn1", 0.0)
                                + phase_state_v4.get("share_dyn_multi", 0.0)
                                + phase_state_v4.get("share_cons1", 0.0)
                                + phase_state_v4.get("share_cons_multi", 0.0)
                            ) > 0.0
                        if phase_state_v5 is not None:
                            path_schedule_active = (
                                phase_state_v5.get("share_dyn1", 0.0)
                                + phase_state_v5.get("share_dyn_multi", 0.0)
                                + phase_state_v5.get("share_cons1", 0.0)
                                + phase_state_v5.get("share_cons_multi", 0.0)
                            ) > 0.0
                        if self.lambda_balancer is not None:
                            # Collect parameters only when gradient norms are needed.
                            if getattr(self.lambda_balancer, "needs_grad_norm", False):
                                balance_params = [p for p in self.ddp_model.parameters() if p.requires_grad]
                            else:
                                balance_params = []
                            bal_stats = {}
                            if self.path_schedule_enabled and not path_schedule_active:
                                lambda_eff = 0.0
                            else:
                                target_ratio_override = None
                                if self.path1_v3_enabled and phase_state_v3 is not None:
                                    target_ratio_override = phase_state_v3.get("target_ratio", None)
                                if self.path1_v4_enabled and phase_state_v4 is not None:
                                    target_ratio_override = phase_state_v4.get("target_ratio", None)
                                if self.path1_v5_enabled and phase_state_v5 is not None:
                                    target_ratio_override = phase_state_v5.get("target_ratio", None)
                                lambda_eff, bal_stats = self.lambda_balancer.compute_lambda(
                                    base_loss=base_loss,
                                    koop_loss=koop_loss,
                                    params=balance_params,
                                    step=self.step + self.resume_step,
                                    base_lambda_override=lambda_k,
                                    target_ratio_override=target_ratio_override,
                                    n_eff=reduce_info["stats"].get("n_eff", None),
                                    active_frac=reduce_info["stats"].get("active_frac", None),
                                )
                        else:
                            lambda_eff = 0.0 if (self.path_schedule_enabled and not path_schedule_active) else lambda_k
                            bal_stats = {}

                        koop_term = lambda_eff * koop_loss
                        loss = loss + koop_term
                        self._path1_v3_state = {}
                        self._path1_v4_state = self._path1_v4_state if self.path1_v4_enabled else {}
                        self._path1_v5_state = self._path1_v5_state if self.path1_v5_enabled else {}
                        if self.path1_v4_enabled and phase_state_v4 is not None:
                            self._path1_v4_state = {
                                "phase": phase_state_v4.get("phase", "unknown"),
                                "phase_state": dict(phase_state_v4),
                                "lambda_koop_eff": float(lambda_eff),
                            }
                        if self.path1_v5_enabled and phase_state_v5 is not None:
                            self._path1_v5_state = {
                                "phase": phase_state_v5.get("phase", "unknown"),
                                "phase_state": dict(phase_state_v5),
                                "lambda_koop_eff": float(lambda_eff),
                            }
                        # Project gradient conflicts and write back gradients.
                        if self.pcgrad_enabled and self.pcgrad_supported:
                            pcgrad_params = [p for p in self.ddp_model.parameters() if p.requires_grad]
                            pcgrad_stats = self._backward_pcgrad(base_loss, koop_term, pcgrad_params)
                            manual_backward_done = True
                        # Log Koopman diagnostics.
                        logger.logkv_mean("koopman/weight", float(weight_koop.mean().item()))
                        logger.logkv_mean("koopman/lambda_k", lambda_k)
                        logger.logkv_mean("koopman/lambda_eff", float(lambda_eff))
                        if effective_schedule_lambdas is not None:
                            logger.logkv_mean("koopman/dyn_scale", effective_schedule_lambdas["dyn_scale"])
                            logger.logkv_mean("koopman/cons_scale", effective_schedule_lambdas["cons_scale"])
                            logger.logkv_mean("koopman/lambda_k1_eff", effective_schedule_lambdas["lambda_k1"])
                            logger.logkv_mean("koopman/lambda_k_multi_eff", effective_schedule_lambdas["lambda_k_multi"])
                            logger.logkv_mean("koopman/lambda_cons1_eff", effective_schedule_lambdas["lambda_cons1"])
                            logger.logkv_mean("koopman/lambda_cons_multi_eff", effective_schedule_lambdas["lambda_cons_multi"])
                        if phase_state_v3 is not None:
                            logger.logkv_mean("koopman/phase_dyn1", float(phase_state_v3["dyn1_phase"]))
                            logger.logkv_mean("koopman/phase_dyn_multi", float(phase_state_v3["dyn_multi_phase"]))
                            logger.logkv_mean("koopman/phase_cons", float(phase_state_v3["cons_phase"]))
                            logger.logkv_mean("koopman/share_dyn1", float(phase_state_v3["share_dyn1"]))
                            logger.logkv_mean("koopman/share_dyn_multi", float(phase_state_v3["share_dyn_multi"]))
                            logger.logkv_mean("koopman/share_cons1", float(phase_state_v3["share_cons1"]))
                            logger.logkv_mean("koopman/share_cons_multi", float(phase_state_v3["share_cons_multi"]))
                            logger.logkv_mean("koopman/target_ratio_phase", float(phase_state_v3["target_ratio"]))
                        if phase_state_v4 is not None:
                            logger.logkv_mean("koopman/phase_dyn1", float(phase_state_v4["dyn1_phase"]))
                            logger.logkv_mean("koopman/phase_dyn_multi", float(phase_state_v4["dyn_multi_phase"]))
                            logger.logkv_mean("koopman/phase_cons", float(phase_state_v4["cons_phase"]))
                            logger.logkv_mean("koopman/share_dyn1", float(phase_state_v4["share_dyn1"]))
                            logger.logkv_mean("koopman/share_dyn_multi", float(phase_state_v4["share_dyn_multi"]))
                            logger.logkv_mean("koopman/share_cons1", float(phase_state_v4["share_cons1"]))
                            logger.logkv_mean("koopman/share_cons_multi", float(phase_state_v4["share_cons_multi"]))
                            logger.logkv_mean("koopman/target_ratio_phase", float(phase_state_v4["target_ratio"]))
                        if phase_state_v5 is not None:
                            logger.logkv_mean("phase/progress_dyn1", float(phase_state_v5["progress_dyn1"]))
                            logger.logkv_mean("phase/progress_dyn_probe", float(phase_state_v5["progress_dyn_probe"]))
                            logger.logkv_mean("phase/progress_dyn", float(phase_state_v5["progress_dyn"]))
                            logger.logkv_mean("phase/progress_cons_probe", float(phase_state_v5["progress_cons_probe"]))
                            logger.logkv_mean("phase/progress_cons", float(phase_state_v5["progress_cons"]))
                            logger.logkv_mean("koopman/share_dyn1_active", float(phase_state_v5["share_dyn1"]))
                            logger.logkv_mean("koopman/share_dyn_multi_active", float(phase_state_v5["share_dyn_multi"]))
                            logger.logkv_mean("koopman/share_cons1_active", float(phase_state_v5["share_cons1"]))
                            logger.logkv_mean("koopman/share_cons_multi_active", float(phase_state_v5["share_cons_multi"]))
                            logger.logkv_mean("koopman/target_ratio_phase", float(phase_state_v5["target_ratio"]))
                        logger.logkv_mean("koopman/beta", beta)
                        # alpha_bar(t) is the cumulative noise-schedule factor.
                        if alpha_bar is not None:
                            logger.logkv_mean("koopman/alpha_bar", float(alpha_bar.mean().item()))
                        logger.logkv_mean("koopman/loss_k", koop_loss.item())
                        logger.logkv_mean("koopman/num", koop_num.item())
                        logger.logkv_mean("koopman/den", koop_den.item())
                        logger.logkv_mean("koopman/lambda_loss", koop_term.item())
                        if base_loss.item() != 0:
                            ratio = (koop_term / base_loss).item()
                            if koop_interval > 1:
                                ratio = ratio / koop_interval
                            logger.logkv_mean("koopman/lambda_loss_ratio", ratio)
                        # GradNorm diagnostics; fixed mode does not update the weight.
                        if bal_stats:
                            if "g_diff" in bal_stats:
                                logger.logkv_mean("gradnorm/g_diff", float(bal_stats["g_diff"]))
                            if "g_koop" in bal_stats:
                                logger.logkv_mean("gradnorm/g_koop", float(bal_stats["g_koop"]))
                            if "target" in bal_stats:
                                logger.logkv_mean("gradnorm/lambda_target", float(bal_stats["target"]))
                            if "raw_target" in bal_stats:
                                logger.logkv_mean("gradnorm/raw_target", float(bal_stats["raw_target"]))
                            if "updated" in bal_stats:
                                logger.logkv_mean("gradnorm/updated", float(bal_stats["updated"]))
                            if "skipped_low_g_koop" in bal_stats:
                                logger.logkv_mean("gradnorm/skipped_low_g_koop", float(bal_stats["skipped_low_g_koop"]))
                            if "skipped_low_support" in bal_stats:
                                logger.logkv_mean("gradnorm/skipped_low_support", float(bal_stats["skipped_low_support"]))
                            if "cos" in bal_stats:
                                logger.logkv_mean("pcgrad/cos", float(bal_stats["cos"]))
                            if "conflict" in bal_stats:
                                logger.logkv_mean("pcgrad/conflict", float(bal_stats["conflict"]))
                            if "proj_coef" in bal_stats:
                                logger.logkv_mean("pcgrad/proj_coef", float(bal_stats["proj_coef"]))
                            g_diff = float(bal_stats.get("g_diff", 0.0) or 0.0)
                            g_koop = float(bal_stats.get("g_koop", 0.0) or 0.0)
                            if g_diff > 0.0 and math.isfinite(g_koop):
                                eff_ratio = float(lambda_eff) * g_koop / g_diff
                                logger.logkv_mean("koopman/effective_grad_ratio", eff_ratio)
                                self._record_support_stat("effective_grad_ratio", eff_ratio)
                        if pcgrad_stats:
                            if "cos" in pcgrad_stats:
                                logger.logkv_mean("pcgrad/cos", float(pcgrad_stats["cos"]))
                            if "conflict" in pcgrad_stats:
                                logger.logkv_mean("pcgrad/conflict", float(pcgrad_stats["conflict"]))
                            if "proj_coef" in pcgrad_stats:
                                logger.logkv_mean("pcgrad/proj_coef", float(pcgrad_stats["proj_coef"]))
                        # Monitor extrapolation, scale drift and weight collapse.
                        if self.koopman_regularizer_mode == "modern":
                            loss_stats = loss_info.get("stats", {})
                            reduce_stats = reduce_info.get("stats", {})
                            loss_dyn_per_sample = loss_info.get("loss_dyn_per_sample", None)
                            loss_dyn1_per_sample = loss_info.get("loss_dyn1_per_sample", None)
                            loss_dyn_multi_per_sample = loss_info.get("loss_dyn_multi_per_sample", None)
                            loss_cons_per_sample = loss_info.get("loss_cons_per_sample", None)
                            loss_cons1_per_sample = loss_info.get("loss_cons1_per_sample", None)
                            loss_cons_multi_per_sample = loss_info.get("loss_cons_multi_per_sample", None)
                            lambda_cons = float(loss_info.get("lambda_cons", 0.0))
                            lambda_cons1 = float(loss_info.get("lambda_cons1", 0.0))
                            lambda_cons_multi = float(loss_info.get("lambda_cons_multi", 0.0))
                            if contrib_info is not None and path_schedule_comp is None:
                                logger.logkv_mean(
                                    "koopman/dyn_contrib",
                                    float(contrib_info["dyn_contrib"].detach().item()),
                                )
                                logger.logkv_mean(
                                    "koopman/cons_contrib_raw",
                                    float(contrib_info["cons_contrib_raw"].detach().item()),
                                )
                                logger.logkv_mean(
                                    "koopman/cons_contrib_capped",
                                    float(contrib_info["cons_contrib_capped"].detach().item()),
                                )
                                logger.logkv_mean(
                                    "koopman/cons_cap_scale",
                                    float(contrib_info["cons_cap_scale"].detach().item()),
                                )
                                lambda_cons1 = float(contrib_info["lambda_cons1_capped"].detach().item())
                                lambda_cons_multi = float(contrib_info["lambda_cons_multi_capped"].detach().item())
                                lambda_cons = lambda_cons1 + lambda_cons_multi
                            wsum = reduce_stats.get("weight_sum", weight_koop.sum())
                            if loss_dyn_per_sample is not None:
                                loss_dyn = (weight_koop * loss_dyn_per_sample).sum() / (wsum + 1e-8)
                                if path_schedule_comp is not None:
                                    loss_dyn = (
                                        path_schedule_comp["share_dyn1"] * path_schedule_comp["branch_losses"]["dyn1_loss"]
                                        + path_schedule_comp["share_dyn_multi"] * path_schedule_comp["branch_losses"]["dyn_multi_loss"]
                                    )
                                logger.logkv_mean("koopman/loss_dyn", float(loss_dyn.item()))
                            if loss_dyn1_per_sample is not None:
                                loss_dyn1 = (weight_koop * loss_dyn1_per_sample).sum() / (wsum + 1e-8)
                                logger.logkv_mean("koopman/loss_dyn1", float(loss_dyn1.item()))
                            if loss_dyn_multi_per_sample is not None:
                                loss_dyn_multi = (weight_koop * loss_dyn_multi_per_sample).sum() / (wsum + 1e-8)
                                logger.logkv_mean("koopman/loss_dyn_multi", float(loss_dyn_multi.item()))
                            if loss_cons_per_sample is not None:
                                if path_schedule_comp is not None:
                                    loss_cons = (
                                        path_schedule_comp["share_cons1_capped"] * path_schedule_comp["branch_losses"]["cons1_loss"]
                                        + path_schedule_comp["share_cons_multi_capped"] * path_schedule_comp["branch_losses"]["cons_multi_loss"]
                                    )
                                elif contrib_info is not None:
                                    loss_cons = contrib_info["cons_contrib_capped"]
                                else:
                                    loss_cons = (weight_koop * loss_cons_per_sample).sum() / (wsum + 1e-8)
                                logger.logkv_mean("koopman/loss_cons", float(loss_cons.item()))
                                logger.logkv_mean("koopman/lambda_cons", lambda_cons)
                            if loss_cons1_per_sample is not None:
                                loss_cons1 = (weight_koop * loss_cons1_per_sample).sum() / (wsum + 1e-8)
                                logger.logkv_mean("koopman/loss_cons1", float(loss_cons1.item()))
                                logger.logkv_mean("koopman/lambda_cons1", lambda_cons1)
                            if loss_cons_multi_per_sample is not None:
                                loss_cons_multi = (weight_koop * loss_cons_multi_per_sample).sum() / (wsum + 1e-8)
                                logger.logkv_mean("koopman/loss_cons_multi", float(loss_cons_multi.item()))
                                logger.logkv_mean("koopman/lambda_cons_multi", lambda_cons_multi)
                            if "overflow_rate" in loss_stats:
                                logger.logkv_mean(
                                    "koopman/overflow_rate",
                                    float(loss_stats["overflow_rate"].item()),
                                )
                            if "scale" in loss_stats:
                                logger.logkv_mean(
                                    "koopman/scale",
                                    float(loss_stats["scale"].item()),
                                )
                            if "scale_dyn" in loss_stats:
                                logger.logkv_mean(
                                    "koopman/scale_dyn",
                                    float(loss_stats["scale_dyn"].item()),
                                )
                            if "scale_dyn1" in loss_stats:
                                logger.logkv_mean(
                                    "koopman/scale_dyn1",
                                    float(loss_stats["scale_dyn1"].item()),
                                )
                            if "scale_dyn_multi" in loss_stats:
                                logger.logkv_mean(
                                    "koopman/scale_dyn_multi",
                                    float(loss_stats["scale_dyn_multi"].item()),
                                )
                            if "raw_dyn1_abs_median" in loss_stats:
                                logger.logkv_mean(
                                    "koopman/raw_dyn1_abs_median",
                                    float(loss_stats["raw_dyn1_abs_median"].item()),
                                )
                            if "raw_dyn_multi_abs_median" in loss_stats:
                                logger.logkv_mean(
                                    "koopman/raw_dyn_multi_abs_median",
                                    float(loss_stats["raw_dyn_multi_abs_median"].item()),
                                )
                            if "scale_cons" in loss_stats:
                                logger.logkv_mean(
                                    "koopman/scale_cons",
                                    float(loss_stats["scale_cons"].item()),
                                )
                            if "scale_cons1" in loss_stats:
                                logger.logkv_mean(
                                    "koopman/scale_cons1",
                                    float(loss_stats["scale_cons1"].item()),
                                )
                            if "scale_cons_multi" in loss_stats:
                                logger.logkv_mean(
                                    "koopman/scale_cons_multi",
                                    float(loss_stats["scale_cons_multi"].item()),
                                )
                            if "n_eff" in reduce_stats:
                                logger.logkv_mean(
                                    "koopman/N_eff",
                                    float(reduce_stats["n_eff"].item()),
                                )
                            if "active_frac" in reduce_stats:
                                logger.logkv_mean(
                                    "koopman/weight_active_frac",
                                    float(reduce_stats["active_frac"].item()),
                                )
                            if "weight_sum" in reduce_stats:
                                logger.logkv_mean(
                                    "koopman/weight_sum",
                                    float(reduce_stats["weight_sum"].item()),
                                )
                            if path_schedule_comp is not None and contrib_info is not None:
                                dyn_contrib = float((lambda_eff * contrib_info["dyn_contrib"]).detach().item())
                                cons_contrib_raw = float((lambda_eff * contrib_info["cons_contrib_raw"]).detach().item())
                                cons_contrib_capped = float((lambda_eff * contrib_info["cons_contrib_capped"]).detach().item())
                                cons_cap_scale = float(contrib_info["cons_cap_scale"].detach().item())
                                logger.logkv_mean("koopman/dyn_contrib", dyn_contrib)
                                logger.logkv_mean("koopman/cons_contrib_raw", cons_contrib_raw)
                                logger.logkv_mean("koopman/cons_contrib_capped", cons_contrib_capped)
                                logger.logkv_mean("koopman/cons_cap_scale", cons_cap_scale)
                            if self.path1_v3_enabled and phase_state_v3 is not None:
                                self._path1_v3_state = {
                                    "phase": phase_state_v3.get("phase", "unknown"),
                                    "phase_state": dict(phase_state_v3),
                                    "lambda_koop_eff": float(lambda_eff),
                                }
                            if self.path1_v4_enabled and phase_state_v4 is not None:
                                self._path1_v4_state = {
                                    "phase": phase_state_v4.get("phase", "unknown"),
                                    "phase_state": dict(phase_state_v4),
                                    "lambda_koop_eff": float(lambda_eff),
                                }
                            if self.path1_v5_enabled and phase_state_v5 is not None:
                                self._path1_v5_state = {
                                    "phase": phase_state_v5.get("phase", "unknown"),
                                    "phase_state": dict(phase_state_v5),
                                    "lambda_koop_eff": float(lambda_eff),
                                }
                        if "n_eff" in reduce_info["stats"]:
                            self._record_support_stat("n_eff", float(reduce_info["stats"]["n_eff"].item()))
                        if "weight_sum" in reduce_info["stats"]:
                            self._record_support_stat("weight_sum", float(reduce_info["stats"]["weight_sum"].item()))
                        overflow_value = None
                        if self.koopman_regularizer_mode == "modern":
                            overflow_value = loss_info.get("stats", {}).get("overflow_rate", None)
                        if overflow_value is not None:
                            self._record_support_stat("overflow_rate", float(overflow_value.item()))
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}, prefix="train"
            )
            if not manual_backward_done:
                self.mp_trainer.backward(loss)
            
    def forward_valid(self, batch, cond={}):
        del batch
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        train_interval_support_summary = self._consume_support_summary()

        if self.valid_data is None:
            return None

        valid_diff_loss = None
        valid_support_summary = {}
        if rank == 0:
            with th.no_grad():
                valid_noise_generator = self._new_valid_noise_generator()
                valid_loader_generator = getattr(self.valid_data, "generator", None)
                if self.validation_rng_enabled:
                    if valid_loader_generator is None:
                        raise RuntimeError(
                            "validation RNG isolation requires DataLoader.generator"
                        )
                    valid_loader_generator.manual_seed(self.valid_loader_seed)
                agg_stats = defaultdict(lambda: [0.0, 0])
                support_values_valid = defaultdict(list)
                do_pathlite_audit = (
                    self.distilled_skeleton_enabled
                    and self.koopman_surrogate is not None
                    and self.koopman_surrogate.audit_enabled
                    and ((self.step + self.resume_step) % self.save_interval == 0)
                )
                audit_sums = defaultdict(float)
                valid_seq_offset = 0
                for full_batch in self.valid_data:
                    if isinstance(full_batch, (list, tuple)):
                        if len(full_batch) == 2:
                            batch_tensor, batch_indices = full_batch
                        elif len(full_batch) == 1:
                            batch_tensor = full_batch[0]
                            batch_indices = None
                        else:
                            raise ValueError(f"unexpected valid batch structure: len={len(full_batch)}")
                    else:
                        batch_tensor = full_batch
                        batch_indices = None

                    for i in range(0, batch_tensor.shape[0], self.microbatch):
                        micro = batch_tensor[i : i + self.microbatch].to(dist_util.dev())
                        micro_cond = {
                            k: v[i : i + self.microbatch].to(dist_util.dev())
                            for k, v in cond.items()
                        }
                        micro_indices = None
                        if batch_indices is not None:
                            micro_indices = batch_indices[i : i + self.microbatch]
                        t = self._get_valid_timesteps(
                            batch_indices=micro_indices,
                            batch_offset=valid_seq_offset,
                            batch_size=micro.shape[0],
                        )
                        valid_seq_offset += micro.shape[0]
                        weights = th.ones_like(t, dtype=th.float32, device=dist_util.dev())
                        valid_noise = self._sample_valid_noise(micro, valid_noise_generator)

                        losses = self.diffusion.training_losses(
                            self.ddp_model,
                            micro,
                            t,
                            model_kwargs=micro_cond,
                            noise=valid_noise,
                            valid=True,
                        )
                        x0_hat_aux = losses.pop("pred_xstart", None)
                        eps_aux = losses.pop("eps", None)
                        accumulate_loss_stats(
                            agg_stats,
                            self.diffusion,
                            t,
                            {k: v * weights for k, v in losses.items()},
                        )
                        if do_pathlite_audit and micro_indices is not None:
                            x0_hat = x0_hat_aux
                            if x0_hat is None and eps_aux is not None:
                                x0_hat = self.diffusion._predict_xstart_from_eps(micro, t, eps_aux)
                            if x0_hat is not None:
                                audit_selector = micro_indices < self.koopman_surrogate.audit_subset_size
                                if bool(audit_selector.any()):
                                    audit_local_idx = audit_selector.nonzero(as_tuple=False).view(-1)
                                    audit_local_idx_dev = audit_local_idx.to(device=dist_util.dev())
                                    audit_info = self.koopman_surrogate.compute_valid_audit(
                                        x0_hat=x0_hat.index_select(0, audit_local_idx_dev),
                                        t=t.index_select(0, audit_local_idx_dev),
                                        batch_indices=micro_indices.index_select(0, audit_local_idx),
                                    )
                                    audit_sums["count"] += float(audit_info["count"])
                                    audit_sums["active_count"] += float(audit_info["active_count"])
                                    audit_sums["loss_raw_sum"] += float(audit_info["loss_raw"]) * float(audit_info["count"])
                                    audit_sums["active_frac_sum"] += float(audit_info["active_frac"]) * float(audit_info["count"])
                                    audit_sums["reliable_frac_sum"] += float(audit_info["reliable_frac"]) * float(audit_info["count"])
                                    audit_sums["noise_gate_mean_sum"] += float(audit_info["noise_gate_mean"]) * float(audit_info["count"])
                                    audit_sums["resid_abs_median_sum"] += float(audit_info["resid_abs_median"]) * float(audit_info["count"])

                        if (
                            self.koopman_model is not None
                            and self.koopman_cfg.get("enabled", False)
                            and (not self.path1_v6_enabled)
                            and (not self.distilled_skeleton_enabled)
                        ):
                            x0_hat = x0_hat_aux
                            if x0_hat is None and eps_aux is not None:
                                x0_hat = self.diffusion._predict_xstart_from_eps(micro, t, eps_aux)
                            if x0_hat is not None:
                                window_size = int(self.koopman_cfg.get("window_size", 16))
                                weight_info = self.koopman_regularizer.sample_weights(
                                    t, schedule_weights=weights
                                )
                                weight_koop = weight_info["weight"]
                                loss_info = self._koopman_loss(x0_hat, window_size)
                                reduce_info = self.koopman_regularizer.reduce_weighted(
                                    weight_koop,
                                    loss_info["loss_per_sample"],
                                    loss_info["num_per_sample"],
                                    loss_info["den_per_sample"],
                                )
                                reduce_stats = reduce_info.get("stats", {})
                                if "n_eff" in reduce_stats:
                                    support_values_valid["n_eff"].append(float(reduce_stats["n_eff"].item()))
                                if "weight_sum" in reduce_stats:
                                    support_values_valid["weight_sum"].append(float(reduce_stats["weight_sum"].item()))
                                overflow_value = loss_info.get("stats", {}).get("overflow_rate", None)
                                if overflow_value is not None:
                                    support_values_valid["overflow_rate"].append(float(overflow_value.item()))

                valid_metrics = finalize_loss_stats(agg_stats, prefix="valid")
                valid_support_summary = self._summarize_support_values(support_values_valid)
                if "valid/loss" in valid_metrics:
                    valid_diff_loss = float(valid_metrics["valid/loss"])
                    logger.logkv("valid_diff_loss", valid_diff_loss)
                for key, value in valid_metrics.items():
                    if key == "valid/loss":
                        continue
                    logger.logkv(key, value)
                if "n_eff_median" in valid_support_summary:
                    logger.logkv("valid_support/N_eff_median", float(valid_support_summary["n_eff_median"]))
                if "weight_sum_median" in valid_support_summary:
                    logger.logkv("valid_support/weight_sum_median", float(valid_support_summary["weight_sum_median"]))
                if "overflow_rate_p95" in valid_support_summary:
                    logger.logkv("valid_support/overflow_rate_p95", float(valid_support_summary["overflow_rate_p95"]))
                if audit_sums.get("count", 0.0) > 0.0:
                    denom = audit_sums["count"]
                    logger.logkv("koopman_audit/loss_raw", audit_sums["loss_raw_sum"] / denom)
                    logger.logkv("koopman_audit/active_frac", audit_sums["active_frac_sum"] / denom)
                    logger.logkv("koopman_audit/reliable_frac", audit_sums["reliable_frac_sum"] / denom)
                    logger.logkv("koopman_audit/noise_gate_mean", audit_sums["noise_gate_mean_sum"] / denom)
                    logger.logkv("koopman_audit/resid_abs_median", audit_sums["resid_abs_median_sum"] / denom)

        valid_support_keys = (
            "n_eff_median",
            "n_eff_p95",
            "weight_sum_median",
            "weight_sum_p95",
            "overflow_rate_median",
            "overflow_rate_p95",
        )
        if world_size > 1:
            sync_tensor = th.tensor(
                [0.0 if valid_diff_loss is None else valid_diff_loss],
                dtype=th.float32,
                device=dist_util.dev(),
            )
            dist.broadcast(sync_tensor, src=0)
            valid_diff_loss = float(sync_tensor.item())
            support_tensor = th.tensor(
                [
                    float(valid_support_summary.get(key, float("nan")))
                    for key in valid_support_keys
                ],
                dtype=th.float32,
                device=dist_util.dev(),
            )
            dist.broadcast(support_tensor, src=0)
            valid_support_summary = {}
            for key, value in zip(valid_support_keys, support_tensor.tolist()):
                if math.isfinite(value):
                    valid_support_summary[key] = float(value)
            dist.barrier()

        if valid_diff_loss is not None:
            info = {"valid_diff_loss": valid_diff_loss}
            if train_interval_support_summary:
                info["support_summary_train_interval"] = train_interval_support_summary
            if valid_support_summary:
                info["support_summary_valid"] = valid_support_summary
            if self.path1_v3_enabled:
                structured_summary = self._merge_path1_v3_structured_support(
                    valid_support_summary,
                    train_interval_support_summary,
                )
                if structured_summary:
                    info["support_summary_guard_merged"] = structured_summary
                    info["support_summary"] = structured_summary
            elif self.path1_v4_enabled:
                structured_summary = self._merge_path1_v4_structured_support(
                    valid_support_summary,
                    train_interval_support_summary,
                )
                if structured_summary:
                    info["support_summary_guard_merged"] = structured_summary
                    info["support_summary"] = structured_summary
            elif self.path1_v5_enabled:
                structured_summary = self._merge_path1_v5_structured_support(
                    valid_support_summary,
                    train_interval_support_summary,
                )
                if structured_summary:
                    info["support_summary_guard_merged"] = structured_summary
                    info["support_summary"] = structured_summary
            else:
                merged_support_summary = self._merge_support_summaries(
                    valid_support_summary,
                    train_interval_support_summary,
                )
                if merged_support_summary:
                    info["support_summary_guard_merged"] = merged_support_summary
                    info["support_summary"] = merged_support_summary
            return info
        return None

    def _maybe_update_best(self, valid_info):
        if self.path1_v2_enabled:
            self._maybe_update_best_window(valid_info)
            return
        if self.path1_v3_enabled:
            self._maybe_update_best_path1_v3(valid_info)
            return
        if self.path1_v4_enabled:
            self._maybe_update_best_path1_v4(valid_info)
            return
        if self.path1_v5_enabled:
            self._maybe_update_best_path1_v5(valid_info)
            return
        if (not self.best_ckpt_enabled) or (not valid_info):
            return

        metric_raw = valid_info.get(self.best_metric_name, None)
        if metric_raw is None:
            return
        metric_raw = float(metric_raw)
        if not math.isfinite(metric_raw):
            logger.logkv_mean("best/updated", 0.0)
            return

        self.valid_count += 1
        if self.best_metric_ema is None:
            self.best_metric_ema = metric_raw
        else:
            decay = self.best_ema_decay
            self.best_metric_ema = decay * float(self.best_metric_ema) + (1.0 - decay) * metric_raw

        logger.logkv_mean("best/raw", metric_raw)
        logger.logkv_mean("best/ema", float(self.best_metric_ema))
        if math.isfinite(self.best_metric):
            logger.logkv_mean("best/current_best", float(self.best_metric))

        if self.valid_count <= self.best_warmup_valid_count:
            logger.logkv_mean("best/updated", 0.0)
            self._append_best_trace(
                metric_raw=metric_raw,
                metric_ema=float(self.best_metric_ema),
                eligible=False,
                updated=False,
            )
            return

        metric_candidate = float(self.best_metric_ema)
        updated = False
        if metric_candidate < (self.best_metric - self.best_min_delta):
            self.best_metric = metric_candidate
            self.best_step = self.step + self.resume_step
            self._save_best_checkpoint(metric_raw=metric_raw, metric_ema=metric_candidate)
            best_artifact = self.best_filename
            if self.checkpoint_policy.lightweight_best:
                model_filename = self._best_artifact_names(self.best_filename)[0]
                suffix = model_filename[len("model_") : -3]
                best_artifact = f"ema_{self.ema_rate[0]}_{suffix}.pt"
            logger.log(
                f"[best] updated step={self.best_step}, "
                f"raw={metric_raw:.6f}, ema={metric_candidate:.6f}, file={best_artifact}"
            )
            logger.logkv_mean("best/updated", 1.0)
            updated = True
        else:
            logger.logkv_mean("best/updated", 0.0)

        if self.best_step >= 0:
            logger.logkv_mean("best/step", float(self.best_step))
        self._append_best_trace(
            metric_raw=metric_raw,
            metric_ema=metric_candidate,
            eligible=True,
            updated=updated,
        )

    def _append_best_trace(self, metric_raw, metric_ema, eligible, updated):
        """Append one machine-readable validation-selector event when configured."""

        if dist.get_rank() != 0 or not self.best_trace_filename:
            return
        payload = {
            "step": int(self.step + self.resume_step),
            "valid_count": int(self.valid_count),
            "metric_raw": float(metric_raw),
            "metric_ema": float(metric_ema),
            "eligible": bool(eligible),
            "updated": bool(updated),
            "best_metric_ema": (
                float(self.best_metric) if math.isfinite(self.best_metric) else None
            ),
            "best_step": int(self.best_step),
            "selector": {
                "ema_decay": float(self.best_ema_decay),
                "warmup_valid_count": int(self.best_warmup_valid_count),
                "min_delta": float(self.best_min_delta),
            },
        }
        trace_path = os.path.join(get_blob_logdir(), self.best_trace_filename)
        with open(trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    @staticmethod
    def _trace_tensor_sha256(tensor):
        value = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
        return digest.hexdigest()

    def _append_training_trace(self, payload):
        """Append stochastic-input provenance for explicitly enabled smoke tests."""

        if dist.get_rank() != 0:
            return
        path = os.path.join(get_blob_logdir(), self.training_trace_filename)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _maybe_update_best_window(self, valid_info):
        if not valid_info:
            return
        metric_raw = valid_info.get(self.selection_metric_name, None)
        if metric_raw is None:
            return
        metric_raw = float(metric_raw)
        if not math.isfinite(metric_raw):
            return
        self.selection_valid_count += 1
        if self.selection_metric_ema is None:
            self.selection_metric_ema = metric_raw
        else:
            alpha = self.selection_ema_alpha
            self.selection_metric_ema = (1.0 - alpha) * float(self.selection_metric_ema) + alpha * metric_raw

        support_summary = valid_info.get("support_summary_guard_merged", valid_info.get("support_summary", {})) or {}
        support_summary_valid = valid_info.get("support_summary_valid", {}) or {}
        support_summary_train = valid_info.get("support_summary_train_interval", {}) or {}
        support_ok = self._support_guard_ok(support_summary)
        logger.logkv_mean("best_window/raw", metric_raw)
        logger.logkv_mean("best_window/ema", float(self.selection_metric_ema))
        logger.logkv_mean("best_window/support_ok", 1.0 if support_ok else 0.0)
        if math.isfinite(self.selection_best_metric):
            logger.logkv_mean("best_window/current_best", float(self.selection_best_metric))

        global_step = self.step + self.resume_step
        improved = False
        if support_ok and float(self.selection_metric_ema) < (
            self.selection_best_metric - self.selection_min_delta
        ):
            self.selection_best_metric = float(self.selection_metric_ema)
            self.selection_best_step = int(global_step)
            self.selection_best_raw = metric_raw
            self.selection_best_support = dict(support_summary)
            self.selection_last_improve_valid = self.selection_valid_count
            self._save_best_checkpoint(
                metric_raw=metric_raw,
                metric_ema=float(self.selection_metric_ema),
                filename=self.selection_filename,
                extra_report={
                    "support_summary_valid": support_summary_valid,
                    "support_summary_train_interval": support_summary_train,
                    "support_summary_guard_merged": support_summary,
                    "mode": "path1_v2",
                },
            )
            logger.log(
                f"[best_window] updated step={self.selection_best_step}, "
                f"raw={metric_raw:.6f}, ema={float(self.selection_metric_ema):.6f}, "
                f"file={self.selection_filename}"
            )
            improved = True

        logger.logkv_mean("best_window/updated", 1.0 if improved else 0.0)
        if self.selection_best_step >= 0:
            logger.logkv_mean("best_window/step", float(self.selection_best_step))
        self._write_best_window_report(
            current_step=global_step,
            metric_raw=metric_raw,
            metric_ema=float(self.selection_metric_ema),
            support_summary_valid=support_summary_valid,
            support_summary_train=support_summary_train,
            support_summary_guard=support_summary,
            support_ok=support_ok,
        )
        if self._should_trigger_path1_v2_early_stop(global_step):
            self._early_stop_triggered = True
            self._early_stop_reason = (
                f"no best_window improvement for {self.selection_patience_valids} validations"
            )
            logger.log(f"[path1_v2] early stop triggered: {self._early_stop_reason}")

    def _maybe_update_best_path1_v3(self, valid_info):
        if not valid_info:
            return
        metric_raw = valid_info.get(self.selection_metric_name, None)
        if metric_raw is None:
            return
        metric_raw = float(metric_raw)
        if not math.isfinite(metric_raw):
            return

        self.selection_valid_count += 1
        if self.selection_metric_ema is None:
            self.selection_metric_ema = metric_raw
        else:
            alpha = self.selection_ema_alpha
            self.selection_metric_ema = (1.0 - alpha) * float(self.selection_metric_ema) + alpha * metric_raw

        global_step = self.step + self.resume_step
        phase_state = self._path1_v3_phase_state(global_step) or {}
        current_phase = phase_state.get("phase", "unknown")
        support_summary_valid = valid_info.get("support_summary_valid", {}) or {}
        support_summary_train = valid_info.get("support_summary_train_interval", {}) or {}
        structured_summary = valid_info.get("support_summary_guard_merged", {}) or {}
        structured_ok = self._path1_v3_structured_guard_ok(structured_summary)

        logger.logkv_mean("best_bc/raw", metric_raw)
        logger.logkv_mean("best_bc/ema", float(self.selection_metric_ema))
        logger.logkv_mean("best_bc/eligible", 1.0 if global_step >= self.path1_v3_dyn_start_step else 0.0)
        logger.logkv_mean("best_structured/raw", metric_raw)
        logger.logkv_mean("best_structured/ema", float(self.selection_metric_ema))
        logger.logkv_mean("best_structured/support_ok", 1.0 if structured_ok else 0.0)
        if math.isfinite(self.primary_best_metric):
            logger.logkv_mean("best_bc/current_best", float(self.primary_best_metric))
        if math.isfinite(self.structured_best_metric):
            logger.logkv_mean("best_structured/current_best", float(self.structured_best_metric))

        primary_improved = False
        if global_step >= self.path1_v3_dyn_start_step and float(self.selection_metric_ema) < (
            self.primary_best_metric - self.selection_min_delta
        ):
            self.primary_best_metric = float(self.selection_metric_ema)
            self.primary_best_step = int(global_step)
            self.primary_best_raw = metric_raw
            self.primary_best_phase = current_phase
            self.primary_last_improve_valid = self.selection_valid_count
            self._save_best_checkpoint(
                metric_raw=metric_raw,
                metric_ema=float(self.selection_metric_ema),
                filename=self.primary_filename,
                extra_report={
                    "mode": "path1_v3",
                    "kind": "best_bc_diff",
                    "phase": current_phase,
                    "phase_state": phase_state,
                    "support_summary_valid": support_summary_valid,
                    "support_summary_train_interval": support_summary_train,
                    "support_summary_guard_merged": structured_summary,
                    "support_ok": bool(structured_ok),
                },
            )
            logger.log(
                f"[best_bc] updated step={self.primary_best_step}, "
                f"raw={metric_raw:.6f}, ema={float(self.selection_metric_ema):.6f}, "
                f"file={self.primary_filename}"
            )
            primary_improved = True

        structured_improved = False
        if structured_ok and global_step >= self.path1_v3_dyn_start_step and float(self.selection_metric_ema) < (
            self.structured_best_metric - self.selection_min_delta
        ):
            self.structured_best_metric = float(self.selection_metric_ema)
            self.structured_best_step = int(global_step)
            self.structured_best_raw = metric_raw
            self.structured_best_phase = current_phase
            self.structured_best_support = dict(structured_summary)
            self.structured_last_improve_valid = self.selection_valid_count
            self._save_best_checkpoint(
                metric_raw=metric_raw,
                metric_ema=float(self.selection_metric_ema),
                filename=self.structured_filename,
                extra_report={
                    "mode": "path1_v3",
                    "kind": "best_structured",
                    "phase": current_phase,
                    "phase_state": phase_state,
                    "support_summary_valid": support_summary_valid,
                    "support_summary_train_interval": support_summary_train,
                    "support_summary_guard_merged": structured_summary,
                    "support_ok": True,
                },
            )
            logger.log(
                f"[best_structured] updated step={self.structured_best_step}, "
                f"raw={metric_raw:.6f}, ema={float(self.selection_metric_ema):.6f}, "
                f"file={self.structured_filename}"
            )
            structured_improved = True

        logger.logkv_mean("best_bc/updated", 1.0 if primary_improved else 0.0)
        logger.logkv_mean("best_structured/updated", 1.0 if structured_improved else 0.0)
        if self.primary_best_step >= 0:
            logger.logkv_mean("best_bc/step", float(self.primary_best_step))
        if self.structured_best_step >= 0:
            logger.logkv_mean("best_structured/step", float(self.structured_best_step))

        self._write_path1_v3_report(
            report_filename=self.primary_report_filename,
            kind="best_bc_diff",
            current_step=global_step,
            current_phase=current_phase,
            phase_state=phase_state,
            metric_raw=metric_raw,
            metric_ema=float(self.selection_metric_ema),
            support_summary_valid=support_summary_valid,
            support_summary_train=support_summary_train,
            support_summary_guard=structured_summary,
            support_ok=structured_ok,
            best_metric=self.primary_best_metric,
            best_raw=self.primary_best_raw,
            best_step=self.primary_best_step,
            best_phase=self.primary_best_phase,
            filename=self.primary_filename,
        )
        self._write_path1_v3_report(
            report_filename=self.structured_report_filename,
            kind="best_structured",
            current_step=global_step,
            current_phase=current_phase,
            phase_state=phase_state,
            metric_raw=metric_raw,
            metric_ema=float(self.selection_metric_ema),
            support_summary_valid=support_summary_valid,
            support_summary_train=support_summary_train,
            support_summary_guard=structured_summary,
            support_ok=structured_ok,
            best_metric=self.structured_best_metric,
            best_raw=self.structured_best_raw,
            best_step=self.structured_best_step,
            best_phase=self.structured_best_phase,
            filename=self.structured_filename,
            best_support_summary=self.structured_best_support,
        )
        if self._should_trigger_path1_v3_early_stop(global_step):
            self._early_stop_triggered = True
            self._early_stop_reason = (
                f"no best_bc improvement for {self.selection_patience_valids} validations after cons_start_step"
            )
            logger.log(f"[path1_v3] early stop triggered: {self._early_stop_reason}")

    def _path1_v4_phase_status(self):
        return {
            "current": self.path1_v4_controller_phase,
            "accepted": dict(self.path1_v4_phase_accepted),
            "rejected": dict(self.path1_v4_phase_rejected),
            "rejection_reason": dict(self.path1_v4_phase_rejection_reason),
            "start_step": dict(self.path1_v4_phase_start_step),
            "start_valid": dict(self.path1_v4_phase_start_valid),
            "best_metric": dict(self.path1_v4_phase_best_metric),
            "best_step": dict(self.path1_v4_phase_best_step),
            "last_improve_valid": dict(self.path1_v4_phase_last_improve_valid),
            "entry_reason": dict(self.path1_v4_phase_entry_reason),
            "structured_state": bool(self.path1_v4_structured_state),
            "overflow_valid_ema": self.path1_v4_overflow_valid_ema,
        }

    def _path1_v4_select_final_candidate(self):
        warmup_ok = self.warmup_best_step >= 0 and math.isfinite(self.warmup_best_metric)
        bc_ok = self.primary_best_step >= 0 and math.isfinite(self.primary_best_metric)
        if bc_ok and (
            (not warmup_ok)
            or float(self.primary_best_metric)
            < float(self.warmup_best_metric) - self.selection_min_delta
        ):
            return {
                "source": "best_bc",
                "metric": float(self.primary_best_metric),
                "raw": self.primary_best_raw,
                "step": int(self.primary_best_step),
                "phase": self.primary_best_phase,
                "filename": self.primary_filename,
            }
        if warmup_ok:
            return {
                "source": "best_warmup",
                "metric": float(self.warmup_best_metric),
                "raw": self.warmup_best_raw,
                "step": int(self.warmup_best_step),
                "phase": "warmup",
                "filename": self.warmup_filename,
            }
        if bc_ok:
            return {
                "source": "best_bc",
                "metric": float(self.primary_best_metric),
                "raw": self.primary_best_raw,
                "step": int(self.primary_best_step),
                "phase": self.primary_best_phase,
                "filename": self.primary_filename,
            }
        return None

    def _maybe_update_path1_v4_final(
        self,
        *,
        warmup_improved,
        primary_improved,
        current_phase,
        phase_state,
        support_summary_valid,
        support_summary_train,
        structured_summary,
    ):
        candidate = self._path1_v4_select_final_candidate()
        if candidate is None:
            return False
        source = candidate["source"]
        source_improved = (source == "best_warmup" and warmup_improved) or (
            source == "best_bc" and primary_improved
        )
        should_update = False
        if source_improved:
            if self.final_best_source != source:
                should_update = True
            elif candidate["step"] != self.final_best_step:
                should_update = True
            elif candidate["metric"] < (self.final_best_metric - self.selection_min_delta):
                should_update = True
        if not should_update:
            return False
        self.final_best_source = source
        self.final_best_metric = float(candidate["metric"])
        self.final_best_raw = candidate["raw"]
        self.final_best_step = int(candidate["step"])
        self.final_best_phase = candidate["phase"]
        self.final_last_improve_valid = self.selection_valid_count
        self._save_best_checkpoint(
            metric_raw=float(candidate["raw"]),
            metric_ema=float(candidate["metric"]),
            filename=self.final_filename,
            extra_report={
                "mode": "path1_v4",
                "kind": "best_final",
                "phase": current_phase,
                "phase_state": phase_state,
                "phase_status": self._path1_v4_phase_status(),
                "support_summary_valid": support_summary_valid,
                "support_summary_train_interval": support_summary_train,
                "support_summary_guard_merged": structured_summary,
                "support_ok": bool(structured_summary.get("structured_state", False)),
                "source": source,
                "source_step": int(candidate["step"]),
                "source_phase": candidate["phase"],
            },
        )
        logger.log(
            f"[best_final] updated source={source}, step={self.final_best_step}, "
            f"raw={float(candidate['raw']):.6f}, ema={float(candidate['metric']):.6f}, "
            f"file={self.final_filename}"
        )
        return True

    def _write_path1_v4_report(
        self,
        *,
        report_filename,
        kind,
        current_step,
        current_phase,
        phase_state,
        metric_raw,
        metric_ema,
        support_summary_valid,
        support_summary_train,
        structured_summary,
        best_metric,
        best_raw,
        best_step,
        best_phase,
        filename,
        best_support_summary=None,
        extra=None,
    ):
        if (not self.path1_v4_enabled) or dist.get_rank() != 0:
            return
        logdir = get_blob_logdir()
        model_filename, _, _ = self._best_artifact_names(filename)
        report = {
            "mode": "path1_v4",
            "kind": kind,
            "current_step": int(current_step),
            "current_phase": current_phase,
            "current_phase_state": phase_state,
            "current_lambda_koop_eff": float(self._path1_v4_state.get("lambda_koop_eff", 0.0)),
            "current_metric_raw": float(metric_raw),
            "current_metric_ema": float(metric_ema),
            "support_summary_valid": support_summary_valid,
            "support_summary_train": support_summary_train,
            "support_summary_guard_merged": structured_summary,
            "structured_state": bool(structured_summary.get("structured_state", False)),
            "phase_status": self._path1_v4_phase_status(),
            "best_metric_name": self.selection_metric_name,
            "best_metric_ema": None if not math.isfinite(best_metric) else float(best_metric),
            "best_metric_raw": best_raw,
            "best_step": None if best_step < 0 else int(best_step),
            "best_phase": best_phase,
            "best_support_summary": best_support_summary or {},
            "selection_valid_count": int(self.selection_valid_count),
            "early_stop_enabled": bool(self._early_stop_enabled),
            "early_stop_triggered": bool(self._early_stop_triggered),
            "early_stop_reason": self._early_stop_reason,
            "artifacts": {
                "checkpoint": bf.join(logdir, filename),
                "model": bf.join(logdir, model_filename),
                "ema": bf.join(logdir, f"ema_{self.ema_rate[0]}_{model_filename[6:]}"),
            },
        }
        if extra:
            report.update(extra)
        with bf.BlobFile(bf.join(logdir, report_filename), "wb") as f:
            f.write(json.dumps(report, indent=2, sort_keys=True).encode("utf-8"))

    def _maybe_update_best_path1_v4(self, valid_info):
        if not valid_info:
            return
        metric_raw = valid_info.get(self.selection_metric_name, None)
        if metric_raw is None:
            return
        metric_raw = float(metric_raw)
        if not math.isfinite(metric_raw):
            return

        self.selection_valid_count += 1
        if self.selection_metric_ema is None:
            self.selection_metric_ema = metric_raw
        else:
            alpha = self.selection_ema_alpha
            self.selection_metric_ema = (1.0 - alpha) * float(self.selection_metric_ema) + alpha * metric_raw

        global_step = self.step + self.resume_step
        current_phase = self.path1_v4_controller_phase
        current_phase_state = self._path1_v4_phase_state(global_step) or {}
        support_summary_valid = valid_info.get("support_summary_valid", {}) or {}
        support_summary_train = valid_info.get("support_summary_train_interval", {}) or {}
        structured_summary = self._path1_v4_update_structured_state(
            valid_info.get("support_summary_guard_merged", {}) or {}
        )
        structured_ok = bool(structured_summary.get("structured_state", False))

        phase_codes = {
            "warmup": 0.0,
            "dyn1": 1.0,
            "dyn": 2.0,
            "cons": 3.0,
            "tail_warmup": 4.0,
            "tail_dyn1": 5.0,
            "tail_dyn": 6.0,
        }
        logger.logkv_mean("phase/current_code", phase_codes.get(current_phase, -1.0))
        logger.logkv_mean("phase/structured_state", 1.0 if structured_ok else 0.0)
        logger.logkv_mean(
            "phase/overflow_valid_ema",
            float(structured_summary.get("overflow_rate_p95_valid_ema", float("nan")) or float("nan")),
        )
        logger.logkv_mean("best_warmup/raw", metric_raw)
        logger.logkv_mean("best_warmup/ema", float(self.selection_metric_ema))
        logger.logkv_mean("best_bc/raw", metric_raw)
        logger.logkv_mean("best_bc/ema", float(self.selection_metric_ema))
        logger.logkv_mean("best_bc/eligible", 1.0 if global_step >= self.path1_v4_dyn1_ramp_start_step else 0.0)
        logger.logkv_mean("best_structured/raw", metric_raw)
        logger.logkv_mean("best_structured/ema", float(self.selection_metric_ema))
        logger.logkv_mean("best_structured/support_ok", 1.0 if structured_ok else 0.0)
        if math.isfinite(self.warmup_best_metric):
            logger.logkv_mean("best_warmup/current_best", float(self.warmup_best_metric))
        if math.isfinite(self.primary_best_metric):
            logger.logkv_mean("best_bc/current_best", float(self.primary_best_metric))
        if math.isfinite(self.structured_best_metric):
            logger.logkv_mean("best_structured/current_best", float(self.structured_best_metric))
        if math.isfinite(self.final_best_metric):
            logger.logkv_mean("best_final/current_best", float(self.final_best_metric))

        warmup_improved = False
        if current_phase == "warmup" and float(self.selection_metric_ema) < (
            self.warmup_best_metric - self.selection_min_delta
        ):
            self.warmup_best_metric = float(self.selection_metric_ema)
            self.warmup_best_step = int(global_step)
            self.warmup_best_raw = metric_raw
            self.warmup_last_improve_valid = self.selection_valid_count
            self._save_best_checkpoint(
                metric_raw=metric_raw,
                metric_ema=float(self.selection_metric_ema),
                filename=self.warmup_filename,
                extra_report={
                    "mode": "path1_v4",
                    "kind": "best_warmup",
                    "phase": current_phase,
                    "phase_state": current_phase_state,
                    "phase_status": self._path1_v4_phase_status(),
                    "support_summary_valid": support_summary_valid,
                    "support_summary_train_interval": support_summary_train,
                    "support_summary_guard_merged": structured_summary,
                    "support_ok": bool(structured_ok),
                },
            )
            warmup_improved = True
            logger.log(
                f"[best_warmup] updated step={self.warmup_best_step}, "
                f"raw={metric_raw:.6f}, ema={float(self.selection_metric_ema):.6f}, "
                f"file={self.warmup_filename}"
            )

        if global_step >= self.path1_v4_dyn1_ramp_start_step:
            self._path1_v4_record_phase_metric(current_phase, float(self.selection_metric_ema), global_step)

        primary_improved = False
        if global_step >= self.path1_v4_dyn1_ramp_start_step and float(self.selection_metric_ema) < (
            self.primary_best_metric - self.selection_min_delta
        ):
            self.primary_best_metric = float(self.selection_metric_ema)
            self.primary_best_step = int(global_step)
            self.primary_best_raw = metric_raw
            self.primary_best_phase = current_phase
            self.primary_last_improve_valid = self.selection_valid_count
            self._save_best_checkpoint(
                metric_raw=metric_raw,
                metric_ema=float(self.selection_metric_ema),
                filename=self.primary_filename,
                extra_report={
                    "mode": "path1_v4",
                    "kind": "best_bc_diff",
                    "phase": current_phase,
                    "phase_state": current_phase_state,
                    "phase_status": self._path1_v4_phase_status(),
                    "support_summary_valid": support_summary_valid,
                    "support_summary_train_interval": support_summary_train,
                    "support_summary_guard_merged": structured_summary,
                    "support_ok": bool(structured_ok),
                },
            )
            primary_improved = True
            logger.log(
                f"[best_bc] updated step={self.primary_best_step}, "
                f"raw={metric_raw:.6f}, ema={float(self.selection_metric_ema):.6f}, "
                f"file={self.primary_filename}"
            )

        structured_improved = False
        if structured_ok and global_step >= self.path1_v4_dyn1_ramp_start_step and float(self.selection_metric_ema) < (
            self.structured_best_metric - self.selection_min_delta
        ):
            self.structured_best_metric = float(self.selection_metric_ema)
            self.structured_best_step = int(global_step)
            self.structured_best_raw = metric_raw
            self.structured_best_phase = current_phase
            self.structured_best_support = dict(structured_summary)
            self.structured_last_improve_valid = self.selection_valid_count
            self._save_best_checkpoint(
                metric_raw=metric_raw,
                metric_ema=float(self.selection_metric_ema),
                filename=self.structured_filename,
                extra_report={
                    "mode": "path1_v4",
                    "kind": "best_structured",
                    "phase": current_phase,
                    "phase_state": current_phase_state,
                    "phase_status": self._path1_v4_phase_status(),
                    "support_summary_valid": support_summary_valid,
                    "support_summary_train_interval": support_summary_train,
                    "support_summary_guard_merged": structured_summary,
                    "support_ok": True,
                },
            )
            structured_improved = True
            logger.log(
                f"[best_structured] updated step={self.structured_best_step}, "
                f"raw={metric_raw:.6f}, ema={float(self.selection_metric_ema):.6f}, "
                f"file={self.structured_filename}"
            )

        final_improved = self._maybe_update_path1_v4_final(
            warmup_improved=warmup_improved,
            primary_improved=primary_improved,
            current_phase=current_phase,
            phase_state=current_phase_state,
            support_summary_valid=support_summary_valid,
            support_summary_train=support_summary_train,
            structured_summary=structured_summary,
        )

        self._path1_v4_handle_phase_control(global_step, float(self.selection_metric_ema), structured_summary)
        current_phase = self.path1_v4_controller_phase
        current_phase_state = self._path1_v4_phase_state(global_step) or {}

        logger.logkv_mean("best_warmup/updated", 1.0 if warmup_improved else 0.0)
        logger.logkv_mean("best_bc/updated", 1.0 if primary_improved else 0.0)
        logger.logkv_mean("best_structured/updated", 1.0 if structured_improved else 0.0)
        logger.logkv_mean("best_final/updated", 1.0 if final_improved else 0.0)
        if self.warmup_best_step >= 0:
            logger.logkv_mean("best_warmup/step", float(self.warmup_best_step))
        if self.primary_best_step >= 0:
            logger.logkv_mean("best_bc/step", float(self.primary_best_step))
        if self.structured_best_step >= 0:
            logger.logkv_mean("best_structured/step", float(self.structured_best_step))
        if self.final_best_step >= 0:
            logger.logkv_mean("best_final/step", float(self.final_best_step))

        self._write_path1_v4_report(
            report_filename=self.warmup_report_filename,
            kind="best_warmup",
            current_step=global_step,
            current_phase=current_phase,
            phase_state=current_phase_state,
            metric_raw=metric_raw,
            metric_ema=float(self.selection_metric_ema),
            support_summary_valid=support_summary_valid,
            support_summary_train=support_summary_train,
            structured_summary=structured_summary,
            best_metric=self.warmup_best_metric,
            best_raw=self.warmup_best_raw,
            best_step=self.warmup_best_step,
            best_phase="warmup",
            filename=self.warmup_filename,
        )
        self._write_path1_v4_report(
            report_filename=self.primary_report_filename,
            kind="best_bc_diff",
            current_step=global_step,
            current_phase=current_phase,
            phase_state=current_phase_state,
            metric_raw=metric_raw,
            metric_ema=float(self.selection_metric_ema),
            support_summary_valid=support_summary_valid,
            support_summary_train=support_summary_train,
            structured_summary=structured_summary,
            best_metric=self.primary_best_metric,
            best_raw=self.primary_best_raw,
            best_step=self.primary_best_step,
            best_phase=self.primary_best_phase,
            filename=self.primary_filename,
        )
        self._write_path1_v4_report(
            report_filename=self.structured_report_filename,
            kind="best_structured",
            current_step=global_step,
            current_phase=current_phase,
            phase_state=current_phase_state,
            metric_raw=metric_raw,
            metric_ema=float(self.selection_metric_ema),
            support_summary_valid=support_summary_valid,
            support_summary_train=support_summary_train,
            structured_summary=structured_summary,
            best_metric=self.structured_best_metric,
            best_raw=self.structured_best_raw,
            best_step=self.structured_best_step,
            best_phase=self.structured_best_phase,
            filename=self.structured_filename,
            best_support_summary=self.structured_best_support,
        )
        self._write_path1_v4_report(
            report_filename=self.final_report_filename,
            kind="best_final",
            current_step=global_step,
            current_phase=current_phase,
            phase_state=current_phase_state,
            metric_raw=metric_raw,
            metric_ema=float(self.selection_metric_ema),
            support_summary_valid=support_summary_valid,
            support_summary_train=support_summary_train,
            structured_summary=structured_summary,
            best_metric=self.final_best_metric,
            best_raw=self.final_best_raw,
            best_step=self.final_best_step,
            best_phase=self.final_best_phase,
            filename=self.final_filename,
            extra={"source": self.final_best_source},
        )
        if self._should_trigger_path1_v4_early_stop(global_step):
            self._early_stop_triggered = True
            self._early_stop_reason = (
                f"no best_final improvement for {self.selection_patience_valids} validations "
                f"in phase={self.path1_v4_controller_phase}"
            )
            logger.log(f"[path1_v4] early stop triggered: {self._early_stop_reason}")

    def _path1_v4_state_dict(self):
        if not self.path1_v4_enabled:
            return None
        return {
            "controller_phase": self.path1_v4_controller_phase,
            "phase_accepted": dict(self.path1_v4_phase_accepted),
            "phase_rejected": dict(self.path1_v4_phase_rejected),
            "phase_rejection_reason": dict(self.path1_v4_phase_rejection_reason),
            "phase_start_step": dict(self.path1_v4_phase_start_step),
            "phase_start_valid": dict(self.path1_v4_phase_start_valid),
            "phase_best_metric": dict(self.path1_v4_phase_best_metric),
            "phase_best_step": dict(self.path1_v4_phase_best_step),
            "phase_last_improve_valid": dict(self.path1_v4_phase_last_improve_valid),
            "phase_entry_reason": dict(self.path1_v4_phase_entry_reason),
            "structured_state": bool(self.path1_v4_structured_state),
            "overflow_valid_ema": self.path1_v4_overflow_valid_ema,
            "selection_valid_count": int(self.selection_valid_count),
            "selection_metric_ema": self.selection_metric_ema,
            "warmup_best_metric": self.warmup_best_metric,
            "warmup_best_step": self.warmup_best_step,
            "warmup_best_raw": self.warmup_best_raw,
            "warmup_last_improve_valid": self.warmup_last_improve_valid,
            "primary_best_metric": self.primary_best_metric,
            "primary_best_step": self.primary_best_step,
            "primary_best_raw": self.primary_best_raw,
            "primary_best_phase": self.primary_best_phase,
            "primary_last_improve_valid": self.primary_last_improve_valid,
            "structured_best_metric": self.structured_best_metric,
            "structured_best_step": self.structured_best_step,
            "structured_best_raw": self.structured_best_raw,
            "structured_best_phase": self.structured_best_phase,
            "structured_best_support": dict(self.structured_best_support),
            "structured_last_improve_valid": self.structured_last_improve_valid,
            "final_best_metric": self.final_best_metric,
            "final_best_step": self.final_best_step,
            "final_best_raw": self.final_best_raw,
            "final_best_phase": self.final_best_phase,
            "final_best_source": self.final_best_source,
            "final_last_improve_valid": self.final_last_improve_valid,
            "early_stop_triggered": bool(self._early_stop_triggered),
            "early_stop_reason": self._early_stop_reason,
            "dyn_rejected": bool(self.path1_v4_dyn_rejected),
            "cons_rejected": bool(self.path1_v4_cons_rejected),
        }

    def _load_path1_v4_state_dict(self, state):
        if (not self.path1_v4_enabled) or (not state):
            return
        self.path1_v4_controller_phase = str(
            state.get("controller_phase", self.path1_v4_controller_phase)
        )
        self.path1_v4_phase_accepted = dict(state.get("phase_accepted", self.path1_v4_phase_accepted))
        self.path1_v4_phase_rejected = dict(state.get("phase_rejected", self.path1_v4_phase_rejected))
        self.path1_v4_phase_rejection_reason = dict(
            state.get("phase_rejection_reason", self.path1_v4_phase_rejection_reason)
        )
        self.path1_v4_phase_start_step = dict(state.get("phase_start_step", self.path1_v4_phase_start_step))
        self.path1_v4_phase_start_valid = dict(state.get("phase_start_valid", self.path1_v4_phase_start_valid))
        self.path1_v4_phase_best_metric = dict(state.get("phase_best_metric", self.path1_v4_phase_best_metric))
        self.path1_v4_phase_best_step = dict(state.get("phase_best_step", self.path1_v4_phase_best_step))
        self.path1_v4_phase_last_improve_valid = dict(
            state.get("phase_last_improve_valid", self.path1_v4_phase_last_improve_valid)
        )
        self.path1_v4_phase_entry_reason = dict(
            state.get("phase_entry_reason", self.path1_v4_phase_entry_reason)
        )
        self.path1_v4_structured_state = bool(
            state.get("structured_state", self.path1_v4_structured_state)
        )
        self.path1_v4_overflow_valid_ema = state.get(
            "overflow_valid_ema", self.path1_v4_overflow_valid_ema
        )
        self.selection_valid_count = int(state.get("selection_valid_count", self.selection_valid_count))
        self.selection_metric_ema = state.get("selection_metric_ema", self.selection_metric_ema)
        self.warmup_best_metric = float(state.get("warmup_best_metric", self.warmup_best_metric))
        self.warmup_best_step = int(state.get("warmup_best_step", self.warmup_best_step))
        self.warmup_best_raw = state.get("warmup_best_raw", self.warmup_best_raw)
        self.warmup_last_improve_valid = int(
            state.get("warmup_last_improve_valid", self.warmup_last_improve_valid)
        )
        self.primary_best_metric = float(state.get("primary_best_metric", self.primary_best_metric))
        self.primary_best_step = int(state.get("primary_best_step", self.primary_best_step))
        self.primary_best_raw = state.get("primary_best_raw", self.primary_best_raw)
        self.primary_best_phase = state.get("primary_best_phase", self.primary_best_phase)
        self.primary_last_improve_valid = int(
            state.get("primary_last_improve_valid", self.primary_last_improve_valid)
        )
        self.structured_best_metric = float(
            state.get("structured_best_metric", self.structured_best_metric)
        )
        self.structured_best_step = int(state.get("structured_best_step", self.structured_best_step))
        self.structured_best_raw = state.get("structured_best_raw", self.structured_best_raw)
        self.structured_best_phase = state.get("structured_best_phase", self.structured_best_phase)
        self.structured_best_support = dict(
            state.get("structured_best_support", self.structured_best_support)
        )
        self.structured_last_improve_valid = int(
            state.get("structured_last_improve_valid", self.structured_last_improve_valid)
        )
        self.final_best_metric = float(state.get("final_best_metric", self.final_best_metric))
        self.final_best_step = int(state.get("final_best_step", self.final_best_step))
        self.final_best_raw = state.get("final_best_raw", self.final_best_raw)
        self.final_best_phase = state.get("final_best_phase", self.final_best_phase)
        self.final_best_source = state.get("final_best_source", self.final_best_source)
        self.final_last_improve_valid = int(
            state.get("final_last_improve_valid", self.final_last_improve_valid)
        )
        self._early_stop_triggered = bool(
            state.get("early_stop_triggered", self._early_stop_triggered)
        )
        self._early_stop_reason = state.get("early_stop_reason", self._early_stop_reason)
        self.path1_v4_dyn_rejected = bool(state.get("dyn_rejected", self.path1_v4_dyn_rejected))
        self.path1_v4_cons_rejected = bool(state.get("cons_rejected", self.path1_v4_cons_rejected))

    @staticmethod
    def _merge_path1_v5_structured_support(valid_summary, train_summary):
        valid_summary = valid_summary or {}
        train_summary = train_summary or {}
        merged = {}
        if "overflow_rate_p95" in valid_summary:
            merged["overflow_rate_p95_valid"] = valid_summary["overflow_rate_p95"]
        if "effective_grad_ratio_median" in train_summary:
            merged["effective_grad_ratio_median_train"] = train_summary["effective_grad_ratio_median"]
        if "n_eff_median" in train_summary:
            merged["n_eff_median_train"] = train_summary["n_eff_median"]
        if "weight_sum_median" in train_summary:
            merged["weight_sum_median_train"] = train_summary["weight_sum_median"]
        return merged

    def _path1_v5_train_guard_ok(self, merged_summary):
        if not self.koopman_cfg.get("enabled", False):
            return True
        if not merged_summary:
            return False
        grad_ratio = merged_summary.get("effective_grad_ratio_median_train", None)
        n_eff_train = merged_summary.get("n_eff_median_train", None)
        weight_sum_train = merged_summary.get("weight_sum_median_train", None)
        if grad_ratio is None:
            return False
        if grad_ratio < self.path1_v5_structured_grad_ratio_min_train:
            return False
        if grad_ratio > self.path1_v5_structured_grad_ratio_max_train:
            return False
        if n_eff_train is None or n_eff_train < self.path1_v5_structured_min_n_eff_train:
            return False
        if (
            weight_sum_train is None
            or weight_sum_train < self.path1_v5_structured_min_weight_sum_train
        ):
            return False
        return True

    def _path1_v5_update_structured_state(self, merged_summary):
        merged_summary = dict(merged_summary or {})
        if not self.koopman_cfg.get("enabled", False):
            merged_summary["overflow_rate_p95_valid_ema"] = 0.0
            merged_summary["train_guard_ok"] = True
            merged_summary["valid_guard_ok"] = True
            merged_summary["structured_hold_count"] = self.path1_v5_structured_hold_valids
            merged_summary["structured_state"] = True
            self.path1_v5_structured_state = True
            self.path1_v5_structured_true_streak = self.path1_v5_structured_hold_valids
            self.path1_v5_last_structured_summary = dict(merged_summary)
            return merged_summary

        overflow_valid = merged_summary.get("overflow_rate_p95_valid", None)
        if overflow_valid is not None and math.isfinite(float(overflow_valid)):
            overflow_valid = float(overflow_valid)
            if self.path1_v5_overflow_valid_ema is None:
                self.path1_v5_overflow_valid_ema = overflow_valid
            else:
                alpha = float(self.path1_v5_overflow_valid_ema_alpha)
                self.path1_v5_overflow_valid_ema = (
                    (1.0 - alpha) * float(self.path1_v5_overflow_valid_ema)
                    + alpha * overflow_valid
                )
        overflow_ema = self.path1_v5_overflow_valid_ema
        train_ok = self._path1_v5_train_guard_ok(merged_summary)
        if overflow_ema is None:
            valid_ok = False
        elif self.path1_v5_structured_state:
            valid_ok = float(overflow_ema) <= float(self.path1_v5_overflow_exit)
        else:
            valid_ok = float(overflow_ema) <= float(self.path1_v5_overflow_enter)
        candidate = bool(train_ok and valid_ok)
        if candidate:
            self.path1_v5_structured_true_streak += 1
        else:
            self.path1_v5_structured_true_streak = 0
        if self.path1_v5_structured_state:
            structured_state = candidate
        else:
            structured_state = candidate and (
                self.path1_v5_structured_true_streak >= self.path1_v5_structured_hold_valids
            )
        self.path1_v5_structured_state = bool(structured_state)
        merged_summary["overflow_rate_p95_valid_ema"] = (
            None if overflow_ema is None else float(overflow_ema)
        )
        merged_summary["train_guard_ok"] = bool(train_ok)
        merged_summary["valid_guard_ok"] = bool(valid_ok)
        merged_summary["structured_hold_count"] = int(self.path1_v5_structured_true_streak)
        merged_summary["structured_state"] = bool(structured_state)
        self.path1_v5_last_structured_summary = dict(merged_summary)
        return merged_summary

    @staticmethod
    def _path1_v5_phase_group(phase_name):
        if phase_name in {"dyn_probe", "dyn"}:
            return "dyn"
        if phase_name in {"cons_probe", "cons"}:
            return "cons"
        return phase_name

    def _path1_v5_best_metric(self, group):
        if group == "warmup":
            return self.warmup_best_metric
        if group == "dyn1":
            return self.dyn1_best_metric
        if group == "dyn":
            return self.dyn_best_metric
        if group == "cons":
            return self.cons_best_metric
        if group == "structured":
            return self.structured_best_metric
        if group == "final":
            return self.final_best_metric
        return float("inf")

    def _path1_v5_best_raw(self, group):
        if group == "warmup":
            return self.warmup_best_raw
        if group == "dyn1":
            return self.dyn1_best_raw
        if group == "dyn":
            return self.dyn_best_raw
        if group == "cons":
            return self.cons_best_raw
        return None

    def _path1_v5_best_step(self, group):
        if group == "warmup":
            return self.warmup_best_step
        if group == "dyn1":
            return self.dyn1_best_step
        if group == "dyn":
            return self.dyn_best_step
        if group == "cons":
            return self.cons_best_step
        return -1

    def _path1_v5_best_filename(self, group):
        if group == "warmup":
            return self.warmup_filename
        if group == "dyn1":
            return self.dyn1_filename
        if group == "dyn":
            return self.dyn_filename
        if group == "cons":
            return self.cons_filename
        return None

    def _path1_v5_phase_local_ema(self, phase_name, metric_raw):
        phase_name = str(phase_name)
        metric_raw = float(metric_raw)
        prev = self.path1_v5_phase_metric_ema.get(phase_name, None)
        if prev is None:
            ema = metric_raw
        else:
            alpha = self.selection_ema_alpha
            ema = (1.0 - alpha) * float(prev) + alpha * metric_raw
        self.path1_v5_phase_metric_ema[phase_name] = float(ema)
        return float(ema)

    def _path1_v5_set_phase(
        self,
        phase_name,
        global_step,
        *,
        reason,
        accepted_phase=None,
        rejected_phase=None,
        blocked_phase=None,
    ):
        phase_name = str(phase_name)
        prev = self.path1_v5_controller_phase
        self.path1_v5_previous_phase = prev
        if accepted_phase is not None:
            self.path1_v5_phase_accepted[str(accepted_phase)] = True
        if rejected_phase is not None:
            rejected_phase = str(rejected_phase)
            self.path1_v5_phase_rejected[rejected_phase] = True
            self.path1_v5_phase_rejection_reason[rejected_phase] = str(reason)
        if blocked_phase is not None:
            self.path1_v5_phase_promotion_blocked_reason[str(blocked_phase)] = str(reason)
        if prev != phase_name:
            self.path1_v5_controller_phase = phase_name
            self.path1_v5_phase_start_step[phase_name] = int(global_step)
            self.path1_v5_phase_start_valid[phase_name] = int(self.selection_valid_count)
            self.path1_v5_phase_entry_reason[phase_name] = str(reason)
            self.path1_v5_phase_metric_ema[phase_name] = None
            logger.log(
                f"[path1_v5] phase transition {prev} -> {phase_name} "
                f"at step={int(global_step)} reason={reason}"
            )

    def _restore_runtime_state_from_checkpoint(self, checkpoint_filename):
        ckpt_path = bf.join(get_blob_logdir(), str(checkpoint_filename))
        if not bf.exists(ckpt_path):
            logger.log(f"[path1_v5] rollback checkpoint missing: {ckpt_path}")
            return False
        logger.log(f"[path1_v5] restoring runtime state from {ckpt_path}")
        payload = dist_util.load_state_dict(ckpt_path, map_location=dist_util.dev())
        model_state = payload.get("model_state_dict", payload)
        self.model.load_state_dict(model_state)
        if self.mp_trainer.use_fp16:
            self.mp_trainer.master_params = self.mp_trainer.state_dict_to_master_params(model_state)
            master_params_to_model_params(
                self.mp_trainer.param_groups_and_shapes, self.mp_trainer.master_params
            )
        dist_util.sync_params(self.model.parameters())

        ema_state_dicts = payload.get("ema_state_dicts", {}) or {}
        new_ema_params = []
        for rate, params in zip(self.ema_rate, self.ema_params):
            state_dict = ema_state_dicts.get(str(rate), None)
            if state_dict is None:
                new_ema_params.append(copy.deepcopy(params))
            else:
                new_ema_params.append(self.mp_trainer.state_dict_to_master_params(state_dict))
        if new_ema_params:
            self.ema_params = new_ema_params

        opt_state = payload.get("optimizer_state_dict", None)
        if opt_state is not None:
            self.opt.load_state_dict(opt_state)
        reg_state = payload.get("koopman_regularizer_state", None)
        if (
            reg_state is not None
            and self.koopman_regularizer is not None
            and hasattr(self.koopman_regularizer, "load_state_dict")
        ):
            self.koopman_regularizer.load_state_dict(reg_state)
        bal_state = payload.get("lambda_balancer_state", None)
        if (
            bal_state is not None
            and self.lambda_balancer is not None
            and hasattr(self.lambda_balancer, "load_state_dict")
        ):
            self.lambda_balancer.load_state_dict(bal_state)
        return True

    def _path1_v5_enter_finalize(
        self,
        global_step,
        *,
        reason,
        rollback_group=None,
        rollback_checkpoint=None,
        rejected_phase=None,
    ):
        rollback_source_phase = rollback_group
        rollback_source_checkpoint = rollback_checkpoint
        if rollback_source_checkpoint is not None:
            self._restore_runtime_state_from_checkpoint(rollback_source_checkpoint)
        self.path1_v5_finalize_reason = str(reason)
        self.path1_v5_rollback_source_phase = rollback_source_phase
        self.path1_v5_rollback_source_checkpoint = rollback_source_checkpoint
        self._path1_v5_state = {
            "phase": "finalize",
            "phase_state": {},
            "lambda_koop_eff": 0.0,
        }
        self._path1_v5_set_phase(
            "finalize",
            global_step,
            reason=reason,
            accepted_phase="finalize",
            rejected_phase=rejected_phase,
        )

    def _path1_v5_update_group_best(self, group, metric_ema, metric_raw, global_step):
        metric_ema = float(metric_ema)
        metric_raw = float(metric_raw)
        improved = False
        if group == "warmup":
            if metric_ema < (self.warmup_best_metric - self.selection_min_delta):
                self.warmup_best_metric = metric_ema
                self.warmup_best_step = int(global_step)
                self.warmup_best_raw = metric_raw
                self.warmup_last_improve_valid = int(self.selection_valid_count)
                improved = True
        elif group == "dyn1":
            if metric_ema < (self.dyn1_best_metric - self.selection_min_delta):
                self.dyn1_best_metric = metric_ema
                self.dyn1_best_step = int(global_step)
                self.dyn1_best_raw = metric_raw
                self.dyn1_last_improve_valid = int(self.selection_valid_count)
                improved = True
        elif group == "dyn":
            if metric_ema < (self.dyn_best_metric - self.selection_min_delta):
                self.dyn_best_metric = metric_ema
                self.dyn_best_step = int(global_step)
                self.dyn_best_raw = metric_raw
                self.dyn_last_improve_valid = int(self.selection_valid_count)
                improved = True
        elif group == "cons":
            if metric_ema < (self.cons_best_metric - self.selection_min_delta):
                self.cons_best_metric = metric_ema
                self.cons_best_step = int(global_step)
                self.cons_best_raw = metric_raw
                self.cons_last_improve_valid = int(self.selection_valid_count)
                improved = True
        if improved:
            self.path1_v5_phase_best_metric[group] = metric_ema
            self.path1_v5_phase_best_step[group] = int(global_step)
            self.path1_v5_phase_best_raw[group] = metric_raw
            self.path1_v5_phase_last_improve_valid[group] = int(self.selection_valid_count)
        return improved

    def _path1_v5_group_last_improve_valid(self, group):
        if group == "warmup":
            return self.warmup_last_improve_valid
        if group == "dyn1":
            return self.dyn1_last_improve_valid
        if group == "dyn":
            return self.dyn_last_improve_valid
        if group == "cons":
            return self.cons_last_improve_valid
        return 0

    def _path1_v5_phase_state(self, global_step):
        if not self.path1_v5_enabled:
            return None
        koopman_enabled = self.koopman_model is not None and self.koopman_cfg.get("enabled", False)
        if not koopman_enabled:
            return None

        phase_name = self.path1_v5_controller_phase
        cons_cfg = self.koopman_cfg.get("consistency", {}) or {}
        cons1_target = max(
            0.0, float(cons_cfg.get("lambda_cons1_target", cons_cfg.get("lambda_cons1", 0.0)))
        )
        cons_multi_target = max(
            0.0, float(cons_cfg.get("lambda_cons_multi_target", cons_cfg.get("lambda_cons_multi", 0.0)))
        )
        cons_weight_sum = cons1_target + cons_multi_target
        cons1_frac = cons1_target / cons_weight_sum if cons_weight_sum > 0.0 else 0.0
        cons_multi_frac = cons_multi_target / cons_weight_sum if cons_weight_sum > 0.0 else 0.0

        progress_dyn1 = self._cosine_ramp(
            global_step, self.path1_v5_dyn1_start_step, self.path1_v5_dyn1_full_step
        )
        progress_dyn_probe = self._cosine_ramp(
            global_step, self.path1_v5_dyn_probe_start_step, self.path1_v5_dyn_probe_full_step
        )
        progress_dyn = self._cosine_ramp(
            global_step, self.path1_v5_dyn_start_step, self.path1_v5_dyn_full_step
        )
        progress_cons_probe = self._cosine_ramp(
            global_step, self.path1_v5_cons_probe_start_step, self.path1_v5_cons_probe_full_step
        )
        progress_cons = self._cosine_ramp(
            global_step, self.path1_v5_cons_start_step, self.path1_v5_cons_full_step
        )

        target_ratio = 0.0
        share_dyn1 = 0.0
        share_dyn_multi = 0.0
        share_cons_total = 0.0

        if phase_name == "warmup":
            target_ratio = 0.0
        elif phase_name == "dyn1":
            target_ratio = self.path1_v5_target_ratio_dyn1_full * progress_dyn1
            share_dyn1 = 1.0 if progress_dyn1 > 0.0 else 0.0
        elif phase_name == "dyn_probe":
            target_ratio = (
                self.path1_v5_target_ratio_dyn1_full
                + (self.path1_v5_target_ratio_dyn_probe_full - self.path1_v5_target_ratio_dyn1_full)
                * progress_dyn_probe
            )
            share_dyn_multi = min(max(self.path1_v5_dyn_multi_probe_share, 0.0), 0.95)
            share_dyn1 = max(0.0, 1.0 - share_dyn_multi)
        elif phase_name == "dyn":
            target_ratio = (
                self.path1_v5_target_ratio_dyn_probe_full
                + (self.path1_v5_target_ratio_dyn_full - self.path1_v5_target_ratio_dyn_probe_full)
                * progress_dyn
            )
            share_dyn_multi = min(max(self.path1_v5_dyn_multi_full_share, 0.0), 0.95)
            share_dyn1 = max(0.0, 1.0 - share_dyn_multi)
        elif phase_name == "cons_probe":
            target_ratio = (
                self.path1_v5_target_ratio_dyn_full
                + (self.path1_v5_target_ratio_cons_probe_full - self.path1_v5_target_ratio_dyn_full)
                * progress_cons_probe
            )
            share_dyn_multi = min(max(self.path1_v5_dyn_multi_full_share, 0.0), 0.95)
            share_cons_total = min(max(self.path1_v5_cons_total_probe_share, 0.0), 0.95)
            share_dyn1 = max(0.0, 1.0 - share_dyn_multi - share_cons_total)
        elif phase_name == "cons":
            target_ratio = (
                self.path1_v5_target_ratio_cons_probe_full
                + (self.path1_v5_target_ratio_cons_full - self.path1_v5_target_ratio_cons_probe_full)
                * progress_cons
            )
            share_dyn_multi = min(max(self.path1_v5_dyn_multi_full_share, 0.0), 0.95)
            share_cons_total = min(max(self.path1_v5_cons_total_full_share, 0.0), 0.95)
            share_dyn1 = max(0.0, 1.0 - share_dyn_multi - share_cons_total)
        elif phase_name == "finalize":
            base_phase = self.path1_v5_rollback_source_phase or "warmup"
            if base_phase == "dyn1":
                target_ratio = self.path1_v5_target_ratio_dyn1_full
                share_dyn1 = 1.0
            elif base_phase == "dyn":
                target_ratio = self.path1_v5_target_ratio_dyn_full
                share_dyn_multi = min(max(self.path1_v5_dyn_multi_full_share, 0.0), 0.95)
                share_dyn1 = max(0.0, 1.0 - share_dyn_multi)
            else:
                target_ratio = 0.0

        share_cons1 = share_cons_total * cons1_frac
        share_cons_multi = share_cons_total * cons_multi_frac
        return {
            "phase": phase_name,
            "progress_dyn1": float(progress_dyn1),
            "progress_dyn_probe": float(progress_dyn_probe),
            "progress_dyn": float(progress_dyn),
            "progress_cons_probe": float(progress_cons_probe),
            "progress_cons": float(progress_cons),
            "target_ratio": float(target_ratio),
            "share_dyn1": float(share_dyn1),
            "share_dyn_multi": float(share_dyn_multi),
            "share_cons1": float(share_cons1),
            "share_cons_multi": float(share_cons_multi),
            "structured_state": bool(self.path1_v5_structured_state),
        }

    def _apply_path1_v5_schedule(self, global_step):
        if (not self.path1_v5_enabled) or self.koopman_regularizer is None:
            return None
        phase_state = self._path1_v5_phase_state(global_step)
        if phase_state is None:
            self.koopman_regularizer.clear_lambda_overrides()
            return None
        self.koopman_regularizer.set_lambda_overrides(
            lambda_k1=1.0 if phase_state["share_dyn1"] > 0.0 else 0.0,
            lambda_k_multi=1.0 if phase_state["share_dyn_multi"] > 0.0 else 0.0,
            lambda_cons1=1.0 if phase_state["share_cons1"] > 0.0 else 0.0,
            lambda_cons_multi=1.0 if phase_state["share_cons_multi"] > 0.0 else 0.0,
        )
        return phase_state

    def _compose_path1_v5_koop_loss(self, *, loss_info, weight_koop, phase_state):
        branch_losses = self._path1_v3_reduce_branch_losses(weight_koop, loss_info)
        if branch_losses is None:
            return None
        share_dyn1 = float(phase_state.get("share_dyn1", 0.0))
        share_dyn_multi = float(phase_state.get("share_dyn_multi", 0.0))
        share_cons1 = float(phase_state.get("share_cons1", 0.0))
        share_cons_multi = float(phase_state.get("share_cons_multi", 0.0))
        cons_cfg = self.koopman_cfg.get("consistency", {}) or {}
        cap_ratio = float(cons_cfg.get("cap_ratio_vs_dyn", cons_cfg.get("cap_ratio", 1.0)))
        contrib_info = self._cap_consistency_contrib(
            dyn1_loss=branch_losses["dyn1_loss"],
            dyn_multi_loss=branch_losses["dyn_multi_loss"],
            cons1_loss=branch_losses["cons1_loss"],
            cons_multi_loss=branch_losses["cons_multi_loss"],
            lambda_k1=share_dyn1,
            lambda_k_multi=share_dyn_multi,
            lambda_cons1=share_cons1,
            lambda_cons_multi=share_cons_multi,
            cap_ratio=cap_ratio,
        )
        return {
            "branch_losses": branch_losses,
            "contrib_info": contrib_info,
            "koop_loss": contrib_info["koop_loss_capped"],
            "share_dyn1": share_dyn1,
            "share_dyn_multi": share_dyn_multi,
            "share_cons1": share_cons1,
            "share_cons_multi": share_cons_multi,
            "share_cons1_capped": float(contrib_info["lambda_cons1_capped"].detach().item()),
            "share_cons_multi_capped": float(contrib_info["lambda_cons_multi_capped"].detach().item()),
        }

    def _path1_v5_phase_status(self):
        return {
            "current": self.path1_v5_controller_phase,
            "previous": self.path1_v5_previous_phase,
            "accepted": dict(self.path1_v5_phase_accepted),
            "rejected": dict(self.path1_v5_phase_rejected),
            "rejection_reason": dict(self.path1_v5_phase_rejection_reason),
            "promotion_blocked_reason": dict(self.path1_v5_phase_promotion_blocked_reason),
            "start_step": dict(self.path1_v5_phase_start_step),
            "start_valid": dict(self.path1_v5_phase_start_valid),
            "best_metric": dict(self.path1_v5_phase_best_metric),
            "best_step": dict(self.path1_v5_phase_best_step),
            "best_raw": dict(self.path1_v5_phase_best_raw),
            "last_improve_valid": dict(self.path1_v5_phase_last_improve_valid),
            "entry_reason": dict(self.path1_v5_phase_entry_reason),
            "structured_state": bool(self.path1_v5_structured_state),
            "structured_true_streak": int(self.path1_v5_structured_true_streak),
            "overflow_valid_ema": self.path1_v5_overflow_valid_ema,
            "rollback_source_phase": self.path1_v5_rollback_source_phase,
            "rollback_source_checkpoint": self.path1_v5_rollback_source_checkpoint,
            "finalize_reason": self.path1_v5_finalize_reason,
        }

    def _path1_v5_select_final_candidate(self):
        warmup_ok = self.warmup_best_step >= 0 and math.isfinite(self.warmup_best_metric)
        dyn1_ok = self.dyn1_best_step >= 0 and math.isfinite(self.dyn1_best_metric)
        dyn_ok = self.dyn_best_step >= 0 and math.isfinite(self.dyn_best_metric)
        cons_ok = self.cons_best_step >= 0 and math.isfinite(self.cons_best_metric)
        delta = float(self.path1_v5_final_min_delta)

        if cons_ok and (not dyn_ok or float(self.cons_best_metric) < float(self.dyn_best_metric) - delta):
            return {
                "source": "best_cons",
                "metric": float(self.cons_best_metric),
                "raw": self.cons_best_raw,
                "step": int(self.cons_best_step),
                "phase": "cons",
                "filename": self.cons_filename,
            }
        if dyn_ok and (not dyn1_ok or float(self.dyn_best_metric) < float(self.dyn1_best_metric) - delta):
            return {
                "source": "best_dyn",
                "metric": float(self.dyn_best_metric),
                "raw": self.dyn_best_raw,
                "step": int(self.dyn_best_step),
                "phase": "dyn",
                "filename": self.dyn_filename,
            }
        if dyn1_ok and (not warmup_ok or float(self.dyn1_best_metric) < float(self.warmup_best_metric) - delta):
            return {
                "source": "best_dyn1",
                "metric": float(self.dyn1_best_metric),
                "raw": self.dyn1_best_raw,
                "step": int(self.dyn1_best_step),
                "phase": "dyn1",
                "filename": self.dyn1_filename,
            }
        if warmup_ok:
            return {
                "source": "best_warmup",
                "metric": float(self.warmup_best_metric),
                "raw": self.warmup_best_raw,
                "step": int(self.warmup_best_step),
                "phase": "warmup",
                "filename": self.warmup_filename,
            }
        return None

    def _maybe_update_path1_v5_final(
        self,
        *,
        warmup_improved,
        dyn1_improved,
        dyn_improved,
        cons_improved,
        current_phase,
        phase_state,
        support_summary_valid,
        support_summary_train,
        structured_summary,
    ):
        candidate = self._path1_v5_select_final_candidate()
        if candidate is None:
            return False
        source = candidate["source"]
        source_improved = (
            (source == "best_warmup" and warmup_improved)
            or (source == "best_dyn1" and dyn1_improved)
            or (source == "best_dyn" and dyn_improved)
            or (source == "best_cons" and cons_improved)
        )
        should_update = False
        if source_improved:
            if self.final_best_source != source:
                should_update = True
            elif candidate["step"] != self.final_best_step:
                should_update = True
            elif candidate["metric"] < (self.final_best_metric - self.selection_min_delta):
                should_update = True
        if not should_update:
            return False
        self.final_best_source = source
        self.final_best_metric = float(candidate["metric"])
        self.final_best_raw = candidate["raw"]
        self.final_best_step = int(candidate["step"])
        self.final_best_phase = candidate["phase"]
        self.final_last_improve_valid = self.selection_valid_count
        self._save_best_checkpoint(
            metric_raw=float(candidate["raw"]),
            metric_ema=float(candidate["metric"]),
            filename=self.final_filename,
            extra_report={
                "mode": "path1_v5",
                "kind": "best_final",
                "phase": current_phase,
                "phase_state": phase_state,
                "phase_status": self._path1_v5_phase_status(),
                "support_summary_valid": support_summary_valid,
                "support_summary_train_interval": support_summary_train,
                "support_summary_guard_merged": structured_summary,
                "support_ok": bool(structured_summary.get("structured_state", False)),
                "source": source,
                "source_step": int(candidate["step"]),
                "source_phase": candidate["phase"],
            },
        )
        logger.log(
            f"[best_final] updated source={source}, step={self.final_best_step}, "
            f"raw={float(candidate['raw']):.6f}, metric={float(candidate['metric']):.6f}, "
            f"file={self.final_filename}"
        )
        return True

    def _write_path1_v5_report(
        self,
        *,
        report_filename,
        kind,
        current_step,
        current_phase,
        phase_state,
        metric_raw,
        metric_ema,
        phase_metric_ema,
        support_summary_valid,
        support_summary_train,
        structured_summary,
        best_metric,
        best_raw,
        best_step,
        filename,
        extra=None,
    ):
        if (not self.path1_v5_enabled) or dist.get_rank() != 0:
            return
        logdir = get_blob_logdir()
        model_filename, _, _ = self._best_artifact_names(filename)
        report = {
            "mode": "path1_v5",
            "kind": kind,
            "current_step": int(current_step),
            "current_phase": current_phase,
            "current_phase_state": phase_state,
            "current_lambda_koop_eff": float(self._path1_v5_state.get("lambda_koop_eff", 0.0)),
            "current_metric_raw": float(metric_raw),
            "current_metric_ema_global": float(metric_ema),
            "current_phase_metric_ema": None if phase_metric_ema is None else float(phase_metric_ema),
            "support_summary_valid": support_summary_valid,
            "support_summary_train": support_summary_train,
            "support_summary_guard_merged": structured_summary,
            "structured_state": bool(structured_summary.get("structured_state", False)),
            "phase_status": self._path1_v5_phase_status(),
            "best_metric_name": self.selection_metric_name,
            "best_metric_ema": None if not math.isfinite(best_metric) else float(best_metric),
            "best_metric_raw": best_raw,
            "best_step": None if best_step < 0 else int(best_step),
            "selection_valid_count": int(self.selection_valid_count),
            "early_stop_enabled": bool(self._early_stop_enabled),
            "early_stop_triggered": bool(self._early_stop_triggered),
            "early_stop_reason": self._early_stop_reason,
            "artifacts": {
                "checkpoint": bf.join(logdir, filename),
                "model": bf.join(logdir, model_filename),
                "ema": bf.join(logdir, f"ema_{self.ema_rate[0]}_{model_filename[6:]}"),
            },
        }
        if extra:
            report.update(extra)
        with bf.BlobFile(bf.join(logdir, report_filename), "wb") as f:
            f.write(json.dumps(report, indent=2, sort_keys=True).encode("utf-8"))

    def _path1_v5_handle_phase_control(self, global_step, metric_raw, phase_metric_ema, structured_summary):
        phase = self.path1_v5_controller_phase
        structured_state = bool((structured_summary or {}).get("structured_state", False))
        current_metric_raw = float(metric_raw)
        phase_valids = self.selection_valid_count - self.path1_v5_phase_start_valid.get(phase, 0)

        if phase == "warmup":
            plateau = (
                global_step >= self.path1_v5_warmup_min_step
                and self.warmup_best_step >= 0
                and (self.selection_valid_count - self.warmup_last_improve_valid)
                >= self.path1_v5_warmup_plateau_valids
            )
            if global_step >= self.path1_v5_warmup_force_step or plateau:
                reason = "force_step" if global_step >= self.path1_v5_warmup_force_step else "plateau"
                self._path1_v5_set_phase(
                    "dyn1",
                    global_step,
                    reason=reason,
                    accepted_phase="dyn1",
                )
            return

        if phase == "dyn1":
            dyn1_last = self.dyn1_last_improve_valid
            if (
                self.warmup_best_raw is not None
                and phase_valids >= self.path1_v5_dyn1_observe_valids
                and (self.selection_valid_count - dyn1_last) >= self.path1_v5_reject_patience_valids
                and current_metric_raw
                > float(self.warmup_best_raw) * (1.0 + self.path1_v5_reject_degrade_vs_prev_phase)
            ):
                self._path1_v5_enter_finalize(
                    global_step,
                    reason="dyn1_reject",
                    rollback_group="warmup",
                    rollback_checkpoint=self.warmup_filename,
                    rejected_phase="dyn1",
                )
                return

            dyn1_ok = (
                global_step >= self.path1_v5_dyn_probe_start_step
                and phase_valids >= self.path1_v5_dyn1_observe_valids
                and structured_state
                and self.warmup_best_step >= 0
                and self.dyn1_best_step >= 0
                and self.dyn1_best_metric
                <= float(self.warmup_best_metric)
                * (1.0 + self.path1_v5_max_degrade_vs_prev_phase_best)
            )
            if dyn1_ok:
                self._path1_v5_set_phase(
                    "dyn_probe",
                    global_step,
                    reason="dyn_probe_admitted",
                    accepted_phase="dyn_probe",
                )
                return
            if global_step >= self.path1_v5_dyn_start_step:
                self._path1_v5_enter_finalize(
                    global_step,
                    reason="dyn_probe_blocked",
                    rollback_group="dyn1" if self.dyn1_best_step >= 0 else "warmup",
                    rollback_checkpoint=self.dyn1_filename if self.dyn1_best_step >= 0 else self.warmup_filename,
                    rejected_phase="dyn_probe",
                )
                self.path1_v5_phase_promotion_blocked_reason["dyn_probe"] = "dyn_probe_blocked"
                return

        if phase == "dyn_probe":
            dyn_last = self.dyn_last_improve_valid
            if (
                self.dyn1_best_raw is not None
                and phase_valids >= self.path1_v5_dyn_probe_observe_valids
                and (self.selection_valid_count - dyn_last) >= self.path1_v5_reject_patience_valids
                and current_metric_raw
                > float(self.dyn1_best_raw) * (1.0 + self.path1_v5_reject_degrade_vs_prev_phase)
            ):
                self._path1_v5_enter_finalize(
                    global_step,
                    reason="dyn_probe_reject",
                    rollback_group="dyn1",
                    rollback_checkpoint=self.dyn1_filename,
                    rejected_phase="dyn_probe",
                )
                return

            dyn_probe_ok = (
                global_step >= self.path1_v5_dyn_start_step
                and phase_valids >= self.path1_v5_dyn_probe_observe_valids
                and structured_state
                and self.dyn1_best_step >= 0
                and self.dyn_best_step >= 0
                and self.dyn_best_metric
                <= float(self.dyn1_best_metric)
                * (1.0 + self.path1_v5_max_degrade_vs_prev_phase_best)
            )
            if dyn_probe_ok:
                self._path1_v5_set_phase(
                    "dyn",
                    global_step,
                    reason="dyn_admitted",
                    accepted_phase="dyn",
                )
                return
            if global_step >= self.path1_v5_dyn_start_step:
                self._path1_v5_enter_finalize(
                    global_step,
                    reason="dyn_probe_blocked",
                    rollback_group="dyn1",
                    rollback_checkpoint=self.dyn1_filename,
                    rejected_phase="dyn_probe",
                )
                self.path1_v5_phase_promotion_blocked_reason["dyn"] = "dyn_probe_blocked"
                return

        if phase == "dyn":
            dyn_last = self.dyn_last_improve_valid
            if (
                self.dyn1_best_raw is not None
                and phase_valids >= self.path1_v5_dyn_observe_valids
                and (self.selection_valid_count - dyn_last) >= self.path1_v5_reject_patience_valids
                and current_metric_raw
                > float(self.dyn1_best_raw) * (1.0 + self.path1_v5_reject_degrade_vs_prev_phase)
            ):
                self._path1_v5_enter_finalize(
                    global_step,
                    reason="dyn_reject",
                    rollback_group="dyn1",
                    rollback_checkpoint=self.dyn1_filename,
                    rejected_phase="dyn",
                )
                return

            dyn_ok = (
                global_step >= self.path1_v5_cons_probe_start_step
                and phase_valids >= self.path1_v5_dyn_observe_valids
                and structured_state
                and self.dyn1_best_step >= 0
                and self.dyn_best_step >= 0
                and self.dyn_best_metric
                <= float(self.dyn1_best_metric)
                * (1.0 + self.path1_v5_max_degrade_vs_prev_phase_best)
            )
            if dyn_ok:
                self._path1_v5_set_phase(
                    "cons_probe",
                    global_step,
                    reason="cons_probe_admitted",
                    accepted_phase="cons_probe",
                )
                return
            if global_step >= self.path1_v5_cons_start_step:
                self._path1_v5_enter_finalize(
                    global_step,
                    reason="cons_probe_blocked",
                    rollback_group="dyn" if self.dyn_best_step >= 0 else "dyn1",
                    rollback_checkpoint=self.dyn_filename if self.dyn_best_step >= 0 else self.dyn1_filename,
                    rejected_phase="cons_probe",
                )
                self.path1_v5_phase_promotion_blocked_reason["cons_probe"] = "cons_probe_blocked"
                return

        if phase == "cons_probe":
            cons_last = self.cons_last_improve_valid
            if (
                self.dyn_best_raw is not None
                and phase_valids >= self.path1_v5_cons_probe_observe_valids
                and (self.selection_valid_count - cons_last) >= self.path1_v5_reject_patience_valids
                and current_metric_raw
                > float(self.dyn_best_raw) * (1.0 + self.path1_v5_reject_degrade_vs_prev_phase)
            ):
                self._path1_v5_enter_finalize(
                    global_step,
                    reason="cons_probe_reject",
                    rollback_group="dyn",
                    rollback_checkpoint=self.dyn_filename,
                    rejected_phase="cons_probe",
                )
                return

            cons_probe_ok = (
                global_step >= self.path1_v5_cons_start_step
                and phase_valids >= self.path1_v5_cons_probe_observe_valids
                and structured_state
                and self.dyn_best_step >= 0
                and self.cons_best_step >= 0
                and self.cons_best_metric
                <= float(self.dyn_best_metric)
                * (1.0 + self.path1_v5_max_degrade_vs_prev_phase_best)
            )
            if cons_probe_ok:
                self._path1_v5_set_phase(
                    "cons",
                    global_step,
                    reason="cons_admitted",
                    accepted_phase="cons",
                )
                return
            if global_step >= self.path1_v5_cons_start_step:
                self._path1_v5_enter_finalize(
                    global_step,
                    reason="cons_probe_blocked",
                    rollback_group="dyn",
                    rollback_checkpoint=self.dyn_filename,
                    rejected_phase="cons_probe",
                )
                self.path1_v5_phase_promotion_blocked_reason["cons"] = "cons_probe_blocked"
                return

        if phase == "cons":
            cons_last = self.cons_last_improve_valid
            if (
                self.dyn_best_raw is not None
                and phase_valids >= self.path1_v5_cons_observe_valids
                and (self.selection_valid_count - cons_last) >= self.path1_v5_reject_patience_valids
                and current_metric_raw
                > float(self.dyn_best_raw) * (1.0 + self.path1_v5_reject_degrade_vs_prev_phase)
            ):
                self._path1_v5_enter_finalize(
                    global_step,
                    reason="cons_reject",
                    rollback_group="dyn",
                    rollback_checkpoint=self.dyn_filename,
                    rejected_phase="cons",
                )

    def _should_trigger_path1_v5_early_stop(self, global_step):
        if not self.path1_v5_enabled:
            return False
        if not self._early_stop_enabled:
            return False
        if self.final_best_step < 0:
            return False
        phase_name = self.path1_v5_controller_phase
        if phase_name == "finalize":
            return (
                self.selection_valid_count - self.final_last_improve_valid
            ) >= self.path1_v5_finalize_patience_valids
        if phase_name == "cons" and global_step >= self._early_stop_start_step:
            return (
                self.selection_valid_count - self.final_last_improve_valid
            ) >= self.selection_patience_valids
        return False

    def _path1_v5_state_dict(self):
        if not self.path1_v5_enabled:
            return None
        return {
            "controller_phase": self.path1_v5_controller_phase,
            "previous_phase": self.path1_v5_previous_phase,
            "phase_accepted": dict(self.path1_v5_phase_accepted),
            "phase_rejected": dict(self.path1_v5_phase_rejected),
            "phase_rejection_reason": dict(self.path1_v5_phase_rejection_reason),
            "phase_promotion_blocked_reason": dict(self.path1_v5_phase_promotion_blocked_reason),
            "phase_start_step": dict(self.path1_v5_phase_start_step),
            "phase_start_valid": dict(self.path1_v5_phase_start_valid),
            "phase_best_metric": dict(self.path1_v5_phase_best_metric),
            "phase_best_step": dict(self.path1_v5_phase_best_step),
            "phase_best_raw": dict(self.path1_v5_phase_best_raw),
            "phase_last_improve_valid": dict(self.path1_v5_phase_last_improve_valid),
            "phase_entry_reason": dict(self.path1_v5_phase_entry_reason),
            "phase_metric_ema": dict(self.path1_v5_phase_metric_ema),
            "structured_state": bool(self.path1_v5_structured_state),
            "structured_true_streak": int(self.path1_v5_structured_true_streak),
            "overflow_valid_ema": self.path1_v5_overflow_valid_ema,
            "selection_valid_count": int(self.selection_valid_count),
            "selection_metric_ema": self.selection_metric_ema,
            "warmup_best_metric": self.warmup_best_metric,
            "warmup_best_step": self.warmup_best_step,
            "warmup_best_raw": self.warmup_best_raw,
            "warmup_last_improve_valid": self.warmup_last_improve_valid,
            "dyn1_best_metric": self.dyn1_best_metric,
            "dyn1_best_step": self.dyn1_best_step,
            "dyn1_best_raw": self.dyn1_best_raw,
            "dyn1_last_improve_valid": self.dyn1_last_improve_valid,
            "dyn_best_metric": self.dyn_best_metric,
            "dyn_best_step": self.dyn_best_step,
            "dyn_best_raw": self.dyn_best_raw,
            "dyn_last_improve_valid": self.dyn_last_improve_valid,
            "cons_best_metric": self.cons_best_metric,
            "cons_best_step": self.cons_best_step,
            "cons_best_raw": self.cons_best_raw,
            "cons_last_improve_valid": self.cons_last_improve_valid,
            "structured_best_metric": self.structured_best_metric,
            "structured_best_step": self.structured_best_step,
            "structured_best_raw": self.structured_best_raw,
            "structured_best_phase": self.structured_best_phase,
            "structured_best_support": dict(self.structured_best_support),
            "structured_last_improve_valid": self.structured_last_improve_valid,
            "final_best_metric": self.final_best_metric,
            "final_best_step": self.final_best_step,
            "final_best_raw": self.final_best_raw,
            "final_best_phase": self.final_best_phase,
            "final_best_source": self.final_best_source,
            "final_last_improve_valid": self.final_last_improve_valid,
            "finalize_reason": self.path1_v5_finalize_reason,
            "rollback_source_phase": self.path1_v5_rollback_source_phase,
            "rollback_source_checkpoint": self.path1_v5_rollback_source_checkpoint,
            "early_stop_triggered": bool(self._early_stop_triggered),
            "early_stop_reason": self._early_stop_reason,
        }

    def _load_path1_v5_state_dict(self, state):
        if (not self.path1_v5_enabled) or (not state):
            return
        self.path1_v5_controller_phase = str(
            state.get("controller_phase", self.path1_v5_controller_phase)
        )
        self.path1_v5_previous_phase = state.get("previous_phase", self.path1_v5_previous_phase)
        self.path1_v5_phase_accepted = dict(state.get("phase_accepted", self.path1_v5_phase_accepted))
        self.path1_v5_phase_rejected = dict(state.get("phase_rejected", self.path1_v5_phase_rejected))
        self.path1_v5_phase_rejection_reason = dict(
            state.get("phase_rejection_reason", self.path1_v5_phase_rejection_reason)
        )
        self.path1_v5_phase_promotion_blocked_reason = dict(
            state.get(
                "phase_promotion_blocked_reason",
                self.path1_v5_phase_promotion_blocked_reason,
            )
        )
        self.path1_v5_phase_start_step = dict(
            state.get("phase_start_step", self.path1_v5_phase_start_step)
        )
        self.path1_v5_phase_start_valid = dict(
            state.get("phase_start_valid", self.path1_v5_phase_start_valid)
        )
        self.path1_v5_phase_best_metric = dict(
            state.get("phase_best_metric", self.path1_v5_phase_best_metric)
        )
        self.path1_v5_phase_best_step = dict(
            state.get("phase_best_step", self.path1_v5_phase_best_step)
        )
        self.path1_v5_phase_best_raw = dict(
            state.get("phase_best_raw", self.path1_v5_phase_best_raw)
        )
        self.path1_v5_phase_last_improve_valid = dict(
            state.get("phase_last_improve_valid", self.path1_v5_phase_last_improve_valid)
        )
        self.path1_v5_phase_entry_reason = dict(
            state.get("phase_entry_reason", self.path1_v5_phase_entry_reason)
        )
        self.path1_v5_phase_metric_ema = dict(
            state.get("phase_metric_ema", self.path1_v5_phase_metric_ema)
        )
        self.path1_v5_structured_state = bool(
            state.get("structured_state", self.path1_v5_structured_state)
        )
        self.path1_v5_structured_true_streak = int(
            state.get("structured_true_streak", self.path1_v5_structured_true_streak)
        )
        self.path1_v5_overflow_valid_ema = state.get(
            "overflow_valid_ema", self.path1_v5_overflow_valid_ema
        )
        self.selection_valid_count = int(state.get("selection_valid_count", self.selection_valid_count))
        self.selection_metric_ema = state.get("selection_metric_ema", self.selection_metric_ema)
        self.warmup_best_metric = float(state.get("warmup_best_metric", self.warmup_best_metric))
        self.warmup_best_step = int(state.get("warmup_best_step", self.warmup_best_step))
        self.warmup_best_raw = state.get("warmup_best_raw", self.warmup_best_raw)
        self.warmup_last_improve_valid = int(
            state.get("warmup_last_improve_valid", self.warmup_last_improve_valid)
        )
        self.dyn1_best_metric = float(state.get("dyn1_best_metric", self.dyn1_best_metric))
        self.dyn1_best_step = int(state.get("dyn1_best_step", self.dyn1_best_step))
        self.dyn1_best_raw = state.get("dyn1_best_raw", self.dyn1_best_raw)
        self.dyn1_last_improve_valid = int(
            state.get("dyn1_last_improve_valid", self.dyn1_last_improve_valid)
        )
        self.dyn_best_metric = float(state.get("dyn_best_metric", self.dyn_best_metric))
        self.dyn_best_step = int(state.get("dyn_best_step", self.dyn_best_step))
        self.dyn_best_raw = state.get("dyn_best_raw", self.dyn_best_raw)
        self.dyn_last_improve_valid = int(
            state.get("dyn_last_improve_valid", self.dyn_last_improve_valid)
        )
        self.cons_best_metric = float(state.get("cons_best_metric", self.cons_best_metric))
        self.cons_best_step = int(state.get("cons_best_step", self.cons_best_step))
        self.cons_best_raw = state.get("cons_best_raw", self.cons_best_raw)
        self.cons_last_improve_valid = int(
            state.get("cons_last_improve_valid", self.cons_last_improve_valid)
        )
        self.structured_best_metric = float(
            state.get("structured_best_metric", self.structured_best_metric)
        )
        self.structured_best_step = int(state.get("structured_best_step", self.structured_best_step))
        self.structured_best_raw = state.get("structured_best_raw", self.structured_best_raw)
        self.structured_best_phase = state.get("structured_best_phase", self.structured_best_phase)
        self.structured_best_support = dict(
            state.get("structured_best_support", self.structured_best_support)
        )
        self.structured_last_improve_valid = int(
            state.get("structured_last_improve_valid", self.structured_last_improve_valid)
        )
        self.final_best_metric = float(state.get("final_best_metric", self.final_best_metric))
        self.final_best_step = int(state.get("final_best_step", self.final_best_step))
        self.final_best_raw = state.get("final_best_raw", self.final_best_raw)
        self.final_best_phase = state.get("final_best_phase", self.final_best_phase)
        self.final_best_source = state.get("final_best_source", self.final_best_source)
        self.final_last_improve_valid = int(
            state.get("final_last_improve_valid", self.final_last_improve_valid)
        )
        self.path1_v5_finalize_reason = state.get("finalize_reason", self.path1_v5_finalize_reason)
        self.path1_v5_rollback_source_phase = state.get(
            "rollback_source_phase", self.path1_v5_rollback_source_phase
        )
        self.path1_v5_rollback_source_checkpoint = state.get(
            "rollback_source_checkpoint", self.path1_v5_rollback_source_checkpoint
        )
        self._early_stop_triggered = bool(
            state.get("early_stop_triggered", self._early_stop_triggered)
        )
        self._early_stop_reason = state.get("early_stop_reason", self._early_stop_reason)

    def _maybe_update_best_path1_v5(self, valid_info):
        if not valid_info:
            return
        metric_raw = valid_info.get(self.selection_metric_name, None)
        if metric_raw is None:
            return
        metric_raw = float(metric_raw)
        if not math.isfinite(metric_raw):
            return

        self.selection_valid_count += 1
        if self.selection_metric_ema is None:
            self.selection_metric_ema = metric_raw
        else:
            alpha = self.selection_ema_alpha
            self.selection_metric_ema = (1.0 - alpha) * float(self.selection_metric_ema) + alpha * metric_raw

        global_step = self.step + self.resume_step
        current_phase = self.path1_v5_controller_phase
        current_phase_state = self._path1_v5_phase_state(global_step) or {}
        support_summary_valid = valid_info.get("support_summary_valid", {}) or {}
        support_summary_train = valid_info.get("support_summary_train_interval", {}) or {}
        structured_summary = self._path1_v5_update_structured_state(
            valid_info.get("support_summary_guard_merged", {}) or {}
        )
        structured_ok = bool(structured_summary.get("structured_state", False))
        phase_metric_ema = self._path1_v5_phase_local_ema(current_phase, metric_raw)
        phase_group = self._path1_v5_phase_group(current_phase)

        phase_codes = {
            "warmup": 0.0,
            "dyn1": 1.0,
            "dyn_probe": 2.0,
            "dyn": 3.0,
            "cons_probe": 4.0,
            "cons": 5.0,
            "finalize": 6.0,
        }
        logger.logkv_mean("phase/current_code", phase_codes.get(current_phase, -1.0))
        logger.logkv_mean("phase/structured_state", 1.0 if structured_ok else 0.0)
        logger.logkv_mean(
            "phase/overflow_valid_ema",
            float(structured_summary.get("overflow_rate_p95_valid_ema", float("nan")) or float("nan")),
        )
        logger.logkv_mean("phase/structured_hold_count", float(structured_summary.get("structured_hold_count", 0)))

        warmup_improved = False
        dyn1_improved = False
        dyn_improved = False
        cons_improved = False

        if current_phase == "warmup":
            warmup_improved = self._path1_v5_update_group_best(
                "warmup", phase_metric_ema, metric_raw, global_step
            )
            if warmup_improved:
                self._save_best_checkpoint(
                    metric_raw=metric_raw,
                    metric_ema=float(self.warmup_best_metric),
                    filename=self.warmup_filename,
                    extra_report={
                        "mode": "path1_v5",
                        "kind": "best_warmup",
                        "phase": current_phase,
                        "phase_state": current_phase_state,
                        "phase_metric_ema": float(phase_metric_ema),
                        "phase_status": self._path1_v5_phase_status(),
                        "support_summary_valid": support_summary_valid,
                        "support_summary_train_interval": support_summary_train,
                        "support_summary_guard_merged": structured_summary,
                        "support_ok": bool(structured_ok),
                    },
                )
        elif phase_group == "dyn1":
            dyn1_improved = self._path1_v5_update_group_best(
                "dyn1", phase_metric_ema, metric_raw, global_step
            )
            if dyn1_improved:
                self._save_best_checkpoint(
                    metric_raw=metric_raw,
                    metric_ema=float(self.dyn1_best_metric),
                    filename=self.dyn1_filename,
                    extra_report={
                        "mode": "path1_v5",
                        "kind": "best_dyn1",
                        "phase": current_phase,
                        "phase_state": current_phase_state,
                        "phase_metric_ema": float(phase_metric_ema),
                        "phase_status": self._path1_v5_phase_status(),
                        "support_summary_valid": support_summary_valid,
                        "support_summary_train_interval": support_summary_train,
                        "support_summary_guard_merged": structured_summary,
                        "support_ok": bool(structured_ok),
                    },
                )
        elif phase_group == "dyn":
            dyn_improved = self._path1_v5_update_group_best(
                "dyn", phase_metric_ema, metric_raw, global_step
            )
            if dyn_improved:
                self._save_best_checkpoint(
                    metric_raw=metric_raw,
                    metric_ema=float(self.dyn_best_metric),
                    filename=self.dyn_filename,
                    extra_report={
                        "mode": "path1_v5",
                        "kind": "best_dyn",
                        "phase": current_phase,
                        "phase_state": current_phase_state,
                        "phase_metric_ema": float(phase_metric_ema),
                        "phase_status": self._path1_v5_phase_status(),
                        "support_summary_valid": support_summary_valid,
                        "support_summary_train_interval": support_summary_train,
                        "support_summary_guard_merged": structured_summary,
                        "support_ok": bool(structured_ok),
                    },
                )
        elif phase_group == "cons":
            cons_improved = self._path1_v5_update_group_best(
                "cons", phase_metric_ema, metric_raw, global_step
            )
            if cons_improved:
                self._save_best_checkpoint(
                    metric_raw=metric_raw,
                    metric_ema=float(self.cons_best_metric),
                    filename=self.cons_filename,
                    extra_report={
                        "mode": "path1_v5",
                        "kind": "best_cons",
                        "phase": current_phase,
                        "phase_state": current_phase_state,
                        "phase_metric_ema": float(phase_metric_ema),
                        "phase_status": self._path1_v5_phase_status(),
                        "support_summary_valid": support_summary_valid,
                        "support_summary_train_interval": support_summary_train,
                        "support_summary_guard_merged": structured_summary,
                        "support_ok": bool(structured_ok),
                    },
                )

        structured_improved = False
        if phase_group in {"dyn1", "dyn", "cons"} and structured_ok and float(phase_metric_ema) < (
            self.structured_best_metric - self.selection_min_delta
        ):
            self.structured_best_metric = float(phase_metric_ema)
            self.structured_best_step = int(global_step)
            self.structured_best_raw = metric_raw
            self.structured_best_phase = current_phase
            self.structured_best_support = dict(structured_summary)
            self.structured_last_improve_valid = self.selection_valid_count
            self._save_best_checkpoint(
                metric_raw=metric_raw,
                metric_ema=float(self.structured_best_metric),
                filename=self.structured_filename,
                extra_report={
                    "mode": "path1_v5",
                    "kind": "best_structured",
                    "phase": current_phase,
                    "phase_state": current_phase_state,
                    "phase_metric_ema": float(phase_metric_ema),
                    "phase_status": self._path1_v5_phase_status(),
                    "support_summary_valid": support_summary_valid,
                    "support_summary_train_interval": support_summary_train,
                    "support_summary_guard_merged": structured_summary,
                    "support_ok": True,
                },
            )
            structured_improved = True

        final_improved = self._maybe_update_path1_v5_final(
            warmup_improved=warmup_improved,
            dyn1_improved=dyn1_improved,
            dyn_improved=dyn_improved,
            cons_improved=cons_improved,
            current_phase=current_phase,
            phase_state=current_phase_state,
            support_summary_valid=support_summary_valid,
            support_summary_train=support_summary_train,
            structured_summary=structured_summary,
        )

        self._path1_v5_handle_phase_control(
            global_step, metric_raw, phase_metric_ema, structured_summary
        )
        current_phase = self.path1_v5_controller_phase
        current_phase_state = self._path1_v5_phase_state(global_step) or {}

        logger.logkv_mean("best_warmup/raw", metric_raw)
        logger.logkv_mean("best_warmup/ema", float(phase_metric_ema if phase_group == "warmup" else self.warmup_best_metric if math.isfinite(self.warmup_best_metric) else metric_raw))
        logger.logkv_mean("best_dyn1/raw", metric_raw)
        logger.logkv_mean("best_dyn1/ema", float(phase_metric_ema if phase_group == "dyn1" else self.dyn1_best_metric if math.isfinite(self.dyn1_best_metric) else metric_raw))
        logger.logkv_mean("best_dyn/raw", metric_raw)
        logger.logkv_mean("best_dyn/ema", float(phase_metric_ema if phase_group == "dyn" else self.dyn_best_metric if math.isfinite(self.dyn_best_metric) else metric_raw))
        logger.logkv_mean("best_cons/raw", metric_raw)
        logger.logkv_mean("best_cons/ema", float(phase_metric_ema if phase_group == "cons" else self.cons_best_metric if math.isfinite(self.cons_best_metric) else metric_raw))
        logger.logkv_mean("best_structured/raw", metric_raw)
        logger.logkv_mean("best_structured/ema", float(self.structured_best_metric if math.isfinite(self.structured_best_metric) else metric_raw))
        logger.logkv_mean("best_structured/support_ok", 1.0 if structured_ok else 0.0)
        logger.logkv_mean("best_final/updated", 1.0 if final_improved else 0.0)
        logger.logkv_mean("best_structured/updated", 1.0 if structured_improved else 0.0)
        logger.logkv_mean("best_warmup/updated", 1.0 if warmup_improved else 0.0)
        logger.logkv_mean("best_dyn1/updated", 1.0 if dyn1_improved else 0.0)
        logger.logkv_mean("best_dyn/updated", 1.0 if dyn_improved else 0.0)
        logger.logkv_mean("best_cons/updated", 1.0 if cons_improved else 0.0)
        logger.logkv_mean("phase/current_code", phase_codes.get(current_phase, -1.0))

        if self.warmup_best_step >= 0:
            logger.logkv_mean("best_warmup/step", float(self.warmup_best_step))
        if self.dyn1_best_step >= 0:
            logger.logkv_mean("best_dyn1/step", float(self.dyn1_best_step))
        if self.dyn_best_step >= 0:
            logger.logkv_mean("best_dyn/step", float(self.dyn_best_step))
        if self.cons_best_step >= 0:
            logger.logkv_mean("best_cons/step", float(self.cons_best_step))
        if self.structured_best_step >= 0:
            logger.logkv_mean("best_structured/step", float(self.structured_best_step))
        if self.final_best_step >= 0:
            logger.logkv_mean("best_final/step", float(self.final_best_step))

        self._write_path1_v5_report(
            report_filename=self.warmup_report_filename,
            kind="best_warmup",
            current_step=global_step,
            current_phase=current_phase,
            phase_state=current_phase_state,
            metric_raw=metric_raw,
            metric_ema=float(self.selection_metric_ema),
            phase_metric_ema=phase_metric_ema,
            support_summary_valid=support_summary_valid,
            support_summary_train=support_summary_train,
            structured_summary=structured_summary,
            best_metric=self.warmup_best_metric,
            best_raw=self.warmup_best_raw,
            best_step=self.warmup_best_step,
            filename=self.warmup_filename,
        )
        self._write_path1_v5_report(
            report_filename=self.dyn1_report_filename,
            kind="best_dyn1",
            current_step=global_step,
            current_phase=current_phase,
            phase_state=current_phase_state,
            metric_raw=metric_raw,
            metric_ema=float(self.selection_metric_ema),
            phase_metric_ema=phase_metric_ema,
            support_summary_valid=support_summary_valid,
            support_summary_train=support_summary_train,
            structured_summary=structured_summary,
            best_metric=self.dyn1_best_metric,
            best_raw=self.dyn1_best_raw,
            best_step=self.dyn1_best_step,
            filename=self.dyn1_filename,
        )
        self._write_path1_v5_report(
            report_filename=self.dyn_report_filename,
            kind="best_dyn",
            current_step=global_step,
            current_phase=current_phase,
            phase_state=current_phase_state,
            metric_raw=metric_raw,
            metric_ema=float(self.selection_metric_ema),
            phase_metric_ema=phase_metric_ema,
            support_summary_valid=support_summary_valid,
            support_summary_train=support_summary_train,
            structured_summary=structured_summary,
            best_metric=self.dyn_best_metric,
            best_raw=self.dyn_best_raw,
            best_step=self.dyn_best_step,
            filename=self.dyn_filename,
        )
        self._write_path1_v5_report(
            report_filename=self.cons_report_filename,
            kind="best_cons",
            current_step=global_step,
            current_phase=current_phase,
            phase_state=current_phase_state,
            metric_raw=metric_raw,
            metric_ema=float(self.selection_metric_ema),
            phase_metric_ema=phase_metric_ema,
            support_summary_valid=support_summary_valid,
            support_summary_train=support_summary_train,
            structured_summary=structured_summary,
            best_metric=self.cons_best_metric,
            best_raw=self.cons_best_raw,
            best_step=self.cons_best_step,
            filename=self.cons_filename,
        )
        self._write_path1_v5_report(
            report_filename=self.structured_report_filename,
            kind="best_structured",
            current_step=global_step,
            current_phase=current_phase,
            phase_state=current_phase_state,
            metric_raw=metric_raw,
            metric_ema=float(self.selection_metric_ema),
            phase_metric_ema=phase_metric_ema,
            support_summary_valid=support_summary_valid,
            support_summary_train=support_summary_train,
            structured_summary=structured_summary,
            best_metric=self.structured_best_metric,
            best_raw=self.structured_best_raw,
            best_step=self.structured_best_step,
            filename=self.structured_filename,
            extra={"best_phase": self.structured_best_phase, "best_support_summary": self.structured_best_support},
        )
        self._write_path1_v5_report(
            report_filename=self.final_report_filename,
            kind="best_final",
            current_step=global_step,
            current_phase=current_phase,
            phase_state=current_phase_state,
            metric_raw=metric_raw,
            metric_ema=float(self.selection_metric_ema),
            phase_metric_ema=phase_metric_ema,
            support_summary_valid=support_summary_valid,
            support_summary_train=support_summary_train,
            structured_summary=structured_summary,
            best_metric=self.final_best_metric,
            best_raw=self.final_best_raw,
            best_step=self.final_best_step,
            filename=self.final_filename,
            extra={"source": self.final_best_source, "best_phase": self.final_best_phase},
        )
        if self._should_trigger_path1_v5_early_stop(global_step):
            self._early_stop_triggered = True
            self._early_stop_reason = (
                f"no best_final improvement for {self.path1_v5_finalize_patience_valids if self.path1_v5_controller_phase == 'finalize' else self.selection_patience_valids} validations "
                f"in phase={self.path1_v5_controller_phase}"
            )
            logger.log(f"[path1_v5] early stop triggered: {self._early_stop_reason}")

    @staticmethod
    def _best_artifact_names(checkpoint_filename: str):
        if checkpoint_filename == "checkpoint_best.pt":
            return "model_best.pt", "opt_best.pt", "best_report.json"
        stem = checkpoint_filename
        if stem.startswith("checkpoint_") and stem.endswith(".pt"):
            stem = stem[len("checkpoint_") : -3]
        else:
            stem = "best_window"
        return f"model_{stem}.pt", f"opt_{stem}.pt", f"{stem}_report.json"

    def _write_path1_v3_report(
        self,
        report_filename,
        kind,
        current_step,
        current_phase,
        phase_state,
        metric_raw,
        metric_ema,
        support_summary_valid,
        support_summary_train,
        support_summary_guard,
        support_ok,
        best_metric,
        best_raw,
        best_step,
        best_phase,
        filename,
        best_support_summary=None,
    ):
        if (not self.path1_v3_enabled) or dist.get_rank() != 0:
            return
        logdir = get_blob_logdir()
        model_filename, _, _ = self._best_artifact_names(filename)
        report = {
            "mode": "path1_v3",
            "kind": kind,
            "current_step": int(current_step),
            "current_phase": current_phase,
            "current_phase_state": phase_state,
            "current_lambda_koop_eff": float(self._path1_v3_state.get("lambda_koop_eff", 0.0)),
            "current_metric_raw": float(metric_raw),
            "current_metric_ema": float(metric_ema),
            "structured_guard_valid": support_summary_valid,
            "structured_guard_train": support_summary_train,
            "structured_guard_merged": support_summary_guard,
            "structured_guard_ok": bool(support_ok),
            "best_metric_name": self.selection_metric_name,
            "best_metric_ema": None if not math.isfinite(best_metric) else float(best_metric),
            "best_metric_raw": best_raw,
            "best_step": None if best_step < 0 else int(best_step),
            "best_phase": best_phase,
            "best_support_summary": best_support_summary or {},
            "selection_valid_count": int(self.selection_valid_count),
            "early_stop_enabled": bool(self._early_stop_enabled),
            "early_stop_triggered": bool(self._early_stop_triggered),
            "early_stop_reason": self._early_stop_reason,
            "artifacts": {
                "checkpoint": bf.join(logdir, filename),
                "model": bf.join(logdir, model_filename),
                "ema": bf.join(logdir, f"ema_{self.ema_rate[0]}_{model_filename[6:]}"),
            },
        }
        with bf.BlobFile(bf.join(logdir, report_filename), "wb") as f:
            f.write(json.dumps(report, indent=2, sort_keys=True).encode("utf-8"))

    def _write_best_window_report(
        self,
        current_step,
        metric_raw,
        metric_ema,
        support_summary_valid,
        support_summary_train,
        support_summary_guard,
        support_ok,
    ):
        if (not self.path1_v2_enabled) or dist.get_rank() != 0:
            return
        logdir = get_blob_logdir()
        report = {
            "mode": "path1_v2",
            "current_step": int(current_step),
            "current_metric_raw": float(metric_raw),
            "current_metric_ema": float(metric_ema),
            "current_support_summary_valid": support_summary_valid,
            "current_support_summary_train_interval": support_summary_train,
            "current_support_summary_guard_merged": support_summary_guard,
            "current_support_ok": bool(support_ok),
            "best_metric_name": self.selection_metric_name,
            "best_metric_ema": None if not math.isfinite(self.selection_best_metric) else float(self.selection_best_metric),
            "best_metric_raw": self.selection_best_raw,
            "best_step": None if self.selection_best_step < 0 else int(self.selection_best_step),
            "best_support_summary": self.selection_best_support,
            "selection_valid_count": int(self.selection_valid_count),
            "last_improve_valid": int(self.selection_last_improve_valid),
            "early_stop_enabled": bool(self._early_stop_enabled),
            "early_stop_triggered": bool(self._early_stop_triggered),
            "early_stop_reason": self._early_stop_reason,
            "artifacts": {
                "checkpoint": bf.join(logdir, self.selection_filename),
                "model": bf.join(logdir, self._best_artifact_names(self.selection_filename)[0]),
                "ema": bf.join(logdir, f"ema_{self.ema_rate[0]}_{self._best_artifact_names(self.selection_filename)[0][6:]}"),
            },
        }
        with bf.BlobFile(bf.join(logdir, self.selection_report_filename), "wb") as f:
            f.write(json.dumps(report, indent=2, sort_keys=True).encode("utf-8"))

    def _save_best_checkpoint(self, metric_raw: float, metric_ema: float, filename=None, extra_report=None):
        if dist.get_rank() != 0:
            return

        logdir = get_blob_logdir()
        checkpoint_filename = str(filename or self.best_filename)
        model_filename, opt_filename, default_report_filename = self._best_artifact_names(
            checkpoint_filename
        )
        if self.checkpoint_policy.lightweight_best:
            suffix = (
                model_filename[len("model_") : -3]
                if model_filename.startswith("model_")
                else "best"
            )
            ema_filenames = []
            for rate, params in zip(self.ema_rate, self.ema_params):
                state = self.mp_trainer.master_params_to_state_dict(params)
                ema_filename = f"ema_{rate}_{suffix}.pt"
                with bf.BlobFile(bf.join(logdir, ema_filename), "wb") as f:
                    th.save(state, f)
                ema_filenames.append(ema_filename)

            report_payload = {
                "step": int(self.step + self.resume_step),
                "metric_name": str(self.best_metric_name),
                "metric_raw": float(metric_raw),
                "metric_ema": float(metric_ema),
                "valid_count": int(self.valid_count),
                "selector": {
                    "ema_decay": float(self.best_ema_decay),
                    "warmup_valid_count": int(self.best_warmup_valid_count),
                    "min_delta": float(self.best_min_delta),
                },
                "artifacts": {"ema": ema_filenames},
            }
            if extra_report:
                report_payload.update(extra_report)
            with bf.BlobFile(bf.join(logdir, default_report_filename), "wb") as f:
                f.write(json.dumps(report_payload, indent=2, sort_keys=True).encode("utf-8"))
            return

        model_state = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
        ema_states = {
            str(rate): self.mp_trainer.master_params_to_state_dict(params)
            for rate, params in zip(self.ema_rate, self.ema_params)
        }
        payload = {
            "step": int(self.step + self.resume_step),
            "best_metric_name": str(
                self.selection_metric_name if self.path_schedule_enabled else self.best_metric_name
            ),
            "best_metric_raw": float(metric_raw),
            "best_metric_ema": float(metric_ema),
            "best_valid_count": int(
                self.selection_valid_count if self.path_schedule_enabled else self.valid_count
            ),
            "model_state_dict": model_state,
            "ema_state_dicts": ema_states,
            "optimizer_state_dict": self.opt.state_dict(),
            "koopman_regularizer_state": (
                self.koopman_regularizer.state_dict()
                if (self.koopman_regularizer is not None and hasattr(self.koopman_regularizer, "state_dict"))
                else None
            ),
            "lambda_balancer_state": (
                self.lambda_balancer.state_dict()
                if (self.lambda_balancer is not None and hasattr(self.lambda_balancer, "state_dict"))
                else None
            ),
            "path1_v4_state": self._path1_v4_state_dict(),
            "path1_v5_state": self._path1_v5_state_dict(),
            "state_version": 1,
        }
        # Save the combined selected-checkpoint metadata.
        with bf.BlobFile(bf.join(logdir, checkpoint_filename), "wb") as f:
            th.save(payload, f)
        # Save selected model weights for sampling and training resumption.
        with bf.BlobFile(bf.join(logdir, model_filename), "wb") as f:
            th.save(model_state, f)
        for rate, state in ema_states.items():
            suffix = model_filename[len("model_") : -3] if model_filename.startswith("model_") else "best"
            with bf.BlobFile(bf.join(logdir, f"ema_{rate}_{suffix}.pt"), "wb") as f:
                th.save(state, f)
        opt_payload = {
            "optimizer_state_dict": self.opt.state_dict(),
            "koopman_regularizer_state": payload["koopman_regularizer_state"],
            "lambda_balancer_state": payload["lambda_balancer_state"],
            "path1_v4_state": payload["path1_v4_state"],
            "path1_v5_state": payload["path1_v5_state"],
            "state_version": 2,
        }
        with bf.BlobFile(bf.join(logdir, opt_filename), "wb") as f:
            th.save(opt_payload, f)
        report_payload = {
            "step": int(self.step + self.resume_step),
            "metric_raw": float(metric_raw),
            "metric_ema": float(metric_ema),
        }
        if extra_report:
            report_payload.update(extra_report)
        with bf.BlobFile(bf.join(logdir, default_report_filename), "wb") as f:
            f.write(json.dumps(report_payload, indent=2, sort_keys=True).encode("utf-8"))

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if self.path_schedule_enabled:
            schedule_name = str(
                self.training_schedule_cfg.get("lr_schedule", "cosine")
            ).lower()
            warmup_steps = self._resolve_schedule_point(
                explicit_step=self.training_schedule_cfg.get("lr_warmup_steps", None),
                frac=self.training_schedule_cfg.get("lr_warmup_frac", 0.01),
                default=0,
            )
            min_ratio = float(self.training_schedule_cfg.get("lr_min_ratio", 0.10))
            global_step = self.step + self.resume_step
            if warmup_steps > 0 and global_step < warmup_steps:
                lr = self.lr * float(global_step + 1) / float(max(1, warmup_steps))
            elif schedule_name == "cosine":
                denom = max(1, self.total_train_steps - warmup_steps)
                progress = min(
                    1.0,
                    max(0.0, float(global_step - warmup_steps) / float(denom)),
                )
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.lr * (min_ratio + (1.0 - min_ratio) * cosine)
            else:
                lr = self.lr
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr
            return
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.final_lr * (frac_done) + self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        total_samples = (self.step + self.resume_step + 1) * self.global_batch
        logger.logkv("system/samples", total_samples)
        logger.logkv("system/samples_per_rank", self.batch_size)

    def _get_valid_timesteps(self, batch_indices, batch_offset, batch_size):
        if self.valid_t_all is not None:
            if batch_indices is None:
                start = batch_offset
                stop = start + batch_size
                if stop > self.valid_t_all.shape[0]:
                    raise ValueError("validation batch offset exceeds precomputed valid timestep table")
                indices = th.arange(start, stop, dtype=th.long)
            else:
                indices = batch_indices.long().cpu()
            return self.valid_t_all[indices].to(dist_util.dev())
        return build_fixed_uniform_timesteps(
            num_samples=batch_size,
            num_timesteps=self.diffusion.num_timesteps,
            seed=self.valid_t_seed + batch_offset,
        ).to(dist_util.dev())

    def _new_valid_noise_generator(self):
        """Create a per-validation RNG stream without advancing training RNG."""

        if not self.validation_rng_enabled:
            return None
        generator = th.Generator(device=dist_util.dev())
        generator.manual_seed(self.valid_noise_seed)
        return generator

    @staticmethod
    def _sample_valid_noise(reference, generator):
        if generator is None:
            return None
        return th.randn(
            reference.shape,
            dtype=reference.dtype,
            device=reference.device,
            generator=generator,
        )

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            extra_state = {
                "optimizer_state_dict": self.opt.state_dict(),
                "koopman_regularizer_state": (
                    self.koopman_regularizer.state_dict()
                    if (self.koopman_regularizer is not None and hasattr(self.koopman_regularizer, "state_dict"))
                    else None
                ),
                "lambda_balancer_state": (
                    self.lambda_balancer.state_dict()
                    if (self.lambda_balancer is not None and hasattr(self.lambda_balancer, "state_dict"))
                    else None
                ),
                "path1_v4_state": self._path1_v4_state_dict(),
                "path1_v5_state": self._path1_v5_state_dict(),
                "state_version": 2,
            }
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(extra_state, f)

        dist.barrier()

    def save_ema_only(self):
        """Save exact EMA states without raw model or optimizer artifacts."""

        global_step = self.step + self.resume_step
        for rate, params in zip(self.ema_rate, self.ema_params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                filename = f"ema_{rate}_{global_step:06d}.pt"
                logger.log(f"saving EMA milestone {rate} at step {global_step}...")
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)
        dist.barrier()

    def snapshot_lightweight_best(self, milestone):
        """Freeze the validation-selected best artifact available at a milestone."""

        if dist.get_rank() != 0:
            return
        if not (
            self.best_ckpt_enabled
            and self.checkpoint_policy.lightweight_best
            and self.checkpoint_policy.should_snapshot_best(milestone)
        ):
            return

        logdir = get_blob_logdir()
        model_filename, _, report_filename = self._best_artifact_names(self.best_filename)
        suffix = (
            model_filename[len("model_") : -3]
            if model_filename.startswith("model_")
            else "best"
        )
        source_report = os.path.join(logdir, report_filename)
        if not os.path.isfile(source_report):
            raise RuntimeError(
                f"cannot snapshot validation best at {milestone}: missing {source_report}"
            )

        snapshot_ema = []
        for rate in self.ema_rate:
            source_name = f"ema_{rate}_{suffix}.pt"
            target_name = f"ema_{rate}_{suffix}_upto{int(milestone):06d}.pt"
            source = os.path.join(logdir, source_name)
            target = os.path.join(logdir, target_name)
            if not os.path.isfile(source):
                raise RuntimeError(
                    f"cannot snapshot validation best at {milestone}: missing {source}"
                )
            temporary = target + ".tmp"
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
            snapshot_ema.append(target_name)

        with open(source_report, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        report["snapshot_milestone"] = int(milestone)
        report["artifacts"] = {"ema": snapshot_ema}
        target_report = os.path.join(
            logdir, f"best_report_upto{int(milestone):06d}.json"
        )
        temporary_report = target_report + ".tmp"
        with open(temporary_report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_report, target_report)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    basename = bf.basename(main_checkpoint)
    if basename.startswith("model_best") or basename.startswith("checkpoint_best"):
        suffix = "best"
        if basename.startswith("model_"):
            suffix = basename[len("model_") : -3]
        elif basename.startswith("checkpoint_"):
            suffix = basename[len("checkpoint_") : -3]
        best_path = bf.join(bf.dirname(main_checkpoint), f"ema_{rate}_{suffix}.pt")
        if bf.exists(best_path):
            return best_path
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def build_fixed_uniform_timesteps(num_samples, num_timesteps, seed):
    generator = th.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return th.randint(low=0, high=int(num_timesteps), size=(int(num_samples),), generator=generator, dtype=th.long)


def accumulate_loss_stats(stats, diffusion, ts, losses):
    for key, values in losses.items():
        values_cpu = values.detach().float().cpu().reshape(-1)
        stats[key][0] += float(values_cpu.sum().item())
        stats[key][1] += int(values_cpu.numel())
        for sub_t, sub_loss in zip(ts.detach().cpu().numpy(), values_cpu.numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            q_key = f"{key}_q{quartile}"
            stats[q_key][0] += float(sub_loss)
            stats[q_key][1] += 1


def finalize_loss_stats(stats, prefix=None):
    metrics = {}
    for key, (value_sum, count) in stats.items():
        if count <= 0:
            continue
        tag = f"{prefix}/{key}" if prefix else key
        metrics[tag] = float(value_sum) / float(count)
    return metrics


def log_loss_dict(diffusion, ts, losses, prefix=None):
    stats = defaultdict(lambda: [0.0, 0])
    accumulate_loss_stats(stats, diffusion, ts, losses)
    for key, value in finalize_loss_stats(stats, prefix=prefix).items():
        logger.logkv_mean(key, value)
