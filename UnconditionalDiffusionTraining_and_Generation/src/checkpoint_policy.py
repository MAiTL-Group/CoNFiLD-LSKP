from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class CheckpointPolicy:
    """Optional sparse checkpoint policy for long uninterrupted training runs."""

    enabled: bool = False
    ema_milestones: Tuple[int, ...] = ()
    skip_step_zero: bool = False
    lightweight_best: bool = False
    terminal_full: bool = False
    best_snapshot_milestones: Tuple[int, ...] = ()

    @classmethod
    def from_config(cls, config: Optional[Mapping]):
        if not config:
            return cls()
        if not isinstance(config, Mapping):
            raise TypeError("checkpoint_policy must be a mapping")

        enabled = bool(config.get("enabled", False))
        milestones = tuple(
            sorted({int(step) for step in config.get("ema_milestones", [])})
        )
        if any(step < 0 for step in milestones):
            raise ValueError("checkpoint_policy.ema_milestones must be non-negative")
        best_snapshots = tuple(
            sorted({int(step) for step in config.get("best_snapshot_milestones", [])})
        )
        if any(step < 0 for step in best_snapshots):
            raise ValueError(
                "checkpoint_policy.best_snapshot_milestones must be non-negative"
            )

        return cls(
            enabled=enabled,
            ema_milestones=milestones if enabled else (),
            skip_step_zero=enabled and bool(config.get("skip_step_zero", False)),
            lightweight_best=enabled and bool(config.get("lightweight_best", False)),
            terminal_full=enabled and bool(config.get("terminal_full", False)),
            best_snapshot_milestones=best_snapshots if enabled else (),
        )

    def should_save_ema(self, step: int) -> bool:
        step = int(step)
        if not self.enabled:
            return False
        if step == 0 and self.skip_step_zero:
            return False
        return step in self.ema_milestones

    def should_snapshot_best(self, step: int) -> bool:
        return self.enabled and int(step) in self.best_snapshot_milestones
