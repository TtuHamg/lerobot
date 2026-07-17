"""Configuration for the out-of-tree Franka Robot plugin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("franka_ros")
@dataclass(kw_only=True)
class FrankaRosConfig(RobotConfig):
    """Phase 1 configuration.

    Only the non-actuating fixture-to-JSONL path exists. A false ``dry_run``
    therefore fails during configuration instead of selecting an unsafe or
    incomplete backend.
    """

    dry_run: bool = True
    fixture_path: Path | None = None
    action_log_path: Path | None = None
    quaternion_norm_tolerance: float = 1e-3

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.dry_run:
            raise ValueError("Franka ROS execution is not implemented; Phase 1 requires dry_run=true")
        if self.quaternion_norm_tolerance <= 0.0 or self.quaternion_norm_tolerance > 0.1:
            raise ValueError("quaternion_norm_tolerance must be in (0, 0.1]")
