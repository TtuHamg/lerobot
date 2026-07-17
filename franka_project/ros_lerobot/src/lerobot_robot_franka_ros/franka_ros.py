"""LeRobot Robot implementation for the staged Franka ROS integration."""

from __future__ import annotations

from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_franka_ros import FrankaRosConfig
from .contract import (
    ABSOLUTE_ACTION_NAMES,
    CAMERA_NAMES,
    CAMERA_SHAPE,
    STATE_NAMES,
    validate_absolute_action,
)
from .dry_run import DryRunBackend


class FrankaRos(Robot):
    """Phase 1 Franka adapter backed only by a fixture and JSONL sink."""

    config_class = FrankaRosConfig
    name = "franka_ros"

    def __init__(self, config: FrankaRosConfig):
        super().__init__(config)
        self.config = config
        self._backend = DryRunBackend(
            fixture_path=config.fixture_path,
            action_log_path=config.action_log_path,
            robot_id=config.id,
        )

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        state = dict.fromkeys(STATE_NAMES, float)
        cameras = dict.fromkeys(CAMERA_NAMES, CAMERA_SHAPE)
        return {**state, **cameras}

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(ABSOLUTE_ACTION_NAMES, float)

    @property
    def is_connected(self) -> bool:
        return self._backend.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """No calibration is performed by the non-actuating Phase 1 backend."""

    def configure(self) -> None:
        """No hardware configuration exists in Phase 1."""

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self._backend.connect()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        return self._backend.get_observation()

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        ordered_action = validate_absolute_action(
            action,
            quaternion_norm_tolerance=self.config.quaternion_norm_tolerance,
        )
        return self._backend.send_action(ordered_action)

    @check_if_not_connected
    def disconnect(self) -> None:
        self._backend.disconnect()
