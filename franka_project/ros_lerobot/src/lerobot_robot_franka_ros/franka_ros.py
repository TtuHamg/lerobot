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
from .ros2_backend import Ros2Backend


class FrankaRos(Robot):
    """Franka adapter with dry-run and isolated, non-actuating ROS2 backends."""

    config_class = FrankaRosConfig
    name = "franka_ros"

    def __init__(self, config: FrankaRosConfig):
        super().__init__(config)
        self.config = config
        if config.dry_run:
            self._backend: DryRunBackend | Ros2Backend = DryRunBackend(
                fixture_path=config.fixture_path,
                action_log_path=config.action_log_path,
                robot_id=config.id,
            )
        else:
            self._backend = Ros2Backend(config=config)

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
        """No calibration is performed by either non-actuating backend."""

    def configure(self) -> None:
        """No controller or hardware configuration is performed here."""

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
    def publish_action_chunk(
        self,
        timed_actions,
        *,
        source_observation_timestep: int,
        source_observation_timestamp: float,
        period_s: float,
    ):
        """Publish one complete absolute plan through the isolated ROS2 topic."""

        if not isinstance(self._backend, Ros2Backend):
            raise RuntimeError("Action chunk publication is only available in ROS2 interface mode")
        return self._backend.publish_action_chunk(
            timed_actions,
            source_observation_timestep=source_observation_timestep,
            source_observation_timestamp=source_observation_timestamp,
            period_s=period_s,
        )

    @check_if_not_connected
    def plan_execution_complete(self) -> bool:
        """True when the gateway finished (or abandoned) the newest published plan."""

        if not isinstance(self._backend, Ros2Backend):
            raise RuntimeError("Plan execution state is only available in ROS2 interface mode")
        return self._backend.plan_execution_complete()

    @check_if_not_connected
    def set_gateway_armed(self, armed: bool, *, timeout_s: float) -> tuple[bool, str]:
        """Arm/disarm the external safety gateway through its existing ROS service."""

        if not isinstance(self._backend, Ros2Backend):
            raise RuntimeError("Gateway arming is only available in ROS2 interface mode")
        return self._backend.set_gateway_armed(armed, timeout_s=timeout_s)

    @check_if_not_connected
    def get_qpos(self):
        """Return the seven-joint sideband without changing canonical Franka state10."""

        if not isinstance(self._backend, Ros2Backend):
            raise RuntimeError("qpos sideband is only available in ROS2 interface mode")
        return self._backend.get_qpos()

    @check_if_not_connected
    def get_ros_sideband(self):
        """Return copied ROS timing, EEF, qpos, and gripper diagnostics."""

        if not isinstance(self._backend, Ros2Backend):
            raise RuntimeError("ROS sideband is only available in ROS2 interface mode")
        return self._backend.get_sideband()

    @check_if_not_connected
    def disconnect(self) -> None:
        self._backend.disconnect()
