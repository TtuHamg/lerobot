"""LeRobot Robot implementation for the Franka joint-space (FastWAM) integration."""

from __future__ import annotations

from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .contract import CAMERA_NAMES, CAMERA_SHAPE
from .joint_config_franka_ros import FrankaJointRosConfig
from .joint_contract import JOINT_ACTION_NAMES, JOINT_STATE_NAMES, validate_joint_action
from .joint_dry_run import JointDryRunBackend
from .joint_ros2_backend import JointRos2Backend


class FrankaJointRos(Robot):
    """Franka joint-space adapter with dry-run and isolated, non-actuating ROS2 backends."""

    config_class = FrankaJointRosConfig
    name = "franka_ros_joint"

    def __init__(self, config: FrankaJointRosConfig):
        super().__init__(config)
        self.config = config
        if config.dry_run:
            self._backend: JointDryRunBackend | JointRos2Backend = JointDryRunBackend(
                fixture_path=config.fixture_path,
                action_log_path=config.action_log_path,
                robot_id=config.id,
            )
        else:
            self._backend = JointRos2Backend(config=config)

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        state = dict.fromkeys(JOINT_STATE_NAMES, float)
        cameras = dict.fromkeys(CAMERA_NAMES, CAMERA_SHAPE)
        return {**state, **cameras}

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(JOINT_ACTION_NAMES, float)

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
        ordered_action = validate_joint_action(action)
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
        """Publish one complete absolute joint-space plan through the isolated ROS2 topic."""

        if not isinstance(self._backend, JointRos2Backend):
            raise RuntimeError("Action chunk publication is only available in ROS2 interface mode")
        return self._backend.publish_action_chunk(
            timed_actions,
            source_observation_timestep=source_observation_timestep,
            source_observation_timestamp=source_observation_timestamp,
            period_s=period_s,
        )

    @check_if_not_connected
    def get_qpos(self):
        """Return the seven-joint ROS sideband."""

        if not isinstance(self._backend, JointRos2Backend):
            raise RuntimeError("qpos sideband is only available in ROS2 interface mode")
        return self._backend.get_qpos()

    @check_if_not_connected
    def get_ros_sideband(self):
        """Return copied ROS timing, qpos, and gripper diagnostics."""

        if not isinstance(self._backend, JointRos2Backend):
            raise RuntimeError("ROS sideband is only available in ROS2 interface mode")
        return self._backend.get_sideband()

    @check_if_not_connected
    def disconnect(self) -> None:
        self._backend.disconnect()
