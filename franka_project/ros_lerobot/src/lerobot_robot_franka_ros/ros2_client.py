"""Franka-specific async client hook for publishing complete ROS2 action chunks."""

import logging
import math
import threading
from dataclasses import asdict, dataclass, field, replace
from pprint import pformat

import draccus

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import TimedAction, visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.utils.import_utils import register_third_party_plugins

from .config_franka_ros import FrankaRosConfig
from .contract import FASTWAM_RENAME_MAP, FRANKA_POLICY_TYPES, PI0_RENAME_MAP
from .franka_ros import FrankaRos
from .visualization_launcher import ActionVisualizationLauncher, ActionVisualizationLaunchSpec


_FASTWAM_SENSOR_CONTRACT = {
    "camera1_topic": "/camera1/camera1/color/image_raw",
    "camera2_topic": "/camera2/camera2/color/image_raw",
    "eef_pose_topic": "/franka_robot_state_broadcaster/current_pose",
    "gripper_topic": "/gripper/joint_states",
    "gripper_joint_name": "robotiq_85_left_knuckle_joint",
}


@dataclass
class FrankaRos2ClientConfig(RobotClientConfig):
    """Client-only options for the chunk-aware Franka ROS2 entry point."""

    visualize_action: bool = field(
        default=False,
        metadata={"help": "Launch the read-only Franka action visualizer with this client."},
    )
    visualization_launch_rviz: bool = field(
        default=True,
        metadata={
            "help": "Also launch RViz when visualize_action=true; false runs only the marker node."
        },
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.visualize_action, bool):
            raise ValueError("visualize_action must be a bool")
        if not isinstance(self.visualization_launch_rviz, bool):
            raise ValueError("visualization_launch_rviz must be a bool")


def _make_action_visualization_launcher(
    config: FrankaRos2ClientConfig,
    *,
    logger: logging.Logger,
) -> ActionVisualizationLauncher:
    robot = config.robot
    if not isinstance(robot, FrankaRosConfig):
        raise TypeError("Action visualization requires robot.type=franka_ros")
    return ActionVisualizationLauncher(
        ActionVisualizationLaunchSpec(
            launch_rviz=config.visualization_launch_rviz,
            fixed_frame=robot.base_frame,
            action_chunk_topic=robot.action_chunk_topic,
            current_pose_topic=robot.eef_pose_topic,
            camera1_topic=robot.camera1_topic,
            camera2_topic=robot.camera2_topic,
        ),
        logger=logger,
    )


def resolve_franka_client_policy_config(config: RobotClientConfig) -> RobotClientConfig:
    """Resolve the policy-specific camera wire profile without changing robot semantics."""

    if not isinstance(config.robot, FrankaRosConfig):
        raise TypeError("FrankaRos2RobotClient requires robot.type=franka_ros")
    if config.policy_type not in FRANKA_POLICY_TYPES:
        raise ValueError(
            f"Franka ROS2 client requires policy_type in {FRANKA_POLICY_TYPES}, "
            f"got {config.policy_type!r}"
        )
    if config.action_offset != 1:
        raise ValueError(
            "Franka ROS2 chunk delivery requires action_offset=1 so the first predicted "
            "waypoint is not discarded as stale"
        )
    if config.robot.base_frame != "base":
        raise ValueError(
            "Franka PI0/FastWAM checkpoints use the frozen Cartesian frame 'base'; "
            f"got robot.base_frame={config.robot.base_frame!r}"
        )
    expected_map = PI0_RENAME_MAP if config.policy_type == "pi0" else FASTWAM_RENAME_MAP
    actual_map = config.rename_map
    # Empty is the natural CLI default.  For PI0, treat it as an omitted wire
    # profile and fill the frozen mapping; FastWAM intentionally keeps it empty.
    if config.policy_type == "pi0" and actual_map == {}:
        actual_map = expected_map
    if actual_map != expected_map:
        raise ValueError(
            f"Franka {config.policy_type} client rename_map mismatch: "
            f"expected={expected_map}, actual={config.rename_map}"
        )
    if config.policy_type == "fastwam":
        robot = config.robot
        for field_name, expected_value in _FASTWAM_SENSOR_CONTRACT.items():
            actual_value = getattr(robot, field_name)
            if actual_value != expected_value:
                raise ValueError(
                    f"The selected FastWAM checkpoint requires robot.{field_name}="
                    f"{expected_value!r}, got {actual_value!r}"
                )
        calibration_matches = math.isclose(
            float(robot.gripper_open_position), 0.0, rel_tol=0.0, abs_tol=1e-12
        ) and math.isclose(
            float(robot.gripper_closed_position), 0.8, rel_tol=0.0, abs_tol=1e-12
        )
        if not calibration_matches:
            raise ValueError(
                "The selected FastWAM checkpoint requires gripper_open_position=0.0 "
                "and gripper_closed_position=0.8; confirm the live joint units before deployment"
            )
        if robot.gripper_max_skew_s > 0.01:
            raise ValueError(
                "The selected FastWAM checkpoint requires gripper_max_skew_s <= 0.01"
            )
        if robot.camera2_max_skew_s > 0.1:
            raise ValueError(
                "The selected FastWAM checkpoint requires camera2_max_skew_s <= 0.1"
            )
        if robot.eef_max_skew_s > 0.05:
            raise ValueError(
                "The selected FastWAM checkpoint requires eef_max_skew_s <= 0.05"
            )
    return replace(config, rename_map=dict(expected_map))


class FrankaRos2RobotClient(RobotClient):
    """Stock async client plus one project-local complete-chunk publication hook.

    The base client still owns gRPC, observation streaming, queue aggregation,
    timing, and bookkeeping.  This subclass only exposes each accepted server
    chunk to the non-actuating ROS2 interface before the base control loop
    consumes its waypoints one at a time.
    """

    def __init__(self, config: RobotClientConfig):
        config = resolve_franka_client_policy_config(config)
        if config.robot.dry_run:
            raise ValueError("FrankaRos2RobotClient requires robot.dry_run=false")
        if not config.robot.ros2_interface_only:
            raise ValueError("FrankaRos2RobotClient only supports ros2_interface_only=true")
        if config.aggregate_fn_name != "latest_only":
            raise ValueError("ROS2 Cartesian chunks require aggregate_fn_name=latest_only")
        super().__init__(config)
        if not isinstance(self.robot, FrankaRos):
            raise TypeError("franka_ros plugin did not construct a FrankaRos instance")

    def _ready_to_send_observation(self):
        if self.config.observation_trigger_mode == "post_action_delay":
            # The base class retains the pending-observation protection and
            # checks only the post-publication monotonic deadline in this mode.
            # Deliberately do not consult the queue or ROS execution status.
            return super()._ready_to_send_observation()

        # The stock check runs first: it owns the pending-observation
        # bookkeeping and the queue-threshold decision on the nominal clock.
        if not super()._ready_to_send_observation():
            return False
        # Gate the next observation on the gateway's real execution progress.
        # When retiming slows execution below real time, the nominal clock
        # otherwise requests a replacement chunk mid-plan and every
        # replacement starts with a catch-up jump.
        return self.robot.plan_execution_complete()

    def _effective_action_chunk_size(self, incoming_actions: list[TimedAction]) -> int:
        received_size = super()._effective_action_chunk_size(incoming_actions)
        limit = self.config.robot.max_action_chunk_waypoints
        return received_size if limit is None else min(received_size, limit)

    def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None) -> int:
        if not isinstance(incoming_actions, list):
            raise TypeError("Incoming action chunk must be a list")
        if incoming_actions and not all(isinstance(action, TimedAction) for action in incoming_actions):
            raise TypeError("Incoming action chunk must contain TimedAction values")

        with self.latest_action_lock:
            latest_action = self.latest_action
        fresh_actions = [
            action for action in incoming_actions if action.get_timestep() > latest_action
        ]
        limit = self.config.robot.max_action_chunk_waypoints
        accepted_actions = fresh_actions if limit is None else fresh_actions[:limit]
        # Every action in a server chunk is timed from the observation used for
        # inference. Preserve that original provenance even when an already
        # executed prefix is removed before publishing the fresh suffix.
        source_timestep = incoming_actions[0].get_timestep() if incoming_actions else None
        source_timestamp = incoming_actions[0].get_timestamp() if incoming_actions else None

        # The local cursor must never consume a waypoint that was intentionally
        # withheld from ROS. Commit the exact same bounded fresh prefix to both
        # the stock local queue and the complete-chunk publisher.
        super()._aggregate_action_queues(accepted_actions, aggregate_fn)

        if len(accepted_actions) < len(fresh_actions):
            self.logger.info(
                "Limited fresh action chunk from %d to %d waypoints before local queue and ROS publication",
                len(fresh_actions),
                len(accepted_actions),
            )

        if accepted_actions:
            try:
                self.robot.publish_action_chunk(
                    accepted_actions,
                    source_observation_timestep=source_timestep,
                    source_observation_timestamp=source_timestamp,
                    period_s=self.config.environment_dt,
                )
            except Exception:
                # No per-waypoint fallback is allowed: it would silently change
                # chunk semantics and could create a second command path.
                self.shutdown_event.set()
                self.logger.exception("ROS2 action chunk publication failed; stopping client")
                raise
        return len(accepted_actions)


@draccus.wrap()
def ros2_async_client(cfg: FrankaRos2ClientConfig) -> None:
    """Run the standard async loops with the Franka chunk hook enabled."""

    cfg = resolve_franka_client_policy_config(cfg)
    logging.info(pformat(asdict(cfg)))
    client = FrankaRos2RobotClient(cfg)
    if not client.start():
        client.stop()
        return

    visualization_launcher: ActionVisualizationLauncher | None = None
    action_receiver_thread: threading.Thread | None = None
    try:
        if cfg.visualize_action:
            visualization_launcher = _make_action_visualization_launcher(
                cfg,
                logger=client.logger,
            )
            visualization_launcher.start()

        client.logger.info("Starting action receiver thread with ROS2 chunk publication")
        action_receiver_thread = threading.Thread(
            target=client.receive_actions,
            name="franka-ros2-action-receiver",
            daemon=True,
        )
        action_receiver_thread.start()
        client.control_loop(task=cfg.task)
    finally:
        try:
            client.stop()
        finally:
            try:
                if visualization_launcher is not None:
                    try:
                        visualization_launcher.stop()
                    except Exception:
                        client.logger.exception("Failed to fully stop action visualization")
            finally:
                if action_receiver_thread is not None:
                    action_receiver_thread.join()
        if cfg.debug_visualize_queue_size:
            visualize_action_queue_size(client.action_queue_size)
        client.logger.info("Franka ROS2 interface client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    ros2_async_client()
