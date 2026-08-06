"""Franka joint-space async client hook for publishing complete ROS2 action chunks."""

import logging
import threading
from dataclasses import asdict
from pprint import pformat

import draccus

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import TimedAction, visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.utils.import_utils import register_third_party_plugins

from .franka_joint_ros import FrankaJointRos
from .joint_config_franka_ros import FrankaJointRosConfig


class FrankaJointRos2RobotClient(RobotClient):
    """Stock async client plus one project-local complete-chunk publication hook.

    The base client still owns gRPC, observation streaming, queue aggregation,
    timing, and bookkeeping.  This subclass only exposes each accepted server
    chunk to the non-actuating ROS2 interface before the base control loop
    consumes its waypoints one at a time.
    """

    def __init__(self, config: RobotClientConfig):
        if not isinstance(config.robot, FrankaJointRosConfig):
            raise TypeError("FrankaJointRos2RobotClient requires robot.type=franka_ros_joint")
        if config.robot.dry_run:
            raise ValueError("FrankaJointRos2RobotClient requires robot.dry_run=false")
        if not config.robot.ros2_interface_only:
            raise ValueError("FrankaJointRos2RobotClient only supports ros2_interface_only=true")
        if config.aggregate_fn_name != "latest_only":
            raise ValueError("ROS2 joint chunks require aggregate_fn_name=latest_only")
        super().__init__(config)
        if not isinstance(self.robot, FrankaJointRos):
            raise TypeError("franka_ros_joint plugin did not construct a FrankaJointRos instance")

    def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
        if not isinstance(incoming_actions, list):
            raise TypeError("Incoming action chunk must be a list")
        if incoming_actions and not all(isinstance(action, TimedAction) for action in incoming_actions):
            raise TypeError("Incoming action chunk must contain TimedAction values")

        with self.latest_action_lock:
            latest_action = self.latest_action
        accepted_actions = [
            action for action in incoming_actions if action.get_timestep() > latest_action
        ]
        # Every action in a server chunk is timed from the observation used for
        # inference. Preserve that original provenance even when an already
        # executed prefix is removed before publishing the fresh suffix.
        source_timestep = incoming_actions[0].get_timestep() if incoming_actions else None
        source_timestamp = incoming_actions[0].get_timestamp() if incoming_actions else None

        # Preserve the stock queue semantics first. With latest_only, the
        # accepted incoming values are exactly the replacements ROS2 should see.
        super()._aggregate_action_queues(incoming_actions, aggregate_fn)

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

    def _ready_to_send_observation(self):
        """Wait for physical ROS execution, not merely local queue depletion."""
        if not super()._ready_to_send_observation():
            return False
        return self.robot._backend.ready_for_next_observation()


@draccus.wrap()
def ros2_joint_async_client(cfg: RobotClientConfig) -> None:
    """Run the standard async loops with the Franka joint-space chunk hook enabled."""

    logging.info(pformat(asdict(cfg)))
    client = FrankaJointRos2RobotClient(cfg)
    if not client.start():
        client.stop()
        return

    client.logger.info("Starting action receiver thread with ROS2 joint chunk publication")
    action_receiver_thread = threading.Thread(
        target=client.receive_actions,
        name="franka-ros2-joint-action-receiver",
        daemon=True,
    )
    action_receiver_thread.start()
    try:
        client.control_loop(task=cfg.task)
    finally:
        client.stop()
        action_receiver_thread.join()
        if cfg.debug_visualize_queue_size:
            visualize_action_queue_size(client.action_queue_size)
        client.logger.info("Franka joint-space ROS2 interface client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    ros2_joint_async_client()
