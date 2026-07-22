"""Franka-specific async client hook for publishing complete ROS2 action chunks."""

import logging
import threading
from dataclasses import asdict
from pprint import pformat

import draccus

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import TimedAction, visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.utils.import_utils import register_third_party_plugins

from .config_franka_ros import FrankaRosConfig
from .franka_ros import FrankaRos


class FrankaRos2RobotClient(RobotClient):
    """Stock async client plus one project-local complete-chunk publication hook.

    The base client still owns gRPC, observation streaming, queue aggregation,
    timing, and bookkeeping.  This subclass only exposes each accepted server
    chunk to the non-actuating ROS2 interface before the base control loop
    consumes its waypoints one at a time.
    """

    def __init__(self, config: RobotClientConfig):
        if not isinstance(config.robot, FrankaRosConfig):
            raise TypeError("FrankaRos2RobotClient requires robot.type=franka_ros")
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
        # The stock check runs first: it owns the pending-observation
        # bookkeeping and the queue-threshold decision on the nominal clock.
        if not super()._ready_to_send_observation():
            return False
        # Gate the next observation on the gateway's real execution progress.
        # When retiming slows execution below real time, the nominal clock
        # otherwise requests a replacement chunk mid-plan and every
        # replacement starts with a catch-up jump.
        return self.robot.plan_execution_complete()

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


@draccus.wrap()
def ros2_async_client(cfg: RobotClientConfig) -> None:
    """Run the standard async loops with the Franka chunk hook enabled."""

    logging.info(pformat(asdict(cfg)))
    client = FrankaRos2RobotClient(cfg)
    if not client.start():
        client.stop()
        return

    client.logger.info("Starting action receiver thread with ROS2 chunk publication")
    action_receiver_thread = threading.Thread(
        target=client.receive_actions,
        name="franka-ros2-action-receiver",
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
        client.logger.info("Franka ROS2 interface client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    ros2_async_client()
