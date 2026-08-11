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
from .web_dashboard_launcher import WebDashboardLauncher, WebDashboardLaunchSpec

_FASTWAM_SENSOR_CONTRACT = {
    "camera1_topic": "/camera1/camera1/color/image_raw",
    "camera2_topic": "/camera2/camera2/color/image_raw",
    "eef_pose_topic": "/franka_robot_state_broadcaster/current_pose",
    "gripper_topic": "/gripper/joint_states",
    "gripper_joint_name": "robotiq_85_left_knuckle_joint",
}


class _InteractiveTaskController:
    """Thread-safe selected-task and generation state for the keyboard loop."""

    def __init__(self, allowed_tasks: tuple[str, ...]):
        if not allowed_tasks:
            raise ValueError("Interactive task control requires a non-empty server task allowlist")
        self.allowed_tasks = allowed_tasks
        self._lock = threading.Lock()
        self._task: str | None = None
        self._generation = 0
        self._armed = False

    def select(self, task: str) -> int:
        if task not in self.allowed_tasks:
            raise ValueError(f"Task is not server-advertised: {task!r}")
        with self._lock:
            self._generation += 1
            self._task = task
            self._armed = False
            return self._generation

    def stop(self) -> int:
        with self._lock:
            self._generation += 1
            self._task = None
            self._armed = False
            return self._generation

    def arm(self) -> tuple[str, int]:
        with self._lock:
            if self._task is None:
                raise RuntimeError("Select a task before arming the gateway")
            self._armed = True
            return self._task, self._generation

    def snapshot(self) -> tuple[str | None, int, bool]:
        with self._lock:
            return self._task, self._generation, self._armed


@dataclass
class FrankaRos2ClientConfig(RobotClientConfig):
    """Client-only options for the chunk-aware Franka ROS2 entry point."""

    visualize_action: bool = field(
        default=False,
        metadata={"help": "Launch the read-only Franka action visualizer with this client."},
    )
    visualization_launch_rviz: bool = field(
        default=True,
        metadata={"help": "Also launch RViz when visualize_action=true; false runs only the marker node."},
    )
    visualize_action_web: bool = field(
        default=False,
        metadata={"help": "Launch the low-load local Franka web dashboard with this client."},
    )
    web_dashboard_port: int = field(
        default=8768,
        metadata={"help": "Localhost port used by the optional Franka web dashboard."},
    )
    web_dashboard_history_seconds: float = field(
        default=60.0,
        metadata={"help": "Maximum rolling history retained by the web dashboard."},
    )
    web_dashboard_camera_fps: float = field(
        default=3.0,
        metadata={"help": "Thumbnail FPS when a web-dashboard camera is explicitly enabled."},
    )
    web_dashboard_open_browser: bool = field(
        default=True,
        metadata={"help": "Open the web dashboard in the local desktop browser after startup."},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.visualize_action, bool):
            raise ValueError("visualize_action must be a bool")
        if not isinstance(self.visualization_launch_rviz, bool):
            raise ValueError("visualization_launch_rviz must be a bool")
        if not isinstance(self.visualize_action_web, bool):
            raise ValueError("visualize_action_web must be a bool")
        if self.visualize_action and self.visualize_action_web:
            raise ValueError("visualize_action and visualize_action_web are mutually exclusive")
        if not isinstance(self.web_dashboard_port, int) or not 1 <= self.web_dashboard_port <= 65535:
            raise ValueError("web_dashboard_port must be an integer in [1, 65535]")
        if not 1.0 <= float(self.web_dashboard_history_seconds) <= 600.0:
            raise ValueError("web_dashboard_history_seconds must be in [1, 600]")
        if not 0.1 <= float(self.web_dashboard_camera_fps) <= 10.0:
            raise ValueError("web_dashboard_camera_fps must be in [0.1, 10]")
        if not isinstance(self.web_dashboard_open_browser, bool):
            raise ValueError("web_dashboard_open_browser must be a bool")


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


def _make_web_dashboard_launcher(
    config: FrankaRos2ClientConfig,
    *,
    logger: logging.Logger,
) -> WebDashboardLauncher:
    robot = config.robot
    if not isinstance(robot, FrankaRosConfig):
        raise TypeError("Web dashboard requires robot.type=franka_ros")
    return WebDashboardLauncher(
        WebDashboardLaunchSpec(
            host="127.0.0.1",
            port=config.web_dashboard_port,
            history_seconds=config.web_dashboard_history_seconds,
            camera_fps=config.web_dashboard_camera_fps,
            open_browser=config.web_dashboard_open_browser,
            default_pose_frame=robot.policy_eef_frame,
            action_chunk_topic=robot.action_chunk_topic,
            ack_topic="/lerobot/franka/action_chunk_ack",
            ik_topic="/lerobot/franka/ik_joint_action_chunk",
            status_topic="/lerobot/franka/safety_gateway_status",
            current_pose_topic=robot.eef_pose_topic,
            robot_state_topic=robot.robot_state_topic,
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
            f"Franka ROS2 client requires policy_type in {FRANKA_POLICY_TYPES}, got {config.policy_type!r}"
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
        if config.gateway_arm_timeout_s <= 0.0 or not math.isfinite(config.gateway_arm_timeout_s):
            raise ValueError("gateway_arm_timeout_s must be finite and positive")
        robot = config.robot
        for field_name, expected_value in _FASTWAM_SENSOR_CONTRACT.items():
            actual_value = getattr(robot, field_name)
            if actual_value != expected_value:
                raise ValueError(
                    f"The selected FastWAM checkpoint requires robot.{field_name}="
                    f"{expected_value!r}, got {actual_value!r}"
                )
        if robot.gripper_max_skew_s > 0.01:
            raise ValueError("The selected FastWAM checkpoint requires gripper_max_skew_s <= 0.01")
        if robot.camera2_max_skew_s > 0.1:
            raise ValueError("The selected FastWAM checkpoint requires camera2_max_skew_s <= 0.1")
        if robot.eef_max_skew_s > 0.05:
            raise ValueError("The selected FastWAM checkpoint requires eef_max_skew_s <= 0.05")
        if (
            robot.max_action_chunk_waypoints is not None
            and config.actions_per_chunk is not None
            and robot.max_action_chunk_waypoints < config.actions_per_chunk
        ):
            raise ValueError(
                "FastWAM deployment must preserve the complete action horizon: "
                f"robot.max_action_chunk_waypoints={robot.max_action_chunk_waypoints}, "
                f"actions_per_chunk={config.actions_per_chunk}"
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
        self._interactive_tasks: _InteractiveTaskController | None = None

    def start(self):
        if self.config.interactive_task_control:
            try:
                self.robot.set_gateway_armed(
                    False,
                    timeout_s=self.config.gateway_arm_timeout_s,
                )
            except Exception:
                self.logger.exception("Could not place the gateway in HOLD before policy setup")
                return False
        started = super().start()
        if not started or not self.config.interactive_task_control:
            return started
        if self.config.policy_type != "fastwam":
            self.logger.error("Interactive task control is supported only for policy_type='fastwam'")
            return False
        try:
            self._interactive_tasks = _InteractiveTaskController(self.allowed_tasks)
            self.logger.info("Interactive client started in HOLD with %d tasks", len(self.allowed_tasks))
            return True
        except Exception:
            self.logger.exception("Could not initialize interactive task control")
            return False

    def stop(self):
        controller = self._interactive_tasks
        if controller is not None and self.robot.is_connected:
            controller.stop()
            self.discard_pending_work()
            try:
                self.robot.set_gateway_armed(
                    False,
                    timeout_s=self.config.gateway_arm_timeout_s,
                )
            except Exception:
                self.logger.exception("Final gateway disarm failed while stopping the client")
        super().stop()

    def _task_snapshot(self, default_task: str) -> tuple[str, int]:
        controller = self._interactive_tasks
        if controller is None:
            return super()._task_snapshot(default_task)
        task, generation, armed = controller.snapshot()
        if task is None or not armed:
            raise RuntimeError("Interactive task control is in HOLD")
        return task, generation

    def _accept_action_delivery(self, task: str, task_generation: int) -> bool:
        controller = self._interactive_tasks
        if controller is None:
            return True
        current_task, current_generation, armed = controller.snapshot()
        return armed and task == current_task and task_generation == current_generation

    def _ready_to_send_observation(self):
        controller = self._interactive_tasks
        if controller is not None:
            task, _, armed = controller.snapshot()
            if task is None or not armed:
                return False
        if self.config.observation_trigger_mode == "post_action_delay":
            # The base class retains the pending-observation protection and
            # checks only the post-publication monotonic deadline in this mode.
            # Deliberately do not consult the queue or ROS execution status.
            return super()._ready_to_send_observation()

        # The stock check runs first: it owns the pending-observation
        # bookkeeping and the queue-threshold decision on the nominal clock.
        if not super()._ready_to_send_observation():
            return False
        return self.robot.plan_execution_complete()

    def select_task(self, task: str) -> int:
        controller = self._interactive_tasks
        if controller is None:
            raise RuntimeError("Interactive task control is not enabled")
        generation = controller.select(task)
        self.discard_pending_work()
        self.robot.set_gateway_armed(False, timeout_s=self.config.gateway_arm_timeout_s)
        self.logger.info("Selected task %r at generation %d; gateway remains in HOLD", task, generation)
        return generation

    def stop_task(self) -> int:
        controller = self._interactive_tasks
        if controller is None:
            raise RuntimeError("Interactive task control is not enabled")
        generation = controller.stop()
        self.discard_pending_work()
        self.robot.set_gateway_armed(False, timeout_s=self.config.gateway_arm_timeout_s)
        self.logger.info("Stopped task at generation %d; gateway is in HOLD", generation)
        return generation

    def arm_selected_task(self) -> tuple[str, int]:
        controller = self._interactive_tasks
        if controller is None:
            raise RuntimeError("Interactive task control is not enabled")
        task, generation, _ = controller.snapshot()
        if task is None:
            raise RuntimeError("Select a task before arming the gateway")
        self.discard_pending_work()
        self.robot.set_gateway_armed(False, timeout_s=self.config.gateway_arm_timeout_s)
        self.robot.set_gateway_armed(True, timeout_s=self.config.gateway_arm_timeout_s)
        result = controller.arm()
        self.must_go.set()
        self.logger.info("Armed task %r at generation %d", *result)
        return result

    def keyboard_task_loop(self) -> None:
        controller = self._interactive_tasks
        if controller is None:
            return
        tasks = controller.allowed_tasks
        self.logger.info(
            "Task keys: %s | s=stop a=arm h=help q=quit",
            " ".join(f"{index + 1}={task!r}" for index, task in enumerate(tasks)),
        )
        while self.running:
            try:
                command = input("task> ").strip().lower()
                if command.isdigit() and 1 <= int(command) <= len(tasks):
                    self.select_task(tasks[int(command) - 1])
                elif command == "s":
                    self.stop_task()
                elif command == "a":
                    self.arm_selected_task()
                elif command == "h":
                    self.logger.info(
                        "Task keys: %s | s=stop a=arm h=help q=quit",
                        " ".join(f"{index + 1}={task!r}" for index, task in enumerate(tasks)),
                    )
                elif command == "q":
                    self.stop_task()
                    self.shutdown_event.set()
                    return
                elif command:
                    self.logger.warning("Unknown task command: %r", command)
            except EOFError:
                self.stop_task()
                self.shutdown_event.set()
                return
            except Exception:
                self.logger.exception("Interactive task command failed")

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
        fresh_actions = [action for action in incoming_actions if action.get_timestep() > latest_action]
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
    web_dashboard_launcher: WebDashboardLauncher | None = None
    action_receiver_thread: threading.Thread | None = None
    keyboard_thread: threading.Thread | None = None
    try:
        if cfg.visualize_action:
            visualization_launcher = _make_action_visualization_launcher(
                cfg,
                logger=client.logger,
            )
            visualization_launcher.start()
        if cfg.visualize_action_web:
            web_dashboard_launcher = _make_web_dashboard_launcher(
                cfg,
                logger=client.logger,
            )
            web_dashboard_launcher.start()

        client.logger.info("Starting action receiver thread with ROS2 chunk publication")
        action_receiver_thread = threading.Thread(
            target=client.receive_actions,
            name="franka-ros2-action-receiver",
            daemon=True,
        )
        action_receiver_thread.start()

        if cfg.interactive_task_control:
            keyboard_thread = threading.Thread(
                target=client.keyboard_task_loop,
                name="franka-task-keyboard",
                daemon=True,
            )
            keyboard_thread.start()
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
                if web_dashboard_launcher is not None:
                    try:
                        web_dashboard_launcher.stop()
                    except Exception:
                        client.logger.exception("Failed to fully stop web dashboard")
            finally:
                if action_receiver_thread is not None:
                    action_receiver_thread.join()
                if keyboard_thread is not None:
                    keyboard_thread.join(timeout=1.0)
        if cfg.debug_visualize_queue_size:
            visualize_action_queue_size(client.action_queue_size)
        client.logger.info("Franka ROS2 interface client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    ros2_async_client()
