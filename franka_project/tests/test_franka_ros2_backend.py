"""No-ROS tests for the Franka ROS2 backend and complete-chunk client hook."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from queue import Queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PLUGIN_SRC = Path(__file__).parents[1] / "ros_lerobot" / "src"
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))

from lerobot.async_inference.configs import RobotClientConfig  # noqa: E402
from lerobot.async_inference.helpers import TimedAction  # noqa: E402
from lerobot_robot_franka_ros.config_franka_ros import FrankaRosConfig  # noqa: E402
from lerobot_robot_franka_ros.contract import CAMERA_SHAPE, STATE_NAMES  # noqa: E402
from lerobot_robot_franka_ros.ros2_backend import Ros2Backend, Ros2BackendError  # noqa: E402
from lerobot_robot_franka_ros.ros2_client import (  # noqa: E402
    FrankaRos2RobotClient,
    resolve_franka_client_policy_config,
)
from lerobot_robot_franka_ros.ros2_contract import (  # noqa: E402
    ImageSample,
    JointStateSample,
    PoseSample,
)


def test_ros2_client_cli_help_is_parseable_without_ros() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(PLUGIN_SRC), environment.get("PYTHONPATH")) if value
    )

    result = subprocess.run(
        [sys.executable, "-m", "lerobot_robot_franka_ros.ros2_client", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert "--action_offset" in result.stdout
    assert "--robot.ros2_interface_only" in result.stdout
    assert "--robot.action_chunk_topic" in result.stdout
    assert "--robot.max_action_chunk_waypoints" in result.stdout


class _ManualClock:
    def __init__(self, nanoseconds: int) -> None:
        self.nanoseconds = nanoseconds

    def __call__(self) -> int:
        return self.nanoseconds


class _FakeRuntime:
    def __init__(self, config: FrankaRosConfig, cache) -> None:
        self.config = config
        self.cache = cache
        self.running = False
        self.closed = False
        self.published = []

    @property
    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        self.running = True

    def publish_action_chunk(self, chunk) -> None:
        if not self.running:
            raise RuntimeError("not running")
        self.published.append(chunk)

    def close(self, *, timeout_s: float | None = None) -> None:
        assert timeout_s is not None and timeout_s > 0.0
        self.running = False
        self.closed = True


def _ros2_config(tmp_path, **kwargs) -> FrankaRosConfig:
    return FrankaRosConfig(
        id="ros2-test",
        calibration_dir=tmp_path / "calibration",
        dry_run=False,
        **kwargs,
    )


@pytest.mark.parametrize("action_offset", [0, 1])
def test_action_offset_config_is_valid_and_serialized(tmp_path, action_offset: int) -> None:
    config = RobotClientConfig(robot=_ros2_config(tmp_path), action_offset=action_offset)

    assert config.action_offset == action_offset
    assert config.to_dict()["action_offset"] == action_offset


@pytest.mark.parametrize("action_offset", [-1, 2, True, 1.5])
def test_action_offset_config_rejects_unsupported_values(tmp_path, action_offset) -> None:
    with pytest.raises(ValueError, match="action_offset"):
        RobotClientConfig(robot=_ros2_config(tmp_path), action_offset=action_offset)


def test_pi0_client_profile_autofills_camera_rename_map(tmp_path) -> None:
    config = RobotClientConfig(
        robot=_ros2_config(tmp_path),
        policy_type="pi0",
        task="task-owned-by-checkpoint",
        fps=30,
        actions_per_chunk=7,
        action_offset=1,
    )

    resolved = resolve_franka_client_policy_config(config)

    assert config.rename_map == {}
    assert resolved.rename_map == {
        "observation.images.camera1": "observation.images.base_0_rgb",
        "observation.images.camera2": "observation.images.left_wrist_0_rgb",
    }
    assert resolved.task == config.task
    assert resolved.fps == config.fps
    assert resolved.actions_per_chunk == config.actions_per_chunk


def test_fastwam_client_profile_requires_raw_camera_keys(tmp_path) -> None:
    robot = FrankaRosConfig(
        id="fastwam-ros2-test",
        calibration_dir=tmp_path / "calibration",
        dry_run=False,
        gripper_open_position=0.0,
        gripper_closed_position=0.8,
        gripper_max_skew_s=0.01,
    )
    config = RobotClientConfig(robot=robot, policy_type="fastwam", action_offset=1)
    assert resolve_franka_client_policy_config(config).rename_map == {}

    config.rename_map = {
        "observation.images.camera1": "observation.images.base_0_rgb",
        "observation.images.camera2": "observation.images.left_wrist_0_rgb",
    }
    with pytest.raises(ValueError, match="fastwam.*rename_map"):
        resolve_franka_client_policy_config(config)


def test_fastwam_client_profile_rejects_wrong_gripper_contract(tmp_path) -> None:
    config = RobotClientConfig(
        robot=_ros2_config(tmp_path), policy_type="fastwam", action_offset=1
    )
    with pytest.raises(ValueError, match="gripper_closed_position=0.8"):
        resolve_franka_client_policy_config(config)


@pytest.mark.parametrize(
    ("override", "error_field"),
    [
        ({"camera1_topic": "/wrong/camera1"}, "camera1_topic"),
        ({"camera2_topic": "/wrong/camera2"}, "camera2_topic"),
        ({"eef_pose_topic": "/wrong/eef"}, "eef_pose_topic"),
        ({"gripper_topic": "/wrong/gripper"}, "gripper_topic"),
        ({"gripper_joint_name": "wrong_joint"}, "gripper_joint_name"),
        ({"camera2_max_skew_s": 0.1001}, "camera2_max_skew_s"),
        ({"eef_max_skew_s": 0.0501}, "eef_max_skew_s"),
    ],
)
def test_fastwam_client_profile_rejects_sensor_contract_drift(
    tmp_path, override: dict, error_field: str
) -> None:
    robot_kwargs = {
        "id": "fastwam-contract-test",
        "calibration_dir": tmp_path / "calibration",
        "dry_run": False,
        "gripper_open_position": 0.0,
        "gripper_closed_position": 0.8,
        "gripper_max_skew_s": 0.01,
    }
    robot_kwargs.update(override)
    config = RobotClientConfig(
        robot=FrankaRosConfig(**robot_kwargs),
        policy_type="fastwam",
        action_offset=1,
    )

    with pytest.raises(ValueError, match=error_field):
        resolve_franka_client_policy_config(config)


@pytest.mark.parametrize("policy_type", [None, "act"])
def test_franka_client_profile_rejects_unknown_policy(tmp_path, policy_type) -> None:
    config = RobotClientConfig(robot=_ros2_config(tmp_path), policy_type=policy_type)
    with pytest.raises(ValueError, match="pi0.*fastwam"):
        resolve_franka_client_policy_config(config)


def test_franka_client_profile_requires_next_timestep_offset(tmp_path) -> None:
    config = RobotClientConfig(robot=_ros2_config(tmp_path), policy_type="pi0")
    with pytest.raises(ValueError, match="action_offset=1"):
        resolve_franka_client_policy_config(config)


def test_franka_client_profile_requires_frozen_base_frame(tmp_path) -> None:
    robot = FrankaRosConfig(
        id="wrong-frame-test",
        calibration_dir=tmp_path / "calibration",
        dry_run=False,
        base_frame="world",
    )
    config = RobotClientConfig(robot=robot, policy_type="pi0", action_offset=1)

    with pytest.raises(ValueError, match="frame 'base'"):
        resolve_franka_client_policy_config(config)


@pytest.mark.parametrize("limit", [None, 1, 30, 50])
def test_max_action_chunk_waypoints_config_accepts_positive_values(tmp_path, limit) -> None:
    config = _ros2_config(tmp_path, max_action_chunk_waypoints=limit)

    assert config.max_action_chunk_waypoints == limit


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_max_action_chunk_waypoints_config_rejects_invalid_values(tmp_path, limit) -> None:
    with pytest.raises(ValueError, match="max_action_chunk_waypoints"):
        _ros2_config(tmp_path, max_action_chunk_waypoints=limit)


def _populate_observation(runtime: _FakeRuntime, clock: _ManualClock) -> None:
    anchor_stamp = 2_000_000_000
    arrival = clock.nanoseconds - 10_000_000
    camera1 = np.zeros(CAMERA_SHAPE, dtype=np.uint8)
    camera2 = np.full(CAMERA_SHAPE, 17, dtype=np.uint8)
    runtime.cache.update_camera1(
        ImageSample(
            stamp_ns=anchor_stamp,
            received_monotonic_ns=arrival,
            image=camera1,
            frame_id="camera1_color_optical_frame",
        )
    )
    runtime.cache.update_camera2(
        ImageSample(
            stamp_ns=anchor_stamp - 10_000_000,
            received_monotonic_ns=arrival,
            image=camera2,
            frame_id="camera2_color_optical_frame",
        )
    )
    runtime.cache.update_eef(
        PoseSample(
            stamp_ns=anchor_stamp - 8_000_000,
            received_monotonic_ns=arrival,
            position=np.asarray([0.5, 0.0, 0.4]),
            quaternion_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
            frame_id="base",
        )
    )
    shuffled_names = tuple(reversed(runtime.config.arm_joint_names))
    runtime.cache.update_qpos(
        JointStateSample(
            stamp_ns=anchor_stamp - 20_000_000,
            received_monotonic_ns=arrival,
            names=shuffled_names,
            positions=np.asarray(list(reversed(range(7))), dtype=np.float64),
        )
    )
    runtime.cache.update_gripper(
        0.25,
        stamp_ns=anchor_stamp - 2_000_000,
        received_monotonic_ns=arrival,
    )


def _timed_chunk(
    *, timestep: int = 4, source_timestamp: float = 1_768_000_000.0, count: int = 2
):
    period = 1.0 / 15.0
    server_send = source_timestamp + 0.2
    rows = (
        [0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.25],
        [0.6, -0.1, 0.45, 0.0, 0.0, 0.0, 1.0, 0.5],
    )
    return [
        TimedAction(
            timestamp=source_timestamp + index * period,
            timestep=timestep + index,
            action=torch.tensor(rows[index % len(rows)], dtype=torch.float32),
            server_send_timestamp=server_send,
        )
        for index in range(count)
    ]


def test_ros2_backend_observation_qpos_and_single_chunk_publication(tmp_path) -> None:
    config = _ros2_config(tmp_path)
    clock = _ManualClock(10_000_000_000)
    runtimes: list[_FakeRuntime] = []

    def runtime_factory(runtime_config, cache):
        runtime = _FakeRuntime(runtime_config, cache)
        runtimes.append(runtime)
        return runtime

    backend = Ros2Backend(config=config, runtime_factory=runtime_factory, monotonic_ns=clock)
    backend.connect()
    runtime = runtimes[0]
    assert backend.is_connected
    _populate_observation(runtime, clock)

    observation = backend.get_observation()
    assert tuple(observation) == (*STATE_NAMES, "camera1", "camera2")
    assert tuple(observation[name] for name in STATE_NAMES) == pytest.approx(
        (0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25)
    )
    np.testing.assert_allclose(backend.get_qpos(), np.arange(7, dtype=np.float64))
    assert "qpos" not in observation
    sideband = backend.get_sideband()
    sideband["qpos"][0] = 999.0
    assert backend.get_qpos()[0] == 0.0

    # Per-step bookkeeping is deliberately not a ROS publication path.
    backend.send_action({"target.x": 0.5})
    assert runtime.published == []

    timed_actions = _timed_chunk()
    chunk = backend.publish_action_chunk(
        timed_actions,
        source_observation_timestep=4,
        source_observation_timestamp=timed_actions[0].timestamp,
        period_s=1.0 / 15.0,
    )
    assert runtime.published == [chunk]
    assert chunk.plan_id == 0
    assert chunk.timesteps == (4, 5)
    assert chunk.actions.shape == (2, 8)
    assert chunk.session_id

    second = backend.publish_action_chunk(
        _timed_chunk(timestep=6, source_timestamp=1_768_000_001.0),
        source_observation_timestep=6,
        source_observation_timestamp=1_768_000_001.0,
        period_s=1.0 / 15.0,
    )
    assert second.plan_id == 1
    assert second.session_id == chunk.session_id
    assert len(runtime.published) == 2

    backend.disconnect()
    assert runtime.closed
    assert not backend.is_connected
    with pytest.raises(Ros2BackendError, match="not connected"):
        backend.get_observation()


def test_ros2_backend_rejects_bad_chunk_without_publish(tmp_path) -> None:
    config = _ros2_config(tmp_path)
    clock = _ManualClock(10_000_000_000)
    runtime: _FakeRuntime | None = None

    def runtime_factory(runtime_config, cache):
        nonlocal runtime
        runtime = _FakeRuntime(runtime_config, cache)
        return runtime

    backend = Ros2Backend(config=config, runtime_factory=runtime_factory, monotonic_ns=clock)
    backend.connect()
    assert runtime is not None
    actions = _timed_chunk()
    actions[1].timestamp += 0.01

    with pytest.raises(Ros2BackendError, match="timestamps do not match"):
        backend.publish_action_chunk(
            actions,
            source_observation_timestep=4,
            source_observation_timestamp=actions[0].timestamp,
            period_s=1.0 / 15.0,
        )
    assert runtime.published == []
    backend.disconnect()


class _FakeChunkRobot:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = []

    def publish_action_chunk(self, actions, **metadata) -> None:
        if self.error is not None:
            raise self.error
        self.calls.append((list(actions), metadata))


def _bare_chunk_client(robot: _FakeChunkRobot) -> FrankaRos2RobotClient:
    client = object.__new__(FrankaRos2RobotClient)
    client.robot = robot
    client.latest_action_lock = threading.Lock()
    client.latest_action = 4
    client.action_queue_lock = threading.Lock()
    client.action_queue = Queue()
    client.action_queue_size = []
    client.shutdown_event = threading.Event()
    client.config = SimpleNamespace(
        environment_dt=1.0 / 15.0,
        action_offset=0,
        robot=SimpleNamespace(max_action_chunk_waypoints=None),
    )
    client.logger = logging.getLogger("test-franka-ros2-client")
    return client


def test_client_hook_publishes_one_fresh_chunk_after_stale_prefix() -> None:
    robot = _FakeChunkRobot()
    client = _bare_chunk_client(robot)
    incoming = _timed_chunk(timestep=4)
    incoming.append(
        TimedAction(
            timestamp=incoming[-1].timestamp + 1.0 / 15.0,
            timestep=6,
            action=incoming[-1].action.clone(),
            server_send_timestamp=incoming[-1].server_send_timestamp,
        )
    )

    client._aggregate_action_queues(incoming, lambda _old, new: new)

    assert len(robot.calls) == 1
    published, metadata = robot.calls[0]
    assert [action.timestep for action in published] == [5, 6]
    assert metadata["source_observation_timestep"] == 4
    assert metadata["source_observation_timestamp"] == incoming[0].timestamp
    assert metadata["period_s"] == pytest.approx(1.0 / 15.0)
    assert [action.timestep for action in client.action_queue.queue] == [5, 6]
    assert not client.shutdown_event.is_set()


@pytest.mark.parametrize(
    ("action_offset", "expected_count", "expected_source_timestep", "expected_last_timestep"),
    [
        (0, 49, 4, 53),
        (1, 50, 5, 54),
    ],
)
def test_action_offset_controls_fixed_overlap_in_fifty_action_chunk(
    action_offset: int,
    expected_count: int,
    expected_source_timestep: int,
    expected_last_timestep: int,
) -> None:
    robot = _FakeChunkRobot()
    client = _bare_chunk_client(robot)
    client.config.action_offset = action_offset
    first_timestep = client._next_observation_timestep(client.latest_action)
    incoming = _timed_chunk(timestep=first_timestep, count=50)

    client._aggregate_action_queues(incoming, lambda _old, new: new)

    assert len(robot.calls) == 1
    published, metadata = robot.calls[0]
    expected_timesteps = list(range(5, expected_last_timestep + 1))
    assert len(published) == expected_count
    assert [action.timestep for action in published] == expected_timesteps
    assert metadata["source_observation_timestep"] == expected_source_timestep
    assert metadata["source_observation_timestamp"] == incoming[0].timestamp
    assert [action.timestep for action in client.action_queue.queue] == expected_timesteps


def test_action_offset_one_still_discards_actions_that_expire_during_inference() -> None:
    robot = _FakeChunkRobot()
    client = _bare_chunk_client(robot)
    client.config.action_offset = 1
    first_timestep = client._next_observation_timestep(client.latest_action)
    incoming = _timed_chunk(timestep=first_timestep, count=50)

    # One queued action was consumed after observation capture but before this
    # chunk arrived. It is genuinely stale and must not be sent to ROS2.
    client.latest_action = 5
    client._aggregate_action_queues(incoming, lambda _old, new: new)

    published, metadata = robot.calls[0]
    assert len(published) == 49
    assert [action.timestep for action in published] == list(range(6, 55))
    assert metadata["source_observation_timestep"] == 5


def test_client_limits_fifty_fresh_actions_to_thirty_for_local_queue_and_ros() -> None:
    robot = _FakeChunkRobot()
    client = _bare_chunk_client(robot)
    client.config.action_offset = 1
    client.config.robot.max_action_chunk_waypoints = 30
    first_timestep = client._next_observation_timestep(client.latest_action)
    incoming = _timed_chunk(timestep=first_timestep, count=50)

    assert client._effective_action_chunk_size(incoming) == 30

    client._aggregate_action_queues(incoming, lambda _old, new: new)

    published, metadata = robot.calls[0]
    expected_timesteps = list(range(5, 35))
    assert [action.timestep for action in published] == expected_timesteps
    assert [action.timestep for action in client.action_queue.queue] == expected_timesteps
    assert metadata["source_observation_timestep"] == 5
    assert metadata["source_observation_timestamp"] == incoming[0].timestamp


def test_client_drops_stale_prefix_before_applying_waypoint_limit() -> None:
    robot = _FakeChunkRobot()
    client = _bare_chunk_client(robot)
    client.config.robot.max_action_chunk_waypoints = 30
    incoming = _timed_chunk(timestep=4, count=50)
    client.latest_action = 9

    client._aggregate_action_queues(incoming, lambda _old, new: new)

    published, metadata = robot.calls[0]
    expected_timesteps = list(range(10, 40))
    assert [action.timestep for action in published] == expected_timesteps
    assert [action.timestep for action in client.action_queue.queue] == expected_timesteps
    assert metadata["source_observation_timestep"] == 4


def test_client_waypoint_limit_does_not_pad_short_fresh_suffix() -> None:
    robot = _FakeChunkRobot()
    client = _bare_chunk_client(robot)
    client.config.robot.max_action_chunk_waypoints = 30
    incoming = _timed_chunk(timestep=4, count=50)
    client.latest_action = 39

    client._aggregate_action_queues(incoming, lambda _old, new: new)

    published, _metadata = robot.calls[0]
    expected_timesteps = list(range(40, 54))
    assert [action.timestep for action in published] == expected_timesteps
    assert [action.timestep for action in client.action_queue.queue] == expected_timesteps


def test_client_hook_fails_closed_when_chunk_publication_fails() -> None:
    client = _bare_chunk_client(_FakeChunkRobot(RuntimeError("publisher failed")))

    with pytest.raises(RuntimeError, match="publisher failed"):
        client._aggregate_action_queues(_timed_chunk(timestep=5), lambda _old, new: new)
    assert client.shutdown_event.is_set()
