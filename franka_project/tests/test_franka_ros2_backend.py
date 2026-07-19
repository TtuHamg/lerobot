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

from lerobot.async_inference.helpers import TimedAction  # noqa: E402
from lerobot_robot_franka_ros.config_franka_ros import FrankaRosConfig  # noqa: E402
from lerobot_robot_franka_ros.contract import CAMERA_SHAPE, STATE_NAMES  # noqa: E402
from lerobot_robot_franka_ros.ros2_backend import Ros2Backend, Ros2BackendError  # noqa: E402
from lerobot_robot_franka_ros.ros2_client import FrankaRos2RobotClient  # noqa: E402
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
    assert "--robot.ros2_interface_only" in result.stdout
    assert "--robot.action_chunk_topic" in result.stdout


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


def _ros2_config(tmp_path) -> FrankaRosConfig:
    return FrankaRosConfig(
        id="ros2-test",
        calibration_dir=tmp_path / "calibration",
        dry_run=False,
    )


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


def _timed_chunk(*, timestep: int = 4, source_timestamp: float = 1_768_000_000.0):
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
            action=torch.tensor(row, dtype=torch.float32),
            server_send_timestamp=server_send,
        )
        for index, row in enumerate(rows)
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
    client.config = SimpleNamespace(environment_dt=1.0 / 15.0)
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


def test_client_hook_fails_closed_when_chunk_publication_fails() -> None:
    client = _bare_chunk_client(_FakeChunkRobot(RuntimeError("publisher failed")))

    with pytest.raises(RuntimeError, match="publisher failed"):
        client._aggregate_action_queues(_timed_chunk(timestep=5), lambda _old, new: new)
    assert client.shutdown_event.is_set()
