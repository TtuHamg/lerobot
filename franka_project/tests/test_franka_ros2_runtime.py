"""ROS-message encoding tests using fakes; no ROS graph or hardware is started."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PLUGIN_SRC = Path(__file__).parents[1] / "ros_lerobot" / "src"
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))

from lerobot_robot_franka_ros import ros2_runtime  # noqa: E402
from lerobot_robot_franka_ros.config_franka_ros import FrankaRosConfig  # noqa: E402
from lerobot_robot_franka_ros.ros2_contract import (  # noqa: E402
    AbsoluteActionChunk,
    RosObservationCache,
)


class _FakeTime:
    def __init__(self) -> None:
        self.sec = 0
        self.nanosec = 0


class _FakeHeader:
    def __init__(self) -> None:
        self.stamp = _FakeTime()
        self.frame_id = ""


class _FakeActionMessage:
    def __init__(self) -> None:
        self.header = _FakeHeader()
        self.schema_version = 0
        self.session_id = ""
        self.plan_id = 0
        self.source_timestep = 0
        self.client_observation_stamp = _FakeTime()
        self.server_send_stamp = _FakeTime()
        self.valid_until = _FakeTime()
        self.period = _FakeTime()
        self.timesteps = []
        self.poses = []
        self.gripper = []


class _FakePose:
    def __init__(self) -> None:
        self.position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)


class _FakeRosClock:
    def __init__(self, nanoseconds: int) -> None:
        self.nanoseconds = nanoseconds

    def now(self):
        return SimpleNamespace(nanoseconds=self.nanoseconds)


class _FakeNode:
    def __init__(self, ros_now_ns: int) -> None:
        self._clock = _FakeRosClock(ros_now_ns)

    def get_clock(self):
        return self._clock


def _cache(
    config: FrankaRosConfig,
    *,
    max_skew_ns: int = 1,
    max_age_ns: int = 1,
    monotonic_ns=lambda: 1,
) -> RosObservationCache:
    return RosObservationCache(
        joint_names=config.arm_joint_names,
        base_frame=config.base_frame,
        max_skew_ns={
            "camera2": max_skew_ns,
            "eef": max_skew_ns,
            "qpos": max_skew_ns,
            "gripper": max_skew_ns,
        },
        max_age_ns=max_age_ns,
        monotonic_ns=monotonic_ns,
    )


def _as_ns(message: _FakeTime) -> int:
    return message.sec * 1_000_000_000 + message.nanosec


def test_runtime_import_is_ros_side_effect_free() -> None:
    # ros2_runtime has been imported above, but binding modules remain lazy.
    assert "rclpy" not in ros2_runtime.__dict__
    assert "sensor_msgs" not in ros2_runtime.__dict__


def test_runtime_callbacks_form_policy_observation_and_qpos_sideband(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = FrankaRosConfig(
        id="runtime-callback-test",
        calibration_dir=tmp_path / "calibration",
        dry_run=False,
    )
    now_ns = 10_000
    cache = _cache(
        config,
        max_skew_ns=100,
        max_age_ns=1_000,
        monotonic_ns=lambda: now_ns,
    )
    runtime = ros2_runtime.Ros2Runtime(config, cache)
    monkeypatch.setattr(ros2_runtime.time, "monotonic_ns", lambda: now_ns - 100)

    stamp = _FakeTime()
    stamp.sec = 2
    header = SimpleNamespace(stamp=stamp, frame_id="camera_frame")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[0, 0] = (1, 2, 3)
    image_message = SimpleNamespace(
        header=header,
        height=480,
        width=640,
        encoding="rgb8",
        step=640 * 3,
        data=image.tobytes(),
    )
    runtime._camera_callback("camera1", image_message)
    runtime._camera_callback("camera2", image_message)

    pose_header = SimpleNamespace(stamp=stamp, frame_id="base")
    runtime._eef_callback(
        SimpleNamespace(
            header=pose_header,
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.5, y=-0.1, z=0.4),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )
    )
    joint_names = tuple(reversed(config.arm_joint_names))
    runtime._arm_joint_state_callback(
        SimpleNamespace(
            header=pose_header,
            name=joint_names,
            position=tuple(reversed(range(7))),
        )
    )
    runtime._gripper_joint_state_callback(
        SimpleNamespace(
            header=pose_header,
            name=(config.gripper_joint_name,),
            position=(0.2,),
        )
    )

    snapshot = cache.get_snapshot()
    observation = snapshot.as_policy_observation()
    assert observation["eef.x"] == pytest.approx(0.5)
    assert observation["gripper.closed_0_1"] == pytest.approx(0.5)
    np.testing.assert_array_equal(snapshot.qpos, np.arange(7))
    np.testing.assert_array_equal(observation["camera1"][0, 0], (1, 2, 3))


def test_action_chunk_ros_message_preserves_metadata_and_remaining_ttl(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = FrankaRosConfig(
        id="runtime-message-test",
        calibration_dir=tmp_path / "calibration",
        dry_run=False,
    )
    runtime = ros2_runtime.Ros2Runtime(config, _cache(config))
    ros_now_ns = 5_000_000_000
    runtime._node = _FakeNode(ros_now_ns)
    runtime._bindings = ros2_runtime._RosBindings(
        rclpy=None,
        context_type=object,
        node_type=object,
        executor_type=object,
        qos_profile_type=object,
        history_policy=None,
        reliability_policy=None,
        durability_policy=None,
        image_type=object,
        pose_stamped_type=object,
        joint_state_type=object,
        pose_type=_FakePose,
        action_chunk_type=_FakeActionMessage,
        action_chunk_ack_type=object,
        gateway_status_type=object,
    )

    received_ns = 10_000
    valid_for_ns = 500
    monkeypatch.setattr(ros2_runtime.time, "monotonic_ns", lambda: received_ns + 200)
    chunk = AbsoluteActionChunk(
        actions=np.asarray(
            [
                [0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.25],
                [0.6, 0.1, 0.45, 0.0, 0.0, 0.0, 1.0, 0.5],
            ],
            dtype=np.float32,
        ),
        timesteps=(7, 8),
        source_timestep=7,
        source_observation_timestamp_ns=1_768_000_000_000_000_000,
        server_send_timestamp_ns=1_768_000_000_200_000_000,
        received_monotonic_ns=received_ns,
        period_ns=66_666_667,
        frame_id="base",
        session_id="session-a",
        plan_id=3,
        valid_for_ns=valid_for_ns,
    )

    message = runtime._action_chunk_message(chunk)

    assert _as_ns(message.header.stamp) == ros_now_ns
    assert message.header.frame_id == "base"
    assert message.schema_version == 1
    assert message.session_id == "session-a"
    assert message.plan_id == 3
    assert message.source_timestep == 7
    assert _as_ns(message.client_observation_stamp) == chunk.source_observation_timestamp_ns
    assert _as_ns(message.server_send_stamp) == chunk.server_send_timestamp_ns
    assert _as_ns(message.valid_until) == ros_now_ns + 300
    assert _as_ns(message.period) == 66_666_667
    assert message.timesteps == [7, 8]
    assert message.gripper == [0.25, 0.5]
    assert len(message.poses) == 2
    assert message.poses[1].position.x == pytest.approx(0.6)
    assert message.poses[1].orientation.w == 1.0
