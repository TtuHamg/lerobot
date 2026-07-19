"""No-ROS tests for the Franka ROS 2 transport-neutral contracts."""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

PLUGIN_SRC = Path(__file__).parents[1] / "ros_lerobot" / "src"
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))

from lerobot_robot_franka_ros.contract import CAMERA_SHAPE, STATE_NAMES  # noqa: E402
from lerobot_robot_franka_ros.ros2_contract import (  # noqa: E402
    AbsoluteActionChunk,
    ImageSample,
    JointStateSample,
    PoseSample,
    Ros2ContractError,
    RosObservationCache,
    decode_ros_image,
)


JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
SKEW_LIMITS = {"camera2": 20, "eef": 10, "qpos": 30, "gripper": 15}


class ManualClock:
    def __init__(self, value_ns: int) -> None:
        self.value_ns = value_ns

    def __call__(self) -> int:
        return self.value_ns


def _image(marker: int = 0) -> np.ndarray:
    image = np.zeros(CAMERA_SHAPE, dtype=np.uint8)
    image[0, 0] = marker
    return image


def _cache(
    clock: ManualClock,
    *,
    skew_limits: dict[str, int] | None = None,
    max_age_ns: int = 1_000,
) -> RosObservationCache:
    return RosObservationCache(
        joint_names=JOINT_NAMES,
        base_frame="base",
        max_skew_ns=SKEW_LIMITS if skew_limits is None else skew_limits,
        max_age_ns=max_age_ns,
        buffer_size=16,
        monotonic_ns=clock,
    )


def _feed_complete_snapshot(
    cache: RosObservationCache,
    *,
    stamp_ns: int,
    received_ns: int,
    marker: int = 0,
    camera1_last: bool = False,
) -> None:
    camera1 = ImageSample(stamp_ns, received_ns, _image(marker), "camera1_optical")
    camera2 = ImageSample(stamp_ns, received_ns, _image(marker), "camera2_optical")
    eef = PoseSample(
        stamp_ns,
        received_ns,
        np.asarray([float(marker), 0.2, 0.3]),
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        "base",
    )
    qpos = JointStateSample(
        stamp_ns,
        received_ns,
        JOINT_NAMES,
        np.full(7, marker, dtype=np.float64),
    )
    if not camera1_last:
        cache.update_camera1(camera1)
    cache.update_camera2(camera2)
    cache.update_eef(eef)
    cache.update_qpos(qpos)
    cache.update_gripper(float(marker), stamp_ns=stamp_ns, received_monotonic_ns=received_ns)
    if camera1_last:
        cache.update_camera1(camera1)


def _valid_actions() -> np.ndarray:
    return np.asarray(
        [
            [0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.25],
            [0.6, 0.1, 0.3, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5), 0.75],
        ],
        dtype=np.float32,
    )


def _chunk(**overrides) -> AbsoluteActionChunk:
    values = {
        "actions": _valid_actions(),
        "timesteps": (4, 5),
        "source_timestep": 4,
        "source_observation_timestamp_ns": 10_000_000_000,
        "server_send_timestamp_ns": 10_100_000_000,
        "received_monotonic_ns": 2_000_000_000,
        "period_ns": 66_666_667,
        "frame_id": "base",
        "session_id": "session-a",
        "plan_id": 7,
        "valid_for_ns": 500_000_000,
    }
    values.update(overrides)
    return AbsoluteActionChunk(**values)


def test_decode_ros_image_rgb_bgr_and_row_padding() -> None:
    rgb = decode_ros_image(
        height=1,
        width=2,
        encoding="rgb8",
        step=6,
        data=bytes([1, 2, 3, 4, 5, 6]),
    )
    assert rgb.tolist() == [[[1, 2, 3], [4, 5, 6]]]
    assert rgb.flags.c_contiguous
    assert not rgb.flags.writeable

    bgr_with_padding = decode_ros_image(
        height=2,
        width=1,
        encoding="bgr8",
        step=4,
        data=bytes([3, 2, 1, 99, 6, 5, 4, 88]),
    )
    assert bgr_with_padding.tolist() == [[[1, 2, 3]], [[4, 5, 6]]]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"height": 0}, "height must be"),
        ({"encoding": "mono8"}, "unsupported image encoding"),
        ({"step": 5}, "step must be at least"),
        ({"data": bytes(5)}, "image data length"),
    ],
)
def test_decode_ros_image_rejects_contract_drift(kwargs: dict, match: str) -> None:
    values = {
        "height": 1,
        "width": 2,
        "encoding": "rgb8",
        "step": 6,
        "data": bytes(6),
    }
    values.update(kwargs)
    with pytest.raises(Ros2ContractError, match=match):
        decode_ros_image(**values)


def test_samples_require_positive_stamps_finite_values_and_unit_quaternion() -> None:
    with pytest.raises(Ros2ContractError, match="stamp_ns must be >= 1"):
        ImageSample(0, 1, _image())
    with pytest.raises(Ros2ContractError, match="received_monotonic_ns must be >= 1"):
        JointStateSample(1, 0, JOINT_NAMES, np.zeros(7))
    with pytest.raises(Ros2ContractError, match="unit length"):
        PoseSample(1, 1, np.zeros(3), np.zeros(4), "base")
    with pytest.raises(Ros2ContractError, match="duplicates"):
        JointStateSample(1, 1, ("joint", "joint"), np.zeros(2))


def test_cache_builds_exact_policy_projection_and_keeps_qpos_sideband_copy() -> None:
    clock = ManualClock(1_000)
    cache = _cache(clock)
    cache.update_camera1(ImageSample(100, 950, _image(1), "camera1_optical"))
    cache.update_camera2(ImageSample(99, 951, _image(2), "camera2_optical"))
    cache.update_eef(
        PoseSample(
            98,
            952,
            np.asarray([0.5, 0.0, 0.4]),
            np.asarray([0.0, 0.0, 0.0, 1.0]),
            "base",
        )
    )
    cache.update_qpos(
        JointStateSample(
            97,
            953,
            tuple(reversed(JOINT_NAMES)),
            np.arange(6, -1, -1, dtype=np.float64),
        )
    )
    cache.update_gripper(0.25, stamp_ns=96, received_monotonic_ns=954)

    snapshot = cache.get_snapshot()
    policy = snapshot.as_policy_observation()
    robot_observation = snapshot.as_robot_observation()
    assert tuple(policy) == (*STATE_NAMES, "camera1", "camera2")
    assert tuple(robot_observation) == tuple(policy)
    assert tuple(policy[name] for name in STATE_NAMES) == pytest.approx(
        (0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25)
    )
    assert snapshot.qpos.tolist() == list(range(7))
    assert snapshot.source_stamps_ns == {
        "camera1": 100,
        "camera2": 99,
        "eef": 98,
        "qpos": 97,
        "gripper": 96,
    }

    policy["camera1"][0, 0, 0] = 255
    qpos = snapshot.qpos
    qpos[0] = 999
    sideband = snapshot.as_sideband()
    sideband["qpos"][1] = 999
    assert snapshot.camera1[0, 0, 0] == 1
    assert snapshot.qpos.tolist() == list(range(7))
    assert "qpos" not in policy


def test_pose_projection_uses_xyzw_and_rotation_matrix_columns() -> None:
    clock = ManualClock(1_000)
    cache = _cache(clock)
    _feed_complete_snapshot(cache, stamp_ns=100, received_ns=950)
    half_sqrt = math.sqrt(0.5)
    cache.update_eef(
        PoseSample(
            100,
            960,
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0.0, 0.0, half_sqrt, half_sqrt]),
            "base",
        )
    )
    state = cache.get_snapshot().state
    assert state[3:9] == pytest.approx((0.0, 1.0, 0.0, -1.0, 0.0, 0.0), abs=1e-7)


def test_cache_is_causal_and_does_not_regress_to_late_old_camera1() -> None:
    clock = ManualClock(10_000)
    cache = _cache(clock)
    cache.update_camera1(ImageSample(100, 9_500, _image(1)))

    cache.update_camera2(ImageSample(95, 9_500, _image(2)))
    cache.update_camera2(ImageSample(101, 9_500, _image(9)))
    cache.update_eef(PoseSample(96, 9_500, [2.0, 0.0, 0.0], [0, 0, 0, 1], "base"))
    cache.update_eef(PoseSample(110, 9_500, [9.0, 0.0, 0.0], [0, 0, 0, 1], "base"))
    cache.update_qpos(JointStateSample(90, 9_500, JOINT_NAMES, np.full(7, 3.0)))
    cache.update_qpos(JointStateSample(105, 9_500, JOINT_NAMES, np.full(7, 9.0)))
    cache.update_gripper(0.4, stamp_ns=99, received_monotonic_ns=9_500)
    cache.update_gripper(0.9, stamp_ns=102, received_monotonic_ns=9_500)

    snapshot = cache.get_snapshot()
    assert snapshot.camera2[0, 0, 0] == 2
    assert snapshot.eef_position[0] == 2.0
    assert snapshot.qpos[0] == 3.0
    assert snapshot.gripper_closed_0_1 == 0.4

    cache.update_camera1(ImageSample(80, 9_600, _image(8)))
    assert cache.latest_anchor_stamp_ns == 100
    assert cache.get_snapshot().anchor_stamp_ns == 100


def test_cache_enforces_independent_skew_limits_and_uniform_freshness_boundaries() -> None:
    clock = ManualClock(1_000)
    cache = _cache(clock, max_age_ns=100)
    cache.update_camera1(ImageSample(100, 900, _image()))
    cache.update_camera2(ImageSample(80, 900, _image()))  # exact camera2 limit
    cache.update_eef(PoseSample(90, 900, [0, 0, 0], [0, 0, 0, 1], "base"))
    cache.update_qpos(JointStateSample(70, 900, JOINT_NAMES, np.zeros(7)))
    cache.update_gripper(0.0, stamp_ns=85, received_monotonic_ns=900)
    assert cache.get_snapshot().anchor_stamp_ns == 100

    clock.value_ns = 1_001
    with pytest.raises(Ros2ContractError, match="stale"):
        cache.get_snapshot()

    clock.value_ns = 1_000
    cache.update_qpos(JointStateSample(69, 901, JOINT_NAMES, np.ones(7)))
    # The newer causal candidate at stamp 70 still satisfies qpos's limit.
    assert cache.get_snapshot().qpos[0] == 0.0

    skewed = _cache(clock, skew_limits={**SKEW_LIMITS, "qpos": 29})
    skewed.update_camera1(ImageSample(100, 950, _image()))
    skewed.update_camera2(ImageSample(100, 950, _image()))
    skewed.update_eef(PoseSample(100, 950, [0, 0, 0], [0, 0, 0, 1], "base"))
    skewed.update_qpos(JointStateSample(70, 950, JOINT_NAMES, np.zeros(7)))
    skewed.update_gripper(0.0, stamp_ns=100, received_monotonic_ns=950)
    with pytest.raises(Ros2ContractError, match="qpos skew exceeds"):
        skewed.get_snapshot()


def test_cache_rejects_incomplete_skew_map_wrong_frame_and_joint_contract() -> None:
    clock = ManualClock(1_000)
    with pytest.raises(Ros2ContractError, match="contain exactly"):
        _cache(clock, skew_limits={"eef": 10})

    cache = _cache(clock)
    with pytest.raises(Ros2ContractError, match="frame_id"):
        cache.update_eef(PoseSample(100, 900, [0, 0, 0], [0, 0, 0, 1], "world"))
    with pytest.raises(Ros2ContractError, match="joint names do not match"):
        cache.update_qpos(
            JointStateSample(100, 900, (*JOINT_NAMES[:-1], "wrong"), np.zeros(7))
        )


def test_cache_concurrent_readers_only_observe_complete_causal_generations() -> None:
    clock = ManualClock(100_000)
    cache = _cache(clock, max_age_ns=100_000)
    _feed_complete_snapshot(cache, stamp_ns=100, received_ns=90_000, marker=0, camera1_last=True)
    barrier = threading.Barrier(3)
    done = threading.Event()
    failures: list[tuple[int, int, int, int, float]] = []

    def reader() -> None:
        barrier.wait()
        while not done.is_set():
            snapshot = cache.get_snapshot()
            marker = snapshot.anchor_stamp_ns % 2
            observed = (
                int(snapshot.camera1[0, 0, 0]),
                int(snapshot.camera2[0, 0, 0]),
                int(snapshot.eef_position[0]),
                int(snapshot.qpos[0]),
                snapshot.gripper_closed_0_1,
            )
            if observed != (marker, marker, marker, marker, float(marker)):
                failures.append(observed)
                done.set()

    readers = [threading.Thread(target=reader) for _ in range(2)]
    for thread in readers:
        thread.start()
    barrier.wait()
    for stamp in range(101, 107):
        _feed_complete_snapshot(
            cache,
            stamp_ns=stamp,
            received_ns=90_000 + stamp,
            marker=stamp % 2,
            camera1_last=True,
        )
    done.set()
    for thread in readers:
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    assert not failures


def test_absolute_action_chunk_preserves_ros_message_fields_and_copies_arrays() -> None:
    source = _valid_actions()
    chunk = _chunk(actions=source, plan_id=2**64 - 1)
    source[0, 0] = 99.0

    assert len(chunk) == 2
    assert chunk.schema_version == 1
    assert chunk.plan_id == 2**64 - 1
    assert chunk.source_timestep == 4
    assert chunk.timesteps == (4, 5)
    assert chunk.actions.shape == (2, 8)
    assert chunk.poses.shape == (2, 7)
    assert chunk.gripper.tolist() == pytest.approx([0.25, 0.75])
    assert chunk.actions[0, 0] == pytest.approx(0.5)
    assert chunk.client_observation_stamp_ns == chunk.source_observation_timestamp_ns
    assert chunk.source_observation_stamp_ns == chunk.source_observation_timestamp_ns
    assert chunk.server_send_stamp_ns == chunk.server_send_timestamp_ns
    assert chunk.action_stamps_ns == (10_000_000_000, 10_066_666_667)

    copied = chunk.actions
    copied[0, 0] = 123.0
    payload = chunk.as_payload()
    payload["poses"][0, 0] = 456.0
    assert chunk.actions[0, 0] == pytest.approx(0.5)
    assert payload["source_timestep"] == 4
    assert payload["valid_for_ns"] == 500_000_000


def test_absolute_action_chunk_preserves_source_observation_for_fresh_suffix() -> None:
    chunk = _chunk(timesteps=(5, 6), source_timestep=4)

    assert chunk.source_timestep == 4
    assert chunk.timesteps == (5, 6)
    assert chunk.action_stamps_ns == (10_066_666_667, 10_133_333_334)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"actions": np.empty((0, 8), dtype=np.float32), "timesteps": ()}, "non-empty shape"),
        ({"actions": np.zeros((2, 7), dtype=np.float32)}, r"shape \[K,8\]"),
        ({"actions": np.zeros((2, 8), dtype=np.int64)}, "floating dtype"),
        ({"actions": np.asarray([[0, 0, 0, 0, 0, 0, 1, np.nan]] * 2)}, "NaN or Inf"),
        ({"actions": np.asarray([[0, 0, 0, 0, 0, 0, 0, 0.5]] * 2)}, "unit length"),
        ({"actions": np.asarray([[0, 0, 0, 0, 0, 0, 1, 1.1]] * 2)}, r"in \[0, 1\]"),
        ({"timesteps": (4,)}, "length must equal"),
        ({"timesteps": (4, 6)}, "strictly contiguous"),
        ({"source_timestep": 5}, "source_timestep must not be later"),
        ({"plan_id": 2**64}, "fit uint64"),
        ({"valid_for_ns": 0}, "valid_for_ns must be >= 1"),
        (
            {"source_observation_timestamp_ns": 0},
            "source_observation_timestamp_ns must be >= 1",
        ),
    ],
)
def test_absolute_action_chunk_rejects_shape_data_timing_and_metadata_drift(
    overrides: dict,
    match: str,
) -> None:
    with pytest.raises(Ros2ContractError, match=match):
        _chunk(**overrides)


def test_absolute_action_chunk_validity_uses_only_local_monotonic_time() -> None:
    chunk = _chunk(received_monotonic_ns=1_000, valid_for_ns=100)
    assert chunk.is_valid_at(1_000)
    assert chunk.is_valid_at(1_100)
    assert not chunk.is_valid_at(1_101)
    assert not chunk.is_valid_at(999)
    with pytest.raises(Ros2ContractError, match="outside its local validity window"):
        chunk.require_valid_at(1_101)
