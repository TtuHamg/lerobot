"""Contract tests for the out-of-tree joint-space (FastWAM) Franka Robot plugin."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN_SRC = Path(__file__).parents[1] / "ros_lerobot" / "src"
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))

from lerobot.robots.config import RobotConfig  # noqa: E402
from lerobot.robots.utils import make_robot_from_config  # noqa: E402
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError  # noqa: E402
from lerobot_robot_franka_ros import (  # noqa: E402
    CAMERA_SHAPE,
    JOINT_ACTION_NAMES,
    JOINT_STATE_NAMES,
    FrankaJointRos,
    FrankaJointRosConfig,
)
from lerobot_robot_franka_ros.joint_ros2_runtime import (  # noqa: E402
    JointRos2RuntimeStateError,
    _clamp_continuous_gripper_commands,
)


def _write_fixture(path: Path, **overrides: np.ndarray) -> None:
    payload = {
        "state": np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.23], dtype=np.float32),
        "camera1": np.zeros(CAMERA_SHAPE, dtype=np.uint8),
        "camera2": np.ones(CAMERA_SHAPE, dtype=np.uint8),
    }
    payload.update(overrides)
    np.savez(path, **payload)


def _config(tmp_path: Path, fixture_path: Path) -> FrankaJointRosConfig:
    return FrankaJointRosConfig(
        id="dry-run-joint-test",
        calibration_dir=tmp_path / "calibration",
        dry_run=True,
        fixture_path=fixture_path,
        action_log_path=tmp_path / "actions.jsonl",
    )


def _valid_action() -> dict[str, float]:
    return {
        "target.fr3_joint1": 0.1,
        "target.fr3_joint2": 0.2,
        "target.fr3_joint3": 0.3,
        "target.fr3_joint4": 0.4,
        "target.fr3_joint5": 0.5,
        "target.fr3_joint6": 0.6,
        "target.fr3_joint7": 0.7,
        "target.gripper.pos": 1.23,
    }


def test_plugin_registration_factory_and_feature_contract(tmp_path: Path) -> None:
    fixture_path = tmp_path / "observation.npz"
    _write_fixture(fixture_path)
    config = _config(tmp_path, fixture_path)

    assert RobotConfig.get_choice_class("franka_ros_joint") is FrankaJointRosConfig
    robot = make_robot_from_config(config)
    assert isinstance(robot, FrankaJointRos)
    assert tuple(robot.observation_features) == (*JOINT_STATE_NAMES, "camera1", "camera2")
    assert tuple(robot.action_features) == JOINT_ACTION_NAMES
    assert all(robot.observation_features[name] is float for name in JOINT_STATE_NAMES)
    assert robot.observation_features["camera1"] == CAMERA_SHAPE
    assert robot.observation_features["camera2"] == CAMERA_SHAPE


def test_lifecycle_fixture_copies_and_jsonl_sink(tmp_path: Path) -> None:
    fixture_path = tmp_path / "observation.npz"
    _write_fixture(fixture_path)
    config = _config(tmp_path, fixture_path)
    robot = FrankaJointRos(config)

    with pytest.raises(DeviceNotConnectedError):
        robot.get_observation()
    with pytest.raises(DeviceNotConnectedError):
        robot.send_action(_valid_action())

    robot.connect()
    assert robot.is_connected
    assert robot.is_calibrated
    with pytest.raises(DeviceAlreadyConnectedError):
        robot.connect()

    observation = robot.get_observation()
    assert tuple(observation[name] for name in JOINT_STATE_NAMES) == pytest.approx(
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.23)
    )
    assert observation["camera1"].shape == CAMERA_SHAPE
    assert observation["camera1"].dtype == np.uint8
    observation["camera1"][0, 0, 0] = 255
    assert robot.get_observation()["camera1"][0, 0, 0] == 0

    sent = robot.send_action(_valid_action())
    assert tuple(sent) == JOINT_ACTION_NAMES
    log_lines = config.action_log_path.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    record = json.loads(log_lines[0])
    assert record["schema_version"] == 1
    assert record["dry_run"] is True
    assert record["robot_id"] == "dry-run-joint-test"
    assert record["sequence"] == 0
    assert record["monotonic_ns"] > 0
    assert record["action"] == _valid_action()

    robot.disconnect()
    assert not robot.is_connected


@pytest.mark.parametrize(
    ("overrides", "error_type", "match"),
    [
        ({"state": np.zeros(7, dtype=np.float32)}, ValueError, "state must have shape"),
        ({"state": np.zeros(8, dtype=np.float64)}, TypeError, "state must have dtype float32"),
        (
            {"state": np.asarray([0.0] * 7 + [np.nan], dtype=np.float32)},
            ValueError,
            "state contains a non-finite",
        ),
        ({"camera1": np.zeros((10, 10, 3), dtype=np.uint8)}, ValueError, "camera1 must have shape"),
        ({"camera2": np.zeros(CAMERA_SHAPE, dtype=np.float32)}, TypeError, "camera2 must have dtype uint8"),
    ],
)
def test_fixture_validation_is_strict(
    tmp_path: Path, overrides: dict[str, np.ndarray], error_type: type[Exception], match: str
) -> None:
    fixture_path = tmp_path / "bad.npz"
    _write_fixture(fixture_path, **overrides)
    robot = FrankaJointRos(_config(tmp_path, fixture_path))

    with pytest.raises(error_type, match=match):
        robot.connect()
    assert not robot.is_connected


def test_fixture_rejects_missing_and_extra_keys(tmp_path: Path) -> None:
    fixture_path = tmp_path / "bad-keys.npz"
    np.savez(
        fixture_path,
        state=np.zeros(8, dtype=np.float32),
        camera1=np.zeros(CAMERA_SHAPE, dtype=np.uint8),
        unexpected=np.zeros(1, dtype=np.uint8),
    )
    robot = FrankaJointRos(_config(tmp_path, fixture_path))

    with pytest.raises(ValueError, match="Fixture keys do not match contract"):
        robot.connect()
    assert not robot.is_connected


@pytest.mark.parametrize(
    ("mutate", "error_type", "match"),
    [
        (lambda action: action.pop("target.fr3_joint1"), ValueError, "Action keys do not match contract"),
        (lambda action: action.update({"extra": 0.0}), ValueError, "Action keys do not match contract"),
        (lambda action: action.update({"target.fr3_joint1": np.nan}), ValueError, "must be finite"),
        (lambda action: action.update({"target.fr3_joint1": [0.5]}), TypeError, "must be a real scalar"),
    ],
)
def test_action_validation_is_strict(tmp_path: Path, mutate, error_type: type[Exception], match: str) -> None:
    fixture_path = tmp_path / "observation.npz"
    _write_fixture(fixture_path)
    config = _config(tmp_path, fixture_path)
    robot = FrankaJointRos(config)
    robot.connect()
    action = _valid_action()
    mutate(action)

    with pytest.raises(error_type, match=match):
        robot.send_action(action)
    assert config.action_log_path.read_text(encoding="utf-8") == ""
    robot.disconnect()


def test_non_dry_run_only_allows_isolated_ros2_interface(tmp_path: Path) -> None:
    config = FrankaJointRosConfig(
        id="ros2-interface",
        calibration_dir=tmp_path / "calibration",
        dry_run=False,
    )
    assert config.ros2_interface_only is True
    assert config.action_chunk_topic.startswith("/lerobot/")

    with pytest.raises(ValueError, match="actuation is not implemented"):
        FrankaJointRosConfig(
            id="unsafe",
            calibration_dir=tmp_path / "calibration",
            dry_run=False,
            ros2_interface_only=False,
        )

    with pytest.raises(ValueError, match="only valid when dry_run=true"):
        FrankaJointRosConfig(
            id="mixed-backends",
            calibration_dir=tmp_path / "calibration",
            dry_run=False,
            fixture_path=tmp_path / "unused.npz",
        )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"max_observation_age_s": float("nan")}, "finite and greater than zero"),
        ({"action_chunk_validity_s": float("inf")}, "finite and greater than zero"),
        ({"observation_buffer_size": True}, "at least 2"),
        ({"gripper_command_min_position": float("nan")}, "finite real values"),
        (
            {"gripper_command_min_position": 0.8, "gripper_command_max_position": 0.8},
            "minimum must be less than maximum",
        ),
    ],
)
def test_ros2_config_rejects_non_finite_or_boolean_numeric_values(
    tmp_path: Path,
    override: dict,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        FrankaJointRosConfig(
            id="bad-ros2-numeric",
            calibration_dir=tmp_path / "calibration",
            dry_run=False,
            **override,
        )


def test_gripper_commands_are_forwarded_continuously_without_quantization() -> None:
    values = [0.0, 0.399, 0.4, 0.782505, 0.8]
    forwarded, raw_min, raw_max, clamped_count = _clamp_continuous_gripper_commands(
        values,
        minimum=0.0,
        maximum=0.8,
    )

    assert forwarded == values
    assert raw_min == pytest.approx(0.0)
    assert raw_max == pytest.approx(0.8)
    assert clamped_count == 0


def test_continuous_gripper_commands_saturate_out_of_range_values() -> None:
    forwarded, raw_min, raw_max, clamped_count = _clamp_continuous_gripper_commands(
        [-0.006074, 0.000392, 0.9],
        minimum=0.0,
        maximum=0.8,
    )

    assert forwarded == pytest.approx([0.0, 0.000392, 0.8])
    assert raw_min == pytest.approx(-0.006074)
    assert raw_max == pytest.approx(0.9)
    assert clamped_count == 2


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")]])
def test_continuous_gripper_commands_reject_empty_or_non_finite_values(values: list[float]) -> None:
    with pytest.raises(JointRos2RuntimeStateError, match="gripper action"):
        _clamp_continuous_gripper_commands(values, minimum=0.0, maximum=0.8)
