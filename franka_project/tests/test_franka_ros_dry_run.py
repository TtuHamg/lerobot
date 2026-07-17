"""Contract tests for the out-of-tree Phase 1 Franka Robot plugin."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN_SRC = Path(__file__).parents[1] / "ros_lerobot" / "src"
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))

from lerobot_robot_franka_ros import (  # noqa: E402
    ABSOLUTE_ACTION_NAMES,
    CAMERA_SHAPE,
    STATE_NAMES,
    FrankaRos,
    FrankaRosConfig,
)
from lerobot.robots.config import RobotConfig  # noqa: E402
from lerobot.robots.utils import make_robot_from_config  # noqa: E402
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError  # noqa: E402


def _write_fixture(path: Path, **overrides: np.ndarray) -> None:
    payload = {
        "state": np.asarray([0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5], dtype=np.float32),
        "camera1": np.zeros(CAMERA_SHAPE, dtype=np.uint8),
        "camera2": np.ones(CAMERA_SHAPE, dtype=np.uint8),
    }
    payload.update(overrides)
    np.savez(path, **payload)


def _config(tmp_path: Path, fixture_path: Path) -> FrankaRosConfig:
    return FrankaRosConfig(
        id="dry-run-test",
        calibration_dir=tmp_path / "calibration",
        dry_run=True,
        fixture_path=fixture_path,
        action_log_path=tmp_path / "actions.jsonl",
    )


def _valid_action() -> dict[str, float]:
    return {
        "target.x": 0.5,
        "target.y": 0.0,
        "target.z": 0.4,
        "target.qx": 0.0,
        "target.qy": 0.0,
        "target.qz": 0.0,
        "target.qw": 1.0,
        "target.gripper.closed_0_1": 0.25,
    }


def test_plugin_registration_factory_and_feature_contract(tmp_path: Path) -> None:
    fixture_path = tmp_path / "observation.npz"
    _write_fixture(fixture_path)
    config = _config(tmp_path, fixture_path)

    assert RobotConfig.get_choice_class("franka_ros") is FrankaRosConfig
    robot = make_robot_from_config(config)
    assert isinstance(robot, FrankaRos)
    assert tuple(robot.observation_features) == (*STATE_NAMES, "camera1", "camera2")
    assert tuple(robot.action_features) == ABSOLUTE_ACTION_NAMES
    assert all(robot.observation_features[name] is float for name in STATE_NAMES)
    assert robot.observation_features["camera1"] == CAMERA_SHAPE
    assert robot.observation_features["camera2"] == CAMERA_SHAPE


def test_lifecycle_fixture_copies_and_jsonl_sink(tmp_path: Path) -> None:
    fixture_path = tmp_path / "observation.npz"
    _write_fixture(fixture_path)
    config = _config(tmp_path, fixture_path)
    robot = FrankaRos(config)

    with pytest.raises(DeviceNotConnectedError):
        robot.get_observation()
    with pytest.raises(DeviceNotConnectedError):
        robot.send_action(_valid_action())
    with pytest.raises(DeviceNotConnectedError):
        robot.disconnect()

    robot.connect()
    assert robot.is_connected
    assert robot.is_calibrated
    with pytest.raises(DeviceAlreadyConnectedError):
        robot.connect()

    observation = robot.get_observation()
    assert tuple(observation[name] for name in STATE_NAMES) == pytest.approx(
        (0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5)
    )
    assert observation["camera1"].shape == CAMERA_SHAPE
    assert observation["camera1"].dtype == np.uint8
    observation["camera1"][0, 0, 0] = 255
    assert robot.get_observation()["camera1"][0, 0, 0] == 0

    sent = robot.send_action(_valid_action())
    assert tuple(sent) == ABSOLUTE_ACTION_NAMES
    log_lines = config.action_log_path.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    record = json.loads(log_lines[0])
    assert record["schema_version"] == 1
    assert record["dry_run"] is True
    assert record["robot_id"] == "dry-run-test"
    assert record["sequence"] == 0
    assert record["monotonic_ns"] > 0
    assert record["action"] == _valid_action()

    robot.disconnect()
    assert not robot.is_connected
    with pytest.raises(DeviceNotConnectedError):
        robot.get_observation()


@pytest.mark.parametrize(
    ("overrides", "error_type", "match"),
    [
        ({"state": np.zeros(9, dtype=np.float32)}, ValueError, "state must have shape"),
        ({"state": np.zeros(10, dtype=np.float64)}, TypeError, "state must have dtype float32"),
        (
            {"state": np.asarray([0.0] * 9 + [np.nan], dtype=np.float32)},
            ValueError,
            "state contains a non-finite",
        ),
        (
            {"state": np.asarray([0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5], dtype=np.float32)},
            ValueError,
            "first column is degenerate",
        ),
        (
            {"state": np.asarray([0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.5], dtype=np.float32)},
            ValueError,
            "columns are collinear",
        ),
        (
            {"state": np.asarray([0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.1], dtype=np.float32)},
            ValueError,
            "gripper state must be in",
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
    robot = FrankaRos(_config(tmp_path, fixture_path))

    with pytest.raises(error_type, match=match):
        robot.connect()
    assert not robot.is_connected


def test_fixture_rejects_missing_and_extra_keys(tmp_path: Path) -> None:
    fixture_path = tmp_path / "bad-keys.npz"
    np.savez(
        fixture_path,
        state=np.zeros(10, dtype=np.float32),
        camera1=np.zeros(CAMERA_SHAPE, dtype=np.uint8),
        unexpected=np.zeros(1, dtype=np.uint8),
    )
    robot = FrankaRos(_config(tmp_path, fixture_path))

    with pytest.raises(ValueError, match="Fixture keys do not match contract"):
        robot.connect()
    assert not robot.is_connected


@pytest.mark.parametrize(
    ("mutate", "error_type", "match"),
    [
        (lambda action: action.pop("target.x"), ValueError, "Action keys do not match contract"),
        (lambda action: action.update({"extra": 0.0}), ValueError, "Action keys do not match contract"),
        (lambda action: action.update({"target.x": np.nan}), ValueError, "must be finite"),
        (lambda action: action.update({"target.x": [0.5]}), TypeError, "must be a real scalar"),
        (lambda action: action.update({"target.qw": 0.0}), ValueError, "quaternion must be unit length"),
        (
            lambda action: action.update({"target.qx": 1.0, "target.qw": 1.0}),
            ValueError,
            "quaternion must be unit length",
        ),
        (
            lambda action: action.update({"target.gripper.closed_0_1": 1.01}),
            ValueError,
            "gripper target must be in",
        ),
    ],
)
def test_action_validation_is_strict(tmp_path: Path, mutate, error_type: type[Exception], match: str) -> None:
    fixture_path = tmp_path / "observation.npz"
    _write_fixture(fixture_path)
    config = _config(tmp_path, fixture_path)
    robot = FrankaRos(config)
    robot.connect()
    action = _valid_action()
    mutate(action)

    with pytest.raises(error_type, match=match):
        robot.send_action(action)
    assert config.action_log_path.read_text(encoding="utf-8") == ""
    robot.disconnect()


def test_non_dry_run_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires dry_run=true"):
        FrankaRosConfig(
            id="unsafe",
            calibration_dir=tmp_path / "calibration",
            dry_run=False,
            fixture_path=tmp_path / "unused.npz",
            action_log_path=tmp_path / "unused.jsonl",
        )


def test_connect_requires_fixture_and_sink_paths(tmp_path: Path) -> None:
    missing_fixture = FrankaRosConfig(
        id="missing-fixture",
        calibration_dir=tmp_path / "calibration-a",
        fixture_path=None,
        action_log_path=tmp_path / "actions.jsonl",
    )
    with pytest.raises(ValueError, match="fixture_path is required"):
        FrankaRos(missing_fixture).connect()

    fixture_path = tmp_path / "observation.npz"
    _write_fixture(fixture_path)
    missing_sink = FrankaRosConfig(
        id="missing-sink",
        calibration_dir=tmp_path / "calibration-b",
        fixture_path=fixture_path,
        action_log_path=None,
    )
    with pytest.raises(ValueError, match="action_log_path is required"):
        FrankaRos(missing_sink).connect()


def test_connect_rejects_fixture_as_action_sink(tmp_path: Path) -> None:
    fixture_path = tmp_path / "observation.npz"
    _write_fixture(fixture_path)
    config = FrankaRosConfig(
        id="same-path",
        calibration_dir=tmp_path / "calibration",
        fixture_path=fixture_path,
        action_log_path=fixture_path,
    )

    with pytest.raises(ValueError, match="must refer to different files"):
        FrankaRos(config).connect()

    # The failed connection must not have appended text to the NPZ fixture.
    with np.load(fixture_path, allow_pickle=False) as fixture:
        assert set(fixture.files) == {"state", "camera1", "camera2"}
