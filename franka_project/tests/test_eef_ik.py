"""Pure-Python tests for the absolute EEF action adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN_SRC = Path(__file__).parents[1] / "ros_lerobot" / "src"
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))

from lerobot_robot_franka_ros.eef_ik import (  # noqa: E402
    EefIkError,
    closed_fraction_to_physical,
    decode_eef_actions,
    quaternion_xyzw_to_matrix,
)
from lerobot_robot_franka_ros.joint_contract import (  # noqa: E402
    EEF_ACTION_NAMES,
    validate_eef_action,
)


def test_eef_rotation_6d_is_orthogonalized_with_right_handed_cross_product() -> None:
    actions = np.asarray(
        [
            [0.4, -0.1, 0.5, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.25],
            [0.5, -0.2, 0.6, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.75],
        ],
        dtype=np.float32,
    )

    positions, rotations, gripper = decode_eef_actions(actions)

    np.testing.assert_allclose(positions, actions[:, :3])
    np.testing.assert_allclose(rotations[0], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rotations[1][:, 0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotations[1][:, 1], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(rotations[1][:, 2], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        np.swapaxes(rotations, 1, 2) @ rotations,
        np.broadcast_to(np.eye(3), rotations.shape),
        atol=1e-12,
    )
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-12)
    np.testing.assert_allclose(gripper, [0.25, 0.75])


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((1, 10), dtype=np.float32),
        np.asarray([[0, 0, 0, 1, 0, 0, 2, 0, 0, 0]], dtype=np.float32),
        np.zeros((1, 9), dtype=np.float32),
        np.asarray([[0, 0, 0, 1, 0, 0, 0, 1, 0, np.nan]], dtype=np.float32),
    ],
)
def test_eef_action_decoder_rejects_invalid_chunks(actions: np.ndarray) -> None:
    with pytest.raises(EefIkError):
        decode_eef_actions(actions)


def test_eef_action_dict_uses_exact_server_vector_order() -> None:
    values = [0.4, -0.1, 0.5, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.25]
    action = dict(zip(EEF_ACTION_NAMES, values, strict=True))

    validated = validate_eef_action(action)

    assert tuple(validated) == EEF_ACTION_NAMES
    assert tuple(validated.values()) == pytest.approx(values)


def test_eef_gripper_fraction_maps_to_physical_range_with_saturation() -> None:
    physical, raw_min, raw_max, clamped_count = closed_fraction_to_physical(
        [-0.1, 0.25, 1.2],
        minimum=0.0,
        maximum=0.8,
    )

    np.testing.assert_allclose(physical, [0.0, 0.2, 0.8])
    assert raw_min == pytest.approx(-0.1)
    assert raw_max == pytest.approx(1.2)
    assert clamped_count == 2


def test_xyzw_quaternion_conversion() -> None:
    half_angle = np.pi / 4.0
    rotation = quaternion_xyzw_to_matrix([0.0, 0.0, np.sin(half_angle), np.cos(half_angle)])
    np.testing.assert_allclose(
        rotation,
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )
