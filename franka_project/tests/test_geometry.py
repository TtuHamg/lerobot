from __future__ import annotations

import numpy as np
import pytest

from franka_eef_pipeline.geometry import (
    decode_relative_action,
    encode_relative_action,
    enforce_quaternion_continuity,
    matrix_to_quaternion_xyzw,
    matrix_to_rotation_6d,
    normalize_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
    rotation_6d_to_matrix,
    rotation_geodesic_angle,
    so3_exp,
    so3_log,
)


def test_quaternion_normalization_and_continuity() -> None:
    trajectory = np.array(
        [
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, -3.0],
            [0.1, 0.0, 0.0, -0.995],
        ]
    )
    continuous = enforce_quaternion_continuity(trajectory)
    np.testing.assert_allclose(np.linalg.norm(continuous, axis=-1), 1.0, atol=1e-12)
    assert np.all(np.sum(continuous[:-1] * continuous[1:], axis=-1) >= 0.0)


def test_quaternion_matrix_random_round_trip() -> None:
    rng = np.random.default_rng(7)
    quaternion = normalize_quaternion_xyzw(rng.normal(size=(1000, 4)))
    matrix = quaternion_xyzw_to_matrix(quaternion)
    recovered = quaternion_xyzw_to_matrix(matrix_to_quaternion_xyzw(matrix))
    assert float(np.max(rotation_geodesic_angle(matrix, recovered))) < 1e-7


def test_rotation_6d_is_first_two_columns_column_major() -> None:
    matrix = so3_exp(np.array([0.2, -0.3, 0.4]))
    encoded = matrix_to_rotation_6d(matrix)
    np.testing.assert_allclose(encoded[:3], matrix[:, 0])
    np.testing.assert_allclose(encoded[3:], matrix[:, 1])
    recovered = rotation_6d_to_matrix(encoded)
    assert float(rotation_geodesic_angle(matrix, recovered)) < 1e-7


@pytest.mark.parametrize(
    "rotation_vector",
    [
        np.zeros(3),
        np.array([1e-10, -2e-10, 3e-10]),
        np.array([np.pi / 2.0, 0.0, 0.0]),
        np.array([0.0, -np.pi + 1e-7, 0.0]),
        np.array([0.3, -0.4, 0.7]),
    ],
)
def test_so3_exp_log_known_round_trip(rotation_vector: np.ndarray) -> None:
    matrix = so3_exp(rotation_vector)
    recovered = so3_exp(so3_log(matrix))
    assert float(rotation_geodesic_angle(matrix, recovered)) < 1e-5


def test_so3_random_round_trip() -> None:
    rng = np.random.default_rng(11)
    axis = rng.normal(size=(2000, 3))
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = rng.uniform(0.0, np.pi - 1e-4, size=(2000, 1))
    matrix = so3_exp(axis * angle)
    recovered = so3_exp(so3_log(matrix))
    assert float(np.max(rotation_geodesic_angle(matrix, recovered))) < 1e-5


def test_relative_action_batch_chunk_round_trip() -> None:
    rng = np.random.default_rng(23)
    batch, chunk = 4, 50
    anchor_position = rng.normal(size=(batch, 3))
    anchor_rotation = so3_exp(rng.normal(scale=0.3, size=(batch, 3)))
    target_position = anchor_position[:, None, :] + rng.normal(scale=0.05, size=(batch, chunk, 3))
    relative_rotation = so3_exp(rng.normal(scale=0.2, size=(batch, chunk, 3)))
    target_rotation = anchor_rotation[:, None, :, :] @ relative_rotation
    target_gripper = rng.uniform(0.0, 1.0, size=(batch, chunk, 1))

    action = encode_relative_action(
        anchor_position,
        anchor_rotation,
        target_position,
        target_rotation,
        target_gripper,
    )
    recovered_position, recovered_rotation, recovered_gripper = decode_relative_action(
        anchor_position,
        anchor_rotation,
        action,
    )

    assert action.shape == (batch, chunk, 7)
    assert float(np.max(np.abs(recovered_position - target_position))) < 1e-6
    assert float(np.max(rotation_geodesic_angle(recovered_rotation, target_rotation))) < 1e-5
    np.testing.assert_allclose(recovered_gripper, target_gripper, atol=1e-12)


@pytest.mark.parametrize(
    "function,value",
    [
        (normalize_quaternion_xyzw, np.zeros(4)),
        (normalize_quaternion_xyzw, np.array([np.nan, 0.0, 0.0, 1.0])),
        (rotation_6d_to_matrix, np.zeros(6)),
        (rotation_6d_to_matrix, np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])),
        (so3_exp, np.array([0.0, np.inf, 0.0])),
        (so3_log, np.zeros((3, 3))),
    ],
)
def test_invalid_geometry_raises(function, value: np.ndarray) -> None:
    with pytest.raises(ValueError):
        function(value)
